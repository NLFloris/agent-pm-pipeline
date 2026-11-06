"""Run Magentic-One against a local model server and record every message.

    python experiments/local_agents/run_magentic.py --model qwen3.8-flash-next

This records a self-instrumented multi-agent corpus. It uses the same
framework as the Magentic-One subset of TraceElephant, so the traces are
structurally comparable, but because the harness is ours the recording keeps
what the public corpora and Claude Code's telemetry both drop:

  * which agent performed each action, not just that one was performed
  * full tool arguments, not their size in bytes
  * the model's own reasoning text, kept as an attribute and never as an
    activity (requirement R2 of the thesis)
  * wall-clock start and end per message, so durations are real

The team is three AssistantAgents with explicit Python tools, under the
Magentic-One orchestrator. AutoGen's own FileSurfer and CodeExecutorAgent were
tried first and rejected: they resolve their tool use internally and surface
only a TextMessage, so a whole run came out as `orchestrate` and `report` with
no named actions at all. AssistantAgent instead emits ToolCallRequestEvent and
ToolCallExecutionEvent, which is what gives the log an activity alphabet.

The web surfer is deliberately absent: it needs a browser stack, and a local
model driving a browser fails for reasons that say nothing about process
structure.

Needs an OpenAI-compatible endpoint. Either a local server (LM Studio:
Developer tab, Start Server, default http://localhost:1234/v1) or a hosted one,
for example the BFH-TI MLMP inference API, which serves the same model line-up
on their hardware:

    export LLM_BASE_URL=https://inference.mlmp.ti.bfh.ch/api/v1
    export LLM_API_KEY=<key from Settings, Account, API keys>

`--model` must match the id the endpoint reports.

Sub-agents. With `--subagents` the Analyst has no tools of its own. It has two
sub-agents instead, each wrapped as a tool (AutoGen's AgentTool) and each with
its own tools: DataChecker (list_files, read_file) and Recalculator
(run_python). Calling one starts it on a task; its tool calls stream into the
run under its own name, between the Analyst's call and the result. The run
file records which agent owns which sub-agent, so the ingester can tell a
sub-agent's start from an ordinary tool call. Use it with
`--tasks experiments/local_agents/tasks_claims.json`.

OpenTelemetry. With `--otel-file capture.jsonl` and/or `--otel-endpoint
http://127.0.0.1:4318` the run is also exported as OTLP/JSON spans: AutoGen's
own invoke_agent and execute_tool spans under one invoke_workflow span per
task (see otel_export.py). The pipeline reads the capture with
`python -m agentpm.cli inventory capture.jsonl --format otel`. Off by default.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import shutil
import sys
import time

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.tools import AgentTool
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

HERE = os.path.dirname(os.path.abspath(__file__))

SEED_FILES = {
    "sales.csv": "region,quarter,units,price\n"
                 "north,Q1,120,9.5\nnorth,Q2,140,9.5\nsouth,Q1,95,11.0\n"
                 "south,Q2,160,11.0\neast,Q1,80,9.5\neast,Q2,75,12.25\n",
    "notes.md": "# Field notes\n\n- The south region changed price in Q1.\n"
                "- East sales fell in Q2 after the price rise.\n"
                "- North held price steady all year.\n",
    "calc.py": 'def revenue(units, price):\n    return units * price\n\n\n'
               'def growth(a, b):\n    # returns percent change from a to b\n'
               '    return (b - a) / a\n',
}


def serialise(content):
    """Message payloads are text, function calls or execution results."""
    if isinstance(content, str):
        return {"text": content}
    if isinstance(content, list):
        out = []
        for item in content:
            if hasattr(item, "name") and hasattr(item, "arguments"):
                out.append({"kind": "call", "id": getattr(item, "id", None),
                            "name": item.name, "arguments": item.arguments})
            elif hasattr(item, "call_id"):
                out.append({"kind": "result", "call_id": item.call_id,
                            "name": getattr(item, "name", None),
                            "is_error": bool(getattr(item, "is_error", False)),
                            "content": str(item.content)})
            else:
                out.append({"kind": "other", "repr": str(item)[:2000]})
        return {"items": out}
    return {"repr": str(content)[:2000]}


def record(msg):
    usage = getattr(msg, "models_usage", None)
    created = getattr(msg, "created_at", None)
    return {
        "type": getattr(msg, "type", msg.__class__.__name__),
        "source": getattr(msg, "source", None),
        "created_at": created.isoformat() if hasattr(created, "isoformat") else created,
        "received_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "payload": serialise(getattr(msg, "content", None)),
    }


def build_team(model, base_url, api_key, timeout, no_cache=False):
    client = OpenAIChatCompletionClient(
        model=model, base_url=base_url, api_key=api_key, timeout=timeout,
        # a local model is not a known OpenAI id, so its capabilities must be declared
        model_info=ModelInfo(vision=False, function_calling=True, json_output=True,
                             family="unknown", structured_output=False,
                             multiple_system_messages=True),
        # the BFH endpoint is a LiteLLM proxy that answers identical requests from
        # a cache; without this, two runs of one task can share an answer
        **({"extra_body": {"cache": {"no-cache": True}}} if no_cache else {}),
    )
    return client


def make_tools(sandbox):
    """Plain functions; AutoGen derives each tool's schema from the signature."""

    def _safe(path: str) -> str:
        full = os.path.abspath(os.path.join(sandbox, path))
        if not full.startswith(os.path.abspath(sandbox)):
            raise ValueError("path escapes the sandbox")
        return full

    async def list_files() -> str:
        """List the files in the working directory with their sizes in bytes."""
        return "\n".join(f"{n}  {os.path.getsize(os.path.join(sandbox, n))} bytes"
                          for n in sorted(os.listdir(sandbox))
                          if os.path.isfile(os.path.join(sandbox, n)))

    async def read_file(path: str) -> str:
        """Read a file in the working directory and return its full contents."""
        with open(_safe(path), encoding="utf-8") as f:
            return f.read()

    async def write_file(path: str, content: str) -> str:
        """Write content to a file in the working directory, replacing it."""
        with open(_safe(path), "w", encoding="utf-8") as f:
            f.write(content)
        return f"wrote {len(content)} characters to {path}"

    async def run_python(code: str) -> str:
        """Run Python code in the working directory and return stdout and stderr."""
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code, cwd=sandbox,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            return "timed out after 30 seconds"
        return (out.decode()[:4000] or "(no output)") + (
            "\nstderr:\n" + err.decode()[:2000] if err else "")

    return list_files, read_file, write_file, run_python


async def run_task(task, sandbox, client, max_turns, otel=False, subagents=False):
    if otel:
        from otel_export import workflow_span
        with workflow_span("magentic_one", task["id"]):
            return await _run_task(task, sandbox, client, max_turns, subagents)
    return await _run_task(task, sandbox, client, max_turns, subagents)


SUBAGENTS = {"DataChecker": {"parent": "Analyst", "tools": ["list_files", "read_file"]},
             "Recalculator": {"parent": "Analyst", "tools": ["run_python"]}}


def make_analyst(client, list_files, read_file, run_python, subagents):
    """The Analyst verifies, either with its own tools or through two sub-agents."""
    if not subagents:
        return AssistantAgent("Analyst", model_client=client,
                              tools=[read_file, run_python],
                              description="Checks claims against the data and verifies results.",
                              system_message="You verify. Re-read the data or re-run a "
                                             "calculation before agreeing with anyone.")
    # a sub-agent answers in text once its tools are done, so it gets a few rounds
    checker = AssistantAgent("DataChecker", model_client=client,
                             tools=[list_files, read_file], max_tool_iterations=4,
                             description="Re-reads the raw data files and quotes the exact "
                                         "rows that bear on a claim.",
                             system_message="You check data. Read the files with your tools "
                                            "and quote the rows that matter. Do not compute.")
    recalc = AssistantAgent("Recalculator", model_client=client,
                            tools=[run_python], max_tool_iterations=4,
                            description="Recomputes a number from scratch in Python, reading "
                                        "sales.csv itself, and reports what the code printed.",
                            system_message="You recompute. Write Python that reads the data "
                                           "file itself, run it, and report what it printed.")
    return AssistantAgent("Analyst", model_client=client,
                          tools=[AgentTool(checker), AgentTool(recalc)],
                          max_tool_iterations=3, reflect_on_tool_use=True,
                          description="Verifies a result independently, by sending sub-agents "
                                      "to re-read the data and to recompute the numbers.",
                          system_message="You verify. You have no file or code tools: send "
                                         "DataChecker to quote the relevant rows and Recalculator "
                                         "to recompute the number, then say whether the result holds.")


async def _run_task(task, sandbox, client, max_turns, subagents=False):
    list_files, read_file, write_file, run_python = make_tools(sandbox)
    team = MagenticOneGroupChat(
        participants=[
            AssistantAgent("FileSurfer", model_client=client,
                           tools=[list_files, read_file],
                           description="Finds and reads files in the working directory.",
                           system_message="You inspect files. Use your tools; never guess "
                                          "at a file's contents."),
            AssistantAgent("Coder", model_client=client,
                           tools=[write_file, run_python],
                           description="Writes and runs Python to compute answers.",
                           system_message="You write and run Python. Always run the code "
                                          "and report what it actually printed."),
            make_analyst(client, list_files, read_file, run_python, subagents),
        ],
        model_client=client,
        max_turns=max_turns,
    )
    messages = []
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.monotonic()
    error = None
    try:
        async for msg in team.run_stream(task=task["prompt"]):
            if hasattr(msg, "messages"):          # the final TaskResult
                continue
            messages.append(record(msg))
    except Exception as exc:                       # a local model may stall or
        error = f"{type(exc).__name__}: {exc}"      # emit an unparseable call
    run = {
        "task_id": task["id"], "shape": task.get("shape"), "prompt": task["prompt"],
        "started_at": started, "seconds": round(time.monotonic() - t0, 1),
        "error": error, "messages": messages,
    }
    if task.get("type"):
        run["task_type"] = task["type"]
    if subagents:
        run["subagents"] = SUBAGENTS
    return run


async def main_async(a):
    with open(a.tasks, encoding="utf-8") as f:
        suite = json.load(f)
    tasks = suite["tasks"]
    for t in tasks:                                # tasks of one type share a template
        t.setdefault("prompt", suite.get("prompt_template", "").format(**t))
    if a.only:
        tasks = [t for t in tasks if t["id"] in set(a.only)]
    os.makedirs(a.out, exist_ok=True)
    client = build_team(a.model, a.base_url, a.api_key, a.timeout, a.no_cache)
    otel = bool(a.otel_file or a.otel_endpoint)
    provider = exporter = None
    if otel:
        sys.path.insert(0, HERE)
        from otel_export import install
        provider, exporter = install(endpoint=a.otel_endpoint, path=a.otel_file)
        print(f"OpenTelemetry on: {a.otel_endpoint or ''} {a.otel_file or ''}".rstrip())

    for i, task in enumerate(tasks, 1):
        sandbox = os.path.abspath(os.path.join(a.out, task["id"]))
        if os.path.exists(sandbox):
            shutil.rmtree(sandbox)
        os.makedirs(sandbox)
        for name, body in SEED_FILES.items():
            with open(os.path.join(sandbox, name), "w", encoding="utf-8") as f:
                f.write(body)

        print(f"[{i}/{len(tasks)}] {task['id']} ({task.get('shape')}) ...", flush=True)
        run = await run_task(task, sandbox, client, a.max_turns, otel=otel, subagents=a.subagents)
        run["model"] = a.model
        run["sandbox"] = sandbox.replace(os.path.expanduser("~"), "~", 1)   # no home folder in the record
        run["no_cache"] = a.no_cache             # whether the proxy was asked to sample afresh
        path = os.path.join(a.out, f"{task['id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run, f, indent=1)
        agents = sorted({m["source"] for m in run["messages"] if m["source"]})
        print(f"    {len(run['messages'])} messages, {run['seconds']}s, "
              f"agents {agents}" + (f", ERROR {run['error']}" if run["error"] else ""))
    await client.close()
    if provider is not None:
        provider.force_flush()
        provider.shutdown()
        print(f"OpenTelemetry: {exporter.exported} spans exported, {exporter.failed} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model id as the local server reports it")
    ap.add_argument("--base-url",
                    default=os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1"),
                    help="OpenAI-compatible endpoint; LM Studio or a hosted one")
    ap.add_argument("--api-key",
                    default=os.environ.get("LLM_API_KEY", "not-needed"),
                    help="read from LLM_API_KEY so it stays out of shell history")
    ap.add_argument("--tasks", default=os.path.join(HERE, "tasks.json"))
    ap.add_argument("--out", default=os.path.expanduser("~/local_agent_runs"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--max-turns", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--otel-file", help="also write the run as OTLP/JSON spans to this file")
    ap.add_argument("--otel-endpoint", help="also post the spans to an OTLP/HTTP JSON collector")
    ap.add_argument("--subagents", action="store_true",
                    help="the Analyst verifies through two sub-agents with their own tools")
    ap.add_argument("--no-cache", action="store_true",
                    help="ask a LiteLLM proxy to sample afresh instead of answering from its cache")
    a = ap.parse_args()
    asyncio.run(main_async(a))
    print(f"\nwrote runs to {a.out}")


if __name__ == "__main__":
    sys.exit(main())
