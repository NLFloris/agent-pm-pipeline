"""The same tasks and team as run_magentic.py, on Microsoft Agent Framework.

    python experiments/local_agents/run_maf.py --only l01_read_csv l02_revenue l04_cross_check

AutoGen, the framework of run_magentic.py, sets neither the arguments nor the
results on its OpenTelemetry convention spans and opens no span for its
orchestrator. This runner tests whether the framework's successor delivers them.
It keeps the agents, tools, prompts and model of run_magentic.py, and it records
only the OpenTelemetry spans, with the framework's option for sensitive data on.
The orchestrator's prompts are the framework's own, so the runs are comparable in
what is recorded, not call by call. Compare with compare_maf.py.
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

from agent_framework import Agent
from agent_framework.observability import enable_sensitive_telemetry
from agent_framework.openai import OpenAIChatCompletionClient
from agent_framework_orchestrations import MagenticBuilder
from opentelemetry import trace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from otel_export import install  # noqa: E402

# copied from run_magentic.py, which cannot be imported here since it needs AutoGen
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


def make_tools(sandbox):
    """Plain functions; the framework derives each tool's schema from the signature."""

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


def build_team(client, sandbox, max_rounds, no_cache):
    list_files, read_file, write_file, run_python = make_tools(sandbox)
    # the BFH endpoint is a LiteLLM proxy that answers identical requests from a cache
    options = {"extra_body": {"cache": {"no-cache": True}}} if no_cache else None

    def agent(name, description, instructions, tools=None):
        return Agent(client=client, name=name, description=description, instructions=instructions,
                     tools=tools, default_options=options)

    return MagenticBuilder(
        participants=[
            agent("FileSurfer", "Finds and reads files in the working directory.",
                  "You inspect files. Use your tools; never guess at a file's contents.",
                  [list_files, read_file]),
            agent("Coder", "Writes and runs Python to compute answers.",
                  "You write and run Python. Always run the code and report what it actually printed.",
                  [write_file, run_python]),
            agent("Analyst", "Checks claims against the data and verifies results.",
                  "You verify. Re-read the data or re-run a calculation before agreeing with anyone.",
                  [read_file, run_python]),
        ],
        manager_agent=agent("Orchestrator", "Plans the work and hands it to the team.",
                            "You coordinate a team of agents."),
        max_round_count=max_rounds,
    ).build()


async def run_task(task, client, sandbox, max_rounds, no_cache):
    workflow = build_team(client, sandbox, max_rounds, no_cache)
    tracer = trace.get_tracer("agentpm-harness")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.monotonic()
    error, final = None, None
    # one root span per run, as in otel_export.workflow_span, so that a run is one case
    with tracer.start_as_current_span(
        f"invoke_workflow {task['id']}",
        attributes={"gen_ai.operation.name": "invoke_workflow", "gen_ai.workflow.name": task["id"],
                    "gen_ai.conversation.id": task["id"], "gen_ai.system": "agent_framework"}):
        try:
            async for event in workflow.run(task["prompt"], stream=True):
                if event.type == "output":
                    final = str(event.data)[:4000]
        except Exception as exc:                   # a model may stall or emit an unparseable call
            error = f"{type(exc).__name__}: {exc}"
    return {"task_id": task["id"], "shape": task.get("shape"), "prompt": task["prompt"],
            "started_at": started, "seconds": round(time.monotonic() - t0, 1),
            "error": error, "final": final}


async def main_async(a):
    with open(os.path.join(HERE, "tasks.json"), encoding="utf-8") as f:
        tasks = json.load(f)["tasks"]
    if a.only:
        tasks = [t for t in tasks if t["id"] in set(a.only)]
    os.makedirs(a.out, exist_ok=True)
    capture = os.path.join(a.out, "capture.jsonl")
    if os.path.exists(capture):
        os.remove(capture)
    provider, exporter = install(service_name="maf-harness", path=capture, batch=False)
    enable_sensitive_telemetry()
    client = OpenAIChatCompletionClient(model=a.model, base_url=a.base_url, api_key=a.api_key)
    for i, task in enumerate(tasks, 1):
        sandbox = os.path.abspath(os.path.join(a.out, task["id"]))
        if os.path.exists(sandbox):
            shutil.rmtree(sandbox)
        os.makedirs(sandbox)
        for name, body in SEED_FILES.items():
            with open(os.path.join(sandbox, name), "w", encoding="utf-8") as f:
                f.write(body)
        print(f"[{i}/{len(tasks)}] {task['id']} ({task.get('shape')}) ...", flush=True)
        run = await run_task(task, client, sandbox, a.max_rounds, a.no_cache)
        run["model"] = a.model
        with open(os.path.join(a.out, f"{task['id']}.json"), "w", encoding="utf-8") as f:
            json.dump(run, f, indent=1)
        print(f"    {run['seconds']}s" + (f", ERROR {run['error']}" if run["error"] else ""), flush=True)
    provider.force_flush()
    provider.shutdown()
    print(f"OpenTelemetry: {exporter.exported} spans exported, {exporter.failed} failed, to {capture}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="gpt-oss:120b")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1"))
    ap.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", "not-needed"))
    ap.add_argument("--out", default="results/local_maf")
    ap.add_argument("--only", nargs="*", help="task ids; all eight when left out")
    ap.add_argument("--max-rounds", type=int, default=24, help="the same limit as run_magentic.py")
    ap.add_argument("--no-cache", action="store_true", help="ask the proxy to sample afresh")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
