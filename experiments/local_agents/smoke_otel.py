"""Prove the OpenTelemetry route end to end with a real framework and no model.

    source ~/venvs/localagents/bin/activate
    python experiments/local_agents/smoke_otel.py --out results/otel_probe

Two AutoGen AssistantAgents with real Python tools take turns in a
RoundRobinGroupChat. Their model is a ReplayChatCompletionClient that returns
scripted responses, so the run needs no endpoint and is reproducible, while
everything else is the real framework: AutoGen's own GenAI spans, the
OpenTelemetry SDK, OTLP/JSON over HTTP to the pipeline's collector, and the
pipeline's OpenTelemetry ingester on the capture.

The harness records the same run through AutoGen's message stream, as
run_magentic.py does. The script then compares the two views event by event:
what the standard route delivers (names, order, attribution) and what it does
not (arguments, results), measured on the same run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from autogen_agentchat.agents import AssistantAgent          # noqa: E402
from autogen_agentchat.teams import RoundRobinGroupChat      # noqa: E402
from autogen_core.models import CreateResult, ModelInfo, RequestUsage   # noqa: E402
from autogen_core import FunctionCall                        # noqa: E402
from autogen_ext.models.replay import ReplayChatCompletionClient  # noqa: E402

from otel_export import install, workflow_span               # noqa: E402
from run_magentic import SEED_FILES, make_tools, record      # noqa: E402


TOOL_MODEL = ModelInfo(vision=False, function_calling=True, json_output=False,
                       family="unknown", structured_output=False)


def _replay(responses):
    return ReplayChatCompletionClient(responses, model_info=TOOL_MODEL)


def _call(cid, name, args):
    return CreateResult(finish_reason="function_calls",
                        content=[FunctionCall(id=cid, name=name, arguments=json.dumps(args))],
                        usage=RequestUsage(prompt_tokens=10, completion_tokens=5), cached=False)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(port, seconds=10):
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


async def run(sandbox, task_id):
    list_files, read_file, write_file, run_python = make_tools(sandbox)
    messages = []
    # agents are created inside the workflow span, as run_magentic.py does, so
    # that AutoGen's create_agent spans belong to the run's trace
    with workflow_span("roundrobin_replay", task_id):
        surfer = AssistantAgent(
            "FileSurfer", tools=[list_files, read_file], description="reads files",
            model_client=_replay([
                _call("call_1", "list_files", {}),
                _call("call_2", "read_file", {"path": "sales.csv"}),
                "FileSurfer: sales.csv has six rows.",
            ]))
        coder = AssistantAgent(
            "Coder", tools=[write_file, run_python], description="runs code",
            model_client=_replay([
                _call("call_3", "run_python", {"code": "print(120*9.5)"}),
                _call("call_4", "write_file", {"path": "answer.txt", "content": "1140.0"}),
                "Coder: wrote the answer.",
            ]))
        team = RoundRobinGroupChat([surfer, coder], max_turns=6)
        async for msg in team.run_stream(task="Compute Q1 revenue for the north region."):
            if hasattr(msg, "messages"):
                continue
            messages.append(record(msg))
    return messages


def harness_tool_calls(messages):
    out = []
    for m in messages:
        for item in m["payload"].get("items", []):
            if item.get("kind") == "call":
                out.append({"tool": item["name"], "agent": m["source"], "arguments": item["arguments"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "results", "otel_probe"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    capture = os.path.join(a.out, "autogen_roundrobin_replay.jsonl")
    harness_path = os.path.join(a.out, "autogen_roundrobin_replay.harness.json")
    if os.path.exists(capture):
        os.remove(capture)

    port = _free_port()
    collector = subprocess.Popen([sys.executable, os.path.join(REPO, "experiments", "otel_capture", "collector.py"),
                                  "--out", capture, "--port", str(port), "--quiet"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _wait(port), "collector did not start"
        provider, exporter = install(service_name="smoke-roundrobin",
                                     endpoint=f"http://127.0.0.1:{port}", batch=False)
        sandbox = tempfile.mkdtemp(prefix="agentpm_smoke_")
        for name, body in SEED_FILES.items():
            with open(os.path.join(sandbox, name), "w", encoding="utf-8") as f:
                f.write(body)
        try:
            messages = asyncio.run(run(sandbox, "smoke-001"))
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)
        provider.force_flush()
        provider.shutdown()
        time.sleep(0.3)
    finally:
        collector.terminate()
        collector.wait(timeout=5)

    with open(harness_path, "w", encoding="utf-8") as f:
        json.dump({"task_id": "smoke-001", "messages": messages}, f, indent=1)

    from agentpm.ingest import otel
    traces = otel.ingest(capture)
    n_payloads = sum(1 for _ in open(capture, encoding="utf-8"))
    print(f"collector received {n_payloads} OTLP/JSON payloads; exporter sent {exporter.exported} spans, "
          f"{exporter.failed} failed")
    print(f"pipeline ingested {len(traces)} case(s) from the capture")
    for t in traces:
        print(f"  case {t.case_id}: {len(t.events)} events, agents {t.outcome['agents']}, "
              f"{t.outcome['spans']} spans in the trace")
        for e in t.ordered():
            print(f"    {e.timestamp:2}  {e.activity:18} by {e.resource:12}"
                  + (f"  -> {e.attributes['to']}" if e.attributes.get("to") else "")
                  + (f"  call {e.attributes.get('call_id')}" if e.attributes.get("call_id") else "")
                  + (f"  args {e.attributes.get('arguments')!r}" if e.activity not in ("invoke_agent", "invoke_workflow") else ""))

    otel_calls = [{"tool": e.activity, "agent": e.resource, "arguments": e.attributes.get("arguments")}
                  for t in traces for e in t.ordered() if e.attributes.get("call_id")]
    h_calls = harness_tool_calls(messages)
    print("\nsame run, two recordings")
    print(f"  {'':3} {'harness: tool / agent / arguments':46} {'OpenTelemetry: tool / agent / arguments'}")
    for i, (h, o) in enumerate(zip(h_calls, otel_calls)):
        print(f"  {i:3} {h['tool'] + ' / ' + h['agent'] + ' / ' + str(h['arguments']):46} "
              f"{o['tool']} / {o['agent']} / {o['arguments']!r}")
    same = [(h["tool"], h["agent"]) for h in h_calls] == [(o["tool"], o["agent"]) for o in otel_calls]
    print(f"\n  tool names and acting agents agree in order: {same}")
    print(f"  arguments present: harness {sum(1 for h in h_calls if h['arguments'])}/{len(h_calls)}, "
          f"OpenTelemetry {sum(1 for o in otel_calls if o['arguments'])}/{len(otel_calls)}")
    print(f"\nwrote {capture}\nwrote {harness_path}")
    return 0 if same and len(traces) == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
