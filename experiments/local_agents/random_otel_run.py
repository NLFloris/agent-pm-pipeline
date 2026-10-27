"""Generate random multi-agent runs with real OpenTelemetry, no model endpoint needed.

    source ~/venvs/localagents/bin/activate
    python experiments/local_agents/random_otel_run.py --runs 6 --seed 1 --out results/otel_demo

Four AutoGen agents in a Swarm: an Orchestrator that hands work to a
FileSurfer, a Coder and an Analyst, each with real Python tools in a sandbox,
each handing back when done. Their "model" is a ReplayChatCompletionClient fed
from a plan drawn with a seed: how many delegations, to whom, and which tool
calls each visit makes. So the decisions are random and reproducible, while
everything that records them is real: AutoGen's own GenAI spans, the
OpenTelemetry SDK, OTLP/JSON, and the pipeline's `otel` ingester afterwards.

Output in --out:
  capture.jsonl   the OTLP/JSON spans, what any collector would have received
  harness.json    the same runs from AutoGen's message stream (has arguments)
  plans.json      the drawn plan per run, to compare against what was recorded

Then: python experiments/otel_report.py results/otel_demo/capture.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from autogen_agentchat.agents import AssistantAgent                       # noqa: E402
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination  # noqa: E402
from autogen_agentchat.teams import Swarm                                 # noqa: E402
from autogen_core import FunctionCall                                     # noqa: E402
from autogen_core.models import CreateResult, ModelInfo, RequestUsage     # noqa: E402
from autogen_ext.models.replay import ReplayChatCompletionClient          # noqa: E402

from otel_export import install, workflow_span                            # noqa: E402
from run_magentic import SEED_FILES, make_tools, record                   # noqa: E402

TOOL_MODEL = ModelInfo(vision=False, function_calling=True, json_output=False,
                       family="unknown", structured_output=False)

NORTH = ("import csv\nrows = list(csv.DictReader(open('sales.csv')))\n"
         "print(sum(int(r['units']) * float(r['price']) for r in rows if r['region'] == 'north'))")
GROWTH = ("import csv\nrows = list(csv.DictReader(open('sales.csv')))\n"
          "q = {(r['region'], r['quarter']): int(r['units']) for r in rows}\n"
          "print({r: (q[(r, 'Q2')] - q[(r, 'Q1')]) / q[(r, 'Q1')] for r in ['north', 'south', 'east']})")
CHECK = "exec(open('calc.py').read()); print(revenue(120, 9.5), growth(120, 140))"

# what each specialist may do on a visit; the plan draws from these
TOOL_POOL = {
    "FileSurfer": [("list_files", {}), ("read_file", {"path": "sales.csv"}),
                   ("read_file", {"path": "notes.md"}), ("read_file", {"path": "calc.py"})],
    "Coder": [("run_python", {"code": NORTH}), ("run_python", {"code": GROWTH}),
              ("write_file", {"path": "answer.txt", "content": "north Q1 revenue 1140.0"}),
              ("write_file", {"path": "growth.txt", "content": "north +16.7%, south +68.4%, east -6.2%"})],
    "Analyst": [("read_file", {"path": "sales.csv"}), ("run_python", {"code": CHECK}),
                ("run_python", {"code": NORTH})],
}
SPECIALISTS = ["FileSurfer", "Coder", "Coder", "Analyst"]   # Coder twice: it is asked for more often


def plan(seed: int) -> list[dict]:
    rng = random.Random(seed)
    visits = []
    for _ in range(rng.randint(3, 6)):
        agent = rng.choice(SPECIALISTS)
        calls = [rng.choice(TOOL_POOL[agent]) for _ in range(rng.randint(1, 3))]
        visits.append({"agent": agent, "calls": [{"tool": t, "arguments": a} for t, a in calls]})
    return visits


def _call(cid, name, args):
    return CreateResult(finish_reason="function_calls",
                        content=[FunctionCall(id=cid, name=name, arguments=json.dumps(args))],
                        usage=RequestUsage(prompt_tokens=10, completion_tokens=5), cached=False)


def _replay(responses):
    return ReplayChatCompletionClient(responses, model_info=TOOL_MODEL)


def scripts(visits):
    """One response list per agent, in the order each agent will be asked."""
    orchestrator, specialist = [], {"FileSurfer": [], "Coder": [], "Analyst": []}
    n = 0
    for i, v in enumerate(visits):
        orchestrator.append(_call(f"h{i}", f"transfer_to_{v['agent'].lower()}", {}))
        for c in v["calls"]:
            n += 1
            specialist[v["agent"]].append(_call(f"c{n}", c["tool"], c["arguments"]))
        specialist[v["agent"]].append(_call(f"b{i}", "transfer_to_orchestrator", {}))
    orchestrator.append("All delegated work is in. DONE")
    return orchestrator, specialist


async def run_one(run_id, visits, sandbox):
    list_files, read_file, write_file, run_python = make_tools(sandbox)
    orch, spec = scripts(visits)
    messages = []
    with workflow_span("swarm_demo", run_id):
        team = Swarm(
            participants=[
                AssistantAgent("Orchestrator", model_client=_replay(orch),
                               handoffs=["FileSurfer", "Coder", "Analyst"],
                               description="Plans the work and delegates it."),
                AssistantAgent("FileSurfer", model_client=_replay(spec["FileSurfer"]),
                               tools=[list_files, read_file], handoffs=["Orchestrator"],
                               description="Finds and reads files."),
                AssistantAgent("Coder", model_client=_replay(spec["Coder"]),
                               tools=[write_file, run_python], handoffs=["Orchestrator"],
                               description="Writes and runs Python."),
                AssistantAgent("Analyst", model_client=_replay(spec["Analyst"]),
                               tools=[read_file, run_python], handoffs=["Orchestrator"],
                               description="Re-checks claims against the data."),
            ],
            termination_condition=TextMentionTermination("DONE") | MaxMessageTermination(120),
        )
        async for msg in team.run_stream(task="Report Q1 revenue for the north region and the Q1 to Q2 growth per region."):
            if hasattr(msg, "messages"):
                continue
            messages.append(record(msg))
    return messages


async def main_async(a):
    os.makedirs(a.out, exist_ok=True)
    capture = os.path.join(a.out, "capture.jsonl")
    if os.path.exists(capture):
        os.remove(capture)
    provider, exporter = install(service_name="random-swarm-demo", path=capture,
                                 endpoint=a.otel_endpoint, batch=False)
    runs, plans = [], {}
    for i in range(a.runs):
        seed = a.seed + i
        run_id = f"demo-{seed:03d}"
        visits = plan(seed)
        plans[run_id] = visits
        sandbox = tempfile.mkdtemp(prefix="agentpm_demo_")
        for name, body in SEED_FILES.items():
            with open(os.path.join(sandbox, name), "w", encoding="utf-8") as f:
                f.write(body)
        try:
            messages = await run_one(run_id, visits, sandbox)
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)
        n_calls = sum(len(v["calls"]) for v in visits)
        print(f"  {run_id}: {len(visits)} delegations ({', '.join(v['agent'] for v in visits)}), "
              f"{n_calls} tool calls, {len(messages)} messages")
        runs.append({"run_id": run_id, "seed": seed, "messages": messages})
    provider.force_flush()
    provider.shutdown()
    with open(os.path.join(a.out, "harness.json"), "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=1)
    with open(os.path.join(a.out, "plans.json"), "w", encoding="utf-8") as f:
        json.dump(plans, f, indent=1)
    print(f"\n{exporter.exported} spans exported, {exporter.failed} failed")
    print(f"wrote {capture}\nwrote {os.path.join(a.out, 'harness.json')}\nwrote {os.path.join(a.out, 'plans.json')}")
    print(f"\nnext: python experiments/otel_report.py {capture} --out {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1, help="first seed; run i uses seed + i")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "otel_demo"))
    ap.add_argument("--otel-endpoint", help="also post the spans to a collector")
    a = ap.parse_args()
    asyncio.run(main_async(a))


if __name__ == "__main__":
    sys.exit(main())
