"""What the OpenTelemetry route delivers on AutoGen and on Microsoft Agent Framework.

    python experiments/local_agents/compare_maf.py > results/local_maf/comparison.txt

For each task run by run_maf.py: the harness recording and the OpenTelemetry
capture of the AutoGen run, against the capture of the Agent Framework run. All
of it goes through the pipeline's generic OpenTelemetry ingester, and the engine
checks the demonstration's reference model on both captures. Different runs of
a model differ anyway, so differing tool sequences say nothing about a framework.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from agentpm.analysis.conformance import check  # noqa: E402
from agentpm.ingest import otel  # noqa: E402
from agentpm.reference import Reference  # noqa: E402

GENERIC = {"create_agent", "invoke_agent", "invoke_workflow", "llm_call"}


def harness_calls(task_id):
    with open(os.path.join(ROOT, "results/local_otel_live", f"{task_id}.json"), encoding="utf-8") as f:
        run = json.load(f)
    calls = [(m["source"], item["name"]) for m in run["messages"] if m["type"] == "ToolCallRequestEvent"
             for item in m["payload"]["items"] if item.get("kind") == "call"]
    return run, calls


def summary(trace, reference):
    tools = [e for e in trace.ordered() if e.activity not in GENERIC]
    verdict = check(trace, reference)
    return {"tools": [e.activity for e in tools],
            "agents": sorted({str(e.resource) for e in trace.events} - {"root"}),
            "with args": sum(1 for e in tools if e.attributes.get("arguments")),
            "with result": sum(1 for e in tools if e.attributes.get("result")),
            "model calls": sum(1 for e in trace.events if e.activity == "llm_call"),
            "conforms": verdict.compliant, "fitness": round(verdict.fitness, 2)}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results/local_maf")
    with open(os.path.join(ROOT, "experiments/otel_demo_reference.json"), encoding="utf-8") as f:
        d = json.load(f)
    reference = Reference(name=d["name"], required=d["required"], precedence=d["precedence"])
    autogen = {t.case_id: t for t in otel.ingest(os.path.join(ROOT, "results/otel_live/capture.jsonl"))}
    maf = {t.case_id: t for t in otel.ingest(os.path.join(out, "capture.jsonl"))
           if t.events}
    for task_id in sorted(maf):
        run, calls = harness_calls(task_id)
        print(f"\n=== {task_id}")
        print(f"  AutoGen, harness recording  {len(calls)} tool calls, agents "
              f"{sorted({a for a, _ in calls})}: {[n for _, n in calls]}")
        for name, traces in (("AutoGen, OpenTelemetry", autogen), ("Agent Framework, OpenTelemetry", maf)):
            if task_id in traces:
                print(f"  {name}")
                for k, v in summary(traces[task_id], reference).items():
                    print(f"    {k:12} {v}")
    other = sorted(set(maf) - {p[:-5] for p in os.listdir(os.path.join(ROOT, "results/local_otel_live"))})
    if other:
        print("\ncases without a counterpart:", other)
    print()
    for task_id in sorted(maf):
        with open(os.path.join(out, f"{task_id}.json"), encoding="utf-8") as f:
            run = json.load(f)
        print(f"--- {task_id} final answer, {run['seconds']}s, error {run['error']}\n{run['final']}\n")


if __name__ == "__main__":
    main()
