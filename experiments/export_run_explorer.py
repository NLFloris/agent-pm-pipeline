"""Export TraceElephant runs with full step content, for the run explorer.

    python experiments/export_run_explorer.py --data ~/Downloads --out runs_te.json

Each step carries what the agent was asked, what it said back, which model
answered and at what token cost, plus any tool call it made. The expert
attribution is resolved to an event index so the page can mark the step in
place.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

from agentpm.ingest import traceelephant
from data_dir import DATA   # noqa: E402

SUBSETS = ["captain-runs-gaia", "magentic-runs-gaia", "swe-agent-runs-swe-bench"]


def export_run(t):
    steps = []
    for i, e in enumerate(t.ordered()):
        a = e.attributes
        steps.append({
            "i": i,
            "step_id": a.get("step_id"),
            "agent": e.resource,
            "activity": e.activity,
            "kind": "tool" if e.activity not in ("orchestrate", "report") else e.activity,
            "prompt": a.get("prompt"),
            "content": a.get("content"),
            "arguments": a.get("arguments"),
            "model": a.get("model"),
            "tokens_in": a.get("prompt_tokens"),
            "tokens_out": a.get("completion_tokens"),
            "finish": a.get("finish_reason"),
        })
    o = t.outcome
    return {
        "id": t.case_id,
        "subset": o.get("subset"),
        "system": o.get("system"),
        "task": o.get("task"),
        "ground_truth": o.get("ground_truth"),
        "agents": o.get("agents"),
        "mistake_agent": o.get("mistake_agent"),
        "mistake_step": o.get("mistake_step"),
        "mistake_reason": o.get("mistake_reason"),
        "decisive": o.get("decisive_event_index"),
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="runs_te.json")
    ap.add_argument("--per-subset", type=int, default=7)
    a = ap.parse_args()

    root = os.path.join(a.data, "traceelephant/data")
    runs = []
    for sub in SUBSETS:
        path = os.path.join(root, sub)
        if not os.path.isdir(path):
            print(f"  [skip] {sub}")
            continue
        traces = traceelephant.ingest_dir(path)
        for t in traces:
            t.outcome["subset"] = sub
        # prefer runs with several agents and a resolvable annotated step,
        # and keep them short enough to read end to end
        usable = [t for t in traces
                  if t.outcome.get("decisive_event_index") is not None
                  and len(t.outcome.get("agents") or []) >= 2]
        usable.sort(key=lambda t: (-len(t.outcome["agents"]), abs(len(t.events) - 16)))
        chosen = usable[:a.per_subset]
        runs.extend(export_run(t) for t in chosen)
        print(f"  {sub:26} {len(traces):3} runs, {len(usable):3} usable, "
              f"exported {len(chosen)}")

    with open(a.out, "w") as f:
        json.dump({"runs": runs}, f, separators=(",", ":"))
    chars = sum(len(s.get("content") or "") for r in runs for s in r["steps"])
    print(f"\n{len(runs)} runs, {sum(len(r['steps']) for r in runs)} steps, "
          f"{chars // 1000}k characters of text, "
          f"{os.path.getsize(a.out) / 1024:.0f} KB -> {a.out}")
    print("  agents:", dict(collections.Counter(
        s["agent"] for r in runs for s in r["steps"]).most_common(8)))


if __name__ == "__main__":
    sys.exit(main())
