"""Export τ² runs with their conformance verdict, for the inspection viewer.

    python experiments/export_conformance_viewer.py --out runs_conformance.json

Picks runs across all four cells of (conformant yes/no) x (harness reward 0/1),
so the viewer shows where the compliance signal agrees with the outcome label
and, just as importantly, where it does not.

Per run it writes what the reference required, what the trace actually contains,
and every violation the engine localised, so a reader can check the verdict
against the evidence instead of trusting the number.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

from agentpm.analysis.conformance import check
from agentpm.ingest import tau2
from agentpm.reference import Reference
from data_dir import DATA   # noqa: E402

MESSAGES = {"assistant_message", "user_message"}
MAX_TEXT = 100000


def clip(v, n=MAX_TEXT):
    """Shorten for display, keeping paragraph breaks so a message stays readable."""
    if v is None:
        return None
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s if len(s) <= n else s[:n] + " …"


def gold_reference(trace):
    names = [a["name"] for a in trace.outcome.get("gold_actions", [])
             if a.get("requestor") == "assistant"]
    pairs = [[a, b] for a, b in zip(names, names[1:]) if a != b]
    return Reference(name="gold", required=sorted(set(names)), precedence=pairs)


def export_run(trace, result):
    gold = [a for a in trace.outcome.get("gold_actions", [])
            if a.get("requestor") == "assistant"]
    present = {a for a in trace.activities}
    steps = []
    for i, e in enumerate(trace.ordered()):
        steps.append({
            "i": i,
            "activity": e.activity,
            "resource": e.resource,
            "kind": "message" if e.activity in MESSAGES else "tool",
            "args": clip(e.attributes.get("arguments"), 2000),
            "text": clip(e.attributes.get("content")),
            "error": bool(e.attributes.get("error")),
        })
    request = next((s["text"] for s in steps
                    if s["activity"] == "user_message" and s["text"]), None)
    return {
        "case_id": trace.case_id,
        "task_id": trace.outcome.get("task_id"),
        "trial": trace.outcome.get("trial"),
        "reward": trace.outcome.get("reward"),
        "request": request,
        "gold": [{"name": a["name"],
                  "args": clip(a.get("arguments"), 2000),
                  "present": a["name"] in present} for a in gold],
        "steps": steps,
        "fitness": result.fitness,
        "compliant": result.compliant,
        "n_constraints": result.n_constraints,
        "violations": [{"constraint": v.constraint, "description": v.description,
                        "at": v.at} for v in result.violations],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="runs_conformance.json")
    ap.add_argument("--per-cell", type=int, default=5)
    ap.add_argument("--all", action="store_true", help="export every run, not a sample")
    a = ap.parse_args()

    path = os.path.join(
        a.data, "tau-data/tau2-bench/submissions/claude-opus-4-5_sierra_2026-02-26",
        "trajectories/claude-opus-4-5_high_airline_gpt-5.2_4trials.json")
    traces = tau2.ingest(path)

    labels = {
        (False, False): "violation found, run failed",
        (True, True): "no violation, run passed",
        (False, True): "violation found, run passed anyway",
        (True, False): "no violation, run failed anyway",
    }
    NO_REF = "no reference actions (refusal task)"

    cells = collections.defaultdict(list)
    for t in traces:
        r = check(t, gold_reference(t))
        key = (NO_REF if not r.n_constraints
               else labels[(r.compliant, t.outcome.get("reward") == 1.0)])
        cells[key].append((t, r))

    picked = []
    for label in list(labels.values()) + [NO_REF]:
        group = sorted(cells.get(label, []), key=lambda x: len(x[0].events))
        if a.all:
            chosen = group
        else:
            # a readable sample: short runs plus one from the middle for realism
            chosen = (group[:a.per_cell - 1]
                      + group[len(group) // 2: len(group) // 2 + 1])[:a.per_cell]
        for t, r in chosen:
            run = export_run(t, r)
            run["cell"] = label
            picked.append(run)
        print(f"  {label:38} available {len(group):3}  exported {len(chosen)}")

    with open(a.out, "w") as f:
        json.dump({"source": os.path.basename(path), "runs": picked}, f,
                  separators=(",", ":"))
    print(f"\n{len(picked)} runs, {os.path.getsize(a.out) / 1024:.0f} KB -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
