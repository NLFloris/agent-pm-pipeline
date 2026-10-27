"""Score the sub-agent corpus: outcome per run, and conformance to one task-type model.

    python experiments/local_agents/check_claims.py > results/local_subagents/conformance.txt

Reads the runs written by `run_magentic.py --subagents --tasks tasks_claims.json`
(default results/local_subagents/runs/), and per run reports:

  outcome      the team's final VERDICT line against the correct answer, which is
               recomputed here from sales.csv and must equal the 'answer' in the
               tasks file, or nothing is scored
  structure    which sub-agents were started, by whom, and which tools they used
  conformance  against claim_check_reference.json, written and hashed before any
               run (results/frozen/claim_check_reference.sha256)

The reference speaks about agents, so the check runs on an agent-qualified view of
each trace: every activity becomes Agent.activity and a sub-agent start becomes
Parent.spawn:Child. That relabelling is declared here as data, as the retail
authentication merge is; the engine and the log are unchanged.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import dataclasses
import hashlib
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from agentpm.analysis.conformance import check        # noqa: E402
from agentpm.ingest import local_magentic             # noqa: E402
from agentpm.model import Trace                       # noqa: E402
from agentpm.reference import load_reference          # noqa: E402

VERDICT = re.compile(r"VERDICT:\s*\**\s*(TRUE|FALSE)", re.I)


def seed_csv():
    """sales.csv as the harness seeds it, read from run_magentic.py without importing AutoGen."""
    with open(os.path.join(HERE, "run_magentic.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "SEED_FILES":
            return ast.literal_eval(node.value)["sales.csv"]
    raise SystemExit("SEED_FILES not found in run_magentic.py")


def truths():
    rows = list(csv.DictReader(io.StringIO(seed_csv())))
    u = {(r["region"], r["quarter"]): int(r["units"]) for r in rows}
    p = {(r["region"], r["quarter"]): float(r["price"]) for r in rows}
    rev = {k: u[k] * p[k] for k in u}
    regions = sorted({r for r, _ in u})
    return {
        "c01_east_units": u["east", "Q2"] < u["east", "Q1"],
        "c02_north_revenue": rev["north", "Q2"] > rev["north", "Q1"],
        "c03_south_top_q2": max(regions, key=lambda r: rev[r, "Q2"]) == "south",
        "c04_east_top_units_q1": all(u["east", "Q1"] > u[r, "Q1"] for r in regions if r != "east"),
        "c05_q1_total_revenue": sum(rev[r, "Q1"] for r in regions) > 3000,
        "c06_south_growth": (u["south", "Q2"] - u["south", "Q1"]) / u["south", "Q1"] > 0.60,
        "c07_north_price": p["north", "Q2"] > p["north", "Q1"],
        "c08_east_revenue": rev["east", "Q2"] > rev["east", "Q1"],
        "c09_average_price": sum(p.values()) / len(p) > 10.5,
        "c10_q2_units_gain": sum(u[r, "Q2"] for r in regions) - sum(u[r, "Q1"] for r in regions) > 70,
    }


def final_verdict(trace):
    """The last VERDICT line the team produced, from the end of the run backwards."""
    for e in reversed(trace.ordered()):
        m = VERDICT.findall(str(e.attributes.get("content") or ""))
        if m:
            return m[-1].upper() == "TRUE"
    return None


def qualified(trace):
    """Agent.activity for every event; Parent.spawn:Child for a sub-agent start."""
    def name(e):
        if e.activity == "spawn":
            return f"{e.resource}.spawn:{e.attributes.get('spawns')}"
        return f"{e.resource}.{e.activity}"
    return Trace(trace.case_id, [dataclasses.replace(e, activity=name(e)) for e in trace.events],
                 trace.outcome)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(REPO, "results", "local_subagents", "runs"))
    ap.add_argument("--tasks", default=os.path.join(HERE, "tasks_claims.json"))
    ap.add_argument("--reference", default=os.path.join(HERE, "claim_check_reference.json"))
    a = ap.parse_args()

    with open(a.tasks, encoding="utf-8") as f:
        answers = {t["id"]: t["answer"] for t in json.load(f)["tasks"]}
    truth = truths()
    wrong = {k for k in answers if truth.get(k) != answers[k]}
    if wrong:
        raise SystemExit(f"the tasks file and sales.csv disagree on {sorted(wrong)}; nothing scored")

    with open(a.reference, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    ref = load_reference(a.reference)
    traces = local_magentic.ingest_dir(a.runs)
    print(f"{len(traces)} runs from {os.path.relpath(a.runs, REPO)}, task type "
          f"{sorted({t.outcome.get('task_type') for t in traces})}")
    print(f"reference {os.path.relpath(a.reference, REPO)}  sha256 {digest[:16]}...")
    print(f"  required    {ref.required}")
    print(f"  precedence  {ref.precedence}\n")

    print(f"{'run':24} {'answer':6} {'team':6} {'ok':3} {'events':>6}  {'sub-agents started (by Analyst)':34} {'conform':8} violations")
    rows = []
    for t in traces:
        expected = answers.get(t.case_id)
        got = final_verdict(t)
        r = check(qualified(t), ref)
        spawns = collections.Counter(e.attributes.get("spawns") for e in t.events if e.activity == "spawn")
        sub_calls = collections.Counter(f"{e.resource}.{e.activity}" for e in t.events
                                        if e.attributes.get("spawned_by") and e.attributes.get("call_id"))
        rows.append({"case_id": t.case_id, "answer": expected, "verdict": got,
                     "correct": got is not None and got == expected, "compliant": r.compliant,
                     "violations": [v.constraint for v in r.violations], "events": len(t.events),
                     "spawns": dict(spawns), "subagent_tool_calls": dict(sub_calls),
                     "error": t.outcome.get("harness_error")})
        shown = ", ".join(f"{k} x{n}" for k, n in sorted(spawns.items())) or "none"
        print(f"{t.case_id:24} {str(expected):6} {str(got):6} {'yes' if rows[-1]['correct'] else 'NO':3} "
              f"{len(t.events):6}  {shown:34} {'yes' if r.compliant else 'NO':8} "
              + "; ".join(v.constraint for v in r.violations)
              + (f"  [harness error: {t.outcome['harness_error'][:60]}]" if t.outcome.get("harness_error") else ""))

    n = len(rows)
    print(f"\ncorrect verdicts       {sum(r['correct'] for r in rows)}/{n}")
    print(f"conforming runs        {sum(r['compliant'] for r in rows)}/{n}")
    both = collections.Counter((r["compliant"], r["correct"]) for r in rows)
    print(f"conforming and correct {both[True, True]}, conforming and wrong {both[True, False]}, "
          f"violating and correct {both[False, True]}, violating and wrong {both[False, False]}")
    total = collections.Counter()
    for r in rows:
        total.update(r["subagent_tool_calls"])
    print(f"sub-agent tool calls   {dict(sorted(total.items()))}")
    kinds = collections.Counter(v for r in rows for v in r["violations"])
    print(f"violations by constraint {dict(kinds.most_common())}")

    out = os.path.join(os.path.dirname(os.path.abspath(a.runs)), "conformance.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"reference_sha256": digest, "runs": rows}, f, indent=1)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    sys.exit(main())
