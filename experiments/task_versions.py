"""Which published tau2-bench runs used an older version of a task, and what that does to the engine.

    python experiments/task_versions.py > results/task_versions.txt

The benchmark corrected 27 airline and 26 retail tasks in release 1.0.0 of its changelog, and a
published run keeps the version it ran on. A submission stores its tasks, so the version can be
read from the file: an agent is on an older version when a task's user scenario differs from the
current tasks.json. A task is affected when that scenario differs, or when agents were scored
against different solutions for it (the writes of the solution list).

The engine's verdicts are the frozen task-type models of cross_agent.py, unchanged. No model is called.
"""
from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_agent import LABELS, files, metrics, violation   # noqa: E402
from task_type_conformance import domain_rules   # noqa: E402

from agentpm.analysis.conformance import check   # noqa: E402
from agentpm.ingest import tau2   # noqa: E402

DOMAINS = os.path.expanduser("~/Downloads/tau-data/tau2-bench/data/tau2/domains")
READS = ("get_", "search_", "list_", "find_", "calculate")


def canon(v):
    """Compare arguments up to dict order, list order and JSON encoding."""
    if isinstance(v, str) and v.strip()[:1] in "{[":
        try:
            v = json.loads(v)
        except ValueError:
            pass
    if isinstance(v, dict):
        return {k: canon(x) for k, x in v.items()}
    if isinstance(v, list):
        return sorted((canon(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True))
    return v


def solution(trace):
    return json.dumps(sorted(json.dumps([a.get("name"), canon(a.get("arguments") or {})], sort_keys=True)
                             for a in (c.get("action") or {} for c in trace.outcome["action_checks"])
                             if not str(a.get("name")).startswith(READS)))


def report(domain):
    with open(f"{DOMAINS}/{domain}/tasks.json", encoding="utf-8") as f:
        current = {str(t["id"]): json.dumps(t.get("user_scenario"), sort_keys=True) for t in json.load(f)}
    task_type, type_model, prepare = domain_rules(domain)
    scenario, solutions, rows = {}, collections.defaultdict(set), []
    for agent, path in files(domain):
        with open(path, encoding="utf-8") as f:
            tasks = json.load(f).get("tasks") or []
        scenario[agent] = {str(t["id"]) for t in tasks if str(t["id"]) in current
                           and json.dumps(t.get("user_scenario"), sort_keys=True) != current[str(t["id"])]}
        for t in tau2.ingest(path):
            if not t.outcome.get("action_checks"):
                continue
            tid = str(t.outcome.get("task_id"))
            solutions[tid].add(solution(t))
            rows.append({"agent": agent, "task": tid, "solution": solution(t),
                         "flag": not check(prepare(t), type_model(task_type(t), False)).compliant,
                         **{lab: violation(t, lab) for lab in LABELS}})
    older = {a for a, s in scenario.items() if s}
    moved = {t for t, s in solutions.items() if len(s) > 1}
    affected = set().union(*scenario.values()) | moved
    scorable = {r["task"] for r in rows}
    old_runs = [r for r in rows if r["agent"] in older and r["task"] in affected]
    print(f"\n=== {domain}: {len(rows)} scorable runs of {len(files(domain))} agents on {len(scorable)} tasks")
    print(f"agents on the older version: {len(older)}")
    for a in sorted(older):
        print(f"  {a}")
    print(f"tasks affected: {len(affected & scorable)} of the scorable ones; solution differs between agents on {len(moved)}")
    print(f"runs on the older version: {len(old_runs)} of {len(rows)}; of these with another solution: "
          f"{sum(r['task'] in moved for r in old_runs)}")
    keep = [r for r in rows if not (r["agent"] in older and r["task"] in affected)]
    for lab in LABELS:
        for name, group in (("all runs", rows), ("without those runs", keep), ("those runs only", old_runs)):
            m = metrics([(r["flag"], r[lab]) for r in group])
            print(f"  {lab:13} {name:19} n {m['n']:5} violations {m['violations']:5} flags {m['flags']:4} "
                  f"prec {m['precision']:.3f} rec {m['recall']:.3f} kappa {m['kappa']:.3f}")


def main():
    for domain in ("airline", "retail"):
        report(domain)


if __name__ == "__main__":
    sys.exit(main())
