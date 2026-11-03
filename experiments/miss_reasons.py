"""Why the engine misses a per-action violation, over the fifteen agents.

    python experiments/miss_reasons.py > results/miss_reasons.txt

cross_agent.py counts a failed gold action as "wrong arguments" whenever an action
of that name occurred at all. That also covers an action the agent repeated too
few times, so this script separates the two. Every missed run (the engine passes
it, the per-action label marks a violation) falls in exactly one group:

  action absent     a failed gold action whose name does not occur in the run
  too few repeats   otherwise, a failed gold action whose name occurs fewer times
                    than the gold list requires
  wrong arguments   otherwise; the name occurs often enough, but no call matches

Uses the frozen confirmation rule, like the main columns of the cross-agent table.
"""
from __future__ import annotations

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_agent import check, domain_rules, files, tau2, violation   # noqa: E402


def group(trace):
    """The group of a missed run, and the action names that put it there."""
    checks = trace.outcome["action_checks"]
    did = collections.Counter(e.activity for e in trace.events)
    gold = collections.Counter((c.get("action") or {}).get("name") for c in checks)
    failed = [(c.get("action") or {}).get("name") for c in checks if not c.get("action_match")]
    absent = sorted({n for n in failed if did[n] == 0})
    if absent:
        return "action absent", absent
    fewer = sorted({n for n in failed if did[n] < gold[n]})
    if fewer:
        return "too few repeats", fewer
    return "wrong arguments", []


def report(domain):
    task_type, type_model, prepare = domain_rules(domain)
    groups, names = collections.Counter(), collections.defaultdict(collections.Counter)
    for agent, path in files(domain):
        for t in tau2.ingest(path):
            if not t.outcome.get("action_checks") or not violation(t, "action"):
                continue
            if not check(prepare(t), type_model(task_type(t), False)).compliant:
                continue
            g, acts = group(t)
            groups[g] += 1
            names[g].update(acts)
    print(f"\n=== {domain}: {sum(groups.values())} missed per-action violations ===")
    for g in ("wrong arguments", "too few repeats", "action absent"):
        print(f"  {g:16} {groups[g]:5}  {names[g].most_common()}")


if __name__ == "__main__":
    for domain in ("airline", "retail"):
        report(domain)
