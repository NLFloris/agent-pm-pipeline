"""Repeated tool use and the outcome: does the difference survive the task and the run length?

    python experiments/repeated_tool_use.py > results/repeated_tool_use.txt

Section 5 reports that among the airline runs of Claude Opus 4.5 with at least
two tool calls, runs with a reward of 0 make 11.7 tool calls on average against
7.4, and that 53% of their calls repeat a tool the run already used, against 34%
(results/three_corpora.txt). Two other explanations predict the same difference:
failed runs are longer, and some tasks need more repeated calls than others.
This script tests both, with the pipeline's own measure,
agentpm.analysis.performance.repeated_tool_use.

  (a) the reproduction of the numbers in Section 5;
  (b) the same task: tasks with both a failed and a successful run, compared
      within the task, so that what the task requires is held fixed;
  (c) the run length: the share per bin of the number of tool calls, and each
      failed run against the successful runs with exactly as many tool calls;
  (d) what the solution list requires: the share of repeated tools in the
      task's own solution list, and the run's share above that;
  (e) every run: task, trial, reward, tool calls, distinct tools, share, and
      the share its solution list requires, so that each number above can be
      recomputed by hand.

The share is 1 - distinct tools / tool calls. With a fixed set of tools it rises
with the number of calls by construction, so a longer run tends to have a higher
share whether or not anything went wrong.

A run counts as failed when its reward is 0. The p-values are a descriptive aid
only. The sign test treats tasks as independent and the runs within a task are
not. Standard library only, no randomness.
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
from task_type_conformance import DEFAULT_DATA   # noqa: E402

from agentpm.analysis.performance import NON_ACTION, repeated_tool_use   # noqa: E402
from agentpm.ingest import tau2   # noqa: E402
from agentpm.model import Event, Trace   # noqa: E402

BINS = [(2, 5), (6, 9), (10, 14), (15, 10 ** 6)]


def calls(trace):
    return sum(1 for a in trace.activities if a not in NON_ACTION)


def solution_share(trace):
    """The repeated-tool share of the run's own solution list, or None below two actions."""
    names = [a["name"] for a in trace.outcome.get("gold_actions") or []
             if a.get("requestor") == "assistant"]
    return repeated_tool_use(Trace(trace.case_id, [Event(n, i) for i, n in enumerate(names)]))


def sign_test(pos, neg):
    """Exact two-sided sign test on the non-tied pairs."""
    n, k = pos + neg, min(pos, neg)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def label(b):
    lo, hi = b
    return f"{lo}+" if hi >= 10 ** 6 else f"{lo}-{hi}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA["airline"],
                    help="trajectory file; defaults to the airline file of Claude Opus 4.5")
    a = ap.parse_args()
    traces = tau2.ingest(a.data)
    runs = []
    for t in traces:
        share = repeated_tool_use(t)
        if share is None:
            continue
        runs.append({"task": t.outcome["task_id"], "trial": t.outcome.get("trial"),
                     "reward": t.outcome.get("reward"), "failed": t.outcome.get("reward") == 0.0,
                     "calls": calls(t), "share": share, "sol": solution_share(t),
                     "has_list": bool(t.outcome.get("gold_actions"))})
    failed = [r for r in runs if r["failed"]]
    passed = [r for r in runs if not r["failed"]]
    mean = statistics.mean

    print(f"{len(traces)} airline runs of Claude Opus 4.5, {len(runs)} with at least two tool calls")
    print("failed = reward 0, succeeded = reward 1\n")

    print("(a) Reproduction of Section 5")
    for name, g in (("failed", failed), ("succeeded", passed)):
        print(f"  {name:10} n={len(g):4}  tool calls {mean(r['calls'] for r in g):5.1f}"
              f"  repeated tool use {mean(r['share'] for r in g):6.1%}"
              f"  distinct tools {mean(r['calls'] * (1 - r['share']) for r in g):4.1f}")

    print("\n(b) Within the same task")
    by_task = collections.defaultdict(list)
    for r in runs:
        by_task[r["task"]].append(r)
    mixed = sorted((k for k, v in by_task.items()
                    if any(r["failed"] for r in v) and any(not r["failed"] for r in v)), key=int)
    print(f"  {len(by_task)} tasks; {len(mixed)} have both a failed and a successful run "
          f"({sum(1 for k in mixed for r in by_task[k] if r['failed'])} failed and "
          f"{sum(1 for k in mixed for r in by_task[k] if not r['failed'])} successful runs)")
    print(f"  the other {sum(1 for r in failed if r['task'] not in mixed)} failed runs belong to tasks "
          f"without a successful run of at least two tool calls")
    for key, unit in (("share", "share"), ("calls", "tool calls")):
        diffs = []
        for k in mixed:
            f = [r[key] for r in by_task[k] if r["failed"]]
            s = [r[key] for r in by_task[k] if not r["failed"]]
            diffs.append(mean(f) - mean(s))
        pos = sum(1 for d in diffs if d > 1e-12)
        neg = sum(1 for d in diffs if d < -1e-12)
        fmt = (lambda x: f"{x:+.1%}") if key == "share" else (lambda x: f"{x:+.1f}")
        print(f"  {unit:11} failed minus succeeded, per task: higher in failed {pos}, lower {neg}, "
              f"equal {len(diffs) - pos - neg}; mean {fmt(mean(diffs))}, "
              f"median {fmt(statistics.median(diffs))}; sign test p = {sign_test(pos, neg):.3f}")
    rest = [r for r in failed if r["task"] not in mixed]
    if rest:
        print(f"  those {len(rest)} failed runs (tasks {', '.join(sorted({r['task'] for r in rest}, key=int))}): "
              f"tool calls {mean(r['calls'] for r in rest):.1f}, share {mean(r['share'] for r in rest):.1%}; "
              f"the {len(failed) - len(rest)} failed runs of the mixed tasks: tool calls "
              f"{mean(r['calls'] for r in failed if r['task'] in mixed):.1f}, "
              f"share {mean(r['share'] for r in failed if r['task'] in mixed):.1%}")
    print("  per task (task: failed runs, mean share failed / succeeded, mean calls failed / succeeded)")
    for k in mixed:
        f = [r for r in by_task[k] if r["failed"]]
        s = [r for r in by_task[k] if not r["failed"]]
        print(f"    {k:>3}: {len(f)} of {len(f) + len(s)}  "
              f"{mean(r['share'] for r in f):6.1%} / {mean(r['share'] for r in s):6.1%}   "
              f"{mean(r['calls'] for r in f):5.1f} / {mean(r['calls'] for r in s):5.1f}")

    print("\n(c) Run length")
    print("  share by the number of tool calls (n, mean share)")
    for b in BINS:
        cells = []
        for g in (failed, passed):
            c = [r["share"] for r in g if b[0] <= r["calls"] <= b[1]]
            cells.append(f"n={len(c):3} {mean(c):6.1%}" if c else f"n={0:3}    -  ")
        print(f"    {label(b):>6} calls   failed {cells[0]}   succeeded {cells[1]}")
    matched = []
    for r in failed:
        same = [s["share"] for s in passed if s["calls"] == r["calls"]]
        if same:
            matched.append((r["share"], mean(same)))
    if matched:
        d = [x - y for x, y in matched]
        pos = sum(1 for x in d if x > 1e-12)
        neg = sum(1 for x in d if x < -1e-12)
        print(f"  each failed run against the successful runs with exactly as many tool calls: "
              f"{len(matched)} of {len(failed)} failed runs have a match")
        print(f"    mean share {mean(x for x, _ in matched):.1%} against {mean(y for _, y in matched):.1%}; "
              f"difference mean {mean(d):+.1%}, median {statistics.median(d):+.1%}; "
              f"higher in the failed run {pos}, lower {neg}, equal {len(d) - pos - neg}")

    print("\n(d) What the solution list requires")
    print("  required share = the repeated-tool share of the task's solution list,")
    print("  taken as 0 when the list has fewer than two actions (no repeat required)")
    for name, g in (("failed", failed), ("succeeded", passed)):
        req = [r["sol"] or 0.0 for r in g]
        exc = [r["share"] - (r["sol"] or 0.0) for r in g]
        print(f"  {name:10} n={len(g):4}  required share {mean(req):6.1%}  run share "
              f"{mean(r['share'] for r in g):6.1%}  excess {mean(exc):+6.1%}  "
              f"list of two or more actions {sum(1 for r in g if r['sol'] is not None)}, "
              f"no solution list {sum(1 for r in g if not r['has_list'])}")
    print("  only runs whose solution list has at least two actions")
    for name, g in (("failed", failed), ("succeeded", passed)):
        g = [r for r in g if r["sol"] is not None]
        print(f"  {name:10} n={len(g):4}  required share {mean(r['sol'] for r in g):6.1%}  run share "
              f"{mean(r['share'] for r in g):6.1%}  excess {mean(r['share'] - r['sol'] for r in g):+6.1%}")

    print("\n(e) Every run with at least two tool calls")
    print("  task trial reward  calls  distinct   share  required (solution list; - below two actions)")
    for r in sorted(runs, key=lambda r: (int(r["task"]), r["trial"])):
        distinct = round(r["calls"] * (1 - r["share"]))
        req = "-" if r["sol"] is None else f"{r['sol']:.1%}"
        print(f"  {r['task']:>4} {r['trial']:>5} {r['reward']:>6} {r['calls']:>6} {distinct:>9} "
              f"{r['share']:>7.1%}  {req}")


if __name__ == "__main__":
    main()
