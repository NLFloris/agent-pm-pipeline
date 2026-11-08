"""Path-sensitivity: does a verdict separate runs with the same outcome and different paths?

    python experiments/path_sensitivity.py > results/path_sensitivity.txt

Claude Opus 4.5 on both domains, with the task-type engine and the LLM Judges.
Two runs form a pair when they belong to the same task, both have reward 1, so
the same outcome, and differ in their path. A path is the sequence of steps, each
step the sorted activities issued together, so the order in which a source lists
calls issued together is not a difference. The tools-only lines leave out the
messages, so that a pair must differ in what the agent did and not only in how
much was said.

Most paths differ without any rule being broken, so separating many pairs is not
in itself good. A separation goes with the label when the per-action label also
separates the pair and the method agrees with it on both runs.

The second table takes the runs with reward 1 that the per-action label marks as
violations, which are the failures an outcome cannot show, next to the runs with
reward 1 that it marks as compliant, and splits those violations by whether only
a lookup or a calculation on the solution list fails, or a change or a transfer.

Reads results/task_type_conformance*.json and results/judge/, so it spends no
tokens. Standard library only.
"""
from __future__ import annotations

import collections
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "judge")]
from cross_agent import WRITES   # noqa: E402
from evaluate import judge_verdicts, tau2   # noqa: E402
from task_type_conformance import DEFAULT_DATA   # noqa: E402

MESSAGES = {"user_message", "assistant_message"}
JUDGES = {
    "airline": [("output-only, gpt-oss", "results/judge/verdicts_output-only_gpt-oss-120b.json"),
                ("policy, gpt-oss", "results/judge/verdicts_full_gpt-oss-120b.json"),
                ("policy, glm", "results/judge/verdicts_full_glm-5-3-flash.json"),
                ("required-actions, gpt-oss", "results/judge/verdicts_gold-actions_gpt-oss-120b.json"),
                ("required-actions, glm", "results/judge/verdicts_gold-actions_glm-5-3-flash.json")],
    "retail": [("output-only, gpt-oss", "results/judge/retail/verdicts_output-only_gpt-oss-120b.json"),
               ("policy, gpt-oss", "results/judge/retail/verdicts_full_gpt-oss-120b.json"),
               ("required-actions, gpt-oss", "results/judge/retail/verdicts_gold-actions_gpt-oss-120b.json")],
}


def path(trace, tools_only=False):
    steps = (sorted(e.activity for e in step
                    if not (tools_only and e.activity in MESSAGES))
             for step in trace.steps())
    return tuple(tuple(s) for s in steps if s)


def pairs_with_other_path(rows, traces, tools_only):
    by_task = collections.defaultdict(list)
    for case, r in rows.items():
        if r["truth_outcome"]:
            by_task[r["task_id"]].append(case)
    out = []
    for cases in by_task.values():
        out += [(a, b) for a, b in itertools.combinations(cases, 2)
                if path(traces[a], tools_only) != path(traces[b], tools_only)]
    return out


def separation(name, verdict, pairs, rows):
    judged = [(a, b) for a, b in pairs if a in verdict and b in verdict]
    sep = [(a, b) for a, b in judged if verdict[a] != verdict[b]]
    right = [(a, b) for a, b in sep
             if rows[a]["truth_process"] != rows[b]["truth_process"]
             and verdict[a] == rows[a]["truth_process"]]
    against = [(a, b) for a, b in sep if rows[a]["truth_process"] != rows[b]["truth_process"]
               and (a, b) not in right]
    both_bad = [(a, b) for a, b in sep if not rows[a]["truth_process"] and not rows[b]["truth_process"]]
    both_ok = len(sep) - len(right) - len(against) - len(both_bad)
    print(f"    {name:26} separates {len(sep):4} of {len(judged):4} ({len(sep) / max(1, len(judged)):6.1%})"
          f"   as the label {len(right):3}   against it {len(against):2}"
          f"   label: both violations {len(both_bad):3}, both compliant {both_ok:3}")


def main():
    for domain in ("airline", "retail"):
        suffix = "" if domain == "airline" else "_retail"
        with open(f"results/task_type_conformance{suffix}.json", encoding="utf-8") as f:
            rows = {r["case_id"]: r for r in json.load(f)}
        traces = {t.case_id: t for t in tau2.ingest(DEFAULT_DATA[domain])}
        methods = [("engine, task type", {c: r["compliant"] for c, r in rows.items()})]
        methods += [(name, judge_verdicts(p)[0]) for name, p in JUDGES[domain]]

        print(f"=== {domain}: {len(rows)} scored runs, "
              f"{sum(r['truth_outcome'] for r in rows.values())} with reward 1 ===")
        for tools_only in (False, True):
            pairs = pairs_with_other_path(rows, traces, tools_only)
            label = [(a, b) for a, b in pairs if rows[a]["truth_process"] != rows[b]["truth_process"]]
            print(f"\n  pairs, same task, reward 1, other path ({'tools only' if tools_only else 'all activities'}): "
                  f"{len(pairs)}, of which the per-action label separates {len(label)} "
                  f"in {len({rows[a]['task_id'] for a, _ in label})} tasks")
            for name, verdict in methods:
                separation(name, verdict, pairs, rows)

        ok = [c for c, r in rows.items() if r["truth_outcome"] and r["truth_process"]]
        bad = [c for c, r in rows.items() if r["truth_outcome"] and not r["truth_process"]]
        print(f"\n  reward 1: {len(bad)} per-action violations, {len(ok)} compliant runs")
        for name, verdict in methods:
            flagged = lambda ids: sum(1 for c in ids if c in verdict and not verdict[c])
            judged = lambda ids: sum(1 for c in ids if c in verdict)
            print(f"    {name:26} flags {flagged(bad):3} of {judged(bad):3} violations, "
                  f"{flagged(ok):3} of {judged(ok):3} compliant runs")

        only_reads = sum(1 for c in bad
                         if all((x.get("action") or {}).get("name") not in WRITES[domain] + ["transfer_to_human_agents"]
                                for x in traces[c].outcome["action_checks"] if not x.get("action_match")))
        print(f"    the violations by what fails: only a lookup or a calculation {only_reads}, "
              f"a change or a transfer {len(bad) - only_reads}")

        reasons = collections.Counter()
        for c in bad:
            if rows[c]["compliant"]:
                failed = {(x.get("action") or {}).get("name")
                          for x in traces[c].outcome["action_checks"] if not x.get("action_match")}
                reasons.update(failed)
        print(f"    actions the engine's misses fail on: {reasons.most_common(8)}")
        caught = [(rows[c]["task_id"], rows[c]["violations"]) for c in bad if not rows[c]["compliant"]]
        print(f"    violations the engine flags: {caught[:12]}\n")


if __name__ == "__main__":
    main()
