"""Compare the process-mining signal against the judge baselines.

    python experiments/judge/evaluate.py --judge results/judge

Ground truth is tau2-bench's own `action_checks`: every gold action of the task
matched, arguments included, means the run followed the protocol. `--label
action-or-db` also counts a run whose final database differs from the expected
one, which closes the check's blind spot for writes the task did not ask for;
`--label reward` uses the benchmark's reward (outcome, not process). It is
computed by the benchmark harness, so it is independent of both methods under
comparison and it is a statement about the process, not the outcome.

For each method the script reports accuracy, precision, recall, F1 and Cohen's
kappa against that truth, each with a bootstrap 95% confidence interval. Methods
are compared pairwise with McNemar's exact test on the runs where they disagree.
Judge reproducibility is the share of runs on which all repeats agree, plus
Fleiss' kappa across repeats. Cost is tokens and seconds per run.

Standard library only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from agentpm.analysis.conformance import check
from agentpm.evaluation import (bootstrap, fleiss_kappa, ground_truth, judge_verdicts, mcnemar,  # noqa: F401
                                metrics, verdict_cost)
from agentpm.ingest import tau2
from agentpm.reference import Reference


# ------------------------------------------------------------------ inputs

def gold_reference(trace):
    names = [a["name"] for a in trace.outcome.get("gold_actions", [])
             if a.get("requestor") == "assistant"]
    pairs = [[a, b] for a, b in zip(names, names[1:]) if a != b]
    return Reference(name="gold", required=sorted(set(names)), precedence=pairs)


def pm_verdicts(traces):
    return {t.case_id: check(t, gold_reference(t)).compliant for t in traces}


# ------------------------------------------------------------------ report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser(
        "~/Downloads/tau-data/tau2-bench/submissions/claude-opus-4-5_sierra_2026-02-26/"
        "trajectories/claude-opus-4-5_high_airline_gpt-5.2_4trials.json"))
    ap.add_argument("--judge", default="results/judge")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--label", choices=["action", "action-or-db", "reward"], default="action")
    ap.add_argument("--pm-verdicts", help="use the verdicts in this JSON (case_id, compliant) as the "
                    "process-mining method, e.g. results/task_type_conformance.json for the "
                    "per-task-type reference, instead of the per-task gold-action reference")
    a = ap.parse_args()

    traces = tau2.ingest(a.data)
    truth = ground_truth(traces, a.label)
    if a.pm_verdicts:
        with open(a.pm_verdicts, encoding="utf-8") as f:
            methods = {"process mining": {r["case_id"]: bool(r["compliant"]) for r in json.load(f)}}
        print(f"process mining verdicts from {a.pm_verdicts}")
    else:
        methods = {"process mining": pm_verdicts(traces)}
    repeats, costs = {}, {}
    for path in sorted(glob.glob(os.path.join(a.judge, "verdicts_*.json"))):
        name = os.path.basename(path)[len("verdicts_"):-len(".json")].replace("_", " ")
        methods[f"judge {name}"], repeats[name], _ = judge_verdicts(path)
        costs[name] = verdict_cost(path)

    pm = methods["process mining"]
    print({"action": "ground truth: tau2 action_checks, all gold actions matched = compliant",
           "action-or-db": "ground truth: all gold actions matched and final database matched = compliant",
           "reward": "ground truth: tau2 reward 1 = compliant (outcome, not process)"}[a.label])
    print(f"runs with ground truth: {len(truth)}   violations: {sum(1 for v in truth.values() if not v)}\n")
    if len(methods) == 1:
        print("no judge verdicts found in", a.judge, "- run run_judge.py first\n")

    print("ALIGNMENT with ground truth (positive class = protocol violation)")
    print("  each method on every run it judged; n differs where a judge did not cover a run")
    print(f"  {'method':30} {'n':>4} {'acc':>6} {'prec':>6} {'rec':>6} {'F1':>6} {'kappa':>6}   95% CI on kappa")
    for name, pred in methods.items():
        ids = sorted(i for i in truth if i in pred)
        m = metrics(pred, truth, ids)
        lo, hi = bootstrap(pred, truth, ids, "kappa", a.rounds)
        print(f"  {name:30} {len(ids):4} {m['accuracy']:6.2f} {m['precision']:6.2f} {m['recall']:6.2f} "
              f"{m['f1']:6.2f} {m['kappa']:6.2f}   [{lo:.2f}, {hi:.2f}]")

    names = [n for n in methods if n != "process mining"]
    if names:
        print("\nPAIRED COMPARISON, McNemar exact test, process mining vs each judge on the runs both scored")
        for other in names:
            ids = sorted(i for i in truth if i in pm and i in methods[other])
            a_only, b_only, both_r, both_w, p = mcnemar(pm, methods[other], truth, ids)
            print(f"  vs {other}  (n={len(ids)})")
            print(f"    both right {both_r:3}   both wrong {both_w:3}   "
                  f"only process mining right {a_only:3}   only judge right {b_only:3}")
            print(f"    p = {p:.4f}  -> "
                  + ("difference is unlikely to be chance" if p < 0.05 else
                     "no evidence the two differ on this sample"))

    if repeats:
        print("\nREPRODUCIBILITY")
        print("  process mining: three full repetitions hash identically (measured in run_three_corpora)")
        for name, rep in repeats.items():
            unanimous = sum(1 for v in rep.values() if len(set(v)) == 1)
            fk = fleiss_kappa(rep)
            print(f"  judge {name}: unanimous on {unanimous}/{len(rep)} runs, "
                  f"Fleiss kappa {fk:.2f}" if fk is not None else f"  judge {name}: n/a")

        print("\nCOST per verdict, over the calls the model answered (cache copies left out)")
        print("  process mining: 0 tokens, well under a millisecond per run")
        for name, c in costs.items():
            if not c:
                continue
            print(f"  judge {name}: {c['prompt_tokens']:,.0f} prompt + {c['completion_tokens']:,.0f} "
                  f"completion tokens, {c['seconds']:.0f} s; {c['verdicts']} answered calls, "
                  f"{c['cached']} cache copies left out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
