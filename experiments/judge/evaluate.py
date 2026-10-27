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
import collections
import glob
import json
import math
import os
import random
import sys

from agentpm.analysis.conformance import check
from agentpm.ingest import tau2
from agentpm.reference import Reference


# ------------------------------------------------------------------ inputs

def gold_reference(trace):
    names = [a["name"] for a in trace.outcome.get("gold_actions", [])
             if a.get("requestor") == "assistant"]
    pairs = [[a, b] for a, b in zip(names, names[1:]) if a != b]
    return Reference(name="gold", required=sorted(set(names)), precedence=pairs)


def ground_truth(traces, label="action"):
    """case_id -> True if every gold action matched, arguments included
    (and, for action-or-db, the final database matched; for reward, reward 1)."""
    out = {}
    for t in traces:
        checks = t.outcome.get("action_checks") or []
        if checks:
            if label == "reward":
                out[t.case_id] = t.outcome.get("reward") == 1.0
                continue
            out[t.case_id] = all(c.get("action_match") for c in checks)
            if label == "action-or-db":
                out[t.case_id] = out[t.case_id] and (t.outcome.get("db_check") or {}).get("db_match", True)
    return out


def pm_verdicts(traces):
    return {t.case_id: check(t, gold_reference(t)).compliant for t in traces}


def judge_verdicts(path):
    """Majority vote over repeats; ties count as non-compliant (the cautious call)."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    majority, per_repeat, cost = {}, {}, {}
    for r in records:
        votes = [x["compliant"] for x in r["repeats"] if x["compliant"] is not None]
        if not votes:
            continue
        yes = sum(1 for v in votes if v)
        majority[r["case_id"]] = yes > len(votes) / 2
        per_repeat[r["case_id"]] = votes
        cost[r["case_id"]] = {
            "seconds": sum(x["seconds"] or 0 for x in r["repeats"]),
            "prompt_tokens": sum(x["prompt_tokens"] or 0 for x in r["repeats"]),
            "completion_tokens": sum(x["completion_tokens"] or 0 for x in r["repeats"]),
            "parsed": sum(1 for x in r["repeats"] if x["parsed"]),
            "n": len(r["repeats"]),
        }
    return majority, per_repeat, cost


def verdict_cost(path):
    """Mean cost of one verdict, over the calls the model actually answered.

    The proxy answers a repeated identical request from its cache, with the
    response id of the first answer; such a copy costs the model nothing and is
    left out. A call counts as fresh if its response id is new for that run and
    the proxy did not flag it as a cache hit. Seconds come from the verdict file,
    tokens from the raw log next to it."""
    with open(path, encoding="utf-8") as f:
        seconds = {(r["case_id"], x["repeat"]): x["seconds"] for r in json.load(f) for x in r["repeats"]}
    raw = os.path.join(os.path.dirname(path), "raw_" + os.path.basename(path)[len("verdicts_"):-5] + ".jsonl")
    seen, fresh, cached = collections.defaultdict(set), [], 0
    with open(raw, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rid = (r.get("response") or {}).get("id")
            if not rid:
                continue                                  # a failed call: no answer, no cost counted
            if rid in seen[r["case_id"]] or r.get("cache_hit"):
                cached += 1
                continue
            seen[r["case_id"]].add(rid)
            usage = r["response"].get("usage") or {}
            fresh.append((usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0,
                          seconds.get((r["case_id"], r["repeat"])) or 0))
    n = len(fresh)
    if not n:
        return None
    return {"verdicts": n, "cached": cached, "prompt_tokens": sum(x[0] for x in fresh) / n,
            "completion_tokens": sum(x[1] for x in fresh) / n, "seconds": sum(x[2] for x in fresh) / n}


# ----------------------------------------------------------------- metrics

def confusion(pred, truth, ids):
    tp = sum(1 for i in ids if not pred[i] and not truth[i])   # positive class = violation
    fp = sum(1 for i in ids if not pred[i] and truth[i])
    fn = sum(1 for i in ids if pred[i] and not truth[i])
    tn = sum(1 for i in ids if pred[i] and truth[i])
    return tp, fp, fn, tn


def metrics(pred, truth, ids):
    tp, fp, fn, tn = confusion(pred, truth, ids)
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    # Cohen's kappa against chance agreement
    po = acc
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n) if n else 0.0
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return {"n": n, "accuracy": acc, "precision": p, "recall": r, "f1": f1, "kappa": kappa}


def bootstrap(pred, truth, ids, key, rounds=1000, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(rounds):
        sample = [rng.choice(ids) for _ in ids]
        vals.append(metrics(pred, truth, sample)[key])
    vals.sort()
    return vals[int(0.025 * rounds)], vals[int(0.975 * rounds)]


def mcnemar(pred_a, pred_b, truth, ids):
    """Exact McNemar: do A and B err on different runs more than chance allows?"""
    a_only = sum(1 for i in ids if pred_a[i] == truth[i] and pred_b[i] != truth[i])
    b_only = sum(1 for i in ids if pred_b[i] == truth[i] and pred_a[i] != truth[i])
    both_right = sum(1 for i in ids if pred_a[i] == truth[i] and pred_b[i] == truth[i])
    both_wrong = len(ids) - a_only - b_only - both_right
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, both_right, both_wrong, 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, j) for j in range(0, k + 1)) / 2 ** n
    return a_only, b_only, both_right, both_wrong, min(1.0, 2 * tail)


def fleiss_kappa(per_repeat):
    """Agreement of the judge with itself across repeats, two categories."""
    rows = [v for v in per_repeat.values() if len(v) >= 2]
    if not rows:
        return None
    n = min(len(v) for v in rows)
    rows = [v[:n] for v in rows]
    p_yes = sum(sum(1 for x in v if x) for v in rows) / (len(rows) * n)
    pe = p_yes ** 2 + (1 - p_yes) ** 2
    po = sum((sum(1 for x in v if x) ** 2 + sum(1 for x in v if not x) ** 2 - n)
             / (n * (n - 1)) for v in rows) / len(rows)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


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
