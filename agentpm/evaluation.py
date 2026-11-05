"""Evaluation of verdicts against labels from outside the agent.

    python -m agentpm.cli evaluate <path> --format tau2 --task-types models.json \
        --judge policy=verdicts.json

It comes after the verdict and needs what the telemetry does not hold, namely
labels and, for a comparison, the verdicts of a Judge. It reports four criteria.

  alignment         precision, recall and Cohen's kappa against each label, kappa
                    with a bootstrap 95% interval, and McNemar's exact test
                    between the engine and each Judge
  reproducibility   for the engine, whether repeated checks hash identically; for
                    a Judge, the share of unanimous runs and Fleiss' kappa
  cost              tokens and time per verdict
  path-sensitivity  on pairs of runs of one task with reward 1 and different steps,
                    how many a method separates, and how many as the label does

A source that carries the benchmark's checks (tau2) brings its labels with it, and
any other source takes them from a JSON file. Judge verdicts are read from files as
experiments/judge/run_judge.py writes them; the package itself calls no model.
Standard library only.
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math
import os
import random
import time

from .analysis.conformance import check
from .model import Trace

MESSAGES = {"user_message", "assistant_message"}
# label name -> how ground_truth reads it, in the order the labels are reported
LABELS = {"per-action": "action", "per-action-or-database": "action-or-db", "reward": "reward"}


# ------------------------------------------------------------------ inputs

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


def labels_from_source(traces):
    """The labels a source carries itself: label name -> (case_id -> compliant)."""
    found = {name: ground_truth(traces, key) for name, key in LABELS.items()}
    return {name: truth for name, truth in found.items() if truth}


def load_labels(path):
    """Labels from a file of the same shape: {"label name": {"case_id": true if compliant}}."""
    with open(path, encoding="utf-8") as f:
        return {name: {c: bool(v) for c, v in truth.items()} for name, truth in json.load(f).items()}


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
    tokens from the raw log next to it. None when there is no answered call or no
    raw log."""
    with open(path, encoding="utf-8") as f:
        seconds = {(r["case_id"], x["repeat"]): x["seconds"] for r in json.load(f) for x in r["repeats"]}
    raw = os.path.join(os.path.dirname(path), "raw_" + os.path.basename(path)[len("verdicts_"):-5] + ".jsonl")
    if not os.path.exists(raw):
        return None
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


def verdict_hash(pairs) -> str:
    """One hash over every verdict, its fitness and its violations."""
    rows = []
    for t, ref in pairs:
        r = check(t, ref)
        rows.append([t.case_id, r.compliant, r.fitness, [(v.constraint, v.at) for v in r.violations]])
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


# -------------------------------------------------------- path-sensitivity

def path(trace: Trace, tools_only: bool = False):
    """The sequence of steps, each step the sorted activities issued together, so
    the order in which a source lists calls issued together is not a difference."""
    steps = (sorted(e.activity for e in step if not (tools_only and e.activity in MESSAGES))
             for step in trace.steps())
    return tuple(tuple(s) for s in steps if s)


def same_outcome_pairs(traces: list[Trace], tools_only: bool = False):
    """Pairs of runs of one task that both have reward 1 and differ in their path."""
    by_task = collections.defaultdict(list)
    for t in traces:
        if t.outcome.get("reward") == 1.0 and t.outcome.get("task_id") is not None:
            by_task[t.outcome["task_id"]].append(t)
    return [(a.case_id, b.case_id) for runs in by_task.values() for a, b in itertools.combinations(runs, 2)
            if path(a, tools_only) != path(b, tools_only)]


def separation(verdict, pairs, truth):
    """How many pairs a method separates, and how that compares with the label."""
    judged = [(a, b) for a, b in pairs if a in verdict and b in verdict]
    sep = [(a, b) for a, b in judged if verdict[a] != verdict[b]]
    as_label = [(a, b) for a, b in sep if truth[a] != truth[b] and verdict[a] == truth[a]]
    against = [(a, b) for a, b in sep if truth[a] != truth[b] and verdict[a] != truth[a]]
    both_bad = [(a, b) for a, b in sep if not truth[a] and not truth[b]]
    return {"judged": len(judged), "separated": len(sep), "as_label": len(as_label), "against": len(against),
            "both_violations": len(both_bad),
            "both_compliant": len(sep) - len(as_label) - len(against) - len(both_bad)}


# ------------------------------------------------------------------ report

def evaluation_text(traces: list[Trace], pairs, judges: dict[str, str], labels: dict[str, dict[str, bool]],
                    rounds: int = 1000, repeats: int = 3) -> str:
    """The four criteria. pairs holds (trace, reference) for the engine's verdicts,
    judges maps a Judge's name to its verdict file, and the first label is the main one."""
    start = time.perf_counter()
    engine = {t.case_id: check(t, ref).compliant for t, ref in pairs}
    ms = (time.perf_counter() - start) * 1000 / max(1, len(engine))
    methods = {"engine": engine}
    votes, tokens = {}, {}
    for name, file in judges.items():
        methods[f"Judge {name}"], votes[name], tokens[name] = judge_verdicts(file)

    lines = ["ALIGNMENT, positive class = violation"]
    for label, truth in labels.items():
        lines.append(f"\n  label {label}: {len(truth)} runs, {sum(1 for v in truth.values() if not v)} violations")
        lines.append(f"    {'method':34} {'n':>4} {'prec':>6} {'rec':>6} {'kappa':>6}   95% interval on kappa")
        for name, pred in methods.items():
            ids = sorted(i for i in truth if i in pred)
            if not ids:
                continue
            m = metrics(pred, truth, ids)
            lo, hi = bootstrap(pred, truth, ids, "kappa", rounds)
            lines.append(f"    {name:34} {len(ids):4} {m['precision']:6.2f} {m['recall']:6.2f} {m['kappa']:6.2f}"
                         f"   [{lo:.2f}, {hi:.2f}]")
        for name in judges:
            other = methods[f"Judge {name}"]
            ids = sorted(i for i in truth if i in engine and i in other)
            if ids:
                a_only, b_only, _, _, p = mcnemar(engine, other, truth, ids)
                lines.append(f"    engine vs Judge {name} (n={len(ids)}): only the engine right {a_only}, "
                             f"only the Judge right {b_only}, McNemar p = {p:.4f}")

    hashes = {verdict_hash(pairs) for _ in range(repeats)}
    lines += ["", "REPRODUCIBILITY",
              f"  engine: {repeats} repetitions give {len(hashes)} distinct verdict hash(es)"]
    for name, rep in votes.items():
        fk = fleiss_kappa(rep)
        if fk is None:
            lines.append(f"  Judge {name}: one verdict per run, so agreement cannot be measured")
            continue
        unanimous = sum(1 for v in rep.values() if len(set(v)) == 1)
        c = verdict_cost(judges[name])
        lines.append(f"  Judge {name}: unanimous on {unanimous}/{len(rep)} runs, Fleiss kappa {fk:.2f}"
                     + (f", but {c['cached']} calls were cache copies, which agree by construction"
                        if c and c["cached"] else ""))

    lines += ["", "COST per verdict", f"  engine: 0 tokens, it calls no model, {ms:.3f} ms"]
    for name, file in judges.items():
        c = verdict_cost(file)
        if c:
            lines.append(f"  Judge {name}: {c['prompt_tokens']:,.0f} prompt + {c['completion_tokens']:,.0f} completion "
                         f"tokens, {c['seconds']:.0f} s, over {c['verdicts']} answered calls")
        else:
            n = max(1, sum(x["n"] for x in tokens[name].values()))
            lines.append(f"  Judge {name}: {sum(x['prompt_tokens'] for x in tokens[name].values()) / n:,.0f} prompt + "
                         f"{sum(x['completion_tokens'] for x in tokens[name].values()) / n:,.0f} completion tokens per "
                         f"call, no raw log to leave out cache copies")

    main_label = next(iter(labels))
    truth = labels[main_label]
    scored = [t for t in traces if t.case_id in truth]
    if any(t.outcome.get("reward") == 1.0 and t.outcome.get("task_id") is not None for t in scored):
        lines += ["", f"PATH-SENSITIVITY, against the label {main_label}"]
        for tools_only in (False, True):
            found = same_outcome_pairs(scored, tools_only)
            lines.append(f"  pairs, same task, reward 1, other path ({'tools only' if tools_only else 'all activities'}): "
                         f"{len(found)}, of which the label separates "
                         f"{sum(1 for a, b in found if truth[a] != truth[b])}")
            for name, verdict in methods.items():
                s = separation(verdict, found, truth)
                lines.append(f"    {name:34} separates {s['separated']:4} of {s['judged']:4}   "
                             f"as the label {s['as_label']:3}   against it {s['against']:2}")
    else:
        lines += ["", "PATH-SENSITIVITY: skipped, the source gives no task and no reward to form pairs"]
    return "\n".join(lines)
