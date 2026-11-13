"""Two Judges and the engine, on airline and retail, against three labels.

    python experiments/judge/compare_judges.py > results/judge/two_judges.txt

The second Judge (glm-5.3-flash) validates the first (gpt-oss:120b): if the two
agree with each other and land in the same place against the labels, the Judge
results are not an artefact of one model. glm judged every airline run and a
stratified sample of 120 retail runs (sample_runs.py), because the proxy serves
it one request at a time; on retail every method is therefore also scored on
those 120 runs alone, so that all of them are compared on the same runs.

Reported per domain and label: alignment of each method (bootstrap 95% interval
on kappa, drawing single runs and, below it, whole tasks, since the
runs repeat each task), McNemar of the engine against each Judge, the two Judges' agreement
with each other per condition, the violations the engine flags and the
output-only Judge passes, how often that Judge flags the violations the engine
misses and the runs the label calls compliant, what the policy Judges cite on
runs the label calls compliant, and cost. Self-agreement (five fresh samples per run, cache
off) is reported from results/judge/self_agreement.

A verdict the Judge returned as malformed JSON counts as not judged, so n
differs per method; that rate is reported too.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from evaluate import (bootstrap, fleiss_kappa, ground_truth, judge_verdicts, mcnemar, metrics,   # noqa: E402
                      verdict_cost)
from task_type_conformance import DEFAULT_DATA   # noqa: E402

from agentpm.evaluation import task_clusters   # noqa: E402
from agentpm.ingest import tau2   # noqa: E402

DOMAINS = {
    "airline": {"judge": "results/judge", "pm": "results/task_type_conformance.json", "sample": None},
    "retail": {"judge": "results/judge/retail", "pm": "results/task_type_conformance_retail.json",
               "sample": "results/judge/retail/sample_120.json"},
}
LABELS = ("action", "action-or-db", "reward")
CONDITIONS = ("gold-actions", "full", "output-only")
SHOWN = {"gold-actions": "required actions", "full": "policy", "output-only": "output only"}
JUDGES = ("gpt-oss-120b", "glm-5-3-flash")


# Rules the policy Judge cites, grouped by keyword. The Judge words the same rule
# differently from run to run, so exact strings undercount; a run counts once
# per theme. Patterns follow the policy's own wording.
THEMES = [("unsupported information", r"information|knowledge|make up"),
          ("transfer to a human agent", r"transfer"),
          ("one tool call at a time", r"one tool call|at a time|same time|simultaneous")]

def cohen(a, b, ids):
    """Agreement of two verdict sets with each other, chance-corrected."""
    n = len(ids)
    if not n:
        return None, 0
    agree = sum(1 for i in ids if a[i] == b[i]) / n
    pa, pb = sum(1 for i in ids if a[i]) / n, sum(1 for i in ids if b[i]) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return ((agree - pe) / (1 - pe) if pe < 1 else 1.0), n


def load(domain):
    cfg = DOMAINS[domain]
    traces = tau2.ingest(DEFAULT_DATA[domain])
    with open(cfg["pm"], encoding="utf-8") as f:
        methods = {"engine, task type": {r["case_id"]: bool(r["compliant"]) for r in json.load(f)}}
    costs, records = {}, {}
    for cond in CONDITIONS:
        for judge in JUDGES:
            path = os.path.join(cfg["judge"], f"verdicts_{cond}_{judge}.json")
            if os.path.exists(path):
                name = f"{SHOWN[cond]}, {judge}"
                methods[name], _, _ = judge_verdicts(path)
                costs[name] = verdict_cost(path)
                with open(path, encoding="utf-8") as f:
                    records[name] = {r["case_id"]: r for r in json.load(f)}
    sample = None
    if cfg["sample"]:
        with open(cfg["sample"], encoding="utf-8") as f:
            sample = set(json.load(f)["case_ids"])
    return traces, methods, costs, records, sample


def alignment(methods, truth, restrict=None, cluster=None):
    pm = methods["engine, task type"]
    print(f"  {'method':34} {'n':>4} {'prec':>5} {'rec':>5} {'kappa':>6}  95% CI       engine/judge only  McNemar p")
    for name, pred in methods.items():
        ids = sorted(i for i in truth if i in pred and (restrict is None or i in restrict))
        if not ids:
            continue
        m = metrics(pred, truth, ids)
        lo, hi = bootstrap(pred, truth, ids, "kappa")
        extra = ""
        if name != "engine, task type":
            both = [i for i in ids if i in pm]
            a_only, b_only, _, _, p = mcnemar(pm, pred, truth, both)
            extra = f"  {a_only:5} / {b_only:<5}      {p:.4f}"
        print(f"  {name:34} {len(ids):4} {m['precision']:5.2f} {m['recall']:5.2f} {m['kappa']:6.2f}  "
              f"[{lo:5.2f}, {hi:4.2f}]{extra}")
    if cluster:
        # The runs repeat each task four times, so they are not independent. These
        # intervals draw whole tasks, with all their runs, instead of single runs.
        tasks = {cluster[i] for i in truth if restrict is None or i in restrict}
        print(f"  95% CI drawing whole tasks ({len(tasks)} tasks):")
        for name, pred in methods.items():
            ids = sorted(i for i in truth if i in pred and (restrict is None or i in restrict))
            if not ids:
                continue
            lo, hi = bootstrap(pred, truth, ids, "kappa", cluster=cluster)
            plo, phi = bootstrap(pred, truth, ids, "precision", cluster=cluster)
            print(f"    {name:34} kappa [{lo:5.2f}, {hi:4.2f}]   precision [{plo:4.2f}, {phi:4.2f}]")


def main():
    for domain in DOMAINS:
        traces, methods, costs, records, sample = load(domain)
        cluster = task_clusters(traces)
        print(f"\n{'=' * 100}\n{domain.upper()}, {len(traces)} runs, agent claude-opus-4-5")
        for lab in LABELS:
            truth = ground_truth(traces, lab)
            print(f"\n-- label {lab}: {len(truth)} scorable runs, {sum(1 for v in truth.values() if not v)} violations")
            alignment({k: v for k, v in methods.items() if sample is None or "glm" not in k}, truth,
                      cluster=cluster)
            if sample:
                st = {i: v for i, v in truth.items() if i in sample}
                print(f"   the same, on the stratified sample only ({len(st)} runs, "
                      f"{sum(1 for v in st.values() if not v)} violations), every method on the same runs:")
                alignment(methods, truth, sample, cluster=cluster)

        print("\n-- the two Judges against each other (Cohen's kappa on the runs both judged)")
        for cond in CONDITIONS:
            a, b = (methods.get(f"{SHOWN[cond]}, {j}") for j in JUDGES)
            if a and b:
                ids = sorted(set(a) & set(b))
                k, n = cohen(a, b, ids)
                same = sum(1 for i in ids if a[i] == b[i])
                print(f"  {SHOWN[cond]:17} n {n:4}  same verdict {same:4} ({same / n:.0%})  kappa {k:.2f}")

        truth = ground_truth(traces, "action")
        pm = methods["engine, task type"]
        for name in [n for n in methods if n.startswith("output only")]:
            passed = sorted(cid for cid, ok in truth.items()
                            if not ok and pm.get(cid) is False and methods[name].get(cid) is True)
            print(f"\n-- violations the engine flags and the {name} Judge passes: {len(passed)}"
                  f" of {sum(1 for cid, ok in truth.items() if not ok and pm.get(cid) is False)}")
            for cid in passed:
                r = records[name][cid]
                print(f"  task {r.get('task_id')} trial {r.get('trial')}")
            missed = [cid for cid, ok in truth.items() if not ok and pm.get(cid) is True]
            compliant = [cid for cid, ok in truth.items() if ok]
            for what, ids in (("violations the engine misses", missed), ("runs the label calls compliant", compliant)):
                judged = [cid for cid in ids if cid in methods[name]]
                print(f"  on the {len(ids)} {what}: flags "
                      f"{sum(1 for cid in judged if methods[name][cid] is False)} of the {len(judged)} it judged")
        where = "of the sample " if sample else ""
        print(f"\n-- what the policy Judges cite on runs {where}the per-action label calls compliant (top 6)")
        flagged_by = {}
        for judge in JUDGES:
            name = f"policy, {judge}"
            if name not in records:
                continue
            rules = collections.Counter()
            themes = collections.Counter()
            flagged = 0
            flagged_by[judge] = set()
            for cid, r in records[name].items():
                if sample and cid not in sample:
                    continue                      # both Judges on the same runs
                if truth.get(cid) and methods[name].get(cid) is False:
                    flagged += 1
                    flagged_by[judge].add(cid)
                    for x in r["repeats"]:
                        cited = [str(rule) for rule in x.get("violated_rules") or []]
                        for rule in cited:
                            rules[rule[:90]] += 1
                        text = " ".join(cited).lower()
                        for theme, pattern in THEMES:
                            if re.search(pattern, text):
                                themes[theme] += 1
                        break
            print(f"  {judge}: flags {flagged} compliant runs")
            for rule, n in rules.most_common(6):
                print(f"    {n:3}  {rule}")
            print("    runs citing, grouped by theme: "
                  + ", ".join(f"{theme} {themes[theme]}" for theme, _ in THEMES))
        if len(flagged_by) == 2:
            print(f"  flagged by both: {len(set.intersection(*flagged_by.values()))}")

        print("\n-- runs judged (a malformed, empty or failed answer is not a verdict), and cost per")
        print("   verdict over the calls the model answered; the proxy's cache copies are left out")
        for name, c in costs.items():
            judged = sum(1 for r in records[name].values() if any(x["parsed"] for x in r["repeats"]))
            print(f"  {name:34} judged {judged}/{len(records[name])}  "
                  f"{c['prompt_tokens']:,.0f} prompt + {c['completion_tokens']:,.0f} completion tokens, "
                  f"{c['seconds']:.0f} s per verdict ({c['cached']} cache copies left out)")
            why = failures(os.path.join(DOMAINS[domain]["judge"], "raw_" + name.split(", ")[0]
                                        .replace("required actions", "gold-actions").replace("policy", "full")
                                        .replace("output only", "output-only") + "_" + name.split(", ")[1] + ".jsonl"),
                           {cid for cid, r in records[name].items() if not any(x["parsed"] for x in r["repeats"])})
            if why:
                print(f"  {'':34} last answer of the runs not judged: {dict(why)}")
        print("  engine: 0 tokens, well under a millisecond per run")

    print(f"\n{'=' * 100}\nSELF-AGREEMENT, five fresh samples per run (proxy cache off), first 30 airline runs")
    for path in sorted(glob.glob("results/judge/self_agreement/verdicts_*.json")):
        _, rep, _ = judge_verdicts(path)
        if len(rep) < 30:
            print(f"  {os.path.basename(path)[9:-5]:34} in progress, {len(rep)} of 30 runs")
            continue
        un = sum(1 for v in rep.values() if len(set(v)) == 1)
        full = {k: v for k, v in rep.items() if len(v) == 5}
        tag = os.path.basename(path)[9:-5]
        majority, _, _ = judge_verdicts(path)
        original, _, _ = judge_verdicts(os.path.join("results/judge", os.path.basename(path)))
        same = sum(1 for k in majority if k in original and majority[k] == original[k])
        print(f"  {tag:34} unanimous on {un}/{len(rep)} runs, Fleiss kappa {fleiss_kappa(rep):.2f} "
              f"(over the first {min(len(v) for v in rep.values())} answers per run; "
              f"{fleiss_kappa(full):.2f} over the {len(full)} runs with five), "
              f"majority equals the original verdict on {same}/{len(majority)}")

    print("\nTHE CACHE, airline gpt-oss:120b: new response ids among the five repeats, and the run at")
    print("temperature 0.7 against the default run (first answer of each run)")
    for cond in ("gold-actions", "full"):
        fresh, copies = cache_split(f"results/judge/raw_{cond}_gpt-oss-120b.jsonl")
        print(f"  {cond:13} {fresh} new ids, {copies} repeats of an earlier id")
    a = first_answers("results/judge/raw_full_gpt-oss-120b.jsonl")
    b = first_answers("results/judge/raw_full_gpt-oss-120b_t0.7.jsonl")
    shared = [c for c in b if c in a]
    print(f"  t0.7 first answer differs from the default first answer on "
          f"{sum(1 for c in shared if a[c] != b[c])}/{len(shared)} runs")
    return 0


def failures(raw, cases):
    """Why the last call of each run without a verdict gave none."""
    sys.path.insert(0, HERE)
    from run_judge import parse_verdict
    last = {}
    with open(raw, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["case_id"] in cases:
                last[r["case_id"]] = r
    why = collections.Counter()
    for r in last.values():
        text = (((r.get("response") or {}).get("choices") or [{}])[0].get("message") or {}).get("content")
        if r.get("error"):
            why["proxy error"] += 1
        elif not (text or "").strip():
            why["empty answer"] += 1
        elif parse_verdict(text) is None:
            why["malformed answer"] += 1
    return why


def cache_split(raw):
    seen, fresh, copies = collections.defaultdict(set), 0, 0
    with open(raw, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rid = (r.get("response") or {}).get("id")
            if not rid:
                continue
            if rid in seen[r["case_id"]]:
                copies += 1
            else:
                fresh += 1
                seen[r["case_id"]].add(rid)
    return fresh, copies


def first_answers(raw):
    out = {}
    with open(raw, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = (((r.get("response") or {}).get("choices") or [{}])[0].get("message") or {}).get("content")
            if text and r["case_id"] not in out:
                out[r["case_id"]] = text
    return out


if __name__ == "__main__":
    sys.exit(main())
