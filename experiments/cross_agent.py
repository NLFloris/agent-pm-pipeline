"""The task-type models on every agent tau2-bench publishes, not only the one they were written on.

    python experiments/cross_agent.py > results/cross_agent.txt

The airline models were written while looking at claude-opus-4-5's runs, so a
result on those runs alone cannot separate the method from the fit. The public
results bucket holds the same 50 airline tasks (and the same 114 retail tasks)
run by other agent models, under the same harness and the same checks. This
script applies the models unchanged to each of them. No model is called; every
agent's runs were produced by Sierra and are read from the mirror.

Three labels, because the experiment showed that the choice matters:

  action        every gold action matched, arguments included. The label of
                Ch. 6. It cannot see a write the task did not ask for, since it
                only checks that the listed actions occurred.
  action-or-db  a violation if any gold action failed to match OR the final
                database differs from the expected one. The benchmark's own
                db_check closes the blind spot for unrequested writes.
  reward        the benchmark's reward; outcome, not process, and in retail it
                also includes natural-language assertions about what was said.

Two variants of the confirmation rule: as frozen, and looking through reads
(see retail_task_types.py for why the second exists).

Excluded: toolorchestra_nvidia, an orchestration system with one trial per task
rather than an agent model with four.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task_type_conformance import TAU2, domain_rules   # noqa: E402

import retail_task_types   # noqa: E402
import task_type_conformance   # noqa: E402
from agentpm.analysis.conformance import check   # noqa: E402
from agentpm.ingest import tau2   # noqa: E402

EXCLUDE = ("toolorchestra",)
LABELS = ("action", "action-or-db", "reward")
DEV_AGENT = "claude-opus-4-5"   # the agent the airline models were written on
WRITES = {"airline": task_type_conformance.WRITES, "retail": retail_task_types.WRITES}


def files(domain):
    out = []
    for f in sorted(glob.glob(f"{TAU2}/*/trajectories/*.json")):
        base = os.path.basename(f)
        if any(x in f for x in EXCLUDE):
            continue
        if f"_{domain}_" in base or base.startswith(f"{domain}_") or base.endswith(f"-{domain}.json"):
            out.append((f.split("/submissions/")[1].split("/")[0], f))
    return out


def violation(t, label):
    o = t.outcome
    act = not all(c.get("action_match") for c in o["action_checks"])
    if label == "action":
        return act
    if label == "reward":
        return o.get("reward") != 1.0
    return act or not (o.get("db_check") or {}).get("db_match", True)


def metrics(pairs):
    """pairs of (flagged, violation); positive class = violation."""
    tp = sum(1 for e, v in pairs if e and v)
    fp = sum(1 for e, v in pairs if e and not v)
    fn = sum(1 for e, v in pairs if not e and v)
    n = len(pairs)
    tn = n - tp - fp - fn
    acc = (tp + tn) / n if n else 0.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n) if n else 0.0
    k = (acc - pe) / (1 - pe) if pe < 1 else 0.0
    return {"n": n, "violations": tp + fn, "flags": tp + fp, "precision": p, "recall": r, "kappa": k}


def bootstrap(pairs, key, rounds=1000, seed=0):
    rng = random.Random(seed)
    vals = sorted(metrics([rng.choice(pairs) for _ in pairs])[key] for _ in range(rounds))
    return vals[int(0.025 * rounds)], vals[int(0.975 * rounds)]


def misses(t, writes):
    """Why the per-action check failed, per failing gold action: write or read,
    and whether an action of that name occurred at all (then the arguments differ)."""
    did = {e.activity for e in t.events}
    out = set()
    for c in t.outcome["action_checks"]:
        if c.get("action_match"):
            continue
        name = (c.get("action") or {}).get("name")
        kind = "write" if name in writes else "read" if name != "transfer_to_human_agents" else "transfer"
        out.add(f"{kind} {'wrong arguments' if name in did else 'missing'}")
    return sorted(out) or ["db only"]


def run(domain, reads_transparent):
    task_type, type_model, prepare = domain_rules(domain)
    writes = WRITES[domain]
    per_agent = {}
    for agent, path in files(domain):
        rows = []
        for t in tau2.ingest(path):
            if not t.outcome.get("action_checks"):
                continue
            r = check(prepare(t), type_model(task_type(t), reads_transparent))
            rows.append({"flag": not r.compliant,
                         "kinds": sorted({v.constraint.split("(")[0] for v in r.violations}),
                         "misses": misses(t, writes),
                         **{lab: violation(t, lab) for lab in LABELS}})
        per_agent[agent] = rows
    return per_agent


def report(domain, reads_transparent):
    per_agent = run(domain, reads_transparent)
    variant = "reads transparent" if reads_transparent else "frozen"
    print(f"\n=== {domain}, confirmation rule {variant}, {len(per_agent)} agents ===")
    print(f"  {'agent':40} {'n':>4} " + "  ".join(f"{lab + ': viol prec  rec kappa':>31}" for lab in LABELS))
    pooled = collections.defaultdict(list)
    held_out = collections.defaultdict(list)
    for agent, rows in per_agent.items():
        cells = []
        for lab in LABELS:
            pairs = [(r["flag"], r[lab]) for r in rows]
            pooled[lab] += pairs
            if not agent.startswith(DEV_AGENT):
                held_out[lab] += pairs
            m = metrics(pairs)
            cells.append(f"{m['violations']:>14} {m['precision']:4.2f} {m['recall']:4.2f} {m['kappa']:5.2f}")
        print(f"  {agent:40} {len(rows):4} " + "  ".join(cells))
    for name, group in (("POOLED, all agents", pooled), (f"POOLED, without {DEV_AGENT}", held_out)):
        print(f"  {name}")
        for lab in LABELS:
            m = metrics(group[lab])
            lo, hi = bootstrap(group[lab], "kappa")
            plo, phi = bootstrap(group[lab], "precision")
            print(f"    {lab:13} n {m['n']:5}  viol {m['violations']:5}  flags {m['flags']:5}  "
                  f"prec {m['precision']:.2f} [{plo:.2f}, {phi:.2f}]  rec {m['recall']:.2f}  "
                  f"kappa {m['kappa']:.2f} [{lo:.2f}, {hi:.2f}]")
    # what the false alarms are, against the Ch. 6 label and the closed one
    for lab in ("action", "action-or-db"):
        fa = collections.Counter("+".join(r["kinds"]) for rows in per_agent.values()
                                 for r in rows if r["flag"] and not r[lab])
        print(f"  false alarms against '{lab}', by constraint kind: {dict(fa.most_common())}")
    # what the missed violations are; a run can miss for several reasons
    for lab in ("action", "action-or-db"):
        rows = [r for rs in per_agent.values() for r in rs if not r["flag"] and r[lab]]
        why = collections.Counter(m for r in rows for m in r["misses"])
        only_args = sum(1 for r in rows if all(m.endswith("wrong arguments") for m in r["misses"]))
        print(f"  missed against '{lab}' ({len(rows)} runs), failing gold actions: {dict(why.most_common())}; "
              f"arguments only: {only_args}")
    return {a: rows for a, rows in per_agent.items()}


def main():
    out = {}
    for domain in ("airline", "retail"):
        for rt in (False, True):
            out[f"{domain}{'_reads_transparent' if rt else ''}"] = report(domain, rt)
    os.makedirs("results", exist_ok=True)
    with open("results/cross_agent.json", "w") as f:
        json.dump(out, f)
    print("\nwrote results/cross_agent.json")


if __name__ == "__main__":
    sys.exit(main())
