"""A sample of runs, stratified by task type, for a Judge that cannot judge them all.

    python experiments/judge/sample_runs.py --domain retail --n 120

glm-5.3-flash is served from one machine with one request at a time for every
user of the proxy, so it judges a sample of the retail runs instead of all 448.
The sample is proportional per task type, at least one run per type, drawn with
a fixed seed and without looking at any label or verdict, so it can be redrawn
exactly and says nothing about which runs are hard.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from task_type_conformance import DEFAULT_DATA, domain_rules   # noqa: E402

from agentpm.ingest import tau2   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=sorted(DEFAULT_DATA), default="retail")
    ap.add_argument("--data")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    task_type = domain_rules(a.domain)[0]
    traces = [t for t in tau2.ingest(a.data or DEFAULT_DATA[a.domain]) if t.outcome.get("action_checks")]
    by_type = collections.defaultdict(list)
    for t in sorted(traces, key=lambda t: t.case_id):
        by_type[task_type(t)].append(t.case_id)

    rng = random.Random(a.seed)
    exact = {k: a.n * len(v) / len(traces) for k, v in by_type.items()}
    share = {k: max(1, int(x)) for k, x in exact.items()}
    # fill up to n by largest remainder, then take any surplus from the largest types
    for k in sorted(exact, key=lambda k: (exact[k] - int(exact[k]), k), reverse=True):
        if sum(share.values()) >= a.n:
            break
        if share[k] < len(by_type[k]):
            share[k] += 1
    while sum(share.values()) > a.n:
        share[max(share, key=lambda k: (share[k], k))] -= 1
    picked = sorted(c for k, v in by_type.items() for c in rng.sample(v, share[k]))

    out = a.out or f"results/judge/{a.domain}/sample_{a.n}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"domain": a.domain, "n": len(picked), "seed": a.seed,
                   "per_type": {k: share[k] for k in sorted(share)}, "case_ids": picked}, f, indent=1)
    print(f"{len(picked)} of {len(traces)} runs over {len(share)} task types -> {out}")


if __name__ == "__main__":
    sys.exit(main())
