"""Run the noise sweep and the task-type engine under ten hash seeds, and compare.

    python experiments/noise_sweep_seeds.py > results/pm4py/noise_sweep_seeds.txt

pm4py's inductive miner returned a different net at noise 0.2 in different runs
of noise_sweep.py (found 2026-10-26). The cause is Python's hash seed, which
changes the iteration order of sets and dicts between processes. This script
runs noise_sweep.py once per PYTHONHASHSEED from 0 to 9, each into its own
temporary directory, and reports per threshold what changes: the activities
the net lacks, fitness, the number of runs flagged, kappa, and the walk-through
case of Chapter 6. As a contrast it computes the verdicts of the declarative
engine with the per-task-type model under the same ten seeds and compares
their hash.
"""
from __future__ import annotations

import collections
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SEEDS = range(10)

ENGINE = r"""
import hashlib, json, sys
sys.path.insert(0, 'experiments')
import task_type_conformance as ttc
from agentpm.analysis.conformance import check
from agentpm.ingest import tau2
for domain in ('airline', 'retail'):
    task_type, type_model, prepare = ttc.domain_rules(domain)
    rows = []
    for t in tau2.ingest(ttc.DEFAULT_DATA[domain]):
        if not t.outcome.get('action_checks'):
            continue
        r = check(prepare(t), type_model(task_type(t)))
        rows.append([t.case_id, r.compliant, r.fitness, [(v.constraint, v.at) for v in r.violations]])
    print(domain, hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16])
"""


def parse(text):
    """Per noise level: the net's size, missing activities, both replay lines, task 29."""
    out = {}
    for block in text.split("\nnoise ")[1:]:
        level = block.split(":")[0]
        head = block.splitlines()[0]
        d = {"net": re.search(r"(\d+ places, \d+ transitions)", head).group(1),
             "missing": re.search(r"missing (.*)$", head).group(1)}
        for name in ("token replay", "alignments"):
            m = re.search(name + r"\s+fitness compliant ([\d.]+) violating ([\d.]+) \| flags\s+(\d+)/172 "
                          r"\| prec ([\d.]+) rec ([\d.]+) F1 [\d.]+ kappa (-?[\d.]+)", block)
            d[name] = m.groups()
        m = re.search(r"task 29 trial 0: token replay ([\d.]+), alignment ([\d.]+); the alignment's log moves: (.*)", block)
        d["task29"] = m.groups()
        out[level] = d
    return out


def main():
    env0 = dict(os.environ)
    runs, engine = {}, {}
    for seed in SEEDS:
        env = dict(env0, PYTHONHASHSEED=str(seed))
        with tempfile.TemporaryDirectory() as tmp:
            res = subprocess.run([sys.executable, os.path.join(HERE, "noise_sweep.py"), "--out", tmp],
                                 cwd=REPO, env=env, capture_output=True, text=True, check=True)
        runs[seed] = parse(res.stdout)
        res = subprocess.run([sys.executable, "-c", ENGINE], cwd=REPO, env=env,
                             capture_output=True, text=True, check=True)
        engine[seed] = res.stdout.split()

    print("Discover-then-check under ten values of PYTHONHASHSEED (0 to 9), tau2 airline, 172 runs\n")
    for level in runs[0]:
        print(f"noise {level}")
        nets = collections.Counter((runs[s][level]["net"], runs[s][level]["missing"]) for s in SEEDS)
        for (net, missing), n in nets.most_common():
            seeds = [s for s in SEEDS if (runs[s][level]["net"], runs[s][level]["missing"]) == (net, missing)]
            print(f"  {n:2} seeds {seeds}: {net}, missing {missing}")
        for name in ("token replay", "alignments"):
            rows = collections.Counter(runs[s][level][name] for s in SEEDS)
            for (ok, bad, flags, p, r, k), n in rows.most_common():
                print(f"  {name:13} {n:2} seeds: fitness compliant {ok} violating {bad} | flags {flags} "
                      f"| prec {p} rec {r} kappa {k}")
        cases = collections.Counter(runs[s][level]["task29"] for s in SEEDS)
        for (tr, al, moves), n in cases.most_common():
            print(f"  task 29 trial 0 {n:2} seeds: token replay {tr}, alignment {al}, log moves {moves}")
        print()

    print("Declarative engine, per-task-type models, verdict hash per seed")
    for domain_i, domain in enumerate(("airline", "retail")):
        hashes = collections.Counter(engine[s][2 * domain_i + 1] for s in SEEDS)
        print(f"  {domain}: {len(hashes)} distinct hash(es) over {len(SEEDS)} seeds: {dict(hashes)}")


if __name__ == "__main__":
    sys.exit(main())
