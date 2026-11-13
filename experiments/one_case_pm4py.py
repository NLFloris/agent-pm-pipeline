"""One run, end to end through pm4py, so the mechanics are visible.

    python experiments/one_case_pm4py.py [--case 2922cd65] [--out results/pm4py/one_case]

Takes one tau2 run and shows, in order: the rows pm4py receives, the model
of that single run, token-based replay against the net mined from the
compliant runs (which tokens were missing or left over), the alignment move
by move (synchronous, model-only, log-only), and our declarative verdict on
the same run, so the three can be read side by side.
"""
from __future__ import annotations

import argparse
import os
import sys

import pm4py

from agentpm.analysis.conformance import check
from agentpm.ingest import tau2
from discover import flat_log, gold_reference
from data_dir import OPUS_AIRLINE   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="2922cd65")
    ap.add_argument("--out", default="results/pm4py/one_case")
    ap.add_argument("--data", default=OPUS_AIRLINE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    traces = tau2.ingest(a.data)
    t = next(x for x in traces if x.case_id.startswith(a.case))
    truth = all(c.get("action_match") for c in t.outcome["action_checks"])
    print(f"CASE {t.case_id}   task {t.outcome['task_id']} trial {t.outcome['trial']}")
    print(f"ground truth: {'compliant' if truth else 'PROTOCOL VIOLATION'}   "
          f"reward {t.outcome['reward']}")
    gold = [x["name"] for x in t.outcome["gold_actions"] if x["requestor"] == "assistant"]
    print(f"gold actions: {' -> '.join(gold)}\n")

    # 1. what pm4py receives
    log = flat_log([t])
    print("1. THE ROWS PM4PY RECEIVES (tool calls only, messages dropped)")
    for i, r in log.iterrows():
        print(f"   {i:2}  {r['time:timestamp'].strftime('%H:%M:%S')}  {r['org:resource']:10} {r['concept:name']}")

    # 2. model of this single run
    dfg, s, e = pm4py.discover_dfg(log)
    pm4py.save_vis_dfg(dfg, s, e, os.path.join(a.out, "this_run_dfg.png"), rankdir="TB")
    net1, im1, fm1 = pm4py.discover_petri_net_inductive(log)
    pm4py.save_vis_petri_net(net1, im1, fm1, os.path.join(a.out, "this_run_petrinet.png"), rankdir="TB")
    print(f"\n2. MODEL OF THIS RUN ALONE   dfg {len(dfg)} edges, petri net "
          f"{len(net1.places)} places / {len(net1.transitions)} transitions -> {a.out}/this_run_*.png")

    # 3. the reference net mined from compliant runs
    good = [x for x in traces if x.outcome.get("action_checks")
            and all(c.get("action_match") for c in x.outcome["action_checks"])]
    net, im, fm = pm4py.discover_petri_net_inductive(flat_log(good), noise_threshold=0.2)
    labels = sorted({tr.label for tr in net.transitions if tr.label})
    print(f"\n3. REFERENCE NET mined from {len(good)} compliant runs, labelled transitions:")
    print("   " + ", ".join(labels))
    missing_from_model = [g for g in dict.fromkeys(gold) if g not in labels]
    if missing_from_model:
        print(f"   NOTE: gold actions absent from the mined net: {missing_from_model}")

    # 4. token-based replay of this run
    tbr = pm4py.conformance_diagnostics_token_based_replay(log, net, im, fm)[0]
    print(f"\n4. TOKEN-BASED REPLAY of this run against the reference net")
    print(f"   trace fitness {tbr['trace_fitness']:.4f}   is fit: {tbr['trace_is_fit']}")
    print(f"   tokens  produced {tbr['produced_tokens']}  consumed {tbr['consumed_tokens']}  "
          f"missing {tbr['missing_tokens']}  remaining {tbr['remaining_tokens']}")
    if tbr.get("activated_transitions"):
        fired = [tr.label for tr in tbr["activated_transitions"] if tr.label]
        print(f"   labelled transitions fired: {fired}")

    # 5. alignment, move by move
    ali = pm4py.conformance_diagnostics_alignments(log, net, im, fm)[0]
    print(f"\n5. ALIGNMENT of this run against the reference net")
    print(f"   fitness {ali['fitness']:.4f}   cost {ali['cost']}")
    print(f"   {'log side':28} {'model side':28} move type")
    for lm, mm in ali["alignment"]:
        if lm == ">>" and mm is None:
            continue                      # silent model move, no information
        if lm == ">>":
            kind = "MODEL MOVE  (model expected this, log did not do it)"
        elif mm == ">>":
            kind = "LOG MOVE    (log did this, model did not allow it here)"
        else:
            kind = "sync"
        print(f"   {str(lm):28} {str(mm):28} {kind}")

    # 6. our declarative engine on the same run
    r = check(t, gold_reference(t))
    print(f"\n6. OUR DECLARATIVE ENGINE on the same run")
    print(f"   fitness {r.fitness:.4f}   compliant {r.compliant}   {r.n_constraints} constraints")
    for v in r.violations:
        print(f"   X {v.constraint}: {v.description}")

    print(f"\n7. SIDE BY SIDE")
    print(f"   truth: {'compliant' if truth else 'violation'}  |  "
          f"token replay fit={tbr['trace_fitness']:.2f}  |  alignment fit={ali['fitness']:.2f}  |  "
          f"declarative fit={r.fitness:.2f} ({'ok' if r.compliant else 'violation'})")


if __name__ == "__main__":
    sys.exit(main())
