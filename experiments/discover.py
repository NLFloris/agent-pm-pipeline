"""Process mining with pm4py's own algorithms, next to our declarative engine.

    python experiments/discover.py --data ~/Downloads --out results/pm4py

Everything else in this repository that draws a model counts directly-follows
pairs itself. This script instead calls the established algorithms, so the
thesis can say which process mining it used and show where the two agree.

Per corpus:
  discovery     pm4py.discover_dfg, discover_petri_net_inductive,
                discover_process_tree_inductive, and for the object-centric
                logs discover_ocdfg and discover_oc_petri_net
  conformance   discover-then-check, the second acquisition route of the
                plan: mine a Petri net from the runs the ground truth marks
                compliant, then replay every run against it with token-based
                replay and with alignments. Both give a fitness per trace.
  comparison    pm4py's procedural fitness against our declarative verdict and
                against the ground truth, on the same runs

Timestamps in the flat log are synthetic and encode position only, as
everywhere in this repository, because parallel tool calls share a wall clock.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pm4py

from agentpm.analysis.conformance import check
from agentpm.ingest import local_magentic, tau2, traceelephant
from agentpm.reference import Reference

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
MESSAGES = {"assistant_message", "user_message"}


def flat_log(traces, drop=MESSAGES):
    rows = []
    for t in traces:
        step = 0
        for e in t.ordered():
            if e.activity in drop:
                continue
            rows.append({"case:concept:name": t.case_id, "concept:name": e.activity,
                         "time:timestamp": BASE + timedelta(seconds=step),
                         "org:resource": e.resource})
            step += 1
    df = pd.DataFrame(rows)
    return pm4py.format_dataframe(df, case_id="case:concept:name",
                                  activity_key="concept:name", timestamp_key="time:timestamp")


def gold_reference(trace):
    names = [a["name"] for a in trace.outcome.get("gold_actions", [])
             if a.get("requestor") == "assistant"]
    pairs = [[a, b] for a, b in zip(names, names[1:]) if a != b]
    return Reference(name="gold", required=sorted(set(names)), precedence=pairs)


def kappa_f1(pred_viol, truth_viol):
    tp = sum(1 for p, t in zip(pred_viol, truth_viol) if p and t)
    fp = sum(1 for p, t in zip(pred_viol, truth_viol) if p and not t)
    fn = sum(1 for p, t in zip(pred_viol, truth_viol) if not p and t)
    tn = len(pred_viol) - tp - fp - fn
    n = len(pred_viol)
    acc = (tp + tn) / n
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n)
    k = (acc - pe) / (1 - pe) if pe < 1 else 0.0
    return p, r, f1, k


def discover(name, traces, out, noise=0.2):
    log = flat_log(traces)
    print(f"\n[{name}]  {len(traces)} traces, {len(log)} tool events, "
          f"{log['concept:name'].nunique()} activities")

    dfg, start, end = pm4py.discover_dfg(log)
    pm4py.save_vis_dfg(dfg, start, end, os.path.join(out, f"{name}_pm4py_dfg.png"), rankdir="TB")
    print(f"  pm4py.discover_dfg                 {len(dfg)} edges")

    net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=noise)
    pm4py.save_vis_petri_net(net, im, fm, os.path.join(out, f"{name}_pm4py_petrinet.png"), rankdir="TB")
    print(f"  discover_petri_net_inductive       {len(net.places)} places, "
          f"{len(net.transitions)} transitions, noise {noise}")

    tree = pm4py.discover_process_tree_inductive(log, noise_threshold=noise)
    pm4py.save_vis_process_tree(tree, os.path.join(out, f"{name}_pm4py_tree.png"))
    print(f"  discover_process_tree_inductive    {str(tree)[:90]}...")
    return log, (net, im, fm)


def discover_ocel(name, path, out):
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return
    ocel = pm4py.read_ocel2_json(path)
    ocdfg = pm4py.discover_ocdfg(ocel)
    pm4py.save_vis_ocdfg(ocdfg, os.path.join(out, f"{name}_pm4py_ocdfg.png"), rankdir="TB",
                         act_threshold=5, edge_threshold=5)
    ocpn = pm4py.discover_oc_petri_net(ocel)
    pm4py.save_vis_ocpn(ocpn, os.path.join(out, f"{name}_pm4py_ocpn.png"), rankdir="TB")
    types = sorted(ocel.objects["ocel:type"].unique())
    print(f"  discover_ocdfg / discover_oc_petri_net on {os.path.basename(path)}: "
          f"{len(ocel.events)} events, object types {types}")


def discover_then_check(traces, out):
    """Mine from compliant runs, replay all runs: pm4py conformance."""
    truth = {t.case_id: all(c.get("action_match") for c in t.outcome["action_checks"])
             for t in traces if t.outcome.get("action_checks")}
    scored = [t for t in traces if t.case_id in truth]
    good = [t for t in scored if truth[t.case_id]]
    print(f"\n[discover-then-check on tau2]  mine from {len(good)} compliant runs, "
          f"replay {len(scored)}")

    train = flat_log(good)
    net, im, fm = pm4py.discover_petri_net_inductive(train, noise_threshold=0.2)
    pm4py.save_vis_petri_net(net, im, fm, os.path.join(out, "tau2_mined_from_compliant.png"), rankdir="TB")
    print(f"  model: {len(net.places)} places, {len(net.transitions)} transitions")

    test = flat_log(scored)
    tbr = pm4py.conformance_diagnostics_token_based_replay(test, net, im, fm)
    ali = pm4py.conformance_diagnostics_alignments(test, net, im, fm)
    cases = list(test["case:concept:name"].unique())
    by_case = {c: t for c, t in zip(cases, scored)}  # format_dataframe keeps first-seen order
    tbr_fit = {c: d["trace_fitness"] for c, d in zip(cases, tbr)}
    ali_fit = {c: d["fitness"] for c, d in zip(cases, ali)}

    rows = []
    for t in scored:
        c = t.case_id
        decl = check(t, gold_reference(t))
        rows.append({
            "case_id": c, "truth_compliant": truth[c],
            "tbr_fitness": round(tbr_fit.get(c, float("nan")), 4),
            "alignment_fitness": round(ali_fit.get(c, float("nan")), 4),
            "declarative_fitness": decl.fitness, "declarative_compliant": decl.compliant,
        })
    with open(os.path.join(out, "tau2_conformance_three_ways.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    truth_viol = [not r["truth_compliant"] for r in rows]
    print("\n  mean fitness by ground truth")
    for key in ("tbr_fitness", "alignment_fitness", "declarative_fitness"):
        ok = statistics.mean(r[key] for r in rows if r["truth_compliant"])
        bad = statistics.mean(r[key] for r in rows if not r["truth_compliant"])
        print(f"    {key:22} compliant {ok:.4f}   violating {bad:.4f}   gap {ok - bad:+.4f}")

    print("\n  as a violation detector (positive = protocol violation)")
    print(f"    {'method':40} {'prec':>5} {'rec':>5} {'F1':>5} {'kappa':>6}")
    for label, pred in (
        ("pm4py token replay, fitness < 1", [r["tbr_fitness"] < 1 for r in rows]),
        ("pm4py alignments, fitness < 1", [r["alignment_fitness"] < 1 for r in rows]),
        ("declarative engine (ours), any violation", [not r["declarative_compliant"] for r in rows]),
    ):
        p, rc, f1, k = kappa_f1(pred, truth_viol)
        print(f"    {label:40} {p:5.2f} {rc:5.2f} {f1:5.2f} {k:6.2f}")

    agree = sum(1 for r in rows if (r["alignment_fitness"] < 1) == (not r["declarative_compliant"]))
    print(f"\n  pm4py alignments and our engine give the same verdict on {agree}/{len(rows)} runs")
    print(f"  wrote {out}/tau2_conformance_three_ways.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads"))
    ap.add_argument("--out", default="results/pm4py")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tau2_path = os.path.join(a.data, "tau-data/tau2-bench/submissions/claude-opus-4-5_sierra_2026-02-26",
                             "trajectories/claude-opus-4-5_high_airline_gpt-5.2_4trials.json")
    print("pm4py", pm4py.__version__)
    tau = tau2.ingest(tau2_path)
    discover("tau2_airline", tau, a.out)
    discover_ocel("tau2_airline", "results/ocel/tau2_airline.ocel.json", a.out)

    te = os.path.join(a.data, "traceelephant/data/magentic-runs-gaia")
    if os.path.isdir(te):
        discover("te_magentic", traceelephant.ingest_dir(te), a.out)
        discover_ocel("te_magentic", "results/ocel/te_magentic.ocel.json", a.out)

    local = "results/local"   # the eight-run corpus of 2026-10-13 that the thesis reports; ~/local_agent_runs holds whatever was recorded last
    if os.path.isdir(local):
        discover("local_magentic", local_magentic.ingest_dir(local), a.out, noise=0.0)
        discover_ocel("local_magentic", "results/ocel/local_magentic.ocel.json", a.out)

    discover_then_check(tau, a.out)


if __name__ == "__main__":
    sys.exit(main())
