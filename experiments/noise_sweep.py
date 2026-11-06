"""Discover-then-check at three noise thresholds, so the procedural result is not one parameter.

    python experiments/noise_sweep.py --data ~/Downloads --out results/pm4py

Mines a Petri net from the 135 compliant tau2 airline runs with pm4py's
inductive miner at noise 0.0, 0.1 and 0.2, replays all 172 scored runs with
token-based replay and alignments, and scores "fitness < 1" as a violation
detector against the benchmark's per-action check. Also prints which
activities each net lacks, which runs the noise-0 net flags, and the
walk-through case of Section 6 (task 29, trial 0) at each threshold.

pm4py's inductive miner does not return the same net in every Python process:
at noise 0.2 the net depends on the interpreter's hash seed (found 2026-10-26;
see noise_sweep_seeds.py, which runs this script under ten seeds). So that a
reported table can be reproduced, the script fixes PYTHONHASHSEED to 0 unless
it is already set, by restarting itself once.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

import pm4py

from agentpm.analysis.conformance import check
from agentpm.ingest import tau2
from discover import flat_log, gold_reference, kappa_f1  # experiments/ is sys.path[0] when run as a script


def main():
    if os.environ.get("PYTHONHASHSEED") is None:     # make the miner's net reproducible
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads"))
    ap.add_argument("--out", default="results/pm4py")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.data, "tau-data/tau2-bench/submissions/claude-opus-4-5_sierra_2026-02-26",
                        "trajectories/claude-opus-4-5_high_airline_gpt-5.2_4trials.json")
    traces = tau2.ingest(path)
    truth = {t.case_id: all(c.get("action_match") for c in t.outcome["action_checks"])
             for t in traces if t.outcome.get("action_checks")}
    scored = [t for t in traces if t.case_id in truth]
    good = [t for t in scored if truth[t.case_id]]
    train, test = flat_log(good), flat_log(scored)
    cases = list(test["case:concept:name"].unique())
    by_id = {t.case_id: t for t in scored}
    truth_viol = [not truth[c] for c in cases]
    all_acts = set(train["concept:name"].unique())
    c29 = next(t.case_id for t in scored if str(t.outcome.get("task_id")) == "29" and str(t.outcome.get("trial")) == "0")

    lines = [f"pm4py {pm4py.__version__}, PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}; "
             f"mined from {len(good)} compliant runs, replayed on {len(scored)}; "
             f"positive = protocol violation ({sum(truth_viol)} of {len(scored)})", ""]
    for noise in (0.0, 0.1, 0.2):
        net, im, fm = pm4py.discover_petri_net_inductive(train, noise_threshold=noise)
        labels = {t.label for t in net.transitions if t.label}
        tbr = pm4py.conformance_diagnostics_token_based_replay(test, net, im, fm)
        ali = pm4py.conformance_diagnostics_alignments(test, net, im, fm)
        lines.append(f"noise {noise}: {len(net.places)} places, {len(net.transitions)} transitions, "
                     f"{len(labels)}/{len(all_acts)} activities labelled; missing {sorted(all_acts - labels) or 'none'}")
        for name, fit in (("token replay", [d["trace_fitness"] for d in tbr]), ("alignments", [d["fitness"] for d in ali])):
            ok = statistics.mean(f for f, v in zip(fit, truth_viol) if not v)
            bad = statistics.mean(f for f, v in zip(fit, truth_viol) if v)
            p, r, f1, k = kappa_f1([f < 1 for f in fit], truth_viol)
            lines.append(f"  {name:13} fitness compliant {ok:.3f} violating {bad:.3f} | flags {sum(f < 1 for f in fit):3}/172 "
                         f"| prec {p:.2f} rec {r:.2f} F1 {f1:.2f} kappa {k:.2f}")
        i = cases.index(c29)
        moves = [m[0] for m in ali[i]["alignment"] if m[1] == ">>"]        # log moves: events the net could not fit
        lines.append(f"  task 29 trial 0: token replay {tbr[i]['trace_fitness']:.3f}, alignment {ali[i]['fitness']:.3f}"
                     f"; the alignment's log moves: {moves or 'none'}")
        if noise == 0.0:
            flagged = [(c, d["fitness"]) for c, d in zip(cases, ali) if d["fitness"] < 1]
            lines.append(f"  runs the noise-0 net flags ({len(flagged)}), with the per-task declarative verdict:")
            for c, f in flagged:
                t = by_id[c]
                dec = check(t, gold_reference(t))
                lines.append(f"    task {t.outcome.get('task_id'):>3} trial {t.outcome.get('trial')}: fitness {f:.3f}, "
                             f"truth {'violation' if not truth[c] else 'compliant'}, per-task declarative "
                             f"{'violation' if not dec.compliant else 'compliant'}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(a.out, "noise_sweep.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
