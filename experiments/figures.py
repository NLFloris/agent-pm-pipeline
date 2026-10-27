"""Render process maps from canonical traces.

    python experiments/figures.py --data ~/Downloads --out results/figures

Two views per corpus, both frequency-filtered because the unfiltered
directly-follows graph over a few hundred agent runs is unreadable (R9):

  <name>_dfg.png       directly-follows graph, each activity coloured by the
                       agent that performs it
  <name>_handover.png  handover-of-work network between agents

Needs the `pm` extra for graphviz plus a `dot` binary on PATH.
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import sys

from agentpm.ingest import agenthallu, local_magentic, tau2, traceelephant

from agentpm.viz import (PALETTE, agent_colors, handover, render_dfg_by_agent,  # noqa: F401
                        render_handover, render_lane_model, render_single_run)

MESSAGES = {"assistant_message", "user_message"}


def emit(name, traces, out, min_edge, drop=frozenset()):
    if not traces:
        print(f"  [skip] {name}: no traces")
        return
    colors = agent_colors(traces)
    render_dfg_by_agent(traces, colors, os.path.join(out, f"{name}_dfg"),
                        f"{name}: activities by acting agent ({len(traces)} runs)",
                        min_edge, drop)
    edges = handover(traces)
    if edges:
        render_handover(edges, colors, os.path.join(out, f"{name}_handover"),
                        f"{name}: handover of work ({len(traces)} runs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads"))
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tau2_path = os.path.join(
        a.data, "tau-data/tau2-bench/submissions/claude-opus-4-5_sierra_2026-02-26",
        "trajectories/claude-opus-4-5_high_airline_gpt-5.2_4trials.json")
    if os.path.exists(tau2_path):
        emit("tau2_airline", tau2.ingest(tau2_path), a.out, 10, drop=MESSAGES)

    te = os.path.join(a.data, "traceelephant/data")
    for sub, tag in [("magentic-runs-gaia", "te_magentic"), ("captain-runs-gaia", "te_captain")]:
        if not os.path.isdir(os.path.join(te, sub)):
            continue
        traces = traceelephant.ingest_dir(os.path.join(te, sub))
        emit(tag, traces, a.out, 5)
        colors = agent_colors(traces)
        render_lane_model(traces, colors, os.path.join(a.out, f"{tag}_lanes"),
                          f"{tag}: one lane per agent, start to end ({len(traces)} runs)",
                          min_edge=8)
        # a single readable run with several agents, drawn event by event
        cand = [t for t in traces
                if len(t.outcome.get("agents", [])) >= 3 and 8 <= len(t.events) <= 22]
        if cand:
            one = sorted(cand, key=lambda t: -len(t.outcome["agents"]))[0]
            render_single_run(one, colors, os.path.join(a.out, f"{tag}_one_run"),
                              f"{tag}: a single run ({one.case_id})")

    local = "results/local"   # the eight-run corpus of 2026-10-13 that the thesis reports; ~/local_agent_runs holds whatever was recorded last
    if os.path.isdir(local):
        traces = local_magentic.ingest_dir(local)
        emit("local_magentic", traces, a.out, 2)
        colors = agent_colors(traces)
        render_lane_model(traces, colors, os.path.join(a.out, "local_magentic_lanes"),
                          f"local Magentic-One: one lane per agent ({len(traces)} runs)",
                          min_edge=2)
        pick = max(traces, key=lambda t: len(t.events))
        render_single_run(pick, colors, os.path.join(a.out, "local_magentic_one_run"),
                          f"local Magentic-One: a single run ({pick.case_id})")

    ah = os.path.join(a.data, "agenthallu")
    for sub, tag in [("Magentic_One", "agenthallu_magentic"), ("Camel", "agenthallu_camel")]:
        if os.path.isdir(os.path.join(ah, sub)):
            emit(tag, agenthallu.ingest_dir(os.path.join(ah, sub)), a.out, 5)



if __name__ == "__main__":
    sys.exit(main())
