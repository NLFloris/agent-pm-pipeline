"""Build the OCEL 2.0 event logs of the thesis through the pipeline's own builder.

    python experiments/build_ocel.py --data ~/Downloads --out results/ocel

This script adds nothing to layer L3. It calls agentpm.logbuild.ocel with the
object rules shipped in agentpm.logbuild.objects, writes each log, reads it
back with pm4py's importer, and prints what the log holds. The same result is
available from the command line:

    python -m agentpm.cli ocel <tau2 file> --format tau2 \\
        --exclude assistant_message user_message -o results/ocel/tau2_airline.ocel.json

Two facts about these logs that the thesis relies on:

  * events a model issued in one turn share a timestamp in the log, which is
    the partial order a flat log cannot hold (see experiments/partial_order.py
    for what discovery does with it);
  * business objects (user, reservation, flight, payment, route) come from
    declared rules over recorded arguments and results, scoped per run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pm4py

from agentpm.ingest import agenthallu, local_magentic, otel, tau2, traceelephant
from agentpm.logbuild.objects import BUILTIN
from agentpm.logbuild.ocel import build_ocel_from_traces, object_sharing
from data_dir import DATA   # noqa: E402

MESSAGES = {"assistant_message", "user_message"}


def emit(name, traces, out, rules=(), exclude=frozenset()):
    ocel = build_ocel_from_traces(traces, object_rules=list(rules), exclude=set(exclude))
    path = os.path.join(out, f"{name}.ocel.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ocel, f, indent=1, sort_keys=True)
    back = pm4py.read_ocel2_json(path)
    share = object_sharing(ocel)
    rels = sum(len(e["relationships"]) for e in ocel["events"])
    n_ties = sum(1 for t in traces for s in _steps(t, exclude) if len(s) > 1)

    print(f"\n[{name}]  -> {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    print(f"  events {len(ocel['events']):6}   objects {len(ocel['objects']):6}   E2O relations {rels:6}")
    print("  object types: " + ", ".join(f"{t}({n})" for t, n in share["objects_by_type"].items()))
    print(f"  events touching a business object (beyond run, agent, tool): "
          f"{share['events_touching_a_business_object']}/{share['events']}  "
          f"({share['events_touching_a_business_object'] / max(1, share['events']):.0%})")
    print(f"  steps with more than one event (shared timestamp): {n_ties}")
    ok = len(back.events) == len(ocel["events"]) and len(back.relations) == rels
    print(f"  reload in pm4py.read_ocel2_json: {'ok' if ok else 'MISMATCH'} "
          f"({len(back.events)} events, {len(back.relations)} relations)")


def _steps(trace, exclude):
    kept = [e for e in trace.ordered() if e.activity not in exclude]
    groups = []
    for e in kept:
        if groups and groups[-1][0].timestamp == e.timestamp:
            groups[-1].append(e)
        else:
            groups.append([e])
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="results/ocel")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("pm4py", pm4py.__version__)

    tau2_path = os.path.join(
        a.data, "tau-data/tau2-bench/submissions/claude-opus-4-5_sierra_2026-02-26",
        "trajectories/claude-opus-4-5_high_airline_gpt-5.2_4trials.json")
    if os.path.exists(tau2_path):
        traces = tau2.ingest(tau2_path)
        emit("tau2_airline", traces, a.out, rules=BUILTIN["tau2"], exclude=MESSAGES)
        emit("tau2_airline_with_messages", traces, a.out, rules=BUILTIN["tau2"])

    for sub, tag in [("magentic-runs-gaia", "te_magentic"), ("swe-agent-runs-swe-bench", "te_swe")]:
        path = os.path.join(a.data, "traceelephant/data", sub)
        if os.path.isdir(path):
            emit(tag, traceelephant.ingest_dir(path), a.out)

    path = os.path.join(a.data, "agenthallu", "Magentic_One")
    if os.path.isdir(path):
        emit("agenthallu_magentic", agenthallu.ingest_dir(path), a.out)

    local = "results/local"   # the eight-run corpus of 2026-10-13 that the thesis reports; ~/local_agent_runs holds whatever was recorded last
    if os.path.isdir(local):
        emit("local_magentic", local_magentic.ingest_dir(local), a.out)
    capture = "results/otel_live/capture.jsonl"   # the eight tasks through AutoGen's OpenTelemetry spans, 2026-10-19
    if os.path.exists(capture):
        emit("otel_live", otel.ingest(capture), a.out)
    sub = "results/local_subagents/runs"   # ten claim_check runs with sub-agents, 2026-10-26
    if os.path.isdir(sub):
        emit("local_subagents", local_magentic.ingest_dir(sub), a.out)


if __name__ == "__main__":
    sys.exit(main())
