"""What discovery does with the partial order, measured.

    python experiments/partial_order.py --data ~/Downloads --out results

The canonical trace keeps the partial order: tool calls a model issued in one
turn share a timestamp. The object-centric log keeps it too. This script asks
whether the discovery algorithms respect it, on tau2-bench airline, tool events
only. Four directly-follows graphs are built from the same 200 runs:

  flat (pipeline)       consecutive pairs over the serialisation
  partial (pipeline)    pairs from one step to the next; concurrent events
                        never follow each other
  pm4py DFG             pm4py.discover_dfg on a flat log whose timestamps carry
                        the ties
  pm4py OC-DFG          pm4py.discover_ocdfg on the OCEL 2.0 log, run object

If an algorithm serialises, its edge multiset equals the flat one and it counts
every within-turn pair as a succession. The output says which of the four do.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pm4py

from agentpm.analysis.discovery import (concurrency_summary, concurrent_pairs,
                                        directly_follows, directly_follows_flat)
from agentpm.ingest import tau2
from agentpm.logbuild.objects import BUILTIN
from agentpm.logbuild.ocel import build_ocel_from_traces
from agentpm.model import Trace
from data_dir import DATA   # noqa: E402

MESSAGES = {"assistant_message", "user_message"}
BASE = datetime(2000, 1, 1, tzinfo=timezone.utc)


def tool_only(traces):
    return [Trace(t.case_id, [e for e in t.events if e.activity not in MESSAGES], t.outcome) for t in traces]


def pm4py_flat_dfg(traces):
    rows = []
    for t in traces:
        for pos, e in enumerate(t.ordered()):
            rows.append({"case:concept:name": t.case_id, "concept:name": e.activity,
                         "time:timestamp": BASE + timedelta(seconds=e.timestamp),   # ties kept
                         "org:resource": e.resource, "pos": pos})
    df = pm4py.format_dataframe(pd.DataFrame(rows), case_id="case:concept:name",
                                activity_key="concept:name", timestamp_key="time:timestamp")
    dfg, _, _ = pm4py.discover_dfg(df)
    return {k: int(v) for k, v in dfg.items()}


def pm4py_ocdfg(ocel_path):
    ocel = pm4py.read_ocel2_json(ocel_path)
    ocdfg = pm4py.discover_ocdfg(ocel)
    couples = ocdfg["edges"]["event_couples"].get("run", {})
    return {k: len(v) for k, v in couples.items()}


def compare(name, edges, flat, partial, within):
    total = sum(edges.values())
    same_as_flat = edges == flat
    same_as_partial = edges == partial
    within_counted = sum(min(edges.get(k, 0), n) for k, n in within.items()) if not same_as_partial else 0
    verdict = "serialises (equals the flat DFG)" if same_as_flat else (
        "respects the partial order (equals the pipeline's)" if same_as_partial else "neither")
    return f"  {name:22} {len(edges):3} distinct edges, {total:5} relations   {verdict}" + (
        f"; {within_counted} relations are within-turn pairs" if within_counted else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    path = os.path.join(a.data, "tau-data/tau2-bench/submissions/claude-opus-4-5_sierra_2026-02-26",
                        "trajectories/claude-opus-4-5_high_airline_gpt-5.2_4trials.json")
    traces = tool_only(tau2.ingest(path))
    lines = []
    s = concurrency_summary(traces)
    lines.append(f"tau2-bench airline, {s['traces']} runs, tool events only")
    lines.append(f"  runs with at least one turn of several calls: {s['traces_with_concurrency']}")
    lines.append(f"  steps {s['steps']}, of which {s['steps_with_more_than_one_event']} hold more than one "
                 f"event (largest {s['largest_step']})")
    lines.append(f"  flat succession relations {s['flat_edges']}, of which {s['within_step_pairs']} are "
                 f"within-turn pairs ({s['within_step_pairs'] / s['flat_edges']:.0%})")
    lines.append(f"  partial-order succession relations {s['partial_order_edges']}")

    flat = directly_follows_flat(traces)
    partial = directly_follows(traces)
    within = concurrent_pairs(traces)
    only_flat = sorted(set(flat) - set(partial))
    lines.append(f"  distinct edges: flat {len(flat)}, partial {len(partial)}, "
                 f"{len(only_flat)} exist only because of serialisation: {only_flat}")

    ocel_dir = os.path.join(a.out, "ocel")
    os.makedirs(ocel_dir, exist_ok=True)
    ocel_path = os.path.join(ocel_dir, "tau2_airline.ocel.json")
    if not os.path.exists(ocel_path):
        import json
        with open(ocel_path, "w", encoding="utf-8") as f:
            json.dump(build_ocel_from_traces(traces, object_rules=BUILTIN["tau2"]), f, indent=1, sort_keys=True)

    lines.append("\nfour directly-follows graphs from the same runs")
    lines.append(compare("flat (pipeline)", flat, flat, partial, within))
    lines.append(compare("partial (pipeline)", partial, flat, partial, within))
    lines.append(compare("pm4py discover_dfg", pm4py_flat_dfg(traces), flat, partial, within))
    lines.append(compare("pm4py discover_ocdfg", pm4py_ocdfg(ocel_path), flat, partial, within))
    lines.append("\nReading: OCEL 2.0 can hold events with an identical time, and the log written here does.")
    lines.append("Neither pm4py algorithm uses that: both sort by time and count consecutive pairs, so a")
    lines.append("shared timestamp changes nothing. Respecting the partial order needs the discovery to read")
    lines.append("steps, which the pipeline's own DFG does. The verdicts that flip under reshuffling")
    lines.append("(experiments/run_three_corpora.py) are the conformance-side cost of the same serialisation.")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(a.out, "partial_order.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
