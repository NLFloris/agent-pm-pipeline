"""Export the raw event logs behind the results, in formats other tools read.

    python experiments/export_event_log.py --data ~/Downloads --out results/logs

Writes, for the tau2 airline corpus:

  tau2_airline.csv              every event, one row: the flat event log
  tau2_airline.xes              the same log in XES, for ProM / Disco / pm4py
  conformance_per_run.csv       one row per run: verdict, fitness, violations,
                                harness reward, and which of the four
                                agreement cells it falls in
  false_negatives.csv           only the runs conformance called compliant and
                                the harness called failed, with the reason the
                                harness gives

Timestamps are synthetic and encode position only. The source records real
wall-clock times, but parallel tool calls in one turn share an identical
timestamp and cannot order each other, so ordering uses the sequence index and
no duration claim is made anywhere. The real value is kept in `epoch_us`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pm4py

from agentpm.analysis.conformance import check
from agentpm.ingest import tau2
from agentpm.reference import Reference

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def gold_reference(trace):
    names = [a["name"] for a in trace.outcome.get("gold_actions", [])
             if a.get("requestor") == "assistant"]
    pairs = [[a, b] for a, b in zip(names, names[1:]) if a != b]
    return Reference(name="gold", required=sorted(set(names)), precedence=pairs)


def cell_of(compliant, passed):
    if compliant and passed:
        return "agree: conforms and passed"
    if not compliant and not passed:
        return "agree: violation and failed"
    if not compliant and passed:
        return "disagree: violation but passed"
    return "disagree: conforms but failed"


def harness_reason(trace):
    """Why the harness failed a run, from its own reward breakdown."""
    reasons = []
    db = trace.outcome.get("db_check")
    if isinstance(db, dict) and db.get("db_match") is False:
        reasons.append("database mismatch")
    acts = trace.outcome.get("action_checks") or []
    bad = [a for a in acts if isinstance(a, dict) and a.get("action_match") is False]
    if bad:
        wanted = bad[0].get("action") or {}
        reasons.append(f"wrong arguments for {wanted.get('name')} "
                       f"(expected {json.dumps(wanted.get('arguments'))})")
    nl = [a for a in (trace.outcome.get("nl_assertions") or [])
          if isinstance(a, dict) and a.get("met") is False]
    if nl:
        reasons.append(f"communication assertion unmet: "
                       f"{str(nl[0].get('nl_assertion'))[:120]}")
    com = [a for a in (trace.outcome.get("communicate_checks") or [])
           if isinstance(a, dict) and a.get("met") is False]
    if com:
        reasons.append("required information not communicated")
    return "; ".join(reasons) or "not decomposable from the shipped breakdown"


def event_rows(traces):
    rows = []
    for t in traces:
        for step, e in enumerate(t.ordered()):
            rows.append({
                "case:concept:name": t.case_id,
                "concept:name": e.activity,
                "time:timestamp": BASE + timedelta(seconds=step),
                "org:resource": e.resource,
                "step": step,
                "arguments": json.dumps(e.attributes.get("arguments"))
                             if e.attributes.get("arguments") is not None else "",
                "tool_error": e.attributes.get("error"),
                "epoch_us": e.attributes.get("epoch_us"),
                "turn_idx": e.attributes.get("turn_idx"),
                "case:task_id": t.outcome.get("task_id"),
                "case:trial": t.outcome.get("trial"),
                "case:reward": t.outcome.get("reward"),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads"))
    ap.add_argument("--out", default="results/logs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    path = os.path.join(
        a.data, "tau-data/tau2-bench/submissions/claude-opus-4-5_sierra_2026-02-26",
        "trajectories/claude-opus-4-5_high_airline_gpt-5.2_4trials.json")
    traces = tau2.ingest(path)

    events = event_rows(traces)
    csv_path = os.path.join(a.out, "tau2_airline.csv")
    events.to_csv(csv_path, index=False)
    print(f"  {len(events)} events -> {csv_path}")

    log = pm4py.format_dataframe(events, case_id="case:concept:name",
                                 activity_key="concept:name",
                                 timestamp_key="time:timestamp")
    xes_path = os.path.join(a.out, "tau2_airline.xes")
    pm4py.write_xes(log, xes_path)
    back = pm4py.read_xes(xes_path)
    print(f"  XES -> {xes_path}  (reloads with {len(back)} rows)")

    per_run, fns = [], []
    for t in traces:
        r = check(t, gold_reference(t))
        if not r.n_constraints:
            continue
        passed = t.outcome.get("reward") == 1.0
        row = {
            "case_id": t.case_id,
            "task_id": t.outcome.get("task_id"),
            "trial": t.outcome.get("trial"),
            "events": len(t.events),
            "tool_calls": sum(1 for e in t.events
                              if e.activity not in ("assistant_message", "user_message")),
            "fitness": r.fitness,
            "compliant": r.compliant,
            "constraints": r.n_constraints,
            "violations": " | ".join(v.constraint for v in r.violations),
            "harness_reward": t.outcome.get("reward"),
            "cell": cell_of(r.compliant, passed),
            "gold_actions": " -> ".join(
                x["name"] for x in t.outcome.get("gold_actions", [])
                if x.get("requestor") == "assistant"),
            "activities": " -> ".join(
                e.activity for e in t.ordered()
                if e.activity not in ("assistant_message", "user_message")),
        }
        per_run.append(row)
        if r.compliant and not passed:
            fns.append(dict(row, harness_reason=harness_reason(t)))

    runs_path = os.path.join(a.out, "conformance_per_run.csv")
    pd.DataFrame(per_run).to_csv(runs_path, index=False)
    print(f"  {len(per_run)} runs -> {runs_path}")

    fn_path = os.path.join(a.out, "false_negatives.csv")
    pd.DataFrame(fns).to_csv(fn_path, index=False)
    print(f"  {len(fns)} false negatives -> {fn_path}")


if __name__ == "__main__":
    sys.exit(main())
