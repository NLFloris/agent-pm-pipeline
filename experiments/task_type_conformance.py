"""Conformance against ONE reference model per task type, not per task.

    python experiments/task_type_conformance.py
    python experiments/task_type_conformance.py --domain retail

The airline models are defined here; the retail models, written later from the
retail policy with the same template, are in retail_task_types.py.

This is the middle level between a global policy (which carried no signal) and
the per-task gold actions (which work but amount to an answer key). A task type
is read mechanically from the WRITE actions its gold list requires: a task whose
gold list contains cancel_reservation is a cancellation, one with
book_reservation a booking, and so on. Every task of a type is checked against
the same model, which is what makes this a process model rather than a key.

Each type's model says only what a process analyst would say about that kind of
work, using nothing that is specific to one task:

  required          the write(s) that define the type, plus the lookup that any
                    write of that kind depends on (reservation lookup before
                    changing or cancelling a reservation, user lookup before
                    booking)
  precedence        that lookup must come before the write
  directly_before   a user message must precede every write, looking through
                    other writes and the agent's own messages, so that one
                    confirmation covers the batch it was asked for. This is the
                    policy's "obtain explicit confirmation" rule. The strict
                    form, immediately before every single write, fired on 33
                    runs of which 25 were batches the user had confirmed once.
  allowed           for tasks that require no write at all, no write may occur

Scored against two truths: the benchmark's per-action check (process) and its
reward (outcome). Also reported per type, because a model that works for
cancellations and fails for bookings is a different finding from one that is
uniformly mediocre.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

from agentpm.analysis.conformance import check
from agentpm.ingest import tau2
from agentpm.reference import Reference
from data_dir import TAU2_SUBMISSIONS   # noqa: E402

WRITES = ["book_reservation", "update_reservation_flights", "update_reservation_baggages",
          "update_reservation_passengers", "cancel_reservation", "send_certificate"]
TOOLS = ["get_user_details", "get_reservation_details", "search_direct_flight",
         "search_onestop_flight", "get_flight_status", "list_all_airports", "calculate",
         "transfer_to_human_agents"] + WRITES
MESSAGES = ["assistant_message", "user_message"]


def task_type(trace):
    """The write actions the task requires, as a canonical key."""
    gold = [a["name"] for a in trace.outcome.get("gold_actions", []) if a.get("requestor") == "assistant"]
    writes = sorted(set(a for a in gold if a in WRITES))
    if writes:
        return "+".join(writes)
    return "transfer" if "transfer_to_human_agents" in gold else "no-write"


def type_model(key, reads_transparent=False):
    """One reference per task type, built from the type alone.

    reads_transparent lets the confirmation rule also look through read-only
    tools; a variant added after the cross-agent run, never the default."""
    if key == "no-write":
        return Reference(name=key, allowed=[t for t in TOOLS if t not in WRITES] + MESSAGES)
    if key == "transfer":
        return Reference(name=key, required=["transfer_to_human_agents"],
                         allowed=[t for t in TOOLS if t not in WRITES] + MESSAGES)
    writes = key.split("+")
    required, precedence = list(writes), []
    if any(w != "book_reservation" for w in writes):
        required.append("get_reservation_details")
        precedence += [["get_reservation_details", w] for w in writes if w != "book_reservation"]
    if "book_reservation" in writes:
        required.append("get_user_details")
        precedence.append(["get_user_details", "book_reservation"])
    return Reference(name=key, required=sorted(set(required)), precedence=precedence,
                     directly_before=[["user_message", w] for w in writes],
                     # one confirmation covers an announced batch of writes
                     transparent=WRITES + ["assistant_message"]
                     + ([t for t in TOOLS if t not in WRITES] if reads_transparent else []))


TAU2 = TAU2_SUBMISSIONS
DEFAULT_DATA = {
    "airline": os.path.join(TAU2, "claude-opus-4-5_sierra_2026-02-26/trajectories/"
                            "claude-opus-4-5_high_airline_gpt-5.2_4trials.json"),
    "retail": os.path.join(TAU2, "claude-opus-4-5_sierra_2026-02-26/trajectories/"
                           "claude-opus-4-5_high_retail_gpt-5.2_4trials.json"),
}


def domain_rules(domain):
    """(task_type, type_model, prepare) for a domain; prepare runs before checking."""
    if domain == "retail":
        import retail_task_types as rt
        return rt.task_type, rt.type_model, rt.abstract
    return task_type, type_model, (lambda t: t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=sorted(DEFAULT_DATA), default="airline")
    ap.add_argument("--data", help="trajectory file; defaults to the Opus 4.5 file of the domain")
    ap.add_argument("--reads-transparent", action="store_true",
                    help="the variant of the confirmation rule that looks through reads")
    a = ap.parse_args()
    path = a.data or DEFAULT_DATA[a.domain]
    task_type, type_model, prepare = domain_rules(a.domain)
    traces = [t for t in tau2.ingest(path) if t.outcome.get("action_checks")]  # scorable only
    print(f"{len(traces)} scorable runs\n")

    rows = []
    for t in traces:
        key = task_type(t)
        r = check(prepare(t), type_model(key, a.reads_transparent))
        rows.append({
            "case_id": t.case_id, "task_id": t.outcome["task_id"], "type": key,
            "compliant": r.compliant, "fitness": r.fitness,
            "violations": [v.constraint for v in r.violations],
            "truth_process": all(c.get("action_match") for c in t.outcome["action_checks"]),
            "truth_outcome": t.outcome.get("reward") == 1.0,
            "truth_db": (t.outcome.get("db_check") or {}).get("db_match", True),
        })

    types = collections.Counter(r["type"] for r in rows)
    print("task types, from the gold write actions, and the model each one gets:")
    for key, n in types.most_common():
        m = type_model(key)
        parts = []
        if m.required: parts.append(f"required {m.required}")
        if m.precedence: parts.append(f"precedence {len(m.precedence)}")
        if m.directly_before: parts.append(f"confirm-before-write {len(m.directly_before)}")
        if m.allowed and not m.required: parts.append("no write allowed")
        print(f"  {n:3} runs  {key:40}  {'; '.join(parts)}")

    def score(label, truth_key, subset=None):
        rs = [r for r in rows if subset is None or r["type"] == subset]
        tp = sum(1 for r in rs if not r["compliant"] and not r[truth_key])
        fp = sum(1 for r in rs if not r["compliant"] and r[truth_key])
        fn = sum(1 for r in rs if r["compliant"] and not r[truth_key])
        tn = len(rs) - tp - fp - fn
        n = len(rs)
        acc = (tp + tn) / n if n else 0
        p = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0
        pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n) if n else 0
        k = (acc - pe) / (1 - pe) if pe < 1 else 0
        print(f"  {label:34} n {n:3}  viol {tp+fn:3}  prec {p:.2f}  rec {rc:.2f}  F1 {f1:.2f}  kappa {k:.2f}")

    print("\nTASK-TYPE MODEL against the two truths, all runs")
    score("vs process truth (action_checks)", "truth_process")
    score("vs outcome truth (reward)", "truth_outcome")

    print("\nper type, against the process truth")
    for key, _ in types.most_common():
        score(key, "truth_process", subset=key)

    fired = collections.Counter(v.split("(")[0] for r in rows for v in r["violations"])
    print(f"\nwhich constraint kinds fire: {dict(fired)}")
    conf = sum(1 for r in rows if any(v.startswith("directly_before") for v in r["violations"]))
    print(f"runs with a write not immediately preceded by a user message: {conf}/{len(rows)}")

    out = "results/task_type_conformance" + ("" if a.domain == "airline" else f"_{a.domain}") \
        + ("_reads_transparent" if a.reads_transparent else "") + ".json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
