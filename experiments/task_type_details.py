"""Three details of the task-type engine that Chapter 6 reports, from one script.

    python experiments/task_type_details.py > results/task_type_details.txt

1. The confirmation rule, strict and batch-aware. The policy says the agent must
   obtain an explicit "yes" before a write. The strictest reading, a user message
   immediately before every single write, is compared with the reading the
   models use, which looks through the other writes of a batch and through the
   agent's own messages. Claude Opus 4.5, airline, against the per-action label.

2. When a violation becomes visible. A violation of `allowed` (a write in a task
   that needs none), `precedence` (a write before its lookup) or
   `directly_before` (a write without a confirmation) is located at a write, so
   a check running beside the agent could see it when the write is issued. A
   violation of `required` (a missing action) is known only when the run ends.
   Counted over the runs the engine flags, with the position of the first
   violation that is located at a write, as a share of the run's events. This
   is a property of the verdicts, not a monitor that was built: in a live
   setting the task type would have to come from the conversation, and here it
   comes from the gold list.

   The same count over all fifteen agents, where the label that also counts a
   wrong final database (action-or-db, see cross_agent.py) is reported too,
   because a write nobody asked for leaves the per-action label satisfied.

3. Transfers to a human agent, and how often a run that transfers fails.

4. Listing dependence. Calls issued together share a step and have no order,
   but the confirmation rule reads the listing and looks only through the
   activities it declares. A lookup listed between the customer's message and a
   change of the same step therefore breaks the chain in one listing and not in
   another. Counted as the runs whose verdict changes under any of 20 seeded
   shuffles of the events within each step, for the frozen models and for the
   variant that also looks through lookups.

Every model is the frozen one from task_type_conformance.py (airline) and
retail_task_types.py (retail); nothing here changes a verdict.
"""
from __future__ import annotations

import dataclasses
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import task_type_conformance as ttc      # noqa: E402

from agentpm.analysis.conformance import check   # noqa: E402
from agentpm.ingest import tau2                   # noqa: E402

AT_A_WRITE = ("allowed", "precedence", "directly_before")


def violated(t):
    return not all(c.get("action_match") for c in t.outcome["action_checks"])


def scorable(domain):
    return [t for t in tau2.ingest(ttc.DEFAULT_DATA[domain]) if t.outcome.get("action_checks")]


def confirmation_ablation(traces):
    print("1. CONFIRMATION RULE, claude-opus-4-5, airline, against the per-action label")
    print(f"   {len(traces)} scorable runs, {sum(violated(t) for t in traces)} violations\n")
    variants = [("strict: a user message immediately before every write", []),
                ("looks through the other writes of a batch", list(ttc.WRITES)),
                ("also looks through the agent's own messages (the model used)", None)]
    for name, transparent in variants:
        tp = fp = conf_tp = conf_fp = 0
        for t in traces:
            m = ttc.type_model(ttc.task_type(t))
            if transparent is not None:
                m = dataclasses.replace(m, transparent=transparent)
            r = check(t, m)
            if r.compliant:
                continue
            v = violated(t)
            tp += v
            fp += not v
            if any(x.constraint.startswith("directly_before") for x in r.violations):
                conf_tp += v
                conf_fp += not v
        total = sum(violated(t) for t in traces)
        print(f"   {name}")
        print(f"     runs flagged {tp + fp:3}  precision {tp / (tp + fp):.2f}  recall {tp / total:.2f}"
              f"  | runs with a confirmation alarm {conf_tp + conf_fp:3}"
              f" ({conf_fp} of them compliant by the label)")


def visibility(domain, traces, task_type, type_model, prepare, db_label=False):
    flagged = at_write = only_end = 0
    tp_write = tp_end = db_write = db_end = 0
    positions = []
    kinds = {}                      # first violation at a write, on runs the database-aware label confirms
    for t in traces:
        r = check(prepare(t), type_model(task_type(t)))
        if r.compliant:
            continue
        flagged += 1
        db = violated(t) or not (t.outcome.get("db_check") or {}).get("db_match", True)
        located = [v.at for v in r.violations if v.constraint.startswith(AT_A_WRITE) and v.at is not None]
        if located:
            at_write += 1
            tp_write += violated(t)
            db_write += db
            positions.append(min(located) / max(1, len(t.ordered()) - 1))
            if db:
                first = min((v for v in r.violations if v.constraint.startswith(AT_A_WRITE) and v.at is not None),
                            key=lambda v: v.at)
                kind = first.constraint.split("(")[0]
                kinds[kind] = kinds.get(kind, 0) + 1
        else:
            only_end += 1
            tp_end += violated(t)
            db_end += db
    extra = (lambda n: f", {n} by the label that also counts the database") if db_label else (lambda n: "")
    print(f"   {domain}: {flagged} runs flagged")
    print(f"     a violation located at a write   {at_write:5}  ({tp_write} of them violations by the per-action label"
          f"{extra(db_write)})")
    print(f"     only a missing action (run end)  {only_end:5}  ({tp_end} of them violations by the per-action label"
          f"{extra(db_end)})")
    if positions:
        print(f"     first violation at a write, as a share of the run: median {statistics.median(positions):.0%},"
              f" earliest {min(positions):.0%}, latest {max(positions):.0%}")
    if db_label and kinds:
        print("     kind of the first violation at a write, on the runs the database-aware label confirms: "
              + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])))


def transfers(domain, traces_all):
    with_transfer = [t for t in traces_all if any(e.activity == "transfer_to_human_agents" for e in t.events)]
    failed = sum(1 for t in with_transfer if t.outcome.get("reward") != 1.0)
    required = sum(1 for t in with_transfer
                   if any(a.get("name") == "transfer_to_human_agents"
                          for a in t.outcome.get("gold_actions", [])))
    print(f"   {domain}: {len(with_transfer)} of {len(traces_all)} runs transfer to a human agent;"
          f" the gold list asks for it in {required}; {failed} of the {len(with_transfer)} have reward below 1")


def shuffled(trace, rng):
    """The same trace with the events of every step listed in a random order."""
    out = []
    for step in trace.steps():
        seqs = sorted(e.sequence if e.sequence is not None else i for i, e in enumerate(step))
        order = list(step)
        rng.shuffle(order)
        out.extend(dataclasses.replace(e, sequence=q) for e, q in zip(order, seqs))
    return dataclasses.replace(trace, events=out)


def listing_dependence(name, traces, task_type, type_model, prepare, shuffles=20, seed=0):
    rng = random.Random(seed)
    for variant in (False, True):
        flips = 0
        for t in traces:
            m = type_model(task_type(t), variant)
            base = check(prepare(t), m).compliant
            if any(check(prepare(shuffled(t, rng)), m).compliant != base for _ in range(shuffles)):
                flips += 1
        label = "variant that also looks through lookups" if variant else "frozen confirmation rule"
        print(f"   {name}, {label}: verdict changes in {flips} of {len(traces)} runs")


def main():
    airline = scorable("airline")
    confirmation_ablation(airline)

    print("\n2. WHEN A VIOLATION BECOMES VISIBLE, claude-opus-4-5, frozen models")
    for domain in ("airline", "retail"):
        task_type, type_model, prepare = ttc.domain_rules(domain)
        visibility(domain, airline if domain == "airline" else scorable(domain), task_type, type_model, prepare)

    import cross_agent
    print("\n   the same over fifteen agents, frozen models, all scorable runs pooled")
    for domain in ("airline", "retail"):
        task_type, type_model, prepare = ttc.domain_rules(domain)
        pooled = [t for _, f in cross_agent.files(domain) for t in tau2.ingest(f) if t.outcome.get("action_checks")]
        visibility(domain, pooled, task_type, type_model, prepare, db_label=True)

    print("\n3. TRANSFERS TO A HUMAN AGENT, claude-opus-4-5, all runs")
    for domain in ("airline", "retail"):
        transfers(domain, list(tau2.ingest(ttc.DEFAULT_DATA[domain])))

    print("\n4. LISTING DEPENDENCE, 20 seeded shuffles of the events within each step")
    for domain in ("airline", "retail"):
        task_type, type_model, prepare = ttc.domain_rules(domain)
        own = airline if domain == "airline" else scorable(domain)
        listing_dependence(f"{domain}, claude-opus-4-5", own, task_type, type_model, prepare)
        pooled = [t for _, f in cross_agent.files(domain) for t in tau2.ingest(f) if t.outcome.get("action_checks")]
        listing_dependence(f"{domain}, fifteen agents", pooled, task_type, type_model, prepare)


if __name__ == "__main__":
    sys.exit(main())
