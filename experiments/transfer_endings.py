"""Runs that the simulated customer ends with ###TRANSFER### although the agent never transferred.

    PYTHONHASHSEED=0 python experiments/transfer_endings.py > results/transfer_endings.txt

In tau2-bench the customer is a language model. It ends a run by writing a
token, ###STOP### or ###TRANSFER###, and after that token the agent gets no
further turn. The token ###TRANSFER### is meant for a customer who has been
transferred to a human agent, but the customer also writes it when the agent
made no transfer_to_human_agents call, for example right after asking for one.
Such a run ends on the customer's side, so an action the task required and the
agent had not yet performed is missing from the log, and the per-action label
and the engine both count it as a violation of the agent.

Definitions:

  customer-ended transfer  a scorable run whose last customer message contains
                           ###TRANSFER### and which has no
                           transfer_to_human_agents call
  agent-ended transfer     the same token after such a call, for comparison

For every customer-ended transfer the script counts how the per-action label,
the per-action-or-database label and the reward mark it, and whether the engine
with the frozen task-type model flags it, for claude-opus-4-5 and pooled over
the fifteen agents of cross_agent.py, on both domains.

Two crude indicators of whether the agent still had a turn to act. Both need
the agent's last message before the customer's final one to ask something (it
contains "confirm", "proceed", "approve", "would you like me to", "shall I" or
"do you want me to"), and both look at the sentences of the customer's final
message that contain "yes", "yeah", "yep", "go ahead", "proceed", "confirm" or
"approve":

  change confirmed at the end   such a sentence does not mention a transfer, as
                                in "Yes, please cancel LU15PA and 8C8K4E.", so
                                the customer confirmed a change in the message
                                that ended the run and the agent never had a
                                turn to make it
  transfer accepted at the end  such a sentence mentions a transfer, as in "Yes,
                                please transfer me to a human agent.", so the
                                agent never had a turn to make the transfer call

A run can have both. They are keyword matches and not a reading of the
conversation: they miss a confirmation worded otherwise and can fire on a "yes"
that answers something else. The Opus airline runs they list were read by hand.

Reads the mirror only; no model is called. Standard library plus agentpm.
"""
from __future__ import annotations

import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cross_agent import DEV_AGENT, LABELS, files, violation   # noqa: E402
from task_type_conformance import domain_rules   # noqa: E402

from agentpm.analysis.conformance import check   # noqa: E402
from agentpm.ingest import tau2   # noqa: E402

TOKEN = "###TRANSFER###"
TRANSFER = "transfer_to_human_agents"
ASKS = re.compile(r"\b(confirm|proceed|approve|would you like me to|shall i|do you want me to)\b", re.I)
YES = re.compile(r"\b(yes|yeah|yep|go ahead|proceed|confirm|approve)\b", re.I)


def ending(t):
    """'customer', 'agent' or None, and the two indicators (change confirmed, transfer accepted)."""
    events = t.ordered()
    said = [i for i, e in enumerate(events) if e.activity == "user_message"]
    if not said or TOKEN not in (events[said[-1]].attributes.get("content") or ""):
        return None, False, False
    last = said[-1]
    if any(e.activity == TRANSFER for e in t.events):
        return "agent", False, False
    before = [e for e in events[:last] if e.activity == "assistant_message"]
    if not before or not ASKS.search(before[-1].attributes.get("content") or ""):
        return "customer", False, False
    text = events[last].attributes["content"].replace(TOKEN, " ")
    yes = [s for s in re.split(r"[.!?\n]+", text) if YES.search(s)]
    return ("customer", any("transfer" not in s.lower() for s in yes),
            any("transfer" in s.lower() for s in yes))


def failed(t):
    return sorted({(c.get("action") or {}).get("name") for c in t.outcome["action_checks"]
                   if not c.get("action_match")})


def rows(domain):
    task_type, type_model, prepare = domain_rules(domain)
    out = {}
    for agent, path in files(domain):
        rs = []
        for t in tau2.ingest(path):
            if not t.outcome.get("action_checks"):
                continue
            r = check(prepare(t), type_model(task_type(t)))
            end, confirmed, accepted = ending(t)
            rs.append({"task": t.outcome["task_id"], "trial": t.outcome["trial"],
                       "end": end, "confirmed": confirmed, "accepted": accepted, "flag": not r.compliant,
                       "kinds": sorted({v.constraint.split("(")[0] for v in r.violations}),
                       "failed": failed(t),
                       **{lab: violation(t, lab) for lab in LABELS}})
        out[agent] = rs
    return out


def share(a, b):
    return f"{a:5} of {b:5} ({a / b:5.1%})" if b else f"{a:5} of {b:5}"


def summary(name, rs):
    cust = [r for r in rs if r["end"] == "customer"]
    print(f"  {name}: {len(rs)} scorable runs")
    print(f"    ends on {TOKEN}: after a transfer call {sum(r['end'] == 'agent' for r in rs)}, "
          f"without one (customer-ended) {len(cust)}")
    for lab in LABELS:
        viol = [r for r in rs if r[lab]]
        ok = [r for r in rs if not r[lab]]
        print(f"    {lab:13} customer-ended among violations {share(sum(r['end'] == 'customer' for r in viol), len(viol))}, "
              f"among compliant runs {share(sum(r['end'] == 'customer' for r in ok), len(ok))}")
    tp = [r for r in rs if r["flag"] and r["action"]]
    print(f"    engine flags {sum(r['flag'] for r in cust)} of the {len(cust)} customer-ended transfers; "
          f"they are {share(sum(r['end'] == 'customer' for r in tp), len(tp))} of the engine's "
          f"true positives against the per-action label")
    kinds = collections.Counter("+".join(r["kinds"]) for r in cust if r["flag"])
    print(f"    constraint kinds of those flags: {dict(kinds.most_common())}")
    for key, what in (("confirmed", "change confirmed at the end"), ("accepted", "transfer accepted at the end")):
        sel = [r for r in cust if r[key]]
        print(f"    {what}: {len(sel)} of the {len(cust)}, of which per-action violations "
              f"{sum(r['action'] for r in sel)}, flagged by the engine {sum(r['flag'] for r in sel)}, "
              f"flagged and a per-action violation {sum(r['flag'] and r['action'] for r in sel)}")


def main():
    print(__doc__.split("Reads the mirror")[0].rstrip())
    for domain in ("airline", "retail"):
        per_agent = rows(domain)
        print(f"\n=== {domain}, {len(per_agent)} agents ===")
        dev = next(a for a in per_agent if a.startswith(DEV_AGENT))
        summary(dev, per_agent[dev])
        summary("POOLED, all agents", [r for rs in per_agent.values() for r in rs])
        print("  per agent: scorable runs, customer-ended transfers, of which per-action violations, "
              "flagged by the engine, change confirmed at the end, transfer accepted at the end")
        for agent, rs in per_agent.items():
            cust = [r for r in rs if r["end"] == "customer"]
            print(f"    {agent:40} {len(rs):4} {len(cust):4} {sum(r['action'] for r in cust):4} "
                  f"{sum(r['flag'] for r in cust):4} {sum(r['confirmed'] for r in cust):4} "
                  f"{sum(r['accepted'] for r in cust):4}")
        print(f"  {dev}, customer-ended transfers that the engine flags "
              "(task, trial, change confirmed / transfer accepted at the end, failing gold actions, "
              "constraint kinds):")
        for r in sorted((r for r in per_agent[dev] if r["end"] == "customer" and r["flag"]),
                        key=lambda r: (int(r["task"]) if r["task"].isdigit() else r["task"], r["trial"])):
            print(f"    task {r['task']:>4} trial {r['trial']}  "
                  f"{'change' if r['confirmed'] else '-':6} {'transfer' if r['accepted'] else '-':8}  "
                  f"{', '.join(r['failed']) or 'db only':45}  "
                  f"{'+'.join(r['kinds'])}")


if __name__ == "__main__":
    sys.exit(main())
