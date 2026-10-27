"""Task types and their reference models for the tau2-bench retail domain.

Written on 2026-10-23 from data/tau2/domains/retail/policy.md alone, with the
same template as the airline models in task_type_conformance.py, and fixed
before any retail run was scored against them. What had been seen of the retail
data at that point: the list of task types and the composition of the
per-action label (which gold actions fail to match), not any conformance result.

A task type is read mechanically from the WRITE actions the task's gold list
requires, exactly as for airline. Retail has 23 types in 114 tasks (no-write and
transfer included), against 11 for airline, because customers combine requests
(cancel one order and return another), but every type gets its model from the
same rules, so there is one template and not 23 models:

  required          the write(s) that define the type, plus what any write of
                    that kind depends on:
                      authenticate_user before any write   "At the beginning of
                        the conversation, you have to authenticate the user
                        identity ... even when the user already provides the
                        user id."
                      get_order_details before an order write   "you should
                        check its status before taking the action" (cancel,
                        modify, return, exchange)
  precedence        that dependency must come before the write
  directly_before   a user message must precede every write, looking through
                    other writes and the agent's own messages: "you must list
                    the action details and obtain explicit user confirmation
                    (yes) to proceed". Identical to the airline rule.
  allowed           for tasks that require no write at all, no write may occur

One step of event abstraction comes first. The policy names two ways to
authenticate, by email or by name and zip code, and either satisfies the rule,
so both are relabelled to one activity, authenticate_user, before checking. The
engine stays untouched; the abstraction is declared here, as data.

Not encoded, because a log of activity names cannot express it: "Exchange or
modify order tools can only be called once per order". That is a constraint on
the order object, not on the run, and needs the order id of each call.

The variant ``reads_transparent`` lets the confirmation rule also look through
read-only tools, so that a lookup between the user's yes and the write does not
break the chain. It exists because the airline models, applied to fifteen other
agents, fired on exactly that (2026-10-23), and is reported next to the frozen
template, never instead of it.
"""
from __future__ import annotations

import copy

from agentpm.reference import Reference

WRITES = ["cancel_pending_order", "modify_pending_order_address", "modify_pending_order_items",
          "modify_pending_order_payment", "modify_user_address",
          "return_delivered_order_items", "exchange_delivered_order_items"]
ORDER_WRITES = [w for w in WRITES if w != "modify_user_address"]
READS = ["find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details",
         "get_order_details", "get_product_details", "get_item_details",
         "list_all_product_types", "calculate"]
TOOLS = READS + ["transfer_to_human_agents"] + WRITES
MESSAGES = ["assistant_message", "user_message"]

# event abstraction: either way of authenticating is the same step of the process
ABSTRACTION = {"find_user_id_by_email": "authenticate_user",
               "find_user_id_by_name_zip": "authenticate_user"}


def abstract(trace):
    """A copy of the trace with the activities in ABSTRACTION relabelled."""
    t = copy.deepcopy(trace)
    for e in t.events:
        e.activity = ABSTRACTION.get(e.activity, e.activity)
    return t


def task_type(trace):
    """The write actions the task requires, as a canonical key."""
    gold = [a["name"] for a in trace.outcome.get("gold_actions", []) if a.get("requestor") == "assistant"]
    writes = sorted(set(a for a in gold if a in WRITES))
    if writes:
        return "+".join(writes)
    return "transfer" if "transfer_to_human_agents" in gold else "no-write"


def type_model(key, reads_transparent=False):
    """One reference per task type, built from the type alone."""
    reads = sorted(set(ABSTRACTION.get(t, t) for t in READS))
    if key == "no-write":
        return Reference(name=key, allowed=reads + ["transfer_to_human_agents"] + MESSAGES)
    if key == "transfer":
        return Reference(name=key, required=["transfer_to_human_agents"],
                         allowed=reads + ["transfer_to_human_agents"] + MESSAGES)
    writes = key.split("+")
    required = list(writes) + ["authenticate_user"]
    precedence = [["authenticate_user", w] for w in writes]
    if any(w in ORDER_WRITES for w in writes):
        required.append("get_order_details")
        precedence += [["get_order_details", w] for w in writes if w in ORDER_WRITES]
    transparent = WRITES + ["assistant_message"] + (reads if reads_transparent else [])
    return Reference(name=key, required=sorted(set(required)), precedence=precedence,
                     directly_before=[["user_message", w] for w in writes],
                     # one confirmation covers an announced batch of writes
                     transparent=transparent)
