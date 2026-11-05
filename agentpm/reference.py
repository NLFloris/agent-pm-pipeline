"""Reference model for conformance checking (thesis R3).

The reference model is the explicit, algorithmic form of a compliance checklist.
It is a small DECLARE-style constraint set over activity names, so that checking a
trace is deterministic (R4) and every violation localises to specific events (R5).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Reference:
    name: str = "reference"
    allowed: list[str] = field(default_factory=list)             # if set, only these activities may occur
    required: list[str] = field(default_factory=list)            # each must occur at least once
    precedence: list[list[str]] = field(default_factory=list)    # [a, b]: a must occur before b if b occurs
    forbidden_after: list[list[str]] = field(default_factory=list)  # [a, b]: b must not occur after a
    directly_before: list[list[str]] = field(default_factory=list)  # [a, b]: every b must be immediately preceded by a
    transparent: list[str] = field(default_factory=list)         # activities directly_before looks through
    end_activities: list[str] = field(default_factory=list)      # if set, the run must end with one of these


def reference_from_dict(d: dict[str, Any]) -> Reference:
    return Reference(
        name=d.get("name", "reference"),
        allowed=list(d.get("allowed", [])),
        required=list(d.get("required", [])),
        precedence=[list(p) for p in d.get("precedence", [])],
        forbidden_after=[list(p) for p in d.get("forbidden_after", [])],
        directly_before=[list(p) for p in d.get("directly_before", [])],
        transparent=list(d.get("transparent", [])),
        end_activities=list(d.get("end_activities", [])),
    )


def load_reference(path: str) -> Reference:
    with open(path, encoding="utf-8") as f:
        return reference_from_dict(json.load(f))


@dataclass
class TaskTypes:
    """One reference model per task type, all from one template declared as data.

    A run that states its type on its outcome (``task_type``) needs nothing else.
    Otherwise the type is read mechanically from the changes its solution list
    requires. Every task of a kind gets the same model and no model names a task. A
    change requires the lookups it depends on, before it, and a confirmation directly
    before it. A task that requires no change may change nothing, and a transfer
    requires the transfer and changes nothing. A new domain brings its own file.
    """
    name: str = "task types"
    changes: list[str] = field(default_factory=list)             # the activities that change something
    lookups: list[list] = field(default_factory=list)            # [lookup, [changes it must precede]]
    no_change_allowed: list[str] = field(default_factory=list)   # all a task without a change may do
    look_through: list[str] = field(default_factory=list)        # what the confirmation rule looks through
    confirmation: str = "user_message"
    transfer: str = "transfer_to_human_agents"
    relabel: dict[str, str] = field(default_factory=dict)        # event abstraction, applied before checking

    def solution(self, trace) -> list[str]:
        return [a["name"] for a in trace.outcome.get("gold_actions", []) if a.get("requestor") == "assistant"]

    def stated_type(self, trace):
        """The type the run states, if this template knows it. A harness can use the
        same field for something else, such as a kind of collaboration."""
        key = trace.outcome.get("task_type")
        known = key and (key in ("no-write", "transfer") or all(c in self.changes for c in key.split("+")))
        return key if known else None

    def type_of(self, trace) -> str:
        stated = self.stated_type(trace)
        if stated:
            return stated
        names = self.solution(trace)
        changes = sorted(set(a for a in names if a in self.changes))
        if changes:
            return "+".join(changes)
        return "transfer" if self.transfer in names else "no-write"

    def model(self, key: str) -> Reference:
        if key == "no-write":
            return Reference(name=key, allowed=list(self.no_change_allowed))
        if key == "transfer":
            return Reference(name=key, required=[self.transfer], allowed=list(self.no_change_allowed))
        changes = key.split("+")
        required, precedence = list(changes), []
        for lookup, before in self.lookups:
            hits = [c for c in changes if c in before]
            if hits:
                required.append(lookup)
                precedence += [[lookup, c] for c in hits]
        return Reference(name=key, required=sorted(set(required)), precedence=precedence,
                         directly_before=[[self.confirmation, c] for c in changes],
                         transparent=list(self.look_through))

    def prepare(self, trace):
        """The trace as the models read it, with the event abstraction applied."""
        if not self.relabel:
            return trace
        t = copy.deepcopy(trace)
        for e in t.events:
            e.activity = self.relabel.get(e.activity, e.activity)
        return t

    def pairs(self, traces: list) -> list:
        """(prepared trace, its type's model) for every run that states a type or has a solution list."""
        return [(self.prepare(t), self.model(self.type_of(t))) for t in traces
                if self.stated_type(t) or self.solution(t)]


def load_task_types(path: str) -> TaskTypes:
    with open(path, encoding="utf-8") as f:
        return TaskTypes(**json.load(f))
