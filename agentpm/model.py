"""Canonical trace data model (pipeline layer L2).

An agent execution trace is normalised into a small, explicit schema before any
process-mining step. Following the thesis design decisions, activities come from
executed actions (tool and model calls); free-text reasoning is kept only as an
event attribute and is never used as an activity label or trusted as ground truth.

Order
-----
Agent traces are not totally ordered. A model that issues several tool calls in
one turn decides them together, and a framework that runs two sub-agents at
once gives their spans overlapping lifetimes. The schema keeps both facts apart:

  timestamp   logical time. Events the source issued together share a value;
              nothing orders them with respect to each other. Sources with a
              real clock may use epoch milliseconds instead.
  sequence    the position of the event in the source's serialisation, unique
              within a trace. It is a stable tiebreak for views that need a
              total order (a flat XES log, a conformance check over a sequence)
              and is never read as evidence that one concurrent event preceded
              another.

``ordered()`` gives that total order; ``steps()`` gives the partial order as a
list of concurrent sets, which is what the object-centric log and the
concurrency-aware discovery consume.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    activity: str                                            # what happened (tool/model call name)
    timestamp: int                                           # logical time (or epoch ms); ties = issued together
    resource: str = "unknown"                                # who did it (agent/role/tool)
    attributes: dict[str, Any] = field(default_factory=dict)  # prompts, args, reasoning, ...
    sequence: int | None = None                              # serialisation position; tiebreak only


@dataclass
class Trace:
    case_id: str
    events: list[Event] = field(default_factory=list)
    # Self-reported outcome is a CLAIM to corroborate against actions, not ground truth.
    outcome: dict[str, Any] = field(default_factory=dict)

    def ordered(self) -> list[Event]:
        """Events in a total order: by timestamp, then sequence, then insertion."""
        return [e for _, e in sorted(
            enumerate(self.events),
            key=lambda ie: (ie[1].timestamp, ie[1].sequence if ie[1].sequence is not None else ie[0], ie[0]))]

    def steps(self) -> list[list[Event]]:
        """The partial order: consecutive groups of events that share a timestamp."""
        groups: list[list[Event]] = []
        for e in self.ordered():
            if groups and groups[-1][0].timestamp == e.timestamp:
                groups[-1].append(e)
            else:
                groups.append([e])
        return groups

    @property
    def activities(self) -> list[str]:
        return [e.activity for e in self.ordered()]

    def step_labels(self) -> list[str]:
        """One label per event in ``ordered()`` order: the step number, with a letter
        when several events share the step. "4a" and "4b" were issued together."""
        labels: list[str] = []
        for k, step in enumerate(self.steps()):
            if len(step) == 1:
                labels.append(str(k))
            else:
                for j, _ in enumerate(step):
                    labels.append(f"{k}{chr(97 + j) if j < 26 else '+' + str(j)}")
        return labels


def trace_from_dict(d: dict[str, Any]) -> Trace:
    events = [
        Event(
            activity=e["activity"],
            timestamp=int(e.get("timestamp", i)),
            resource=e.get("resource", "unknown"),
            attributes=e.get("attributes", {}),
            sequence=int(e["sequence"]) if e.get("sequence") is not None else None,
        )
        for i, e in enumerate(d.get("events", []))
    ]
    return Trace(case_id=d["case_id"], events=events, outcome=d.get("outcome", {}))


def load_trace(path: str) -> Trace:
    with open(path, encoding="utf-8") as f:
        return trace_from_dict(json.load(f))
