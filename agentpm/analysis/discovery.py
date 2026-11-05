"""Process discovery over canonical traces (R8).

Two directly-follows relations are offered, and the difference between them is
a result of the thesis rather than an implementation detail.

``directly_follows_flat`` reads each trace as the sequence ``ordered()`` gives
and counts consecutive pairs. That is what every mainstream discovery algorithm
does with a log, and it counts a pair of tool calls issued in one turn as a
succession, although nothing in the source orders them.

``directly_follows`` reads the partial order ``steps()`` gives. A pair is
counted only between an event and the events of the next step, so concurrent
events never follow each other. Within-step pairs are returned separately so
that the share of succession edges that are serialisation artefacts can be
reported.

Petri-net and process-tree discovery is delegated to pm4py in the experiments;
this module stays free of that dependency.
"""
from __future__ import annotations

from ..model import Trace

Edge = tuple[str, str]


def directly_follows_flat(traces: list[Trace]) -> dict[Edge, int]:
    """Consecutive pairs over the total order, as a flat log would give them."""
    counts: dict[Edge, int] = {}
    for t in traces:
        acts = t.activities
        for a, b in zip(acts, acts[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def directly_follows(traces: list[Trace]) -> dict[Edge, int]:
    """Pairs between one step and the next; concurrent events never follow each other."""
    counts: dict[Edge, int] = {}
    for t in traces:
        steps = t.steps()
        for cur, nxt in zip(steps, steps[1:]):
            for a in cur:
                for b in nxt:
                    counts[(a.activity, b.activity)] = counts.get((a.activity, b.activity), 0) + 1
    return counts


def concurrent_pairs(traces: list[Trace]) -> dict[Edge, int]:
    """Pairs that share a step, in serialisation order: the edges a flat log invents."""
    counts: dict[Edge, int] = {}
    for t in traces:
        for step in t.steps():
            for a, b in zip(step, step[1:]):
                counts[(a.activity, b.activity)] = counts.get((a.activity, b.activity), 0) + 1
    return counts


def concurrency_summary(traces: list[Trace]) -> dict[str, int]:
    """How much of the log is partially ordered."""
    steps = [s for t in traces for s in t.steps()]
    multi = [s for s in steps if len(s) > 1]
    return {
        "traces": len(traces),
        "traces_with_concurrency": sum(1 for t in traces if any(len(s) > 1 for s in t.steps())),
        "steps": len(steps),
        "steps_with_more_than_one_event": len(multi),
        "largest_step": max((len(s) for s in steps), default=0),
        "flat_edges": sum(directly_follows_flat(traces).values()),
        "partial_order_edges": sum(directly_follows(traces).values()),
        "within_step_pairs": sum(concurrent_pairs(traces).values()),
    }
