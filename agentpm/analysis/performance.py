"""Secondary lenses on how a run went (R8). They describe runs and enter no verdict.

Repeated tool use is the share of a run's tool calls that repeat a tool the run
has already used. It counts repeats, whatever their cause, so it is not rework in
the sense of work redone after an error. A handover is a transition at which the
acting agent changes.
"""
from __future__ import annotations

import collections
from typing import Optional

from ..model import Trace

# activities that are not an executed tool call: communication, coordination,
# framework bookkeeping
NON_ACTION = {"assistant_message", "user_message", "orchestrate", "report", "delegate",
              "invoke_agent", "invoke_workflow", "create_agent", "llm_call", "user_prompt",
              "subagent_completed", "handoff", "guardrail"}


def repeated_tool_use(trace: Trace) -> Optional[float]:
    """The share of the run's tool calls that repeat a tool the run has already used,
    or None for a run with fewer than two tool calls."""
    calls = [a for a in trace.activities if a not in NON_ACTION]
    return 1 - len(set(calls)) / len(calls) if len(calls) >= 2 else None


def handovers(traces: list[Trace]) -> collections.Counter:
    """(agent, next agent) -> how often the acting agent changed that way."""
    edges: collections.Counter = collections.Counter()
    for t in traces:
        res = [e.resource for e in t.ordered()]
        for a, b in zip(res, res[1:]):
            if a != b:
                edges[(a, b)] += 1
    return edges
