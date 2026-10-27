"""Performance analysis (secondary lens, R7).

Planned: bottlenecks, waiting times, redundant loops, rework, and localisation of
the first divergence from a healthy path. As a first step this module reports a
crude rework signal, namely the number of immediately repeated activities.
"""
from __future__ import annotations

from ..model import Trace


def repeat_count(trace: Trace) -> int:
    """Number of immediately repeated activities (a simple rework signal)."""
    acts = trace.activities
    return sum(1 for a, b in zip(acts, acts[1:]) if a == b)
