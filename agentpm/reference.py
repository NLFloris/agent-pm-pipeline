"""Reference model for conformance checking (thesis R3).

The reference model is the explicit, algorithmic form of a compliance checklist.
It is a small DECLARE-style constraint set over activity names, so that checking a
trace is deterministic (R4) and every violation localises to specific events (R5).
"""
from __future__ import annotations

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
