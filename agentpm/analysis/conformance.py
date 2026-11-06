"""Conformance checking (pipeline layer L4, primary analysis).

Deterministic, rule-based conformance of a trace against a Reference. Every
violation names the constraint and the specific event responsible, so the verdict
is reproducible (R4) and every violation can be traced to its cause (R5). This is
the algorithmic form of a compliance checklist, replacing a prompt-based judge for
the compliance signal.

Fitness is computed with one unit of weight per constraint, so it is independent
of trace length and comparable across runs. A single constraint may still emit
several localized violations (for example a closed-world 'allowed' rule flags each
disallowed event), but it counts once toward fitness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..model import Trace
from ..reference import Reference


@dataclass
class Violation:
    constraint: str                 # e.g. "precedence(retrieve,answer)"
    description: str
    at: Optional[int] = None        # index in the ordered trace, if localizable


@dataclass
class ConformanceResult:
    case_id: str
    compliant: bool
    fitness: float                  # fraction of constraints satisfied, one weight each, in [0, 1]
    violations: list[Violation] = field(default_factory=list)
    n_constraints: int = 0          # number of constraints checked (not events)


def check(trace: Trace, ref: Reference) -> ConformanceResult:
    """Check one trace against a reference.

    Order is read from the trace's steps, not from the serialisation: events the
    source issued together share a step, and the engine never orders them with
    respect to each other. "a before b" holds when some a is in the same step as
    b or an earlier one; "b not after a" is violated only by a b in a later step
    than the first a. The verdict is therefore invariant to the order in which
    calls issued together happen to be listed. Positions in ``at`` still index
    the serialised trace, so a violation can be shown.
    """
    ordered = trace.ordered()
    acts = [e.activity for e in ordered]
    step_of = [e.timestamp for e in ordered]
    first: dict[str, int] = {}
    for i, a in enumerate(acts):
        first.setdefault(a, i)          # first occurrence index (serialised)

    violations: list[Violation] = []
    n_constraints = 0
    failed = 0

    # allowed (closed-world): one constraint; still localize every disallowed event
    if ref.allowed:
        n_constraints += 1
        allowed = set(ref.allowed)
        before = len(violations)
        for i, a in enumerate(acts):
            if a not in allowed:
                violations.append(Violation(f"allowed({a})", f"activity '{a}' is not allowed", i))
        if len(violations) > before:
            failed += 1

    # required: one constraint per required activity
    for r in ref.required:
        n_constraints += 1
        if r not in first:
            violations.append(Violation(f"required({r})", f"required activity '{r}' is missing"))
            failed += 1

    # precedence: one constraint per pair; a must occur before b if b occurs.
    # "Before" is by step: an a issued together with the first b counts.
    for pair in ref.precedence:
        a, b = pair
        n_constraints += 1
        if b in first and (a not in first or step_of[first[a]] > step_of[first[b]]):
            violations.append(Violation(f"precedence({a},{b})", f"'{a}' must occur before '{b}'", first[b]))
            failed += 1

    # forbidden_after: one constraint per pair; b must not occur after a, where
    # "after" means a later step than the first a
    for pair in ref.forbidden_after:
        a, b = pair
        n_constraints += 1
        if a in first:
            for j in range(first[a] + 1, len(acts)):
                if acts[j] == b and step_of[j] > step_of[first[a]]:
                    violations.append(
                        Violation(f"forbidden_after({a},{b})", f"'{b}' must not occur after '{a}'", j)
                    )
                    failed += 1
                    break

    # directly_before (chain precedence): one constraint per pair; every b must have a
    # immediately before it. This is what "confirm with the user right before you
    # write" needs, which plain precedence cannot express.
    # Activities in ref.transparent are looked through: one confirmation covers a
    # batch of writes that the agent announces together, and the agent may say
    # something while performing them. Without this the rule fires on every
    # second write of a batch, which is not what a policy means by "confirm".
    transparent = set(ref.transparent)
    for pair in ref.directly_before:
        a, b = pair
        n_constraints += 1
        before = len(violations)
        for j, act in enumerate(acts):
            if act != b:
                continue
            k = j - 1
            while k >= 0 and acts[k] in transparent:
                k -= 1
            if k < 0 or acts[k] != a:
                violations.append(
                    Violation(f"directly_before({a},{b})",
                              f"'{b}' at {j} is not preceded by '{a}' (looking through "
                              f"{sorted(transparent)})", j)
                )
        if len(violations) > before:
            failed += 1

    # end activities: one constraint
    if ref.end_activities:
        n_constraints += 1
        ends = set(ref.end_activities)
        if not acts or acts[-1] not in ends:
            last = acts[-1] if acts else "<empty>"
            violations.append(
                Violation("end_activities", f"run ends with '{last}', expected one of {ref.end_activities}",
                          len(acts) - 1 if acts else None)
            )
            failed += 1

    fitness = 1.0 if n_constraints == 0 else round((n_constraints - failed) / n_constraints, 4)
    return ConformanceResult(
        case_id=trace.case_id,
        compliant=not violations,
        fitness=fitness,
        violations=violations,
        n_constraints=n_constraints,
    )
