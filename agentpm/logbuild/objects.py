"""Object extraction rules for the object-centric log (layer L3, R6).

An object-centric log relates every event to the business objects it touches.
Which objects exist is a property of the source, so it is declared per source
as data, not written into the builder. A rule names an object type and where
in the event's attributes its identifier is found:

  {"type": "reservation", "path": "arguments.reservation_id"}
  {"type": "reservation", "path": "result.reservation_id", "activity": "book_reservation"}
  {"type": "flight",      "path": "arguments.flights[].flight_number"}
  {"type": "route",       "compose": ["arguments.origin", "arguments.destination",
                                      "arguments.date"], "join": ">"}

A path walks nested dicts with dots; ``[]`` fans out over a list. A value that
is a JSON string is decoded before the walk, because most sources record tool
arguments and results as text. ``activity`` restricts a rule to one activity.
``compose`` joins several paths into one identifier, for objects such as a
route that have no identifier of their own in the source.

Identifiers are scoped per run by the builder: two runs that touch the same
reservation id are independent replays against the same seeded database, not
one reservation moving through one process.

Rules can be loaded from a JSON file (a list of rule objects) with
:func:`load_rules`. The rules for the sources shipped with the pipeline are in
:data:`BUILTIN`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..model import Event


@dataclass(frozen=True)
class ObjectRule:
    type: str
    path: str | None = None
    activity: str | None = None
    compose: tuple[str, ...] = field(default_factory=tuple)
    join: str = ">"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObjectRule":
        if not d.get("type"):
            raise ValueError(f"object rule without a type: {d}")
        if not d.get("path") and not d.get("compose"):
            raise ValueError(f"object rule {d['type']!r} needs a path or a compose list")
        return cls(type=str(d["type"]), path=d.get("path"), activity=d.get("activity"),
                   compose=tuple(d.get("compose") or ()), join=str(d.get("join", ">")))


def _decode(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "{[":
            try:
                return json.loads(s)
            except ValueError:
                return v
    return v


def _walk(obj: Any, parts: list[str]) -> list[Any]:
    """All values reached by a dotted path with ``[]`` fan-out; missing -> []."""
    if not parts:
        return [] if obj is None or obj == "" else [obj]
    head, rest = parts[0], parts[1:]
    obj = _decode(obj)
    if head.endswith("[]"):
        key = head[:-2]
        seq = _decode(obj.get(key)) if isinstance(obj, dict) and key else obj
        if not isinstance(seq, list):
            return []
        return [v for item in seq for v in _walk(item, rest)]
    if not isinstance(obj, dict):
        return []
    return _walk(obj.get(head), rest)


def _values(event: Event, path: str) -> list[str]:
    return [str(v) for v in _walk(event.attributes, path.split(".")) if v is not None]


def extract(event: Event, rules: Iterable[ObjectRule]) -> list[tuple[str, str]]:
    """(object type, raw identifier) pairs the event touches, in rule order, deduplicated."""
    out: list[tuple[str, str]] = []
    for r in rules:
        if r.activity and r.activity != event.activity:
            continue
        if r.compose:
            parts = [_values(event, p) for p in r.compose]
            if all(parts):
                out.append((r.type, r.join.join(p[0] for p in parts)))
        elif r.path:
            for v in _values(event, r.path):
                out.append((r.type, v))
    seen: set[tuple[str, str]] = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def rules_from_json(items: list[dict[str, Any]]) -> list[ObjectRule]:
    return [ObjectRule.from_dict(d) for d in items]


def load_rules(path: str) -> list[ObjectRule]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("rules", [])
    return rules_from_json(data)


# Rules for the shipped sources. The airline rules name the objects of the
# tau2-bench airline domain as its tool schemas spell them.
BUILTIN: dict[str, list[ObjectRule]] = {
    "tau2": rules_from_json([
        {"type": "user", "path": "arguments.user_id"},
        {"type": "reservation", "path": "arguments.reservation_id"},
        {"type": "reservation", "path": "result.reservation_id", "activity": "book_reservation"},
        {"type": "flight", "path": "arguments.flight_number"},
        {"type": "flight", "path": "arguments.flights[].flight_number"},
        {"type": "payment", "path": "arguments.payment_id"},
        {"type": "payment", "path": "arguments.payment_methods[].payment_id"},
        {"type": "route", "compose": ["arguments.origin", "arguments.destination", "arguments.date"]},
    ]),
}
