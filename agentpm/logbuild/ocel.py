"""Build an OCEL 2.0 object-centric event log from OpenAI agent runs (layer L3).

Output format
-------------
OCEL 2.0 JSON exchange format (Berti et al., OCEL 2.0 Specification, PADS RWTH
Aachen, 2023; www.ocel-standard.org). Key layout verified against the reference
importer in pm4py (``pm4py/objects/ocel/importer/jsonocel/variants/
ocel20_standard.py``, accessed 2026-07-08):

  top level    {"objectTypes": [...], "eventTypes": [...], "objects": [...], "events": [...]}
  event        {"id", "type", "time", "attributes": [{"name", "value"}],
                "relationships": [{"objectId", "qualifier"}]}
  object       {"id", "type", "attributes": [{"name", "time", "value"}],
                "relationships": [{"objectId", "qualifier"}]}
  type decl    {"name", "attributes": [{"name", "type"}]}

Mapping (thesis requirement R6, object-centric case notion)
-----------------------------------------------------------
One log may hold many runs, which is where object-centric analysis pays off.

  objects   run   (one per SDK trace, attribute workflow_name)
            agent (one per distinct agent name, across runs)
            tool  (one per distinct function name, across runs)
  O2O       agent -participates_in-> run, tool -used_in-> run
  events    executed calls only (R2): tool calls under their tool name,
            generation/response as llm_call, handoff, guardrail
  E2O       every event -run-> its run; -executed_by-> its agent, except
            handoffs, which instead relate -from-> and -to-> the two agents
            involved (one qualifier per event-object pair); tool calls also
            -uses-> their tool

The builder is deterministic (thesis R4): objects, events, and type declarations
are emitted in sorted order, so identical input yields byte-identical output.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from ..ingest.openai_agents import AgentRun, _activity, _event_attributes, owning_agent
from ..model import Trace
from .objects import ObjectRule, extract

_EPOCH = "1970-01-01T00:00:00+00:00"


def _parse_iso(iso: Any) -> datetime | None:
    if not isinstance(iso, str):
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None


def _ms_to_iso(ms: int) -> str:
    """Canonical epoch-millisecond timestamps back to ISO 8601 for OCEL."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# Logical timestamps are small integers; epoch milliseconds are not. Anything
# below this bound is read as a logical time and placed on a synthetic clock.
_LOGICAL_BOUND = 10_000_000_000
_LOGICAL_BASE = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _event_time(ts: int) -> str:
    """OCEL needs a datetime; a logical time becomes seconds on a fixed clock,
    so that events issued together keep an identical time in the log."""
    if ts < _LOGICAL_BOUND:
        return (_LOGICAL_BASE + timedelta(seconds=ts)).isoformat()
    return _ms_to_iso(ts)


def _val(v: Any) -> Any:
    """OCEL attribute values as JSON scalars; nested payloads as JSON strings."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return json.dumps(v, sort_keys=True)


def _ocel_type(v: Any) -> str:
    """OCEL 2.0 attribute type matching an observed value."""
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "float"
    return "string"


def _declared_type(types: set[str]) -> str:
    """Single declared type per attribute; mixed observations degrade to string."""
    return next(iter(types)) if len(types) == 1 else "string"


def build_ocel(runs: list[AgentRun]) -> dict[str, Any]:
    """Map OpenAI agent runs onto one OCEL 2.0 log (dict, JSON-serialisable)."""
    objects: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    event_type_attrs: dict[str, dict[str, set[str]]] = {}

    def ensure_object(oid: str, otype: str, attrs: dict[str, Any], time: str) -> None:
        if oid not in objects:
            objects[oid] = {
                "id": oid,
                "type": otype,
                "attributes": [{"name": k, "time": time, "value": _val(v)} for k, v in sorted(attrs.items())],
                "relationships": [],
            }

    def relate_object(oid: str, target: str, qualifier: str) -> None:
        rels = objects[oid]["relationships"]
        entry = {"objectId": target, "qualifier": qualifier}
        if entry not in rels:
            rels.append(entry)

    for run in runs:
        starts = [d for s in run.spans if (d := _parse_iso(s.started_at)) is not None]
        run_start_dt = min(starts) if starts else datetime(1970, 1, 1, tzinfo=timezone.utc)
        run_start = run_start_dt.isoformat()
        ensure_object(run.trace_id, "run", {"workflow_name": run.workflow_name}, run_start)
        by_id = {s.id: s for s in run.spans}

        for i, span in enumerate(run.spans):
            act = _activity(span)
            if act is None:
                continue
            # normalise the event time; an unparseable started_at falls back to the
            # run start plus the span's file position, preserving order (finding of
            # the design review, keeps pm4py able to parse every emitted time)
            t_dt = _parse_iso(span.started_at)
            time = (t_dt if t_dt is not None else run_start_dt + timedelta(milliseconds=i)).isoformat()
            agent = owning_agent(span, by_id)
            ensure_object(f"agent:{agent}", "agent", {"name": agent}, time)
            relate_object(f"agent:{agent}", run.trace_id, "participates_in")

            rels = [{"objectId": run.trace_id, "qualifier": "run"}]
            if span.type != "handoff":
                # handoffs carry from/to instead, keeping one qualifier per event-object pair
                rels.append({"objectId": f"agent:{agent}", "qualifier": "executed_by"})
            if span.type == "function":
                tool_id = f"tool:{span.data.get('name', 'function')}"
                ensure_object(tool_id, "tool", {"name": span.data.get("name", "function")}, time)
                relate_object(tool_id, run.trace_id, "used_in")
                rels.append({"objectId": tool_id, "qualifier": "uses"})
            elif span.type == "handoff":
                for key, qual in (("from_agent", "from"), ("to_agent", "to")):
                    name = span.data.get(key)
                    if name:
                        ensure_object(f"agent:{name}", "agent", {"name": name}, time)
                        relate_object(f"agent:{name}", run.trace_id, "participates_in")
                        rels.append({"objectId": f"agent:{name}", "qualifier": qual})

            attrs = {k: _val(v) for k, v in _event_attributes(span).items()}
            for k, v in attrs.items():
                event_type_attrs.setdefault(act, {}).setdefault(k, set()).add(_ocel_type(v))
            events.append(
                {
                    "id": span.id,
                    "type": act,
                    "time": time,
                    "attributes": [{"name": k, "value": v} for k, v in sorted(attrs.items())],
                    "relationships": rels,
                }
            )

    events.sort(key=lambda e: (e["time"], e["id"]))
    object_type_attrs = {"run": ["workflow_name"], "agent": ["name"], "tool": ["name"]}
    used_types = sorted({o["type"] for o in objects.values()})
    return {
        "objectTypes": [
            {"name": t, "attributes": [{"name": a, "type": "string"} for a in object_type_attrs[t]]}
            for t in used_types
        ],
        "eventTypes": [
            {"name": t, "attributes": [{"name": a, "type": _declared_type(ts)} for a, ts in sorted(attrs.items())]}
            for t, attrs in sorted(event_type_attrs.items())
        ],
        "objects": [objects[k] for k in sorted(objects)],
        "events": events,
    }


def write_ocel(ocel: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ocel, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_ocel_from_traces(
    traces: list[Trace],
    object_rules: list[ObjectRule] | None = None,
    tool_objects: bool = True,
    exclude: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Map canonical traces onto one OCEL 2.0 log (source-agnostic, layer L3).

    Unlike :func:`build_ocel`, which consumes OpenAI-specific runs and can express
    handoff semantics, this builder works from the canonical model (L2) and is fed
    by any ingester. Object types:

      run        one per trace (the case)
      agent      one per distinct event resource
      tool       one per distinct activity, when ``tool_objects`` is set
      <declared> whatever ``object_rules`` extract from the event attributes,
                 scoped per run (see ``objects.py``); these are the business
                 objects, the reason an object-centric log exists (R6)

    Every event relates to its run (qualifier "run") and its agent
    ("executed_by"), to its tool ("uses") when tool objects are emitted, and to
    each extracted object ("touches"). An event that starts a sub-agent (a
    ``spawns`` attribute) also relates to the sub-agent ("spawns"), and the
    sub-agent's object points at its parent ("spawned_by").

    Time: events keep the canonical timestamp, so events the source issued
    together carry an identical time in the log. OCEL 2.0 permits that; it is
    the partial order the flat log cannot hold. The event id carries the
    serialisation position for anyone who needs the tiebreak back.

    ``exclude`` drops activities from the log, for views that want executed
    actions only and not the communication turns.

    Emission is sorted, so identical input yields byte-identical output (R4).
    """
    rules = list(object_rules or [])
    skip = set(exclude or ())
    objects: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    event_type_attrs: dict[str, dict[str, set[str]]] = {}

    def ensure_object(oid: str, otype: str, attrs: dict[str, Any], time: str) -> None:
        if oid not in objects:
            objects[oid] = {
                "id": oid,
                "type": otype,
                "attributes": [{"name": k, "time": time, "value": _val(v)} for k, v in sorted(attrs.items())],
                "relationships": [],
            }

    def relate_object(oid: str, target: str, qualifier: str) -> None:
        rels = objects[oid]["relationships"]
        entry = {"objectId": target, "qualifier": qualifier}
        if entry not in rels:
            rels.append(entry)

    for trace in traces:
        ordered = [e for e in trace.ordered() if e.activity not in skip]
        run_time = _event_time(ordered[0].timestamp) if ordered else _EPOCH
        ensure_object(trace.case_id, "run", dict(trace.outcome or {}), run_time)

        for idx, ev in enumerate(ordered):
            time = _event_time(ev.timestamp)
            agent_id = f"agent:{ev.resource}"
            ensure_object(agent_id, "agent", {"name": ev.resource}, time)
            relate_object(agent_id, trace.case_id, "participates_in")

            rels = [
                {"objectId": trace.case_id, "qualifier": "run"},
                {"objectId": agent_id, "qualifier": "executed_by"},
            ]
            if tool_objects:
                tool_id = f"tool:{ev.activity}"
                ensure_object(tool_id, "tool", {"name": ev.activity}, time)
                relate_object(tool_id, trace.case_id, "used_in")
                rels.append({"objectId": tool_id, "qualifier": "uses"})
            child = ev.attributes.get("spawns")
            if child:
                child_id = f"agent:{child}"
                ensure_object(child_id, "agent", {"name": child}, time)
                relate_object(child_id, trace.case_id, "participates_in")
                relate_object(child_id, agent_id, "spawned_by")
                rels.append({"objectId": child_id, "qualifier": "spawns"})
            for otype, raw in extract(ev, rules):
                oid = f"{otype}:{trace.case_id}|{raw}"
                ensure_object(oid, otype, {"key": raw}, time)
                relate_object(oid, trace.case_id, "belongs_to")
                rels.append({"objectId": oid, "qualifier": "touches"})

            attrs = {k: _val(v) for k, v in ev.attributes.items()}
            for k, v in attrs.items():
                event_type_attrs.setdefault(ev.activity, {}).setdefault(k, set()).add(_ocel_type(v))
            events.append(
                {
                    "id": f"{trace.case_id}:{idx:06d}",
                    "type": ev.activity,
                    "time": time,
                    "attributes": [{"name": k, "value": v} for k, v in sorted(attrs.items())],
                    "relationships": rels,
                }
            )

    events.sort(key=lambda e: (e["time"], e["id"]))
    obj_attr_names: dict[str, set[str]] = {}
    for o in objects.values():
        obj_attr_names.setdefault(o["type"], set()).update(a["name"] for a in o["attributes"])
    return {
        "objectTypes": [
            {"name": t, "attributes": [{"name": a, "type": "string"} for a in sorted(names)]}
            for t, names in sorted(obj_attr_names.items())
        ],
        "eventTypes": [
            {"name": t, "attributes": [{"name": a, "type": _declared_type(ts)} for a, ts in sorted(attrs.items())]}
            for t, attrs in sorted(event_type_attrs.items())
        ],
        "objects": [objects[k] for k in sorted(objects)],
        "events": events,
    }


_STRUCTURAL = {"run", "agent", "tool"}


def object_sharing(ocel: dict[str, Any]) -> dict[str, Any]:
    """How many events relate to more than one object type, and how many touch a
    business object, one that the rules extracted rather than the builder
    supplied. Run, agent and tool are structural: every event has them."""
    by_type: dict[str, str] = {o["id"]: o["type"] for o in ocel["objects"]}
    shared = business = 0
    for e in ocel["events"]:
        types = {by_type[r["objectId"]] for r in e["relationships"]}
        if len(types) > 1:
            shared += 1
        if types - _STRUCTURAL:
            business += 1
    counts: dict[str, int] = {}
    for o in ocel["objects"]:
        counts[o["type"]] = counts.get(o["type"], 0) + 1
    return {"events": len(ocel["events"]), "events_with_more_than_one_object_type": shared,
            "events_touching_a_business_object": business,
            "objects_by_type": dict(sorted(counts.items()))}
