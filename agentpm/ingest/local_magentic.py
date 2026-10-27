"""Ingest locally recorded Magentic-One runs (pipeline layer L1).

Input format
------------
One JSON file per run, written by ``experiments/local_agents/run_magentic.py``
while driving AutoGen's Magentic-One against a local model server:

  run      {"task_id", "shape", "prompt", "model", "started_at", "seconds",
            "error", "sandbox", "messages": [...],
            optional "task_type", optional "subagents": {name: {"parent", "tools"}}}
  message  {"type", "source", "created_at", "received_at",
            "prompt_tokens", "completion_tokens", "payload"}
  payload  {"text": str}                                    a spoken turn
           {"items": [{"kind": "call", "id", "name", "arguments"} |
                      {"kind": "result", "call_id", "name", "is_error", "content"}]}

``source`` names the agent, which is what this source has and the two public
multi-agent corpora do not: Claude Code's telemetry carries no agent on its tool
events, and TraceElephant records the acting agent but not the arguments.

Mapping to the canonical model (L2)
-----------------------------------
  activity   a tool call becomes its function name. A speaking turn becomes
             ``orchestrate`` for the Magentic-One orchestrator and ``report``
             for anyone else. A speaker selection becomes ``delegate``, with
             the chosen agent on the ``to`` attribute.
  sub-agent  a call whose name is a declared sub-agent becomes ``spawn`` by the
             parent, with the sub-agent on the ``spawns`` attribute and its
             task as the arguments. The sub-agent's own events follow with
             ``spawned_by`` set to the parent. The task message the framework
             hands the sub-agent (source ``user``) is not a separate event.
  reasoning  ``ThoughtEvent`` is never an activity (decision L6). Its text is
             attached to the next event from the same agent, or to the trace.
  timestamp  step in the run: the calls of one message were issued together and
             share a step, every other event is a step of its own (the
             serialisation is the event order). ``created_at`` is kept as an attribute and is
             real wall-clock, so durations are available here, unlike in the
             benchmark corpora.
  resource   the acting agent.
  case       the task id.
  outcome    the task's declared collaboration ``shape``, the model that ran it,
             and any harness error. The shape is what the task asked for, not a
             judgement about what happened, so it stays a claim to check the
             discovered model against.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from ..model import Event, Trace

_ORCHESTRATOR = re.compile(r"orchestrat|magenticone", re.I)
_MAX_ATTR_CHARS = 4000


def _truncate(v: Any) -> Any:
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return s if len(s) <= _MAX_ATTR_CHARS else s[:_MAX_ATTR_CHARS] + f"... [{len(s)} chars]"


def _results_by_call(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for m in messages:
        for item in (m.get("payload") or {}).get("items") or []:
            if item.get("kind") == "result" and item.get("call_id"):
                out[str(item["call_id"])] = item
    return out


def _selected_speaker(text: Any) -> Optional[str]:
    """A speaker-selection turn names the agent it hands to."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    return t if t and len(t) < 64 and "\n" not in t else None


def _events(messages: list[dict[str, Any]],
            subagents: Optional[dict[str, Any]] = None) -> tuple[list[Event], list[str], list[str]]:
    results = _results_by_call(messages)
    subagents = subagents or {}
    events: list[Event] = []
    thoughts: list[str] = []
    pending: dict[str, str] = {}          # agent -> reasoning awaiting an event
    running: dict[str, str] = {}          # call id of a spawn -> sub-agent still at work
    step = 0

    for m in messages:
        mtype = str(m.get("type") or "")
        source = str(m.get("source") or "unknown")
        payload = m.get("payload") or {}
        base = {
            "created_at": m.get("created_at"),
            "prompt_tokens": m.get("prompt_tokens"),
            "completion_tokens": m.get("completion_tokens"),
            "message_type": mtype,
        }

        if mtype == "ThoughtEvent":
            text = payload.get("text")
            if isinstance(text, str) and text.strip():
                # an agent can think twice before it acts; keep both thoughts
                pending[source] = f"{pending[source]}\n\n{text}" if source in pending else text
                thoughts.append(text)
            continue

        def attach(attrs: dict[str, Any]) -> dict[str, Any]:
            if source in pending:
                attrs["reasoning"] = _truncate(pending.pop(source))
            if source in subagents:
                attrs["spawned_by"] = subagents[source].get("parent")
            return attrs

        items = payload.get("items") or []
        calls = [i for i in items if i.get("kind") == "call"]
        if calls:
            for call in calls:
                name = str(call.get("name") or "tool_call")
                result = results.get(str(call.get("id")))
                attrs = attach({
                    **base,
                    "arguments": _truncate(call.get("arguments")),
                    "call_id": call.get("id"),
                })
                if result is not None:
                    attrs["result"] = _truncate(result.get("content"))
                    attrs["error"] = bool(result.get("is_error"))
                if name in subagents:
                    attrs["spawns"] = name
                    running[str(call.get("id"))] = name
                events.append(Event(activity="spawn" if name in subagents else name,
                                    timestamp=step, resource=source,
                                    attributes=attrs))
            step += 1
            continue

        if mtype == "ToolCallExecutionEvent":
            for item in items:             # a sub-agent is done when its result is back
                running.pop(str(item.get("call_id")), None)
            continue                       # already attached to its call

        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if source == "user" and running:
            continue                       # a sub-agent's task, already on its spawn
        if mtype in ("SelectSpeakerEvent", "SelectorEvent"):
            target = _selected_speaker(text)
            step += 1
            events.append(Event(activity="delegate", timestamp=step - 1,
                                resource=source,
                                attributes=attach({**base, "to": target,
                                                   "content": _truncate(text)})))
            continue

        activity = "orchestrate" if _ORCHESTRATOR.search(source) else "report"
        step += 1
        events.append(Event(activity=activity, timestamp=step - 1, resource=source,
                            attributes=attach({**base, "content": _truncate(text)})))

    return events, thoughts, list(pending.values())


def ingest(path: str) -> list[Trace]:
    """Read one recorded local run into a canonical trace."""
    with open(path, encoding="utf-8") as f:
        run = json.load(f)

    events, thoughts, unattached = _events(run.get("messages") or [], run.get("subagents"))
    outcome = {
        "source": "local_magentic",
        "model": run.get("model"),
        "shape": run.get("shape"),
        "prompt": _truncate(run.get("prompt")),
        "seconds": run.get("seconds"),
        "harness_error": run.get("error"),
        "agents": sorted({e.resource for e in events}),
        "reasoning_turns": len(thoughts),
    }
    if run.get("task_type"):
        outcome["task_type"] = run["task_type"]
    if run.get("subagents"):
        outcome["subagents"] = {k: v.get("parent") for k, v in sorted(run["subagents"].items())}
    if unattached:
        # thoughts with no later event from the same agent attach to the trace
        outcome["unattached_reasoning"] = [_truncate(t) for t in unattached]
    return [Trace(
        case_id=str(run.get("task_id") or os.path.splitext(os.path.basename(path))[0]),
        events=events,
        outcome=outcome,
    )]


def ingest_dir(path: str) -> list[Trace]:
    """Read every recorded run in a directory."""
    traces: list[Trace] = []
    for name in sorted(os.listdir(path)):
        if name.endswith(".json"):
            traces.extend(ingest(os.path.join(path, name)))
    return traces
