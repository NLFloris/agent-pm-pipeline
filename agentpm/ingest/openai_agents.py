"""Ingest raw OpenAI Agents SDK telemetry (pipeline layer L1).

Input format
------------
The OpenAI Agents SDK traces every agent run as one *trace* with *spans*. A
custom trace processor (see ``examples/openai_capture.py``) writes the SDK's own
export dicts to a JSONL file, one object per line. The shapes below were
verified against the SDK source (``agents/tracing/traces.py``, ``spans.py``,
``span_data.py``, github.com/openai/openai-agents-python, accessed 2026-07-08):

  trace line   {"object": "trace", "id": "trace_...", "workflow_name": ...,
                "group_id": ..., "metadata": ...}
  span line    {"object": "trace.span", "id": "span_...", "trace_id": ...,
                "parent_id": ..., "started_at": <ISO 8601>, "ended_at": <ISO 8601>,
                "span_data": {...}, "error": ...}

Relevant ``span_data`` payloads (field ``type`` discriminates):
  agent        {"type": "agent", "name", "handoffs", "tools", "output_type"}
  function     {"type": "function", "name", "input", "output", "mcp_data"}
  generation   {"type": "generation", "input", "output", "model", "model_config", "usage"}
  response     {"type": "response", "response_id", "usage"}
  handoff      {"type": "handoff", "from_agent", "to_agent"}
  guardrail    {"type": "guardrail", "name", "triggered"}

Mapping to the canonical model (L2)
-----------------------------------
Only executed calls become activities (thesis requirement R2); agent spans are
containers, not activities, and free-text payloads stay in event attributes.
Other SDK span types (for example custom or MCP tool-listing spans) may occur in
the raw input and are ignored by this mapping.

  activity   function span -> its tool name; generation/response -> "llm_call";
             handoff -> "handoff"; guardrail -> "guardrail"
  timestamp  started_at parsed to epoch milliseconds; a span without a parseable
             started_at gets the last parsed value plus one, so file order is
             preserved among its neighbours
  resource   name of the nearest ancestor agent span, else "unknown"
  case       the trace_id (one SDK trace = one run)

Function input/output and generation input/output (prompts and completions) are
carried verbatim as attributes; response spans expose only an identifier and
usage in the SDK export. Attributes may contain sensitive data; redact before
sharing logs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..model import Event, Trace


@dataclass
class SdkSpan:
    """One exported SDK span (subset of the envelope we need)."""
    id: str
    trace_id: str
    parent_id: Optional[str]
    started_at: Optional[str]
    ended_at: Optional[str]
    data: dict[str, Any]          # the span_data payload
    error: Any = None

    @property
    def type(self) -> str:
        return self.data.get("type", "unknown")


@dataclass
class AgentRun:
    """One SDK trace with its spans, as exported."""
    trace_id: str
    workflow_name: str = "unknown"
    group_id: Optional[str] = None
    metadata: Any = None
    spans: list[SdkSpan] = field(default_factory=list)


def _iso_to_epoch_ms(iso: Optional[str]) -> Optional[int]:
    if not iso:
        return None
    try:
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def parse_export_jsonl(path: str) -> list[AgentRun]:
    """Parse a JSONL file of SDK export dicts into AgentRun structures."""
    runs: dict[str, AgentRun] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = obj.get("object")
            if kind == "trace":
                run = runs.setdefault(obj["id"], AgentRun(trace_id=obj["id"]))
                run.workflow_name = obj.get("workflow_name") or run.workflow_name
                run.group_id = obj.get("group_id")
                run.metadata = obj.get("metadata")
            elif kind == "trace.span":
                run = runs.setdefault(obj["trace_id"], AgentRun(trace_id=obj["trace_id"]))
                run.spans.append(
                    SdkSpan(
                        id=obj["id"],
                        trace_id=obj["trace_id"],
                        parent_id=obj.get("parent_id"),
                        started_at=obj.get("started_at"),
                        ended_at=obj.get("ended_at"),
                        data=obj.get("span_data", {}),
                        error=obj.get("error"),
                    )
                )
    return list(runs.values())


def owning_agent(span: SdkSpan, by_id: dict[str, SdkSpan]) -> str:
    """Name of the nearest ancestor agent span, else 'unknown'."""
    cur: Optional[SdkSpan] = span
    seen: set[str] = set()
    while cur is not None and cur.id not in seen:
        seen.add(cur.id)
        if cur.type == "agent":
            return cur.data.get("name", "unknown")
        cur = by_id.get(cur.parent_id) if cur.parent_id else None
    return "unknown"


def _activity(span: SdkSpan) -> Optional[str]:
    t = span.type
    if t == "function":
        return span.data.get("name", "function")
    if t in ("generation", "response"):
        return "llm_call"
    if t in ("handoff", "guardrail"):
        return t
    return None  # agent/custom/other spans are not activities


def _event_attributes(span: SdkSpan) -> dict[str, Any]:
    attrs: dict[str, Any] = {"span_id": span.id, "span_type": span.type}
    if span.ended_at:
        attrs["ended_at"] = span.ended_at
    if span.error is not None:
        attrs["error"] = span.error
    d = span.data
    if span.type == "function":
        attrs.update({k: d[k] for k in ("input", "output", "mcp_data") if d.get(k) is not None})
    elif span.type == "generation":
        attrs.update({k: d[k] for k in ("input", "output", "model", "usage") if d.get(k) is not None})
    elif span.type == "response":
        attrs.update({k: d[k] for k in ("response_id", "usage") if d.get(k) is not None})
    elif span.type == "handoff":
        attrs.update({k: d[k] for k in ("from_agent", "to_agent") if d.get(k) is not None})
    elif span.type == "guardrail":
        attrs.update({k: d[k] for k in ("name", "triggered") if d.get(k) is not None})
    return attrs


def run_to_trace(run: AgentRun) -> Trace:
    """Flatten one AgentRun into a canonical Trace (activities = executed calls)."""
    by_id = {s.id: s for s in run.spans}
    events: list[Event] = []
    last_ms = -1  # monotonic fallback keeps file order for unparseable timestamps
    for span in run.spans:
        act = _activity(span)
        if act is None:
            continue
        ms = _iso_to_epoch_ms(span.started_at)
        if ms is None:
            ms = last_ms + 1
        last_ms = ms
        events.append(
            Event(
                activity=act,
                timestamp=ms,
                resource=owning_agent(span, by_id),
                attributes=_event_attributes(span),
            )
        )
    return Trace(case_id=run.trace_id, events=events,
                 outcome={"workflow_name": run.workflow_name})


class OpenAIAgentsIngester:
    """L1 ingester for OpenAI Agents SDK export JSONL."""

    def ingest(self, path: str) -> list[Trace]:
        return [run_to_trace(r) for r in parse_export_jsonl(path)]
