"""Ingest OpenTelemetry telemetry into canonical traces (pipeline layer L1).

Two dialects are read from the same file. The file is what
``experiments/otel_capture/collector.py`` writes: one JSON object per line, each
holding one OTLP/JSON payload under ``payload`` with its ``signal``
(``traces`` or ``logs``). A bare OTLP/JSON body (``resourceSpans`` or
``resourceLogs`` at the top level) is accepted too.

1. OpenTelemetry GenAI semantic conventions (spans)
---------------------------------------------------
Checked against github.com/open-telemetry/semantic-conventions-genai, main
branch as of 2026-10-14, status **Development** (not stable), which pins the
general conventions at v1.44.0. The parts this ingester relies on:

  gen_ai.operation.name   Required on every GenAI span. Well-known values used
                          here: execute_tool, invoke_agent, invoke_workflow,
                          chat, generate_content, text_completion, plan.
  execute_tool spans      name "execute_tool {gen_ai.tool.name}";
                          gen_ai.tool.name Required, gen_ai.tool.call.id
                          Recommended, gen_ai.tool.call.arguments and
                          gen_ai.tool.call.result Opt-In.
  invoke_agent spans      name "invoke_agent {gen_ai.agent.name}";
                          gen_ai.agent.name / gen_ai.agent.id when available,
                          gen_ai.conversation.id to correlate a session.
  invoke_workflow spans   an orchestrator coordinating several agents.
  nesting                 tool spans are children of the invoke_agent span
                          that issued them, and a delegated agent is a nested
                          invoke_agent span. Parent links therefore give full
                          attribution, and real start/end times give real
                          concurrency.

Tool calls are spans, not events: the events document defines only
gen_ai.evaluation.result. Anything tool-shaped arriving as a log record is a
vendor dialect, which is the second case.

Mapping:
  case       the root span of each trace, or gen_ai.conversation.id when set
  activity   execute_tool -> gen_ai.tool.name; invoke_agent -> "invoke_agent";
             invoke_workflow -> "invoke_workflow"; chat and the other inference
             operations -> "llm_call"; plan -> "plan"
  resource   gen_ai.agent.name of the nearest enclosing invoke_agent span,
             which for a sub-agent's tools is the sub-agent (R6)
  timestamp  the step: the position of the span's start in the trace, except
             that a tool span starting before the previous tool span under the
             same parent has ended shares its step, since the agent issued them
             together. Real start and end are kept as started_ns / ended_ns.
  sequence   the serialised position, a tiebreak for flat views
  attributes call id, arguments, result, agent id, parent agent, duration

2. Claude Code's OpenTelemetry dialect (log records)
----------------------------------------------------
Verified on a real capture, tests/fixtures/claude_code_otel_subagents.jsonl,
taken 2026-10-12 from Claude Code 2.1.278. It emits logs and metrics, no spans.
Each record has event.name, session.id, event.timestamp, and a monotonic
event.sequence, which orders events that share a timestamp. Events used:

  tool_result          tool_name, tool_use_id, success, duration_ms, sizes
  subagent_completed   agent_type, duration_ms, total_tool_uses, total_tokens
  api_request          model, tokens, cost_usd, duration_ms
  user_prompt          prompt.id, prompt_length (text is redacted)

Mapping:
  case       session.id
  activity   tool_result -> tool_name; an "Agent" tool result is a sub-agent
             returning and becomes "subagent_return"; subagent_completed ->
             "subagent_completed"; api_request -> "llm_call"; user_prompt ->
             "user_prompt"
  resource   "main". This dialect carries no agent attribution on tool events,
             so a sub-agent's tool calls cannot be assigned to it. That is a
             documented limitation of the source, not of the mapping, and it
             is why the semantic-convention path above exists.
  timestamp  event.sequence
"""
from __future__ import annotations

import json
from typing import Any, Iterator, Optional

from ..model import Event, Trace

_INFERENCE_OPS = {"chat", "generate_content", "text_completion", "embeddings",
                  "generate_image", "transcribe", "translate"}
_MAX_ATTR_CHARS = 4000


def _truncate(v: Any) -> Any:
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return s if len(s) <= _MAX_ATTR_CHARS else s[:_MAX_ATTR_CHARS] + f"... [{len(s)} chars]"


def _attr_value(v: Any) -> Any:
    """Collapse an OTLP AnyValue to a plain Python value."""
    if not isinstance(v, dict):
        return v
    for k in ("stringValue", "boolValue", "doubleValue"):
        if k in v:
            return v[k]
    if "intValue" in v:
        try:
            return int(v["intValue"])
        except (TypeError, ValueError):
            return v["intValue"]
    if "arrayValue" in v:
        return [_attr_value(x) for x in (v["arrayValue"] or {}).get("values", [])]
    if "kvlistValue" in v:
        return {kv.get("key"): _attr_value(kv.get("value"))
                for kv in (v["kvlistValue"] or {}).get("values", [])}
    return v


def _attrs(node: dict[str, Any]) -> dict[str, Any]:
    return {a.get("key"): _attr_value(a.get("value")) for a in node.get("attributes") or []
            if isinstance(a, dict) and a.get("key")}


def _payloads(path: str) -> Iterator[dict[str, Any]]:
    """Yield OTLP bodies from a collector JSONL or a bare OTLP JSON file."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    stripped = text.strip()
    if stripped.startswith("{") and "\n" not in stripped:
        obj = json.loads(stripped)
        yield obj.get("payload", obj)
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        yield obj.get("payload", obj) if isinstance(obj, dict) else {}


# --------------------------------------------------------- semantic conventions

def _spans(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in payloads:
        for rs in p.get("resourceSpans") or []:
            for ss in rs.get("scopeSpans") or []:
                for sp in ss.get("spans") or []:
                    a = _attrs(sp)
                    out.append({
                        "span_id": sp.get("spanId"), "parent": sp.get("parentSpanId") or None,
                        "trace_id": sp.get("traceId"), "name": sp.get("name") or "",
                        "start": int(sp.get("startTimeUnixNano") or 0),
                        "end": int(sp.get("endTimeUnixNano") or 0),
                        "status": (sp.get("status") or {}).get("code"),
                        "attrs": a,
                    })
    return out


def _agent_of(span: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> Optional[str]:
    """Name of the nearest enclosing invoke_agent span, this span included."""
    cur: Optional[dict[str, Any]] = span
    while cur is not None:
        if cur["attrs"].get("gen_ai.operation.name") == "invoke_agent":
            return str(cur["attrs"].get("gen_ai.agent.name") or cur["attrs"].get("gen_ai.agent.id")
                       or cur["name"].removeprefix("invoke_agent").strip() or "agent")
        cur = by_id.get(cur["parent"]) if cur["parent"] else None
    return None


def _semconv_traces(spans: list[dict[str, Any]]) -> list[Trace]:
    by_id = {s["span_id"]: s for s in spans if s["span_id"]}
    # one trace id is one run; gen_ai.conversation.id, when any span in the run
    # carries it, names the case so that resumed sessions share one case
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for s in spans:
        by_trace.setdefault(str(s["trace_id"] or "trace"), []).append(s)
    groups: dict[str, list[dict[str, Any]]] = {}
    for tid, group in by_trace.items():
        conv = next((s["attrs"].get("gen_ai.conversation.id") for s in group
                     if s["attrs"].get("gen_ai.conversation.id")), None)
        groups.setdefault(str(conv or tid), []).extend(group)

    traces: list[Trace] = []
    for case, group in sorted(groups.items()):
        group.sort(key=lambda s: (s["start"], s["name"]))
        events: list[Event] = []
        # a step is one logical time; a span that starts before the previous event
        # span under the same parent has ended was issued together with it
        step = -1
        prev_parent: Optional[str] = None
        prev_end = 0
        for s in group:
            op = s["attrs"].get("gen_ai.operation.name")
            if not op:
                continue
            together = (s["parent"] is not None and s["parent"] == prev_parent
                        and s["start"] < prev_end and op == "execute_tool")
            if not together:
                step += 1
            prev_parent, prev_end = s["parent"], max(prev_end if together else 0, s["end"])
            # a span with no enclosing invoke_agent runs in the root context, the
            # caller of the first agent, which is also the parent of top-level
            # invoke_agent spans
            agent = _agent_of(s, by_id) or "root"
            parent = by_id.get(s["parent"]) if s["parent"] else None
            parent_agent = _agent_of(parent, by_id) if parent else None
            base = {
                "span_id": s["span_id"], "parent_span_id": s["parent"],
                "started_ns": s["start"], "ended_ns": s["end"],
                "duration_ms": (s["end"] - s["start"]) / 1e6 if s["end"] and s["start"] else None,
                "error": s["status"] == 2 or s["status"] == "STATUS_CODE_ERROR",
            }
            if op == "execute_tool":
                events.append(Event(
                    activity=str(s["attrs"].get("gen_ai.tool.name") or s["name"]),
                    timestamp=step, sequence=len(events), resource=agent,
                    attributes={**base,
                                "call_id": s["attrs"].get("gen_ai.tool.call.id"),
                                "arguments": _truncate(s["attrs"].get("gen_ai.tool.call.arguments")),
                                "result": _truncate(s["attrs"].get("gen_ai.tool.call.result")),
                                "tool_type": s["attrs"].get("gen_ai.tool.type")}))
            elif op == "invoke_agent":
                # the delegation itself is performed by the enclosing agent;
                # the invoked agent's own work follows as its child spans
                events.append(Event(
                    activity="invoke_agent", timestamp=step, sequence=len(events),
                    resource=parent_agent or "root",
                    attributes={**base, "to": agent,
                                "agent_id": s["attrs"].get("gen_ai.agent.id"),
                                "agent_description": s["attrs"].get("gen_ai.agent.description")}))
            elif op == "invoke_workflow":
                events.append(Event(activity="invoke_workflow", timestamp=step, sequence=len(events),
                                    resource=agent,
                                    attributes={**base, "workflow": s["attrs"].get("gen_ai.workflow.name")}))
            elif op in _INFERENCE_OPS:
                events.append(Event(
                    activity="llm_call", timestamp=step, sequence=len(events), resource=agent,
                    attributes={**base, "operation": op,
                                "model": s["attrs"].get("gen_ai.request.model"),
                                "input_tokens": s["attrs"].get("gen_ai.usage.input_tokens"),
                                "output_tokens": s["attrs"].get("gen_ai.usage.output_tokens")}))
            else:
                events.append(Event(activity=str(op), timestamp=step, sequence=len(events), resource=agent,
                                    attributes=base))
        agents = sorted({e.resource for e in events} | {e.attributes.get("to") for e in events
                                                         if e.attributes.get("to")})
        traces.append(Trace(case_id=case, events=events,
                            outcome={"source": "otel_semconv", "agents": agents,
                                     "spans": len(group)}))
    return traces


# ---------------------------------------------------------- claude code dialect

def _log_records(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in payloads:
        for rl in p.get("resourceLogs") or []:
            for sl in rl.get("scopeLogs") or []:
                for lr in sl.get("logRecords") or []:
                    out.append(_attrs(lr))
    return out


def _claude_traces(records: list[dict[str, Any]]) -> list[Trace]:
    by_session: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r.get("event.name"):
            by_session.setdefault(str(r.get("session.id") or "session"), []).append(r)

    traces: list[Trace] = []
    for sid, recs in sorted(by_session.items()):
        recs.sort(key=lambda r: int(r.get("event.sequence") or 0))
        events: list[Event] = []
        spawns: list[str] = []
        for r in recs:
            name = r.get("event.name")
            base = {"sequence": r.get("event.sequence"), "timestamp_iso": r.get("event.timestamp"),
                    "prompt_id": r.get("prompt.id")}
            if name == "tool_result":
                tool = str(r.get("tool_name") or "tool")
                attrs = {**base, "tool_use_id": r.get("tool_use_id"),
                         "error": str(r.get("success")).lower() == "false",
                         "duration_ms": r.get("duration_ms"),
                         "input_bytes": r.get("tool_input_size_bytes"),
                         "result_bytes": r.get("tool_result_size_bytes")}
                if tool == "Agent":
                    events.append(Event("subagent_return", len(events), "main", attrs))
                else:
                    events.append(Event(tool, len(events), "main", attrs))
            elif name == "tool_decision" and r.get("tool_name") == "Agent":
                spawns.append(str(r.get("tool_use_id")))
                events.append(Event("subagent_spawn", len(events), "main",
                                    {**base, "tool_use_id": r.get("tool_use_id")}))
            elif name == "subagent_completed":
                events.append(Event("subagent_completed", len(events), "main",
                                    {**base, "agent_type": r.get("agent_type"),
                                     "duration_ms": r.get("duration_ms"),
                                     "total_tool_uses": r.get("total_tool_uses"),
                                     "total_tokens": r.get("total_tokens"),
                                     "model": r.get("model")}))
            elif name == "api_request":
                events.append(Event("llm_call", len(events), "main",
                                    {**base, "model": r.get("model"),
                                     "input_tokens": r.get("input_tokens"),
                                     "output_tokens": r.get("output_tokens"),
                                     "cost_usd": r.get("cost_usd"),
                                     "duration_ms": r.get("duration_ms")}))
            elif name == "user_prompt":
                events.append(Event("user_prompt", len(events), "user",
                                    {**base, "prompt_length": r.get("prompt_length")}))
        traces.append(Trace(case_id=sid, events=events,
                            outcome={"source": "otel_claude_code", "subagents_spawned": len(spawns),
                                     "attribution": "none; this dialect does not name the "
                                                    "agent on tool events"}))
    return traces


# -------------------------------------------------------------------- entry

def ingest(path: str) -> list[Trace]:
    """Read an OpenTelemetry capture; spans per the GenAI conventions, else Claude Code logs."""
    payloads = list(_payloads(path))
    spans = _spans(payloads)
    if any(s["attrs"].get("gen_ai.operation.name") for s in spans):
        return _semconv_traces(spans)
    records = _log_records(payloads)
    if records:
        return _claude_traces(records)
    raise ValueError(f"{path}: no GenAI spans and no log records found")


class OtelIngester:
    def ingest(self, path: str) -> list[Trace]:
        return ingest(path)
