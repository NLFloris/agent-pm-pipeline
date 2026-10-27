"""Ingest AgentHallu multi-agent trajectories (pipeline layer L1).

Input format
------------
One JSON file per run, from github.com/liuxuannan/AgentHallu under
``AgentHallu/<framework>/NNN.json``. Verified first-hand against the
``Magentic_One`` (94 runs) and ``Camel`` (93 runs) subsets, accessed 2026-09-22:

  run    {"model_id", "agent_type", "question", "true_answer", "history": [...],
          "agent_answer", "is_hallucination", "hallucination_step",
          "hallucination_category", "hallucination_subcategory",
          "hallucination_reason"}
  step   {"step", "role", "content"} and, when the step executed something,
          {"tool_calls": [...], "tool_responses": [...]}

``role`` names the *acting agent*, which is what makes these subsets genuinely
multi-agent: Magentic-One has MagenticOneOrchestrator, WebSurfer, Coder and
ComputerTerminal; Camel has Coordinator, Planner, Search Agent, Developer Agent
and Multi-Modal Processing Agent.

Delegation and reporting back
-----------------------------
Magentic-One's orchestrator delegates in the step text, as
``Call Next Speaker: <Agent> with arguments: (...)``. The delegated agent's next
step is its answer to that instruction. Both are executed control actions, not
reasoning, so both become activities, the same way the OpenAI Agents SDK
ingester treats a handoff span as an activity:

  delegate   an orchestrator step that names a next speaker. resource is the
             orchestrator, attribute ``to`` is the agent being handed to.
  report     a step by a non-orchestrating agent that carries text but executed
             no tool. resource is that agent, attribute ``to`` is the agent it
             reports back to, when a delegation is open.
  <tool>     a tool call, one event per call, resource is the calling agent.

Frameworks without an explicit delegation phrase (Camel) yield no ``delegate``
events; the handover is then visible only in the change of resource between
consecutive events, which is what a handover-of-work analysis reads anyway.

Mapping to the canonical model (L2)
-----------------------------------
  case       "<framework>:<file stem>", since one file is one run
  timestamp  the step index; this source carries no wall-clock time
  resource   the acting agent's role
  outcome    the shipped human annotation (is_hallucination, the responsible
             step, category and written reason) plus the question and answers.
             This is a label about the *process*, localised to a step, which is
             rare among public agent datasets, and it is still a claim to
             corroborate rather than ground truth for conformance.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from ..model import Event, Trace

_NEXT_SPEAKER = re.compile(r"Call Next Speaker:\s*([A-Za-z][A-Za-z0-9_ ]*?)(?:\s+with\s+arguments|[:\(]|$)")
_ORCHESTRATORS = ("orchestrator", "coordinator", "planner")
_MAX_ATTR_CHARS = 300


def _truncate(v: Any) -> Any:
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return s if len(s) <= _MAX_ATTR_CHARS else s[:_MAX_ATTR_CHARS] + f"... [{len(s)} chars]"


def _is_orchestrator(role: Optional[str]) -> bool:
    return bool(role) and any(k in role.lower() for k in _ORCHESTRATORS)


def _delegation_target(content: Any) -> Optional[str]:
    if not isinstance(content, str):
        return None
    m = _NEXT_SPEAKER.search(content)
    return m.group(1).strip() if m else None


def _tool_name(call: Any) -> str:
    if isinstance(call, dict):
        for key in ("name", "tool_name", "function"):
            v = call.get(key)
            if isinstance(v, str):
                return v
            if isinstance(v, dict) and isinstance(v.get("name"), str):
                return v["name"]
    if isinstance(call, str):
        # some frameworks record the call as free text "Tool(name=..., args=...)"
        m = re.match(r"\s*([A-Za-z_][\w.]*)", call)
        if m:
            return m.group(1)
    return "tool_call"


def _events(history: list[Any]) -> list[Event]:
    events: list[Event] = []
    seq = 0
    pending_from: Optional[str] = None   # who is waiting for a report back
    for step in history:
        if not isinstance(step, dict):
            continue
        role = step.get("role") or "unknown"
        content = step.get("content")
        calls = step.get("tool_calls") or []
        responses = step.get("tool_responses") or []
        if not isinstance(calls, list):
            calls = [calls]

        target = _delegation_target(content) if _is_orchestrator(role) else None
        if target:
            events.append(Event(
                activity="delegate", timestamp=seq, resource=str(role),
                attributes={"to": target, "instruction": _truncate(content),
                            "step": step.get("step")},
            ))
            seq += 1
            pending_from = str(role)
            continue

        for i, call in enumerate(calls):
            response = responses[i] if isinstance(responses, list) and i < len(responses) else None
            events.append(Event(
                activity=_tool_name(call), timestamp=seq, resource=str(role),
                attributes={"arguments": _truncate(call), "result": _truncate(response),
                            "step": step.get("step")},
            ))
            seq += 1

        if not calls and isinstance(content, str) and content.strip():
            if _is_orchestrator(role):
                activity, to = "orchestrate", None
            else:
                activity, to = "report", pending_from
                pending_from = None
            events.append(Event(
                activity=activity, timestamp=seq, resource=str(role),
                attributes={"to": to, "content": _truncate(content), "step": step.get("step")},
            ))
            seq += 1
    return events


def ingest(path: str) -> list[Trace]:
    """Read one AgentHallu run file into a single canonical trace."""
    with open(path, encoding="utf-8") as f:
        run = json.load(f)

    framework = run.get("agent_type") or os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    halluc = str(run.get("is_hallucination", "")).strip().lower() in ("true", "yes", "1")

    return [Trace(
        case_id=f"{framework}:{stem}",
        events=_events(run.get("history") or []),
        outcome={
            "source": "agenthallu",
            "framework": framework,
            "model_id": run.get("model_id"),
            "question": _truncate(run.get("question")),
            "true_answer": _truncate(run.get("true_answer")),
            "agent_answer": _truncate(run.get("agent_answer")),
            "is_hallucination": halluc,
            "hallucination_step": run.get("hallucination_step"),
            "hallucination_category": run.get("hallucination_category"),
            "hallucination_subcategory": run.get("hallucination_subcategory"),
            "hallucination_reason": _truncate(run.get("hallucination_reason")),
        },
    )]


def ingest_dir(path: str) -> list[Trace]:
    """Read every run file in a framework directory."""
    traces: list[Trace] = []
    for name in sorted(os.listdir(path)):
        if name.endswith(".json"):
            traces.extend(ingest(os.path.join(path, name)))
    return traces
