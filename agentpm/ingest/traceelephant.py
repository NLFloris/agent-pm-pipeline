"""Ingest TraceElephant failure-attribution traces (pipeline layer L1).

Input format
------------
One directory per run, from huggingface.co/datasets/TraceElephant/TraceElephant
(CC BY 4.0; Chen et al., ACL 2026, arXiv 2604.22708). Verified first-hand over
all 220 released runs, accessed 2026-09-23:

  <subset>/<run_id>/trace_metadata.json
      {"task_id", "task_instruction", "system_name", "agent_configuration",
       "run_id", "ground_truth", "mistake_agent", "mistake_step",
       "mistake_reason", "agent_system_intro"}
  <subset>/<run_id>/step_records.json
      [{"step_id", "agent_id", "agent_name",
        "input": {"messages": [...], "model", "temperature", ...},
        "output": <an OpenAI chat-completion envelope>,
        "tool_logs": [{"id", "type", "function": {"name", "arguments"}}]}]

``output`` is the provider's raw response, not the text: the agent's words are
at ``output.choices[0].message.content`` and the model and token counts sit
beside them. Verified over all 5,960 steps: 3,701 carry only content, 1,074
only tool calls, 930 both and 255 neither.

Subsets are ``captain-runs-gaia``, ``captain-runs-assistantbench``,
``magentic-runs-gaia``, ``magentic-runs-assistant-bench`` and
``swe-agent-runs-swe-bench``. Only the 220 *failed* runs of the paper's 380
executions are released, so a corpus built from this source has no negative
class: every trace is known to contain a mistake.

Mapping to the canonical model (L2)
-----------------------------------
  activity   each entry of ``tool_logs`` becomes one event under its function
             name. A step that executed no tool becomes a single communication
             event: ``orchestrate`` for an orchestrating agent, ``report``
             otherwise. The prompt a step received is an attribute, never an
             activity (R2).
  timestamp  the step's position in the run; this source carries no clock.
  resource   ``agent_name``. For the SWE-Agent subset this field names the tool
             rather than an agent, because that system is single-agent; callers
             that care about the distinction should check ``system_name``.
  case       the run directory name.
  outcome    the expert attribution (``mistake_agent``, ``mistake_step``,
             ``mistake_reason``) plus the task and its gold answer.

``mistake_step`` indexes the source ``step_id``, which one step may expand into
several events. ``decisive_event_index`` on the outcome resolves it to the first
event of that step, so a localisation claim can be scored directly against the
canonical trace.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Iterator, Optional

from ..model import Event, Trace

_ORCHESTRATOR = re.compile(r"orchestrat|coordinat|planner", re.I)
_MAX_ATTR_CHARS = 4000


def _truncate(v: Any) -> Any:
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return s if len(s) <= _MAX_ATTR_CHARS else s[:_MAX_ATTR_CHARS] + f"... [{len(s)} chars]"


def _completion(step: dict[str, Any]) -> dict[str, Any]:
    """Unwrap the chat-completion envelope into the parts worth keeping."""
    out = step.get("output")
    if not isinstance(out, dict):
        return {"content": _truncate(out)}
    choices = out.get("choices")
    message = {}
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
    usage = out.get("usage") or {}
    return {
        "content": _truncate(message.get("content")),
        "model": out.get("model"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "finish_reason": (choices[0].get("finish_reason")
                          if isinstance(choices, list) and choices else None),
    }


def _last_prompt(step: dict[str, Any]) -> Optional[str]:
    """The final message handed to the agent, which is the step's real input."""
    inp = step.get("input")
    if not isinstance(inp, dict):
        return _truncate(inp)
    messages = inp.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict):
            return _truncate(last.get("content"))
    return _truncate(messages)


def _tool_calls(step: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    for call in step.get("tool_logs") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        yield str(fn.get("name") or "tool_call"), fn.get("arguments")


def _events(records: list[Any]) -> tuple[list[Event], dict[int, int]]:
    """Canonical events plus a map from source step_id to first event index."""
    events: list[Event] = []
    first_of_step: dict[int, int] = {}
    for step in records:
        if not isinstance(step, dict):
            continue
        agent = str(step.get("agent_name") or "unknown")
        step_id = step.get("step_id")
        if isinstance(step_id, int):
            first_of_step.setdefault(step_id, len(events))
        said = _completion(step)
        base = {"step_id": step_id, "agent_id": step.get("agent_id"),
                "prompt": _last_prompt(step), "model": said.get("model"),
                "completion_tokens": said.get("completion_tokens"),
                "prompt_tokens": said.get("prompt_tokens"),
                "finish_reason": said.get("finish_reason")}

        calls = list(_tool_calls(step))
        for i, (name, arguments) in enumerate(calls):
            # 930 steps carry text alongside their tool calls; keep it on the
            # first call rather than inventing an extra event for it
            events.append(Event(
                activity=name, timestamp=len(events), resource=agent,
                attributes=dict(base, arguments=_truncate(arguments),
                                content=said.get("content") if i == 0 else None),
            ))
        if not calls:
            events.append(Event(
                activity="orchestrate" if _ORCHESTRATOR.search(agent) else "report",
                timestamp=len(events), resource=agent,
                attributes=dict(base, content=said.get("content")),
            ))
    return events, first_of_step


def ingest(path: str) -> list[Trace]:
    """Read one TraceElephant run directory into a single canonical trace."""
    meta_path = os.path.join(path, "trace_metadata.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(path, "step_records.json"), encoding="utf-8") as f:
        records = json.load(f)

    events, first_of_step = _events(records)
    try:
        mistake_step = int(meta.get("mistake_step"))
    except (TypeError, ValueError):
        mistake_step = None

    return [Trace(
        case_id=meta.get("run_id") or os.path.basename(path.rstrip("/")),
        events=events,
        outcome={
            "source": "traceelephant",
            "system": meta.get("system_name"),
            "task_id": meta.get("task_id"),
            "task": _truncate(meta.get("task_instruction")),
            "ground_truth": _truncate(meta.get("ground_truth")),
            "agents": sorted({e.resource for e in events}),
            # expert attribution: a claim about the process, localized to a step
            "mistake_agent": meta.get("mistake_agent"),
            "mistake_step": mistake_step,
            "mistake_reason": _truncate(meta.get("mistake_reason")),
            "decisive_event_index": first_of_step.get(mistake_step),
        },
    )]


def ingest_dir(path: str) -> list[Trace]:
    """Read every run directory under a subset directory."""
    traces: list[Trace] = []
    for name in sorted(os.listdir(path)):
        run = os.path.join(path, name)
        if os.path.isfile(os.path.join(run, "trace_metadata.json")):
            traces.extend(ingest(run))
    return traces
