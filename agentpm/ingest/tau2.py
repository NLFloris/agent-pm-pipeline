"""Ingest tau2-bench trajectory dumps (pipeline layer L1).

Input format
------------
One JSON file per (model, domain) submission, downloaded from the public
tau2-bench results bucket ``sierra-tau-bench-public``, under
``submissions/<model>_<org>_<date>/trajectories/*.json``. The shapes below were
verified first-hand against
``claude-opus-4-5_sierra_2026-02-26/trajectories/claude-opus-4-5_high_airline_gpt-5.2_4trials.json``
(200 simulations, 50 tasks, 4 trials each, accessed 2026-09-22):

  file          {"timestamp", "info", "tasks": [...], "simulations": [...]}
  info          {"environment_info": {"domain_name", "policy", "tool_defs"},
                 "agent_info", "user_info", "num_trials", "seed", ...}
  task          {"id", "description", "user_scenario", "initial_state",
                 "evaluation_criteria": {"actions", "env_assertions",
                 "communicate_info", "nl_assertions", "reward_basis"}, ...}
  gold action   {"action_id", "requestor", "name", "arguments", "compare_args"}
  simulation    {"id" (UUID), "task_id", "trial", "seed", "start_time",
                 "end_time", "duration", "termination_reason", "agent_cost",
                 "user_cost", "messages": [...], "reward_info": {...}}
  message       {"role", "content", "tool_calls", "turn_idx", "timestamp",
                 "cost", "usage", "raw_data", ...}
  tool call     {"id", "name", "arguments", "requestor"}
  tool result   {"role": "tool", "id", "content", "error", "requestor",
                 "turn_idx", "timestamp"}

Older submissions may omit ``turn_idx`` or ``raw_data``, and voice submissions
use a different per-simulation layout; keys are probed defensively and a
simulation that cannot be parsed is skipped rather than crashing the batch.

Mapping to the canonical model (L2)
-----------------------------------
  activity   a tool call becomes its tool name; an assistant or user message
             that carries text becomes ``assistant_message`` / ``user_message``.
             Tool *results* do not become separate events: their content is
             attached to the call they answer, so one executed action is one
             event. Reasoning content (``reasoning_content`` and
             ``thinking_blocks`` on ``raw_data.choices[0].message``) is never
             an activity, only an attribute (R2), on the calls and the text
             turn of the message that carries it.
  timestamp  the index of the source message the event came from, so that the
             tool calls one model turn issues together share a timestamp and
             are not ordered with respect to each other. The source does carry
             real ISO timestamps, but calls issued in one turn share a
             byte-identical one, which is the same fact. The real value is kept
             in the ``timestamp_iso`` and ``epoch_us`` attributes so that
             performance analysis can still use it.
  sequence   the position of the event in the simulation, the order in which
             the harness executed the calls. A tiebreak for flat views only.
  resource   the ``requestor`` of a tool call (``assistant`` or ``user``), which
             is what makes tau2's dual-control domains multi-actor, else the
             message role.
  case       the simulation UUID (one simulation = one run = one case).
  outcome    ``reward_info`` verbatim plus task identity. This is the harness's
             claim about the run, to be corroborated against the recorded
             actions, never ground truth in itself.

Gold actions from ``evaluation_criteria.actions`` are carried on the trace
outcome under ``gold_actions`` so that a per-task reference model can be built
from them without re-reading the source file.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterator, Optional

from ..model import Event, Trace

# Roles whose free-text turns become communication activities.
_TEXT_ACTIVITY = {"assistant": "assistant_message", "user": "user_message"}


def _epoch_us(value: Any) -> Optional[int]:
    """Parse an ISO 8601 timestamp to epoch microseconds, or None."""
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1_000_000)
    except ValueError:
        return None


def _tool_results(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index tool-result messages by the id of the call they answer."""
    out: dict[str, dict[str, Any]] = {}
    for m in messages:
        if m.get("role") == "tool" and m.get("id") is not None:
            out[str(m["id"])] = m
    return out


def _reasoning(message: dict[str, Any]) -> Optional[str]:
    """Reasoning text carried alongside a message, if the submission has it."""
    raw = message.get("raw_data")
    if not isinstance(raw, dict):
        return None
    # submissions store the provider response, so the fields sit on
    # raw_data.choices[0].message; a flat raw_data is accepted as well
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        inner = choices[0].get("message")
        if isinstance(inner, dict):
            raw = inner
    if isinstance(raw.get("reasoning_content"), str) and raw["reasoning_content"].strip():
        return raw["reasoning_content"]
    blocks = raw.get("thinking_blocks")
    if isinstance(blocks, list):
        parts = [b.get("thinking") for b in blocks if isinstance(b, dict) and b.get("thinking")]
        if parts:
            return "\n".join(parts)
    return None


def _events(messages: list[dict[str, Any]]) -> Iterator[Event]:
    """Executed actions and communication turns, in file order."""
    results = _tool_results(messages)
    seq = 0
    turn = -1
    for m in messages:
        role = m.get("role")
        if role == "tool":
            continue  # attached to its call below, not an event of its own
        turn += 1
        reasoning = _reasoning(m)
        calls = m.get("tool_calls") or []
        for call in calls:
            result = results.get(str(call.get("id")))
            attributes: dict[str, Any] = {
                "arguments": call.get("arguments"),
                "tool_call_id": call.get("id"),
                "turn_idx": m.get("turn_idx"),
                "timestamp_iso": m.get("timestamp"),
                "epoch_us": _epoch_us(m.get("timestamp")),
            }
            if result is not None:
                attributes["result"] = result.get("content")
                attributes["error"] = bool(result.get("error"))
            if reasoning:
                attributes["reasoning"] = reasoning
            yield Event(
                activity=str(call.get("name")),
                timestamp=turn,
                resource=str(call.get("requestor") or role or "unknown"),
                attributes=attributes,
                sequence=seq,
            )
            seq += 1
        content = m.get("content")
        if isinstance(content, str) and content.strip() and role in _TEXT_ACTIVITY:
            attributes = {
                "content": content,
                "turn_idx": m.get("turn_idx"),
                "timestamp_iso": m.get("timestamp"),
                "epoch_us": _epoch_us(m.get("timestamp")),
                "cost": m.get("cost"),
            }
            if reasoning:
                attributes["reasoning"] = reasoning
            yield Event(
                activity=_TEXT_ACTIVITY[role],
                timestamp=turn,
                resource=str(role),
                attributes=attributes,
                sequence=seq,
            )
            seq += 1


def ingest(path: str) -> list[Trace]:
    """Read one tau2-bench trajectory file into canonical traces."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    env = payload.get("info", {}).get("environment_info") or payload.get("info", {}).get("environment") or {}
    domain = env.get("domain_name", "unknown")
    gold: dict[str, list[dict[str, Any]]] = {}
    for task in payload.get("tasks") or []:
        criteria = task.get("evaluation_criteria") or {}
        gold[str(task.get("id"))] = criteria.get("actions") or []

    traces: list[Trace] = []
    for sim in payload.get("simulations") or []:
        messages = sim.get("messages")
        if not isinstance(messages, list):
            continue  # voice or otherwise unexpected layout; skip, do not crash
        task_id = str(sim.get("task_id"))
        reward_info = sim.get("reward_info") or {}
        traces.append(
            Trace(
                case_id=str(sim.get("id")),
                events=list(_events(messages)),
                outcome={
                    "source": "tau2",
                    "domain": domain,
                    "task_id": task_id,
                    "trial": sim.get("trial"),
                    "seed": sim.get("seed"),
                    "reward": reward_info.get("reward"),
                    "reward_breakdown": reward_info.get("reward_breakdown"),
                    "db_check": reward_info.get("db_check"),
                    "action_checks": reward_info.get("action_checks"),
                    "nl_assertions": reward_info.get("nl_assertions"),
                    "communicate_checks": reward_info.get("communicate_checks"),
                    "termination_reason": sim.get("termination_reason"),
                    "duration": sim.get("duration"),
                    "agent_cost": sim.get("agent_cost"),
                    "user_cost": sim.get("user_cost"),
                    "gold_actions": gold.get(task_id, []),
                },
            )
        )
    return traces
