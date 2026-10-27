"""Ingest a Claude Code session transcript (pipeline layer L1).

Input format
------------
Claude Code writes one JSONL transcript per session to
``~/.claude/projects/<project-dir>/<session-id>.jsonl``. Each line is a JSON
object. The shapes below were observed on transcripts produced by Claude Code
v2.x (inspected 2026-07-15); this is an internal file whose format is not a
published interface and may change between versions, so it is read here as a
read-only source for offline analysis.

  assistant line  {"type": "assistant", "timestamp": <ISO 8601>, "uuid": ...,
                   "sessionId": ..., "isSidechain": <bool>, "cwd": ..., "gitBranch": ...,
                   "message": {"model": ..., "content": [ <block>, ... ]}}
  tool_use block  {"type": "tool_use", "id": "toolu_...", "name": "Bash",
                   "input": {...}, "caller": {"type": "direct"}}
  result line     {"type": "user", "timestamp": <ISO 8601>,
                   "sourceToolAssistantUUID": ...,
                   "message": {"content": [{"type": "tool_result",
                               "tool_use_id": "toolu_...", "is_error": <bool>,
                               "content": ...}]}}

Mapping to the canonical model (L2)
-----------------------------------
Only executed tool calls become activities (thesis requirement R2). Assistant
prose and reasoning never become activities. A ``thinking`` block with text is
kept as the ``reasoning`` attribute of the next tool call from the same session
and side of the sidechain; most transcripts store thinking without its text, and
those blocks are skipped.

  case       the sessionId (one session = one run)
  activity   the tool name (Bash, Read, Write, Edit, Agent, ...)
  timestamp  the assistant line timestamp, parsed to epoch milliseconds
  resource   "subagent" when isSidechain is true, otherwise the model name
  attributes tool_use_id, truncated input, is_error, and duration_ms when the
             matching tool_result is present

Tool inputs can contain shell commands, file paths, and whole file contents, so
values are truncated on ingestion and the resulting log should be treated as
sensitive. Redact further before sharing.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from ..model import Event, Trace

_MAX_ATTR_CHARS = 200
_MAX_REASONING_CHARS = 4000


def _iso_to_epoch_ms(iso: Optional[str]) -> Optional[int]:
    if not isinstance(iso, str):
        return None
    try:
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _truncate(v: Any, limit: int = _MAX_ATTR_CHARS) -> Any:
    """Keep attributes small and reduce incidental payload exposure."""
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return s if len(s) <= limit else s[:limit] + f"... [{len(s)} chars]"


def _iter_lines(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # transcripts may hold partially written lines


def parse_transcript(path: str) -> list[Trace]:
    """Parse one Claude Code transcript into canonical traces (one per session)."""
    calls: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    thinking: dict[tuple[str, bool], list[str]] = {}   # reasoning awaiting a call

    for o in _iter_lines(path):
        kind = o.get("type")
        if kind == "assistant":
            msg = o.get("message") or {}
            key = (o.get("sessionId", "unknown"), bool(o.get("isSidechain")))
            for block in msg.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "thinking":
                    text = block.get("thinking")
                    if isinstance(text, str) and text.strip():
                        thinking.setdefault(key, []).append(text)
                elif block.get("type") == "tool_use":
                    pending = thinking.pop(key, None)
                    calls.append(
                        {
                            "session": key[0],
                            "ts": o.get("timestamp"),
                            "tool_use_id": block.get("id"),
                            "name": block.get("name", "unknown"),
                            "input": block.get("input"),
                            "sidechain": key[1],
                            "model": msg.get("model", "unknown"),
                            "reasoning": "\n\n".join(pending) if pending else None,
                        }
                    )
        elif kind == "user":
            for block in (o.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tid = block.get("tool_use_id")
                    if tid:
                        results[tid] = {"ts": o.get("timestamp"), "is_error": bool(block.get("is_error"))}

    by_session: dict[str, list[Event]] = {}
    last_ms: dict[str, int] = {}
    for c in calls:
        session = c["session"]
        ms = _iso_to_epoch_ms(c["ts"])
        if ms is None:  # keep file order when a timestamp is missing or unparseable
            ms = last_ms.get(session, -1) + 1
        last_ms[session] = ms

        attrs: dict[str, Any] = {"tool_use_id": c["tool_use_id"], "model": c["model"]}
        if c["input"] is not None:
            attrs["input"] = _truncate(c["input"])
        if c["reasoning"]:
            attrs["reasoning"] = _truncate(c["reasoning"], _MAX_REASONING_CHARS)
        res = results.get(c["tool_use_id"])
        if res:
            attrs["is_error"] = res["is_error"]
            end_ms = _iso_to_epoch_ms(res["ts"])
            if end_ms is not None:
                attrs["duration_ms"] = max(0, end_ms - ms)

        by_session.setdefault(session, []).append(
            Event(
                activity=c["name"],
                timestamp=ms,
                resource="subagent" if c["sidechain"] else c["model"],
                attributes=attrs,
            )
        )

    return [
        Trace(case_id=s, events=evts, outcome={"source": "claude_code_transcript"})
        for s, evts in sorted(by_session.items())
    ]


class ClaudeCodeIngester:
    """L1 ingester for Claude Code session transcripts."""

    def ingest(self, path: str) -> list[Trace]:
        return parse_transcript(path)
