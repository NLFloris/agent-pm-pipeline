"""Capture raw OpenAI Agents SDK telemetry to a JSONL file.

Register this processor in any application built on the OpenAI Agents SDK
(``pip install openai-agents``, OPENAI_API_KEY set) and every trace and span the
SDK produces is appended to a JSONL file, using the SDK's own ``export()`` dicts.
That file is the raw-telemetry input of this pipeline (see
``agentpm/ingest/openai_agents.py`` for the exact shapes).

Usage in your agent application:

    from examples.openai_capture import JsonlExportProcessor
    from agents import add_trace_processor

    add_trace_processor(JsonlExportProcessor("runs.jsonl"))
    # ... run your agents as usual; then:
    # python3 -m agentpm.cli ocel runs.jsonl -o runs.ocel.json

The processor implements the SDK's TracingProcessor interface
(on_trace_start/end, on_span_start/end, shutdown, force_flush; verified against
``agents/tracing/processor_interface.py``, accessed 2026-07-08). It is additive,
so the SDK's default exporter keeps working. Captured payloads can contain
prompts, tool arguments, and outputs; treat the file as sensitive.
"""
from __future__ import annotations

import json
import threading


class JsonlExportProcessor:
    """Append every finished trace and span export dict to a JSONL file."""

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()

    def _write(self, payload) -> None:
        if payload is None:
            return
        with self._lock, open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    # TracingProcessor interface
    def on_trace_start(self, trace) -> None:
        self._write(trace.export())

    def on_trace_end(self, trace) -> None:
        pass  # already written at start; end adds nothing to the export dict

    def on_span_start(self, span) -> None:
        pass  # spans are written once finished, when timestamps are complete

    def on_span_end(self, span) -> None:
        self._write(span.export())

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass
