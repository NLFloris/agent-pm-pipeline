"""OpenTelemetry for the Magentic-One harness: AutoGen's own GenAI spans, on the wire.

AutoGen 0.7 instruments itself against the GenAI semantic conventions: every
agent turn is an ``invoke_agent {name}`` span and every tool execution an
``execute_tool {tool}`` span nested inside it, with gen_ai.operation.name,
gen_ai.agent.name, gen_ai.tool.name and gen_ai.tool.call.id. The spans go to
whatever tracer provider is installed globally. This module installs one that
exports OTLP/JSON, either to an HTTP collector (experiments/otel_capture/collector.py)
or straight to a file in the collector's line format, so that the pipeline's
OpenTelemetry ingester reads the result without any harness-specific code.

The exporter uses the official protobuf encoder of the OTLP exporter package
and the proto3 JSON mapping, with one deviation the OTLP specification requires:
trace and span ids are hex, not base64.

What AutoGen's spans do not carry, and the ingester therefore cannot have:
tool arguments and results (gen_ai.tool.call.arguments / .result are opt-in in
the convention and AutoGen does not set them), model usage, reasoning text.
The harness's own JSON recording keeps those; the comparison is the point.
"""
from __future__ import annotations

import base64
import datetime
import json
import os
import threading
import urllib.request
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from google.protobuf.json_format import MessageToDict
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (BatchSpanProcessor, SimpleSpanProcessor,
                                            SpanExporter, SpanExportResult)

_ID_KEYS = {"traceId", "spanId", "parentSpanId"}


def _hex_ids(node: Any) -> None:
    """OTLP/JSON carries ids as hex; the proto3 JSON mapping gives base64."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in _ID_KEYS and isinstance(v, str) and v:
                node[k] = base64.b64decode(v).hex()
            else:
                _hex_ids(v)
    elif isinstance(node, list):
        for v in node:
            _hex_ids(v)


class OtlpJsonExporter(SpanExporter):
    """Export spans as OTLP/JSON to a collector URL and/or a capture file."""

    def __init__(self, endpoint: str | None = None, path: str | None = None) -> None:
        if not endpoint and not path:
            raise ValueError("give an endpoint, a path, or both")
        self.endpoint = endpoint.rstrip("/") if endpoint else None
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self.exported = 0
        self.failed = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        body = MessageToDict(encode_spans(spans))
        _hex_ids(body)
        ok = True
        if self.endpoint:
            req = urllib.request.Request(self.endpoint + "/v1/traces", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    ok = 200 <= r.status < 300
            except Exception:
                ok = False
        if self.path:
            record = {"signal": "traces",
                      "received_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "payload": body}
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        with self._lock:
            if ok:
                self.exported += len(spans)
            else:
                self.failed += len(spans)
        return SpanExportResult.SUCCESS if ok else SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass


def install(service_name: str = "magentic-one-harness", endpoint: str | None = None,
            path: str | None = None, batch: bool = True) -> tuple[TracerProvider, OtlpJsonExporter]:
    """Install a global tracer provider that AutoGen's spans will reach."""
    exporter = OtlpJsonExporter(endpoint, path)
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider, exporter


@contextmanager
def workflow_span(name: str, conversation_id: str) -> Iterator[trace.Span]:
    """One root span per run, so that every span of the run shares a trace and a case."""
    tracer = trace.get_tracer("agentpm-harness")
    with tracer.start_as_current_span(
        f"invoke_workflow {name}",
        attributes={"gen_ai.operation.name": "invoke_workflow", "gen_ai.workflow.name": name,
                    "gen_ai.conversation.id": conversation_id, "gen_ai.system": "autogen"},
    ) as span:
        yield span
