"""Minimal OTLP/JSON collector: capture agent telemetry to a file.

    python collector.py --out capture.jsonl [--port 4318]

Claude Code (and anything else speaking OTLP over HTTP with
``OTEL_EXPORTER_OTLP_PROTOCOL=http/json``) posts to ``/v1/traces``,
``/v1/logs`` and ``/v1/metrics``. Each request body is appended verbatim as one
JSON line, wrapped with the signal name and the receive time, so nothing the
exporter sent is lost and the capture can be re-parsed later when the mapping
changes.

Standard library only, so it runs in the same environment as the pipeline core
and adds no dependency to the thesis artifact.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_lock = threading.Lock()
_counts: dict[str, int] = {}


class Handler(BaseHTTPRequestHandler):
    out_path = "capture.jsonl"
    quiet = False

    def _signal(self) -> str | None:
        for name in ("traces", "logs", "metrics"):
            if self.path.rstrip("/").endswith("/v1/" + name):
                return name
        return None

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        signal = self._signal()
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.headers.get("Content-Encoding") == "gzip":
            try:
                body = gzip.decompress(body)
            except OSError:
                pass

        if signal is None:
            self.send_response(404)
            self.end_headers()
            return

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            # protobuf or anything else we cannot read: keep the bytes, flag it
            payload = {"_unparsed_base64_len": len(body)}

        record = {
            "signal": signal,
            "received_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "payload": payload,
        }
        with _lock:
            with open(self.out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            _counts[signal] = _counts.get(signal, 0) + 1
            total = sum(_counts.values())
        if not self.quiet:
            print(f"\r  captured {total} payloads  {_counts}", end="", flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args) -> None:
        pass  # the counter above is the only progress output


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="capture.jsonl")
    ap.add_argument("--port", type=int, default=4318)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    Handler.out_path = a.out
    Handler.quiet = a.quiet
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"OTLP/JSON collector on http://127.0.0.1:{a.port} -> {a.out}")
    print("  point the agent at it with:")
    print("    export CLAUDE_CODE_ENABLE_TELEMETRY=1")
    print("    export OTEL_EXPORTER_OTLP_PROTOCOL=http/json")
    print(f"    export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:{a.port}")
    print("  ctrl-c to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped. {sum(_counts.values())} payloads in {a.out}  {_counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
