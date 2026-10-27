"""The one-command report on a real capture, without pm4py or graphviz."""
import json
import os

from agentpm.cli import main
from agentpm.ingest import otel
from agentpm.report import report

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "autogen_otel_roundrobin.jsonl")


def test_report_writes_logs_and_flags_what_the_source_lacks(tmp_path):
    checks = dict((line.split(".")[0][:20], status) for status, line in
                  report(otel.ingest(FIX), str(tmp_path), source="autogen", pictures=False))
    files = set(os.listdir(tmp_path))
    assert {"CHECKS.txt", "inventory.txt", "event_log.csv", "runs.ocel.json", "README.md"} <= files
    assert "conformance.txt" not in files              # no reference, no gold actions
    text = (tmp_path / "CHECKS.txt").read_text()
    assert "4 tool calls" in text
    assert "!!  no tool arguments" in text               # OpenTelemetry route: opt-in, not set
    assert "attributed to" in text and "FileSurfer" in text
    assert "byte-identical" in text


def test_report_command_with_a_reference(tmp_path):
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"name": "r", "required": ["read_file", "write_file"],
                               "precedence": [["read_file", "write_file"]]}))
    out = tmp_path / "out"
    assert main(["report", FIX, "--format", "otel", "-o", str(out), "--reference", str(ref), "--no-pictures"]) == 0
    verdicts = json.loads((out / "verdicts.json").read_text())
    assert len(verdicts) == 1 and verdicts[0]["compliant"] is True


def test_web_detects_the_format():
    from agentpm.web import load_any
    fmt, traces = load_any(FIX, "detect")
    assert fmt == "otel" and len(traces) == 1
    fmt, _ = load_any(os.path.join(os.path.dirname(__file__), "..", "examples", "claude_transcript_sample.jsonl"), "detect")
    assert fmt == "claude"
