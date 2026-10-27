from datetime import datetime

from agentpm.analysis.conformance import check
from agentpm.ingest.claude_code import ClaudeCodeIngester
from agentpm.logbuild.ocel import build_ocel_from_traces
from agentpm.reference import Reference

SAMPLE = "examples/claude_transcript_sample.jsonl"


def test_transcript_to_canonical_trace():
    traces = ClaudeCodeIngester().ingest(SAMPLE)
    assert len(traces) == 1
    t = traces[0]
    assert t.case_id == "sess-demo-001"
    # only executed tool calls become activities; assistant prose never does (R2)
    assert t.activities == ["Read", "Edit", "Bash", "Grep"]


def test_tool_results_are_paired_back_to_calls():
    t = ClaudeCodeIngester().ingest(SAMPLE)[0]
    by_act = {e.activity: e for e in t.ordered()}
    # duration comes from pairing tool_use with its tool_result via tool_use_id
    assert by_act["Bash"].attributes["duration_ms"] == 4000
    assert by_act["Bash"].attributes["is_error"] is True
    assert by_act["Read"].attributes["is_error"] is False


def test_sidechain_events_are_attributed_to_subagent():
    t = ClaudeCodeIngester().ingest(SAMPLE)[0]
    by_act = {e.activity: e for e in t.ordered()}
    assert by_act["Grep"].resource == "subagent"
    assert by_act["Read"].resource == "claude-opus-4-8"


def test_large_inputs_are_truncated():
    from agentpm.ingest.claude_code import _truncate
    assert _truncate("x" * 5000).endswith("chars]")
    assert len(_truncate("x" * 5000)) < 260
    assert _truncate(42) == 42


def test_real_protocol_read_before_edit_holds():
    t = ClaudeCodeIngester().ingest(SAMPLE)[0]
    ref = Reference(required=["Read", "Edit"], precedence=[["Read", "Edit"]])
    res = check(t, ref)
    assert res.compliant and res.fitness == 1.0


def test_violated_protocol_is_localized():
    t = ClaudeCodeIngester().ingest(SAMPLE)[0]
    ref = Reference(precedence=[["Edit", "Read"]])  # deliberately false for this trace
    res = check(t, ref)
    assert not res.compliant
    assert res.violations[0].constraint == "precedence(Edit,Read)"
    assert res.violations[0].at == 0


def test_generic_ocel_from_claude_traces():
    traces = ClaudeCodeIngester().ingest(SAMPLE)
    ocel = build_ocel_from_traces(traces)
    assert set(ocel) == {"objectTypes", "eventTypes", "objects", "events"}
    object_ids = {o["id"] for o in ocel["objects"]}
    declared = {t["name"] for t in ocel["eventTypes"]}
    for e in ocel["events"]:
        assert e["type"] in declared
        datetime.fromisoformat(e["time"])
        for rel in e["relationships"]:
            assert rel["objectId"] in object_ids and rel["qualifier"]
    # one run, two agents (main model + subagent), four tools
    counts = {}
    for o in ocel["objects"]:
        counts[o["type"]] = counts.get(o["type"], 0) + 1
    assert counts == {"run": 1, "agent": 2, "tool": 4}


def test_generic_ocel_is_deterministic():
    traces = ClaudeCodeIngester().ingest(SAMPLE)
    assert build_ocel_from_traces(traces) == build_ocel_from_traces(ClaudeCodeIngester().ingest(SAMPLE))


def test_thinking_with_text_becomes_reasoning_attribute(tmp_path):
    import json
    lines = [
        {"type": "assistant", "sessionId": "s", "timestamp": "2026-07-15T10:00:00.000Z",
         "message": {"model": "m", "content": [{"type": "thinking", "thinking": "", "signature": "x"}]}},
        {"type": "assistant", "sessionId": "s", "timestamp": "2026-07-15T10:00:01.000Z",
         "message": {"model": "m", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]}},
        {"type": "assistant", "sessionId": "s", "timestamp": "2026-07-15T10:00:02.000Z",
         "message": {"model": "m", "content": [{"type": "thinking", "thinking": "check the test first"}]}},
        {"type": "assistant", "sessionId": "s", "timestamp": "2026-07-15T10:00:03.000Z",
         "message": {"model": "m", "content": [{"type": "tool_use", "id": "t2", "name": "Bash", "input": {}}]}},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines), encoding="utf-8")
    t = ClaudeCodeIngester().ingest(str(p))[0]
    assert t.activities == ["Read", "Bash"]                       # thinking is never an activity
    assert "reasoning" not in t.events[0].attributes              # empty (signature-only) block skipped
    assert t.events[1].attributes["reasoning"] == "check the test first"
