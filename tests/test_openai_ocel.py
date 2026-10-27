from datetime import datetime

from agentpm.analysis.conformance import check
from agentpm.ingest.openai_agents import OpenAIAgentsIngester, parse_export_jsonl
from agentpm.logbuild.ocel import build_ocel
from agentpm.reference import load_reference

SAMPLE = "examples/openai_spans.jsonl"


def test_openai_ingest_flat_trace():
    traces = OpenAIAgentsIngester().ingest(SAMPLE)
    assert len(traces) == 1
    t = traces[0]
    assert t.case_id == "trace_demo_retrieval_001"
    # executed calls only, ordered by started_at; agent spans are not activities
    assert t.activities == ["llm_call", "handoff", "llm_call", "retrieve", "llm_call", "answer"]
    # resource = nearest ancestor agent span
    by_act = {e.activity: e for e in t.ordered()}
    assert by_act["retrieve"].resource == "Research"
    assert by_act["handoff"].resource == "Triage"
    # payloads stay in attributes, never in the activity label (R2)
    assert "input" in by_act["retrieve"].attributes


def test_openai_trace_is_compliant_with_reference():
    t = OpenAIAgentsIngester().ingest(SAMPLE)[0]
    res = check(t, load_reference("examples/reference_model.json"))
    assert res.compliant and res.fitness == 1.0


def test_ocel_structure_and_referential_integrity():
    ocel = build_ocel(parse_export_jsonl(SAMPLE))
    assert set(ocel) == {"objectTypes", "eventTypes", "objects", "events"}
    object_ids = {o["id"] for o in ocel["objects"]}
    declared_event_types = {t["name"] for t in ocel["eventTypes"]}
    declared_object_types = {t["name"] for t in ocel["objectTypes"]}
    assert {o["type"] for o in ocel["objects"]} <= declared_object_types
    for e in ocel["events"]:
        assert e["type"] in declared_event_types
        datetime.fromisoformat(e["time"])  # ISO 8601 parseable
        for rel in e["relationships"]:
            assert rel["objectId"] in object_ids and rel["qualifier"]
    for o in ocel["objects"]:
        for rel in o["relationships"]:
            assert rel["objectId"] in object_ids
    # object-centric content: one run, two agents, two tools
    types = sorted(o["type"] for o in ocel["objects"])
    assert types == ["agent", "agent", "run", "tool", "tool"]
    # handoff event carries qualified from/to relations to both agents, and at
    # most one qualifier per event-object pair
    handoff = next(e for e in ocel["events"] if e["type"] == "handoff")
    quals = {r["qualifier"] for r in handoff["relationships"]}
    assert {"from", "to", "run"} == quals
    for e in ocel["events"]:
        pairs = [r["objectId"] for r in e["relationships"]]
        assert len(pairs) == len(set(pairs))


def test_ocel_build_is_deterministic():
    runs = parse_export_jsonl(SAMPLE)
    assert build_ocel(runs) == build_ocel(parse_export_jsonl(SAMPLE))


def test_generation_payloads_are_attached():
    # regression: prompts and completions must reach the log as attributes
    t = OpenAIAgentsIngester().ingest(SAMPLE)[0]
    gens = [e for e in t.ordered() if e.activity == "llm_call"]
    assert gens and all("input" in e.attributes and "output" in e.attributes for e in gens)


def test_timestamp_fallback_preserves_file_order():
    # regression: an unparseable started_at must keep its file position, not sort first
    from agentpm.ingest.openai_agents import AgentRun, SdkSpan, run_to_trace
    spans = [
        SdkSpan("s1", "tr", None, "2026-07-08T09:00:00+00:00", None, {"type": "function", "name": "a"}),
        SdkSpan("s2", "tr", None, "not-a-date", None, {"type": "function", "name": "b"}),
        SdkSpan("s3", "tr", None, "2026-07-08T09:00:05+00:00", None, {"type": "function", "name": "c"}),
    ]
    t = run_to_trace(AgentRun(trace_id="tr", spans=spans))
    assert t.activities == ["a", "b", "c"]


def test_ocel_times_parseable_and_boolean_typed():
    # regression: unparseable times are normalised, and boolean attrs are declared boolean
    from agentpm.ingest.openai_agents import AgentRun, SdkSpan
    spans = [
        SdkSpan("s1", "tr", None, "2026-07-08T09:00:00+00:00", None, {"type": "function", "name": "a"}),
        SdkSpan("s2", "tr", None, "garbage", None, {"type": "guardrail", "name": "safety", "triggered": False}),
    ]
    ocel = build_ocel([AgentRun(trace_id="tr", spans=spans)])
    for e in ocel["events"]:
        datetime.fromisoformat(e["time"])
    guard = next(t for t in ocel["eventTypes"] if t["name"] == "guardrail")
    assert next(a for a in guard["attributes"] if a["name"] == "triggered")["type"] == "boolean"


def test_ocel_deterministic_across_processes():
    # regression: byte-identical output under different hash seeds (thesis R4)
    import os
    import subprocess
    import sys
    code = (
        "import json;"
        "from agentpm.ingest.openai_agents import parse_export_jsonl;"
        "from agentpm.logbuild.ocel import build_ocel;"
        "print(json.dumps(build_ocel(parse_export_jsonl('examples/openai_spans.jsonl'))))"
    )
    outs = set()
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout)
    assert len(outs) == 1
