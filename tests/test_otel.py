"""OpenTelemetry ingestion: the GenAI semantic conventions and Claude Code's dialect.

The semantic-convention fixture is built in code against the convention text
(semantic-conventions-genai, main @ 2026-10-14, status Development). It plants a
known structure: an orchestrator that invokes two sub-agents concurrently, each
of which executes its own tools. The test checks that the ingester recovers
exactly that structure, with every tool attributed to the agent that ran it and
the two sub-agents' lifetimes overlapping. That is instrument validation, not
evidence about agents.
"""
import json
import os

from agentpm.ingest import otel

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _span(sid, parent, name, start_ms, end_ms, **attrs):
    def val(v):
        if isinstance(v, bool):
            return {"boolValue": v}
        if isinstance(v, int):
            return {"intValue": str(v)}
        return {"stringValue": str(v)}
    return {"spanId": sid, "parentSpanId": parent, "traceId": "t1", "name": name,
            "startTimeUnixNano": str(start_ms * 1_000_000), "endTimeUnixNano": str(end_ms * 1_000_000),
            "attributes": [{"key": k, "value": val(v)} for k, v in attrs.items()]}


def semconv_fixture():
    """Orchestrator -> two concurrent sub-agents, each with its own tool spans."""
    spans = [
        _span("w", None, "invoke_workflow research", 0, 1000,
              **{"gen_ai.operation.name": "invoke_workflow", "gen_ai.workflow.name": "research"}),
        _span("o", "w", "invoke_agent Orchestrator", 0, 1000,
              **{"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "Orchestrator",
                 "gen_ai.conversation.id": "conv-1"}),
        _span("o-llm", "o", "chat gpt-x", 10, 50,
              **{"gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-x",
                 "gen_ai.usage.input_tokens": 100, "gen_ai.usage.output_tokens": 20}),
        # two sub-agents, overlapping in time
        _span("a", "o", "invoke_agent Searcher", 60, 600,
              **{"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "Searcher"}),
        _span("b", "o", "invoke_agent Coder", 80, 700,
              **{"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "Coder"}),
        _span("a1", "a", "execute_tool web_search", 100, 300,
              **{"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "web_search",
                 "gen_ai.tool.call.id": "c1", "gen_ai.tool.call.arguments": '{"q":"otel"}',
                 "gen_ai.tool.call.result": "3 hits"}),
        _span("b1", "b", "execute_tool run_python", 120, 400,
              **{"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "run_python",
                 "gen_ai.tool.call.id": "c2", "gen_ai.tool.call.arguments": '{"code":"1+1"}'}),
        _span("a2", "a", "execute_tool read_file", 320, 500,
              **{"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "read_file",
                 "gen_ai.tool.call.id": "c3"}),
        _span("o-tool", "o", "execute_tool write_report", 800, 900,
              **{"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "write_report"}),
    ]
    return {"resourceSpans": [{"resource": {"attributes": []},
                               "scopeSpans": [{"scope": {"name": "test"}, "spans": spans}]}]}


def test_semconv_attributes_every_tool_to_its_agent(tmp_path):
    p = tmp_path / "spans.json"
    p.write_text(json.dumps(semconv_fixture()), encoding="utf-8")
    traces = otel.ingest(str(p))
    assert len(traces) == 1
    t = traces[0]
    assert t.case_id == "conv-1"
    assert t.outcome["source"] == "otel_semconv"

    by_act = {(e.activity, e.resource) for e in t.events}
    assert ("web_search", "Searcher") in by_act
    assert ("read_file", "Searcher") in by_act
    assert ("run_python", "Coder") in by_act
    assert ("write_report", "Orchestrator") in by_act
    assert ("llm_call", "Orchestrator") in by_act

    # the delegations are performed by the orchestrator and name their target
    inv = [e for e in t.events if e.activity == "invoke_agent"]
    assert {(e.resource, e.attributes["to"]) for e in inv} == {
        ("Orchestrator", "Searcher"), ("Orchestrator", "Coder"), ("root", "Orchestrator")}
    assert sorted(t.outcome["agents"]) == ["Coder", "Orchestrator", "Searcher", "root"]

    # arguments and results survive, and so does the call id
    ws = next(e for e in t.events if e.activity == "web_search")
    assert ws.attributes["arguments"] == '{"q":"otel"}'
    assert ws.attributes["result"] == "3 hits"
    assert ws.attributes["call_id"] == "c1"


def test_semconv_preserves_concurrency(tmp_path):
    p = tmp_path / "spans.json"
    p.write_text(json.dumps(semconv_fixture()), encoding="utf-8")
    t = otel.ingest(str(p))[0]
    a = next(e for e in t.events if e.activity == "invoke_agent" and e.attributes["to"] == "Searcher")
    b = next(e for e in t.events if e.activity == "invoke_agent" and e.attributes["to"] == "Coder")
    # the two sub-agents' lifetimes overlap: a real AND-split, measurable from the log
    assert a.attributes["started_ns"] < b.attributes["ended_ns"]
    assert b.attributes["started_ns"] < a.attributes["ended_ns"]
    # and their tools interleave in start order while keeping their owners
    order = [(e.activity, e.resource) for e in t.ordered() if e.activity not in ("invoke_agent", "invoke_workflow", "llm_call")]
    assert order == [("web_search", "Searcher"), ("run_python", "Coder"),
                     ("read_file", "Searcher"), ("write_report", "Orchestrator")]
    # timestamps are positions; real times are attributes
    assert [e.timestamp for e in t.events] == list(range(len(t.events)))


def test_claude_code_dialect_from_real_capture():
    t = otel.ingest(os.path.join(FIXTURES, "claude_code_otel_subagents.jsonl"))[0]
    assert t.outcome["source"] == "otel_claude_code"
    acts = t.activities
    assert acts.count("subagent_spawn") == 2
    assert acts.count("subagent_completed") == 2
    assert acts.count("subagent_return") == 2
    # the sub-agents' own tool calls are present (Read, Read, Bash, Bash) ...
    assert sorted(a for a in acts if a in ("Read", "Bash")) == ["Bash", "Bash", "Read", "Read"]
    # ... but cannot be attributed: the dialect names no agent on tool events
    assert {e.resource for e in t.events if e.activity in ("Read", "Bash")} == {"main"}
    assert t.outcome["subagents_spawned"] == 2
    # ordering comes from event.sequence, not the shared timestamps
    seqs = [e.attributes["sequence"] for e in t.ordered()]
    assert seqs == sorted(seqs)
    done = [e for e in t.events if e.activity == "subagent_completed"]
    assert all(e.attributes["duration_ms"] for e in done)


def test_otel_is_deterministic(tmp_path):
    p = tmp_path / "spans.json"
    p.write_text(json.dumps(semconv_fixture()), encoding="utf-8")
    a, b = otel.ingest(str(p))[0], otel.ingest(str(p))[0]
    assert [(e.activity, e.timestamp, e.resource) for e in a.events] == \
           [(e.activity, e.timestamp, e.resource) for e in b.events]


def test_otel_rejects_files_with_nothing_usable(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"resourceMetrics": []}), encoding="utf-8")
    try:
        otel.ingest(str(p))
        assert False, "expected ValueError"
    except ValueError:
        pass


# ------------------------------------------------- a real framework's emission

def test_autogen_capture_through_collector_and_ingester():
    """tests/fixtures/autogen_otel_roundrobin.jsonl was recorded on 2026-10-19 by
    experiments/local_agents/smoke_otel.py: AutoGen 0.7.5's own GenAI spans, the
    OpenTelemetry SDK 1.45, OTLP/JSON over HTTP into experiments/otel_capture/collector.py.
    The agents' model is a replay client, so the run is scripted; the spans are
    the framework's. This is the semantic-convention route on a capture nobody
    hand-wrote."""
    traces = otel.ingest(os.path.join(FIXTURES, "autogen_otel_roundrobin.jsonl"))
    assert len(traces) == 1
    t = traces[0]
    assert t.case_id == "smoke-001"                       # gen_ai.conversation.id on the root span
    assert t.outcome["source"] == "otel_semconv"
    assert t.outcome["spans"] == 109                      # runtime spans are kept for the parent chain
    tools = [(e.activity, e.resource, e.attributes["call_id"]) for e in t.ordered() if e.attributes.get("call_id")]
    assert tools == [("list_files", "FileSurfer", "call_1"), ("run_python", "Coder", "call_3"),
                     ("read_file", "FileSurfer", "call_2"), ("write_file", "Coder", "call_4")]
    # the delegation is recorded as the enclosing context invoking each agent in turn
    invoked = [e.attributes["to"] for e in t.ordered() if e.activity == "invoke_agent"]
    assert invoked == ["FileSurfer", "Coder", "FileSurfer", "Coder", "FileSurfer", "Coder"]
    assert all(e.resource == "root" for e in t.events if e.activity == "invoke_agent")
    # what the framework's spans do not carry: arguments and results are opt-in
    # in the convention and AutoGen 0.7.5 does not set them
    assert all(e.attributes["arguments"] is None and e.attributes["result"] is None
               for e in t.events if e.attributes.get("call_id"))
    # real clocks: a tool span lies inside its agent's turn
    by_id = {e.attributes["span_id"]: e for e in t.events}
    for e in t.events:
        if e.attributes.get("call_id"):
            parent = by_id.get(e.attributes["parent_span_id"])
            while parent is not None and parent.activity != "invoke_agent":
                parent = by_id.get(parent.attributes.get("parent_span_id"))
            assert parent is None or (parent.attributes["started_ns"] <= e.attributes["started_ns"]
                                      and e.attributes["ended_ns"] <= parent.attributes["ended_ns"])
