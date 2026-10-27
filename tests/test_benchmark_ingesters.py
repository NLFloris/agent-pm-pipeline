"""Fixture-based tests for the benchmark trajectory ingesters.

Fixtures are hand-built miniatures of the real file layouts (schemas documented
in each ingester's module docstring), so the suite stays fast and the repo stays
free of multi-MB corpora.
"""
import json

import pytest

from agentpm.ingest import agenthallu, tau2, traceelephant


# --------------------------------------------------------------------- tau2

TAU2_FILE = {
    "info": {"environment_info": {"domain_name": "airline", "policy": "...", "tool_defs": None}},
    "tasks": [{
        "id": "t1",
        "evaluation_criteria": {"actions": [
            {"action_id": "1_0", "requestor": "assistant", "name": "get_user_details",
             "arguments": {"user_id": "u1"}},
            {"action_id": "1_1", "requestor": "user", "name": "check_status_bar", "arguments": {}},
        ]},
    }],
    "simulations": [{
        "id": "sim-1", "task_id": "t1", "trial": 0, "seed": 7,
        "termination_reason": "user_stop", "duration": 12.5,
        "agent_cost": 0.1, "user_cost": 0.02,
        "reward_info": {"reward": 1.0, "db_check": None, "action_checks": []},
        "messages": [
            {"role": "user", "content": "hello", "turn_idx": 0,
             "timestamp": "2026-02-24T10:30:50.666494"},
            {"role": "assistant", "content": None, "turn_idx": 1,
             "timestamp": "2026-02-24T10:30:52.000000",
             "raw_data": {"choices": [{"message": {"reasoning_content": "think first"}}]},
             "tool_calls": [{"id": "c1", "name": "get_user_details",
                             "arguments": {"user_id": "u1"}, "requestor": "assistant"}]},
            {"role": "tool", "id": "c1", "content": "{\"ok\":true}", "error": False,
             "requestor": "assistant", "turn_idx": 2,
             "timestamp": "2026-02-24T10:30:52.100000"},
            {"role": "assistant", "content": "all set", "turn_idx": 3,
             "timestamp": "2026-02-24T10:30:53.000000",
             "raw_data": {"choices": [{"message": {"reasoning_content": "wrap up"}}]}},
        ],
    }],
}


def _write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def test_tau2_maps_calls_results_and_outcome(tmp_path):
    traces = tau2.ingest(_write(tmp_path, "sub.json", TAU2_FILE))
    assert len(traces) == 1
    t = traces[0]
    assert t.case_id == "sim-1"
    # one tool call + the user turn + the final assistant turn; the tool RESULT
    # is attached to its call rather than becoming an event of its own
    assert t.activities == ["user_message", "get_user_details", "assistant_message"]
    call = t.events[1]
    assert call.resource == "assistant"
    assert call.attributes["result"] == '{"ok":true}'
    assert call.attributes["error"] is False
    assert call.attributes["reasoning"] == "think first"      # R2: attribute, not activity
    assert t.events[2].attributes["reasoning"] == "wrap up"   # text turns keep theirs too
    assert call.attributes["epoch_us"] > 0
    assert t.outcome["reward"] == 1.0
    assert t.outcome["domain"] == "airline"
    assert [a["name"] for a in t.outcome["gold_actions"]] == [
        "get_user_details", "check_status_bar"]


def test_tau2_skips_unparseable_simulation(tmp_path):
    payload = json.loads(json.dumps(TAU2_FILE))
    payload["simulations"].append({"id": "sim-2", "messages": "not a list"})
    traces = tau2.ingest(_write(tmp_path, "sub.json", payload))
    assert [t.case_id for t in traces] == ["sim-1"]


def test_tau2_timestamps_are_logical_order(tmp_path):
    """Parallel calls share a wall clock, so ordering uses the sequence index."""
    traces = tau2.ingest(_write(tmp_path, "sub.json", TAU2_FILE))
    stamps = [e.timestamp for e in traces[0].events]
    assert stamps == sorted(stamps) == list(range(len(stamps)))


# --------------------------------------------------------------- agenthallu

AGENTHALLU_RUN = {
    "model_id": "gpt-4.1", "agent_type": "Magentic-One",
    "question": "how tall?", "true_answer": "185", "agent_answer": "184",
    "is_hallucination": "True", "hallucination_step": "3",
    "hallucination_category": "factual", "hallucination_subcategory": "entity",
    "hallucination_reason": "wrong landmark",
    "history": [
        {"step": 1, "role": "MagenticOneOrchestrator", "content": "fact sheet"},
        {"step": 2, "role": "MagenticOneOrchestrator",
         "content": "Call Next Speaker: WebSurfer with arguments: (\"q\": find it)"},
        {"step": 3, "role": "WebSurfer", "content": None,
         "tool_calls": [{"function": {"name": "web_search"}}],
         "tool_responses": ["results"]},
        {"step": 4, "role": "WebSurfer", "content": "I found the answer"},
    ],
}


def test_agenthallu_delegation_and_report(tmp_path):
    traces = agenthallu.ingest(_write(tmp_path, "000.json", AGENTHALLU_RUN))
    t = traces[0]
    assert t.activities == ["orchestrate", "delegate", "web_search", "report"]
    delegate = t.events[1]
    assert delegate.resource == "MagenticOneOrchestrator"
    assert delegate.attributes["to"] == "WebSurfer"
    # the report closes the open delegation, naming who it goes back to
    assert t.events[3].attributes["to"] == "MagenticOneOrchestrator"
    assert t.events[2].resource == "WebSurfer"
    assert t.outcome["is_hallucination"] is True
    assert t.outcome["framework"] == "Magentic-One"


def test_agenthallu_handles_framework_without_delegation_phrase(tmp_path):
    run = json.loads(json.dumps(AGENTHALLU_RUN))
    run["history"][1]["content"] = "let us plan"
    traces = agenthallu.ingest(_write(tmp_path, "000.json", run))
    assert "delegate" not in traces[0].activities


# ------------------------------------------------------------ traceelephant

TE_META = {
    "run_id": "gaia_task_1_x", "system_name": "magentic-one", "task_id": "a1",
    "task_instruction": "how tall?", "ground_truth": "185",
    "mistake_agent": "WebSurfer", "mistake_step": "2",
    "mistake_reason": "misread the cover",
}
TE_STEPS = [
    {"step_id": 1, "agent_id": 1, "agent_name": "Orchestrator",
     "input": {"messages": [{"role": "user", "content": "plan it"}]},
     "output": "here is the plan", "tool_logs": []},
    {"step_id": 2, "agent_id": 2, "agent_name": "WebSurfer",
     "input": {"messages": [{"role": "user", "content": "search"}]},
     "output": None,
     "tool_logs": [
         {"function": {"name": "web_search", "arguments": "{\"q\":\"vogue\"}"}},
         {"function": {"name": "visit_url", "arguments": "{\"u\":\"x\"}"}},
     ]},
]


def _te_run(tmp_path, meta=TE_META, steps=TE_STEPS):
    run = tmp_path / meta["run_id"]
    run.mkdir()
    (run / "trace_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (run / "step_records.json").write_text(json.dumps(steps), encoding="utf-8")
    return str(run)


def test_traceelephant_expands_tool_logs_and_locates_mistake(tmp_path):
    t = traceelephant.ingest(_te_run(tmp_path))[0]
    assert t.case_id == "gaia_task_1_x"
    assert t.activities == ["orchestrate", "web_search", "visit_url"]
    assert [e.resource for e in t.events] == ["Orchestrator", "WebSurfer", "WebSurfer"]
    assert t.events[1].attributes["prompt"] == "search"
    assert t.outcome["mistake_agent"] == "WebSurfer"
    assert t.outcome["mistake_step"] == 2
    # step 2 expands into two events; the decisive one is the first of them
    assert t.outcome["decisive_event_index"] == 1


def test_traceelephant_tolerates_missing_mistake_step(tmp_path):
    meta = dict(TE_META, mistake_step="unknown")
    t = traceelephant.ingest(_te_run(tmp_path, meta=meta))[0]
    assert t.outcome["mistake_step"] is None
    assert t.outcome["decisive_event_index"] is None


@pytest.mark.parametrize("module", [tau2, agenthallu, traceelephant])
def test_ingesters_are_deterministic(tmp_path, module):
    """Identical input must yield an identical trace (R4)."""
    if module is tau2:
        make = lambda: module.ingest(_write(tmp_path, "sub.json", TAU2_FILE))
    elif module is agenthallu:
        make = lambda: module.ingest(_write(tmp_path, "000.json", AGENTHALLU_RUN))
    else:
        path = _te_run(tmp_path)
        make = lambda: module.ingest(path)
    first, second = make(), make()
    assert [(e.activity, e.timestamp, e.resource) for e in first[0].events] == \
           [(e.activity, e.timestamp, e.resource) for e in second[0].events]
    assert first[0].outcome == second[0].outcome


TE_ENVELOPE_STEPS = [
    {"step_id": 1, "agent_id": 1, "agent_name": "InformationExtraction_Expert",
     "input": {"messages": [{"role": "user", "content": "find the papers"}]},
     "output": {"model": "gpt-4o-2024-08-06",
                "usage": {"completion_tokens": 254, "prompt_tokens": 2247},
                "choices": [{"finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": "I will search OpenReview."}}]},
     "tool_logs": []},
    {"step_id": 2, "agent_id": 2, "agent_name": "WebSurfer",
     "input": {"messages": [{"role": "user", "content": "go"}]},
     "output": {"choices": [{"message": {"content": "Searching now.",
                                         "tool_calls": [{"id": "c1"}]}}]},
     "tool_logs": [{"function": {"name": "web_search", "arguments": "{\"q\":\"yuri\"}"}}]},
]


def test_traceelephant_unwraps_the_completion_envelope(tmp_path):
    """`output` is the provider's raw response; the words are inside it."""
    t = traceelephant.ingest(_te_run(tmp_path, steps=TE_ENVELOPE_STEPS))[0]
    said = t.events[0]
    assert said.activity == "report"
    assert said.attributes["content"] == "I will search OpenReview."
    assert said.attributes["model"] == "gpt-4o-2024-08-06"
    assert said.attributes["completion_tokens"] == 254
    assert said.attributes["finish_reason"] == "stop"
    # text accompanying a tool call rides on that call, not on an extra event
    tool = t.events[1]
    assert tool.activity == "web_search"
    assert tool.attributes["content"] == "Searching now."
    assert len(t.events) == 2


# ------------------------------------------------------------ local_magentic

LOCAL_RUN = {
    "task_id": "l02_revenue", "shape": "relay", "model": "qwen3.8-flash-next",
    "prompt": "compute revenue per region", "seconds": 41.2, "error": None,
    "messages": [
        {"type": "TextMessage", "source": "user", "created_at": "2026-10-13T10:00:00+00:00",
         "payload": {"text": "compute revenue per region"}},
        {"type": "SelectSpeakerEvent", "source": "MagenticOneOrchestrator",
         "created_at": "2026-10-13T10:00:02+00:00", "payload": {"text": "FileSurfer"}},
        {"type": "ThoughtEvent", "source": "FileSurfer",
         "created_at": "2026-10-13T10:00:03+00:00",
         "payload": {"text": "I should open the csv before computing anything."}},
        {"type": "ToolCallRequestEvent", "source": "FileSurfer",
         "created_at": "2026-10-13T10:00:04+00:00", "completion_tokens": 31,
         "payload": {"items": [{"kind": "call", "id": "c1", "name": "open_path",
                                "arguments": "{\"path\": \"sales.csv\"}"}]}},
        {"type": "ToolCallExecutionEvent", "source": "FileSurfer",
         "created_at": "2026-10-13T10:00:04+00:00",
         "payload": {"items": [{"kind": "result", "call_id": "c1", "name": "open_path",
                                "is_error": False, "content": "region,quarter,units,price"}]}},
        {"type": "TextMessage", "source": "Coder",
         "created_at": "2026-10-13T10:00:09+00:00",
         "payload": {"text": "north 2470, south 2805, east 1678.75"}},
    ],
}


def test_local_magentic_keeps_agent_arguments_and_reasoning(tmp_path):
    from agentpm.ingest import local_magentic
    t = local_magentic.ingest(_write(tmp_path, "l02_revenue.json", LOCAL_RUN))[0]
    assert t.case_id == "l02_revenue"
    assert t.activities == ["report", "delegate", "open_path", "report"]
    assert [e.resource for e in t.events] == [
        "user", "MagenticOneOrchestrator", "FileSurfer", "Coder"]

    delegate = t.events[1]
    assert delegate.attributes["to"] == "FileSurfer"

    call = t.events[2]
    assert call.attributes["arguments"] == '{"path": "sales.csv"}'
    assert call.attributes["result"] == "region,quarter,units,price"
    assert call.attributes["error"] is False
    # reasoning rides on the next event from that agent, never as an activity (L6)
    assert call.attributes["reasoning"].startswith("I should open the csv")
    assert "ThoughtEvent" not in t.activities

    assert t.outcome["shape"] == "relay"
    assert t.outcome["model"] == "qwen3.8-flash-next"
    assert t.outcome["reasoning_turns"] == 1
    assert t.outcome["agents"] == ["Coder", "FileSurfer", "MagenticOneOrchestrator", "user"]


def test_local_magentic_keeps_every_thought(tmp_path):
    from agentpm.ingest import local_magentic
    run = {**LOCAL_RUN, "messages": [
        {"type": "ThoughtEvent", "source": "Coder", "payload": {"text": "first idea"}},
        {"type": "TextMessage", "source": "Coder", "payload": {"text": ""}},
        {"type": "ThoughtEvent", "source": "Coder", "payload": {"text": "second idea"}},
        {"type": "TextMessage", "source": "Coder", "payload": {"text": "done"}},
        {"type": "ThoughtEvent", "source": "Analyst", "payload": {"text": "never acted"}},
    ]}
    t = local_magentic.ingest(_write(tmp_path, "l08.json", run))[0]
    # an empty turn is no event, so both thoughts ride on the next one
    assert t.events[0].attributes["reasoning"] == "first idea\n\nsecond idea"
    assert t.outcome["reasoning_turns"] == 3
    assert t.outcome["unattached_reasoning"] == ["never acted"]
