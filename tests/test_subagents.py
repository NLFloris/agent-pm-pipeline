"""Sub-agents in the local Magentic-One corpus: ingester and OCEL builder.

The fixture is a real AutoGen 0.7.5 run of the --subagents team, with scripted
model responses instead of an endpoint: the Coder computes, then the Analyst
starts DataChecker and Recalculator as tools, and each uses its own tool.
"""
import os

from agentpm.ingest import local_magentic
from agentpm.logbuild.ocel import build_ocel_from_traces

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "local_subagents_replay.json")


def _trace():
    return local_magentic.ingest(FIXTURE)[0]


def test_subagent_start_is_a_spawn_by_the_parent():
    t = _trace()
    assert [(e.resource, e.activity) for e in t.events] == [
        ("user", "report"),
        ("Coder", "run_python"), ("Coder", "report"),
        ("Analyst", "spawn"), ("DataChecker", "read_file"), ("DataChecker", "report"),
        ("Analyst", "spawn"), ("Recalculator", "run_python"), ("Recalculator", "report"),
        ("Analyst", "report"),
    ]
    spawns = [e for e in t.events if e.activity == "spawn"]
    assert [e.attributes["spawns"] for e in spawns] == ["DataChecker", "Recalculator"]
    # the task the parent gave is the spawn's arguments, the sub-agent's answer its result
    assert "east rows" in spawns[0].attributes["arguments"]
    assert "east,Q1,80" in spawns[0].attributes["result"]


def test_subagent_events_carry_their_parent_and_arguments():
    t = _trace()
    sub = [e for e in t.events if e.resource in ("DataChecker", "Recalculator")]
    assert {e.attributes["spawned_by"] for e in sub} == {"Analyst"}
    read = next(e for e in sub if e.activity == "read_file")
    assert read.attributes["arguments"] == '{"path": "sales.csv"}'
    assert "spawned_by" not in next(e for e in t.events if e.resource == "Coder").attributes
    assert t.outcome["subagents"] == {"DataChecker": "Analyst", "Recalculator": "Analyst"}
    assert t.outcome["task_type"] == "claim_check"


def test_the_framework_task_message_is_not_an_event():
    # AutoGen hands a sub-agent its task as a message from "user"; only the
    # run's own first message is the customer's request
    t = _trace()
    assert [e.resource for e in t.events].count("user") == 1


def test_ocel_links_sub_agent_to_parent():
    log = build_ocel_from_traces([_trace()])
    objects = {o["id"]: o for o in log["objects"]}
    assert {"objectId": "agent:Analyst", "qualifier": "spawned_by"} in objects["agent:DataChecker"]["relationships"]
    assert {"objectId": "agent:Analyst", "qualifier": "spawned_by"} in objects["agent:Recalculator"]["relationships"]
    spawn = next(e for e in log["events"] if e["type"] == "spawn")
    assert {"objectId": "agent:Analyst", "qualifier": "executed_by"} in spawn["relationships"]
    assert {"objectId": "agent:DataChecker", "qualifier": "spawns"} in spawn["relationships"]


def test_calls_issued_together_share_a_step(tmp_path):
    import json
    run = {"task_id": "t", "messages": [
        {"type": "ToolCallRequestEvent", "source": "Recalculator", "payload": {"items": [
            {"kind": "call", "id": "r1", "name": "run_python", "arguments": "{}"},
            {"kind": "call", "id": "r2", "name": "run_python", "arguments": "{}"}]}},
        {"type": "TextMessage", "source": "Recalculator", "payload": {"text": "done"}},
    ]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(run))
    t = local_magentic.ingest(str(p))[0]
    assert [e.timestamp for e in t.events] == [0, 0, 1]
    assert [len(s) for s in t.steps()] == [2, 1]


def test_runs_without_subagents_are_unchanged(tmp_path):
    import json
    with open(FIXTURE, encoding="utf-8") as f:
        run = json.load(f)
    run.pop("subagents")
    run.pop("task_type")
    p = tmp_path / "plain.json"
    p.write_text(json.dumps(run))
    t = local_magentic.ingest(str(p))[0]
    # without the declaration a sub-agent call is an ordinary tool call
    assert "spawn" not in t.activities and "DataChecker" in t.activities
    assert "subagents" not in t.outcome and "task_type" not in t.outcome


def test_report_checks_a_reference_that_names_agents(tmp_path):
    # the frozen claim_check model speaks of Agent.activity; --by-agent makes the
    # report read the run that way, and the plain names fail every agent constraint
    import json
    from agentpm.reference import load_reference
    from agentpm.report import report
    ref = load_reference(os.path.join(os.path.dirname(__file__), "..", "experiments", "local_agents",
                                      "claim_check_reference.json"))
    report([_trace()], str(tmp_path / "agent"), reference=ref, pictures=False, by_agent=True)
    assert json.loads((tmp_path / "agent" / "verdicts.json").read_text())[0]["compliant"] is True
    report([_trace()], str(tmp_path / "plain"), reference=ref, pictures=False)
    assert json.loads((tmp_path / "plain" / "verdicts.json").read_text())[0]["compliant"] is False
    # the log keeps the plain tool names
    assert "Analyst.spawn" not in (tmp_path / "agent" / "event_log.csv").read_text()
