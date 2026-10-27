"""The partial order: concurrent events share a timestamp and never follow each other.

Covers the model (steps, tiebreak), the tau2 ingester (calls issued in one turn
share a timestamp), the concurrency-aware discovery, the object rules and the
object-centric builder that keeps the ties and the business objects.
"""
import json

from agentpm.analysis.discovery import (concurrency_summary, concurrent_pairs,
                                        directly_follows, directly_follows_flat)
from agentpm.ingest import tau2
from agentpm.logbuild.objects import BUILTIN, ObjectRule, extract, rules_from_json
from agentpm.logbuild.ocel import build_ocel_from_traces, object_sharing
from agentpm.model import Event, Trace


def _trace():
    # turn 0: a | turn 1: b and c issued together | turn 2: d
    return Trace("r1", [Event("a", 0, "x", sequence=0), Event("b", 1, "x", sequence=1),
                        Event("c", 1, "x", sequence=2), Event("d", 2, "x", sequence=3)])


def test_steps_group_events_that_share_a_timestamp():
    assert [[e.activity for e in s] for s in _trace().steps()] == [["a"], ["b", "c"], ["d"]]


def test_ordered_breaks_ties_by_sequence_not_insertion():
    t = Trace("r", [Event("c", 1, sequence=2), Event("b", 1, sequence=1), Event("a", 0, sequence=0)])
    assert t.activities == ["a", "b", "c"]


def test_partial_order_dfg_has_no_edge_inside_a_step():
    t = [_trace()]
    flat = directly_follows_flat(t)
    partial = directly_follows(t)
    assert ("b", "c") in flat and ("b", "c") not in partial
    assert partial == {("a", "b"): 1, ("a", "c"): 1, ("b", "d"): 1, ("c", "d"): 1}
    assert concurrent_pairs(t) == {("b", "c"): 1}
    s = concurrency_summary(t)
    assert s["steps_with_more_than_one_event"] == 1 and s["largest_step"] == 2
    assert s["flat_edges"] == 3 and s["partial_order_edges"] == 4 and s["within_step_pairs"] == 1


TAU2_TWO_CALLS = {
    "info": {"environment_info": {"domain_name": "airline"}},
    "tasks": [{"id": "1", "evaluation_criteria": {"actions": []}}],
    "simulations": [{
        "id": "sim-1", "task_id": "1", "trial": 0,
        "messages": [
            {"role": "user", "content": "change my flight", "turn_idx": 0, "timestamp": "2026-01-01T00:00:00"},
            {"role": "assistant", "content": None, "turn_idx": 1, "timestamp": "2026-01-01T00:00:01",
             "tool_calls": [
                 {"id": "c1", "name": "get_user_details", "arguments": {"user_id": "u1"}, "requestor": "assistant"},
                 {"id": "c2", "name": "search_direct_flight", "requestor": "assistant",
                  "arguments": {"origin": "AMS", "destination": "ZRH", "date": "2026-02-01"}}]},
            {"role": "tool", "id": "c1", "content": "{\"user_id\": \"u1\", \"reservations\": [\"R9\"]}",
             "requestor": "assistant", "turn_idx": 1},
            {"role": "tool", "id": "c2", "content": "[]", "requestor": "assistant", "turn_idx": 1},
            {"role": "assistant", "content": None, "turn_idx": 2, "timestamp": "2026-01-01T00:00:02",
             "tool_calls": [{"id": "c3", "name": "book_reservation", "requestor": "assistant",
                             "arguments": {"user_id": "u1", "flights": [{"flight_number": "HAT001"}],
                                           "payment_methods": [{"payment_id": "P1"}]}}]},
            {"role": "tool", "id": "c3", "content": "{\"reservation_id\": \"R10\"}", "requestor": "assistant"},
            {"role": "assistant", "content": "done", "turn_idx": 3, "timestamp": "2026-01-01T00:00:03"},
        ],
        "reward_info": {"reward": 1.0},
    }],
}


def _tau2_traces(tmp_path):
    p = tmp_path / "sub.json"
    p.write_text(json.dumps(TAU2_TWO_CALLS), encoding="utf-8")
    return tau2.ingest(str(p))


def test_tau2_calls_issued_in_one_turn_share_a_timestamp(tmp_path):
    t = _tau2_traces(tmp_path)[0]
    by = {e.activity: e for e in t.events}
    assert by["get_user_details"].timestamp == by["search_direct_flight"].timestamp
    assert by["get_user_details"].sequence < by["search_direct_flight"].sequence
    assert by["search_direct_flight"].timestamp < by["book_reservation"].timestamp
    assert [[e.activity for e in s] for s in t.steps()] == [
        ["user_message"], ["get_user_details", "search_direct_flight"], ["book_reservation"], ["assistant_message"]]
    # the serialisation is still available and unchanged
    assert [e.sequence for e in t.ordered()] == list(range(5))


def test_object_rules_read_arguments_results_and_lists(tmp_path):
    t = _tau2_traces(tmp_path)[0]
    by = {e.activity: e for e in t.events}
    assert extract(by["get_user_details"], BUILTIN["tau2"]) == [("user", "u1")]
    assert extract(by["search_direct_flight"], BUILTIN["tau2"]) == [("route", "AMS>ZRH>2026-02-01")]
    assert extract(by["book_reservation"], BUILTIN["tau2"]) == [
        ("user", "u1"), ("reservation", "R10"), ("flight", "HAT001"), ("payment", "P1")]
    assert extract(by["assistant_message"], BUILTIN["tau2"]) == []
    # a rule restricted to one activity does not fire elsewhere
    only_book = rules_from_json([{"type": "reservation", "path": "result.reservation_id", "activity": "book_reservation"}])
    assert extract(by["get_user_details"], only_book) == []
    try:
        ObjectRule.from_dict({"type": "x"})
        assert False, "a rule needs a path or a compose list"
    except ValueError:
        pass


def test_ocel_keeps_ties_and_relates_business_objects(tmp_path):
    traces = _tau2_traces(tmp_path)
    ocel = build_ocel_from_traces(traces, object_rules=BUILTIN["tau2"],
                                  exclude={"assistant_message", "user_message"})
    times = {e["type"]: e["time"] for e in ocel["events"]}
    assert times["get_user_details"] == times["search_direct_flight"] != times["book_reservation"]
    types = {t["name"] for t in ocel["objectTypes"]}
    assert types == {"run", "agent", "tool", "user", "route", "reservation", "flight", "payment"}
    book = next(e for e in ocel["events"] if e["type"] == "book_reservation")
    touched = {r["objectId"] for r in book["relationships"] if r["qualifier"] == "touches"}
    assert touched == {"user:sim-1|u1", "reservation:sim-1|R10", "flight:sim-1|HAT001", "payment:sim-1|P1"}
    assert {e["type"] for e in ocel["events"]} == {"get_user_details", "search_direct_flight", "book_reservation"}
    share = object_sharing(ocel)
    assert share["events_with_more_than_one_object_type"] == 3
    assert ocel == build_ocel_from_traces(_tau2_traces(tmp_path), object_rules=BUILTIN["tau2"],
                                          exclude={"assistant_message", "user_message"})


def test_precedence_is_by_step_not_by_listing_order():
    from agentpm.analysis.conformance import check
    from agentpm.reference import Reference
    # user lookup and reservation lookup issued together, listed in the "wrong" order
    t = Trace("r", [Event("get_reservation_details", 1, "assistant", sequence=0),
                    Event("get_user_details", 1, "assistant", sequence=1),
                    Event("cancel_reservation", 2, "assistant", sequence=2)])
    ref = Reference(name="x", precedence=[["get_user_details", "get_reservation_details"],
                                          ["get_reservation_details", "cancel_reservation"]])
    assert check(t, ref).compliant                      # same step counts as before, either way
    late = Trace("r", [Event("cancel_reservation", 1, "assistant", sequence=0),
                       Event("get_reservation_details", 2, "assistant", sequence=1)])
    r = check(late, Reference(name="x", precedence=[["get_reservation_details", "cancel_reservation"]]))
    assert not r.compliant and r.violations[0].at == 0
    # forbidden_after: a b issued together with a is not "after" it
    same = Trace("r", [Event("write_file", 3, "c", sequence=0), Event("read_file", 3, "c", sequence=1),
                       Event("read_file", 4, "c", sequence=2)])
    r = check(same, Reference(name="x", forbidden_after=[["write_file", "read_file"]]))
    assert len(r.violations) == 1 and r.violations[0].at == 2


def test_step_labels_share_a_number_for_events_issued_together():
    t = Trace("r", [Event("a", 0, sequence=0), Event("b", 1, sequence=1), Event("c", 1, sequence=2), Event("d", 2, sequence=3)])
    assert t.step_labels() == ["0", "1a", "1b", "2"]


def test_otel_tool_spans_issued_together_share_a_step(tmp_path):
    import json
    from agentpm.ingest import otel
    from tests.test_otel import _span
    spans = [
        _span("o", None, "invoke_agent A", 0, 1000, **{"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "A"}),
        _span("t1", "o", "execute_tool x", 10, 300, **{"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "x"}),
        _span("t2", "o", "execute_tool y", 20, 250, **{"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "y"}),
        _span("t3", "o", "execute_tool z", 400, 500, **{"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "z"}),
    ]
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"resourceSpans": [{"resource": {"attributes": []},
                                                "scopeSpans": [{"scope": {"name": "t"}, "spans": spans}]}]}))
    t = otel.ingest(str(p))[0]
    by = {e.activity: e for e in t.events}
    assert by["x"].timestamp == by["y"].timestamp          # y started before x ended, same parent
    assert by["z"].timestamp > by["y"].timestamp           # z started after both had ended
    assert [[e.activity for e in s] for s in t.steps()] == [["invoke_agent"], ["x", "y"], ["z"]]
