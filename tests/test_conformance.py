from agentpm.analysis.conformance import check
from agentpm.model import load_trace
from agentpm.reference import load_reference

REF = "examples/reference_model.json"


def test_compliant_trace_passes():
    res = check(load_trace("examples/sample_trace.json"), load_reference(REF))
    assert res.compliant
    assert res.fitness == 1.0
    assert res.violations == []


def test_faulted_trace_is_flagged():
    res = check(load_trace("examples/sample_trace_faulted.json"), load_reference(REF))
    assert not res.compliant
    kinds = {v.constraint.split("(")[0] for v in res.violations}
    # answering before retrieving violates precedence, forbidden_after, and the end activity
    assert "precedence" in kinds
    assert "forbidden_after" in kinds
    assert res.fitness < 1.0


def test_determinism():
    a = check(load_trace("examples/sample_trace_faulted.json"), load_reference(REF))
    b = check(load_trace("examples/sample_trace_faulted.json"), load_reference(REF))
    assert a == b


def test_fitness_is_length_independent_under_allowed():
    # Regression: closed-world 'allowed' must count once, not per event, so the
    # same single precedence violation scores equally on a short and a long trace.
    from agentpm.model import Trace, Event
    from agentpm.reference import Reference
    ref = Reference(allowed=["u", "retrieve", "answer"], precedence=[["retrieve", "answer"]])
    short = Trace("s", [Event("answer", 0), Event("retrieve", 1)])
    long = Trace("l", [Event("u", i) for i in range(20)] + [Event("answer", 20), Event("retrieve", 21)])
    rs, rl = check(short, ref), check(long, ref)
    assert not rs.compliant and not rl.compliant
    assert rs.fitness == rl.fitness == 0.5
    assert rs.n_constraints == rl.n_constraints == 2


def test_directly_before_requires_immediate_predecessor():
    from agentpm.model import Event, Trace
    from agentpm.reference import Reference
    from agentpm.analysis.conformance import check
    ref = Reference(directly_before=[["user_message", "cancel_reservation"]])
    ok = Trace("ok", [Event("assistant_message", 0), Event("user_message", 1),
                      Event("cancel_reservation", 2)])
    bad = Trace("bad", [Event("user_message", 0), Event("assistant_message", 1),
                        Event("cancel_reservation", 2)])
    assert check(ok, ref).compliant
    r = check(bad, ref)
    assert not r.compliant
    assert r.violations[0].constraint == "directly_before(user_message,cancel_reservation)"
    assert r.violations[0].at == 2


def test_directly_before_looks_through_transparent_activities():
    from agentpm.model import Event, Trace
    from agentpm.reference import Reference
    from agentpm.analysis.conformance import check
    ref = Reference(directly_before=[["user_message", "cancel_reservation"]],
                    transparent=["cancel_reservation", "assistant_message"])
    # one confirmation, then a batch of three cancellations with chatter between
    batch = Trace("batch", [Event("user_message", 0), Event("cancel_reservation", 1),
                            Event("assistant_message", 2), Event("cancel_reservation", 3),
                            Event("cancel_reservation", 4)])
    assert check(batch, ref).compliant
    # a lookup between the confirmation and the write breaks the chain
    broken = Trace("broken", [Event("user_message", 0), Event("get_reservation_details", 1),
                              Event("cancel_reservation", 2)])
    assert not check(broken, ref).compliant
