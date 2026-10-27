from agentpm.model import load_trace


def test_load_and_order():
    t = load_trace("examples/sample_trace.json")
    assert t.case_id == "run-compliant-001"
    assert len(t.events) == 5
    assert t.activities[0] == "user_request"
    assert t.activities[-1] == "answer"


def test_ordering_is_by_timestamp():
    t = load_trace("examples/sample_trace_faulted.json")
    # events are stored in file order but 'activities' must follow timestamp order
    assert t.activities == ["user_request", "answer", "retrieve"]
