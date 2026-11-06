"""The four criteria of the evaluation command."""
import json

import pytest

from agentpm.cli import main
from agentpm.evaluation import flags_at_reward_one, metrics, path, same_outcome_pairs, separation
from agentpm.model import Event, Trace

RUNS = "examples/demos/tau2_airline/claude-opus-4-5_high_airline_gpt-5.2_4trials.json"
TEMPLATE = "examples/task_types_tau2_airline.json"
REQUIRED = "results/judge/verdicts_gold-actions_gpt-oss-120b.json"
POLICY = "results/judge/verdicts_full_gpt-oss-120b.json"


def _run(case, steps, reward=1.0, task="t"):
    events = [Event(a, k) for k, step in enumerate(steps) for a in step]
    return Trace(case, events, {"reward": reward, "task_id": task})


def test_kappa_and_precision_treat_a_violation_as_the_positive_class():
    truth = {"a": False, "b": False, "c": True, "d": True}
    pred = {"a": False, "b": True, "c": True, "d": True}
    m = metrics(pred, truth, list(truth))
    assert (m["precision"], m["recall"]) == (1.0, 0.5)
    assert m["kappa"] == pytest.approx(0.5)


def test_the_listing_of_calls_issued_together_is_not_another_path():
    assert path(_run("a", [["x", "y"]])) == path(_run("b", [["y", "x"]]))
    assert path(_run("a", [["x"], ["y"]])) != path(_run("b", [["y"], ["x"]]))


def test_pairs_need_the_same_task_reward_one_and_another_path():
    runs = [_run("a", [["x"], ["y"]]), _run("b", [["y"], ["x"]]), _run("c", [["x"], ["y"]]),
            _run("d", [["z"]], reward=0.0), _run("e", [["z"]], task="other")]
    assert sorted(same_outcome_pairs(runs)) == [("a", "b"), ("b", "c")]
    assert same_outcome_pairs(runs, tools_only=True) == same_outcome_pairs(runs)


def test_separation_compares_a_method_with_the_label():
    truth = {"a": True, "b": False, "c": True, "d": True}
    verdict = {"a": True, "b": False, "c": True, "d": False}
    s = separation(verdict, [("a", "b"), ("a", "c"), ("c", "d")], truth)
    assert (s["judged"], s["separated"], s["as_label"], s["against"], s["both_compliant"]) == (3, 2, 1, 0, 1)


def test_flags_at_reward_one_count_only_correct_outcomes_and_judged_runs():
    runs = [_run("a", [["x"]]), _run("b", [["x"]]), _run("c", [["x"]]), _run("d", [["x"]]),
            _run("e", [["x"]], reward=0.0)]
    truth = {"a": False, "b": False, "c": True, "d": True, "e": False}
    verdict = {"a": False, "c": True, "d": False, "e": False}
    assert flags_at_reward_one(verdict, runs, truth) == [(1, 1), (1, 2)]


def test_evaluate_reproduces_the_reported_airline_comparison(capsys):
    assert main(["evaluate", RUNS, "--format", "tau2", "--task-types", TEMPLATE,
                 "--judge", f"required={REQUIRED}", "--rounds", "20"]) == 0
    out = capsys.readouterr().out
    assert "label per-action: 172 runs, 37 violations" in out
    assert "0.94   0.41   0.50" in out            # the engine
    assert "0.93   1.00   0.95" in out            # the required-actions Judge
    assert "3 repetitions give 1 distinct verdict hash" in out
    assert "engine: 0 tokens" in out
    assert "flags   1 of  11 violations,   0 of 131 compliant runs" in out    # the engine
    assert "flags  11 of  11 violations,   3 of 131 compliant runs" in out    # the required-actions Judge
    assert "other path (all activities): 167, of which the label separates 6" in out


def test_evaluate_names_the_cache_copies_behind_a_judges_agreement(capsys):
    main(["evaluate", RUNS, "--format", "tau2", "--task-types", TEMPLATE, "--judge", f"policy={POLICY}", "--rounds", "5"])
    assert "cache copies, which agree by construction" in capsys.readouterr().out


def test_evaluate_takes_labels_from_a_file_and_skips_what_the_source_lacks(tmp_path, capsys):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"human": {"sample": True}}))
    args = ["evaluate", "examples/sample_trace.json", "--format", "generic", "--reference", "examples/reference_model.json"]
    with pytest.raises(SystemExit):
        main(args)
    capsys.readouterr()
    case = json.load(open("examples/sample_trace.json"))["case_id"]
    labels.write_text(json.dumps({"human": {case: True}}))
    assert main(args + ["--labels", str(labels), "--rounds", "5"]) == 0
    out = capsys.readouterr().out
    assert "label human: 1 runs, 0 violations" in out
    assert "PATH-SENSITIVITY: skipped" in out
