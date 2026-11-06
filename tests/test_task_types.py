"""One reference model per task type: the template files against the frozen models and the reported verdicts."""
import html
import json
import sys
import time

from agentpm.analysis.conformance import check
from agentpm.cli import main
from agentpm.ingest import tau2
from agentpm.model import Event, Trace
from agentpm.reference import load_task_types
from agentpm.report import report

sys.path.insert(0, "experiments")
import retail_task_types  # noqa: E402
import task_type_conformance  # noqa: E402

RUNS = "examples/demos/tau2_airline/claude-opus-4-5_high_airline_gpt-5.2_4trials.json"
AIRLINE = "examples/task_types_tau2_airline.json"
RETAIL = "examples/task_types_tau2_retail.json"


def _results(domain):
    suffix = "" if domain == "airline" else "_retail"
    with open(f"results/task_type_conformance{suffix}.json", encoding="utf-8") as f:
        return json.load(f)


def test_the_template_files_give_the_frozen_models():
    for domain, path, module in (("airline", AIRLINE, task_type_conformance), ("retail", RETAIL, retail_task_types)):
        types = load_task_types(path)
        keys = {r["type"] for r in _results(domain)}
        assert len(keys) >= 11
        for key in keys:
            assert types.model(key) == module.type_model(key)


def test_the_package_reproduces_the_reported_airline_verdicts():
    rows = {r["case_id"]: r for r in _results("airline")}
    pairs = load_task_types(AIRLINE).pairs(tau2.ingest(RUNS))
    assert {t.case_id for t, _ in pairs} == set(rows)
    for t, model in pairs:
        r = check(t, model)
        expected = rows[t.case_id]
        assert (model.name, r.compliant, r.fitness) == (expected["type"], expected["compliant"], expected["fitness"])
        assert [v.constraint for v in r.violations] == expected["violations"]


def test_the_relabeling_leaves_the_original_trace_alone():
    types = load_task_types(RETAIL)
    t = Trace("r", [Event("find_user_id_by_email", 0)])
    assert types.prepare(t).activities == ["authenticate_user"]
    assert t.activities == ["find_user_id_by_email"]


def test_report_and_run_take_the_template(tmp_path, capsys):
    traces = tau2.ingest(RUNS)[:12]
    report(traces, str(tmp_path), pictures=False, task_types=load_task_types(AIRLINE))
    verdicts = json.loads((tmp_path / "verdicts.json").read_text())
    assert verdicts and all(v["reference"] != "gold" for v in verdicts)
    assert "answer key" not in (tmp_path / "README.md").read_text()
    report(traces, str(tmp_path / "default"), pictures=False)
    assert "answer key" in (tmp_path / "default" / "README.md").read_text()
    assert main(["run", RUNS, "--format", "tau2", "--task-types", AIRLINE]) == 0
    assert "172 traces checked, 156 compliant" in capsys.readouterr().out


def test_the_website_demo_checks_the_airline_runs_against_their_task_types(tmp_path):
    from agentpm.web import _run_job, demos, report_page
    form = dict({d["key"]: d for d in demos()}["tau2"], pictures="0")
    job = {"lines": [], "progress": None, "status": "queued", "rid": None, "error": None, "t0": time.time()}
    _run_job(job, form, str(tmp_path), lambda text, raw=False: None)
    assert job["status"] == "done", job["error"]
    verdicts = json.loads((tmp_path / job["rid"] / "verdicts.json").read_text())
    assert (len(verdicts), sum(v["compliant"] for v in verdicts)) == (172, 156)
    page = html.unescape(report_page(job["rid"], str(tmp_path / job["rid"])).decode())
    assert "Reference: the model of each run's task type, 11 models." in page


def test_a_run_that_states_its_type_needs_no_solution_list():
    types = load_task_types(AIRLINE)
    run = Trace("r", [Event("get_reservation_details", 0), Event("user_message", 1), Event("cancel_reservation", 2)],
                {"task_type": "cancel_reservation"})
    ((checked, model),) = types.pairs([run])
    assert model == types.model("cancel_reservation")
    assert check(checked, model).compliant
    assert types.pairs([Trace("r", [Event("cancel_reservation", 0)])]) == []
    assert types.pairs([Trace("r", [Event("cancel_reservation", 0)], {"task_type": "pipeline_collaboration"})]) == []
