"""The run that Appendix A of the thesis follows through the pipeline.

    PYTHONHASHSEED=0 python experiments/example_run.py

Appendix A takes trial 1 of airline task 40 by Claude Opus 4.5, one of the
shortest of the 172 scored airline runs, and shows what each layer of the
pipeline receives and returns. This script cuts that run out of the bundled
results file of tau2-bench, keeping the file's own shape (the timestamp, the
info with the policy, the one task and the one simulation), writes it to
examples/tau2_airline_task40_trial1.json, and runs the report on it with the
task-type template of the airline domain, as the appendix does:

    python -m agentpm.cli report examples/tau2_airline_task40_trial1.json --format tau2 \\
        --task-types examples/task_types_tau2_airline.json -o results/example_run

The report folder holds every listing of the appendix: the canonical trace,
the OCEL log (runs.ocel.json), the verdict and the secondary lenses
(lenses.txt). The Judge verdicts of Table A.2 are not computed here; they are
the first answers in results/judge/verdicts_*_gpt-oss-120b.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "examples", "demos", "tau2_airline",
                      "claude-opus-4-5_high_airline_gpt-5.2_4trials.json")
TARGET = os.path.join(ROOT, "examples", "tau2_airline_task40_trial1.json")
SIMULATION = "871f33ef-fa83-42e7-8d9a-da8474280864"


def main() -> int:
    with open(SOURCE, encoding="utf-8") as f:
        data = json.load(f)
    sims = [s for s in data["simulations"] if s["id"] == SIMULATION]
    if len(sims) != 1:
        sys.exit(f"simulation {SIMULATION} not found once in {SOURCE}")
    sim = sims[0]
    task = [t for t in data["tasks"] if str(t["id"]) == str(sim["task_id"])]
    out = {"timestamp": data["timestamp"], "info": data["info"],
           "tasks": task, "simulations": [sim]}
    with open(TARGET, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {os.path.relpath(TARGET, ROOT)}: task {sim['task_id']}, trial {sim['trial']}, "
          f"{len(sim['messages'])} messages, reward {sim['reward_info']['reward']}")
    return subprocess.call([sys.executable, "-m", "agentpm.cli", "report", TARGET,
                            "--format", "tau2",
                            "--task-types", os.path.join(ROOT, "examples", "task_types_tau2_airline.json"),
                            "-o", os.path.join(ROOT, "results", "example_run")], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
