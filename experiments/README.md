# Experiments

Every number and figure in `results/` and in the thesis comes from one of these
scripts. Each script states its exact command in its docstring. Run them from
the repository root, with the `pm` extra installed (`pip install -e ".[pm]"`)
and the Graphviz binary on the `PATH`. Unlike the `agentpm` package, the
scripts may use pm4py and pandas.

## Data

Most scripts read the public corpora from a data folder, `~/Downloads` unless
the environment variable `AGENTPM_DATA` names another one (`data_dir.py`), or
`--data` says otherwise, laid out as follows:

```
~/Downloads/
├── tau-data/          python experiments/mirror_tau.py ~/Downloads/tau-data   (6.19 GB, sha256 manifest)
├── agenthallu/        git clone https://github.com/liuxuannan/AgentHallu ~/Downloads/agenthallu
└── traceelephant/     optional, huggingface.co/datasets/TraceElephant/TraceElephant, data.zip unpacked
```

The harness corpora (`results/local*`) and the Judge outputs (`results/judge/`)
are in the repository, so the scripts that only need those run on a fresh clone.

## Scripts, outputs and where the thesis uses them

"supporting" marks a script whose output the thesis does not report but that
helps to inspect the data. Sections are those of the thesis.

| script | writes | thesis |
|---|---|---|
| `run_three_corpora.py` | `results/three_corpora.txt` (stdout), `three_corpora_summary.json`: inventory, conformance, discovery and repeated tool use per corpus | Section 5 |
| `repeated_tool_use.py` | `results/repeated_tool_use.txt`: whether the higher share of repeated tool calls in failed airline runs survives a comparison within the same task, at the same run length, and against the repeats the solution list requires | Section 5 |
| `build_ocel.py` | `results/ocel/*.ocel.json`, read back with pm4py; stdout in `results/ocel/BUILD.txt` | Section 5 |
| `export_event_log.py` | `results/logs/`: the flat τ² event log as CSV and XES, verdicts per run | supporting |
| `partial_order.py` | `results/partial_order.txt`: what discovery does with calls issued together | Section 5 |
| `figures.py` | `results/figures/*.png`: directly-follows graphs, handover networks, lane models | supporting |
| `thesis_figures.py` | `results/figures/thesis/`: the printed sub-agent figure and the airline directly-follows graph | Section 5, appendix |
| `otel_report.py` | the reports in `results/otel_live/` and `results/otel_demo/` (same as `agentpm report`) | appendix |
| `task_type_conformance.py` | `results/task_type_conformance.json` (airline), `--domain retail` writes `task_type_conformance_retail.json`; the task-type reference models | Section 6 |
| `retail_task_types.py` | the retail task-type models, frozen before scoring (`results/frozen/retail_task_types.sha256`) | Section 6 |
| `task_type_details.py` | `results/task_type_details.txt`: confirmation rule, when a violation becomes visible, transfers, listing dependence | Sections 6 and 7, appendix |
| `cross_agent.py` | `results/cross_agent.txt` and `.json`: the task-type models on the 15 agents per domain | Section 6 |
| `miss_reasons.py` | `results/miss_reasons.txt`: the engine's misses over the 15 agents, split into absent actions, too few repeats and wrong arguments | Section 6 |
| `task_versions.py` | `results/task_versions.txt`: which of the published runs used an older version of a task, and the engine's scores with and without them | Section 7 |
| `transfer_endings.py` | `results/transfer_endings.txt`: runs the simulated customer ends with a transfer the agent never made | Section 7 |
| `path_sensitivity.py` | `results/path_sensitivity.txt`: how often the engine and the Judges flag the violations and the compliant runs among the runs with reward 1, and whether they separate runs with the same outcome and different paths | Section 6, appendix |
| `example_run.py` | `examples/tau2_airline_task40_trial1.json` and the report in `results/example_run/`: the one run that the thesis follows through the pipeline | Appendix A |
| `first_divergence.py` | `results/first_divergence.txt`: first divergence against the step labels of AgentHallu | Section 5 |
| `discover.py` | `results/pm4py/`: pm4py's discovery and conformance next to the engine | supporting |
| `noise_sweep.py` | `results/pm4py/noise_sweep.txt`: discover-then-check at three noise thresholds (fixes `PYTHONHASHSEED=0`) | Section 6 |
| `noise_sweep_seeds.py` | `results/pm4py/noise_sweep_seeds.txt`: the same under ten hash seeds | supporting |
| `one_case_pm4py.py` | `results/pm4py/one_case/`: one τ² run through pm4py step by step | supporting |
| `judge/` | the LLM-as-a-Judge baseline and the comparison, see `judge/README.md` | Section 6 |
| `local_agents/` | the Magentic-One harness and its corpora, and the same tasks on Microsoft Agent Framework (`run_maf.py`, `compare_maf.py`), see `local_agents/README.md` | Section 5, appendix |
| `otel_capture/` | an OTLP/JSON collector and the Claude Code capture suite, see `otel_capture/README.md` | appendix |
| `mirror_tau.py` | downloads τ²-bench and τ-bench with a sha256 manifest | |
| `export_conformance_viewer.py`, `export_run_explorer.py` | JSON for two inspection viewers, not used in the thesis | |

## Regenerating everything that needs no model endpoint

```bash
export PATH=/opt/homebrew/bin:$PATH          # the graphviz binary, on macOS with Homebrew
export PYTHONHASHSEED=0
python experiments/run_three_corpora.py --data ~/Downloads --out results > results/three_corpora.txt
python experiments/export_event_log.py --data ~/Downloads --out results/logs
python experiments/build_ocel.py --data ~/Downloads --out results/ocel > results/ocel/BUILD.txt
python experiments/partial_order.py --data ~/Downloads --out results
python experiments/figures.py --data ~/Downloads --out results/figures
python experiments/thesis_figures.py --out results/figures/thesis
python experiments/task_type_conformance.py
python experiments/task_type_conformance.py --domain retail
python experiments/task_type_details.py > results/task_type_details.txt
python experiments/cross_agent.py > results/cross_agent.txt
python experiments/miss_reasons.py > results/miss_reasons.txt
python experiments/task_versions.py > results/task_versions.txt
python experiments/transfer_endings.py > results/transfer_endings.txt
python experiments/path_sensitivity.py > results/path_sensitivity.txt
python experiments/repeated_tool_use.py > results/repeated_tool_use.txt
python experiments/first_divergence.py > results/first_divergence.txt
python experiments/discover.py --data ~/Downloads --out results/pm4py
python experiments/noise_sweep.py --data ~/Downloads --out results/pm4py
python experiments/noise_sweep_seeds.py > results/pm4py/noise_sweep_seeds.txt
python experiments/one_case_pm4py.py
python experiments/example_run.py
python experiments/judge/evaluate.py > results/judge/evaluation.txt
python experiments/judge/evaluate.py --pm-verdicts results/task_type_conformance.json > results/judge/evaluation_per_type.txt
python experiments/judge/compare_judges.py > results/judge/two_judges.txt
python experiments/local_agents/check_claims.py > results/local_subagents/conformance.txt
python experiments/local_agents/compare_maf.py > results/local_maf/comparison.txt
python -m agentpm.cli report results/local_subagents/runs --format local_magentic -o results/local_subagents/report
python -m agentpm.cli report results/otel_live/capture.jsonl --format otel -o results/otel_live
python -m agentpm.cli report results/otel_demo/capture.jsonl --format otel -o results/otel_demo \
    --reference experiments/otel_demo_reference.json
```

All text outputs come back byte-identical with `PYTHONHASHSEED=0`; this was last
checked on 6 November 2026 by running every command above. Two outputs
depend on Python's hash seed through pm4py's inductive miner: the per-run
fitness values in `results/pm4py/tau2_conformance_three_ways.csv` and the
noise-0.2 net, which is why `noise_sweep.py` fixes the seed and
`noise_sweep_seeds.py` reports the variation. Pictures, among them the pm4py
models, the single-run drawings and the thesis PDFs, can differ in bytes and in
layout from one run to the next even with the same code and seed, so pictures
are compared by eye, not by checksum.

The Judge runs (`judge/run_judge.py`) and the harness runs
(`local_agents/run_magentic.py`) call a model endpoint and are not repeatable
byte for byte; their raw outputs are kept in `results/` for that reason.
