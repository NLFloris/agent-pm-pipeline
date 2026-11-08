# agent-pm-pipeline

Turn the execution traces of LLM-based agents into event logs, and check them
with process mining.

An agent does its work by calling tools. This pipeline reads the agent's
telemetry, turns every executed tool call into an event, writes the runs as an
[OCEL 2.0](https://www.ocel-standard.org/) object-centric event log, and checks
each run against an explicit reference model with conformance checking. The
verdict is computed rather than judged: it costs no tokens, gives the same
answer every time, and names the constraint that was broken and where.

This repository is the artifact of the master's thesis *From Agent Traces to
Event Logs: A Process Mining Framework and Pipeline for Trajectory-Level
Evaluation of LLM-Based Agents* (Floris Kuijpers, University of Twente and Bern
University of Applied Sciences, 2026). It holds the pipeline, the experiments,
and every result the thesis reports, with the event logs and the recorded runs
behind them.

## Quick start

Python 3.10 or newer. The pipeline itself uses only the standard library;
pm4py, pandas and Graphviz are needed for the pictures and the experiments.

```bash
git clone https://github.com/NLFloris/agent-pm-pipeline.git
cd agent-pm-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[pm,dev]"
pytest -q                                   # 79 passed
```

For the pictures, also install the Graphviz binary (`brew install graphviz` on
macOS, `apt install graphviz` on Debian and Ubuntu).

### The website

```bash
python -m agentpm.cli web                   # opens http://127.0.0.1:8765
```

A local website: drop in a trace file, watch the pipeline's log while it runs,
and read every output on one page, from what the source exposes to the
conformance verdicts, the pm4py models, the secondary lenses, each run step by
step, and downloads of the event logs. It runs the same `report` as the command
line and listens on this machine only, so it is not hosted anywhere. Three demos
fill the form in, one per corpus of the thesis, all bundled with the repository:

| demo | runs | what it shows |
|---|---|---|
| τ²-bench airline | 200 customer-service conversations by Claude Opus 4.5 | a single agent with a database; each run checked against the model of its task type (16 of 172 flagged); users, reservations, flights and payments as objects in the OCEL log |
| AgentHallu, Magentic-One | 94 runs of a team of agents | an orchestrator handing work to a web surfer, a coder and a terminal; handover network and one lane per agent; human hallucination labels per run |
| Own harness, sub-agents | 10 runs of one task type by a Magentic-One team on gpt-oss:120b | an agent that starts two sub-agents with their own tool calls; checked against a reference model that names agents, written before any run (9 of 10 conform) |

### The command line

```bash
python -m agentpm.cli report <file or folder> --format <format> -o <output folder>
```

runs the whole pipeline: it checks what the source exposes, writes the flat
and the object-centric event log, runs conformance, and draws the pm4py models
and the pipeline's own pictures. `CHECKS.txt` in the output folder is the
place to start.

On a source that carries a solution list, such as τ²-bench, a report without
`--reference` or `--task-types` checks every run against the list of its own task.
That list is an answer key, so these verdicts show what the engine reaches when it is
handed the solution, and this is not the reference model the thesis recommends. For
one model per task type, add `--task-types examples/task_types_tau2_airline.json`.

For example, on the bundled examples:

```bash
python -m agentpm.cli run examples/sample_trace.json         --format generic --reference examples/reference_model.json
python -m agentpm.cli run examples/sample_trace_faulted.json --format generic --reference examples/reference_model.json -v
```

The first run passes. The second reports success, but the agent answered
before retrieving, so conformance flags a `precedence`, a `forbidden_after` and
an `end_activities` violation: the failure an output-only check cannot see.

[`USAGE.md`](USAGE.md) is the full guide: every command, the reference-model
format, object rules, OpenTelemetry, and how to add a source.

## How it works

```
agent telemetry  ->  L1 ingest  ->  L2 canonical trace  ->  L3 event log  ->  L4 conformance
```

**L1, `agentpm/ingest/`.** One module per source format maps it to canonical
traces, and documents in its docstring the schema it was verified against.

| format | source |
|---|---|
| `otel` | OpenTelemetry: GenAI semantic-convention spans, with full agent attribution, or Claude Code's log dialect |
| `tau2` | τ²-bench trajectory files |
| `agenthallu` | AgentHallu run files |
| `local_magentic` | runs recorded by the Magentic-One harness in `experiments/local_agents/`, including sub-agents |
| `traceelephant` | TraceElephant run folders |
| `openai` | OpenAI Agents SDK trace export |
| `claude` | Claude Code session transcripts |
| `generic` | a trace already in the canonical shape |

**L2, `agentpm/model.py`.** A `Trace` is a list of `Event`s. An activity is an
executed action; reasoning text is kept as an attribute and never becomes an
activity, and the agent's own report of success is a claim, not ground truth.
Calls a model issued together share a logical timestamp, so the partial order
is kept.

**L3, `agentpm/logbuild/`.** `ocel.py` writes OCEL 2.0 JSON that
`pm4py.read_ocel2_json` reads back. Every event is related to its run, its
agent and its tool, plus the business objects that object rules
(`objects.py`) find in the arguments, such as a reservation or a flight.

**L4, `agentpm/analysis/conformance.py`.** A reference model is a small set of
declarative constraints: `required`, `precedence`, `directly_before`,
`forbidden_after`, `allowed` and `end_activities`. Fitness is the share of
constraints that hold, and every violation names its constraint and the event
responsible. `reference.py` also builds one model per task type from a template
declared as data. `analysis/discovery.py` and `analysis/performance.py` hold the
secondary lenses: the directly-follows graph over the listing and over the steps,
handovers and rework. `report.py`, `viz.py` and `web.py` build the report, the
pictures and the website on top.

**After the verdict, `agentpm/evaluation.py`.** Scores the verdicts against labels
on four criteria: alignment, reproducibility, cost and path-sensitivity. Labels come
from the source or a file, and a Judge's verdicts from the file `experiments/judge/`
writes; the package itself calls no model.

## Repository layout

```
agent-pm-pipeline/
├── agentpm/          the pipeline, standard library only
├── tests/            tests on hand-made and recorded fixtures
├── examples/         small inputs, and the demo data in examples/demos/
├── experiments/      the scripts that produce every result (may use pm4py)
│   ├── judge/          the LLM-as-a-Judge baseline and the comparison
│   ├── local_agents/   the Magentic-One harness that records the own corpus
│   └── otel_capture/   an OTLP/JSON collector and the Claude Code capture suite
├── results/          every output, with VALIDATION.md and RESULTS.md as reports
├── USAGE.md          the full guide
├── CITATION.cff
└── LICENSE
```

[`experiments/README.md`](experiments/README.md) lists every script, what it
writes and which section of the thesis uses it, and how to regenerate everything.
[`results/README.md`](results/README.md) says what each result file is.

## Main results

All numbers are produced by the scripts in `experiments/`; the thesis gives the
full picture.

- **One reference model per task type** (one for all cancellations, one for all
  rebookings, and so on) on Claude Opus 4.5's 172 scorable τ²-bench airline
  runs: it flags 16 runs at precision 0.94 and recall 0.41 (κ 0.50) against the
  benchmark's own per-action check, at zero tokens. About half of its misses
  are the right action with a wrong argument value, which a check on the order
  and presence of actions does not decide.
  (`results/task_type_conformance.json`)
- **Applied unchanged to the 15 agents per domain** that τ²-bench publishes:
  pooled precision 0.74 and recall 0.44 on airline, 0.89 and 0.31 on retail; on
  airline, precision is 0.94 once writes the task never asked for also count as
  violations.
  (`results/cross_agent.txt`)
- **LLM-as-a-Judge** (gpt-oss:120b and glm-5.3-flash) on the same airline runs:
  handed only the policy, the Judge aligns at chance (κ 0.09 and 0.10), and the
  two models agree with each other little more than chance; handed the
  solution list from which the label is computed, it reaches κ 0.95 and 0.97,
  an upper bound rather than a realistic Judge, at about 4,500 tokens per verdict.
  (`results/judge/two_judges.txt`)
- **Agent teams and sub-agents**: the pipeline turns AgentHallu's Magentic-One
  runs and the harness runs into OCEL logs with the agents as objects, and the
  sub-agent runs into logs where each sub-agent's tool calls are attributed to
  it. On the 10 sub-agent runs, 10 answers are correct and 9 runs conform; the
  tenth skipped the verification the task type requires.
  (`results/local_subagents/conformance.txt`)

## Data and licences

| corpus | in this repository | licence |
|---|---|---|
| [τ²-bench](https://github.com/sierra-research/tau2-bench) | the airline file of the Claude Opus 4.5 submission, in `examples/demos/`; the rest via `experiments/mirror_tau.py` | MIT |
| [AgentHallu](https://github.com/liuxuannan/AgentHallu) | the 94 Magentic-One runs, in `examples/demos/` | CC BY 4.0 |
| [TraceElephant](https://huggingface.co/datasets/TraceElephant/TraceElephant) | derived logs only; explored, not used in the thesis | CC BY 4.0 |
| harness runs, OpenTelemetry captures, event logs, verdicts, Judge outputs | `results/` | CC BY 4.0 (see `results/LICENSE.md`) |

The code is MIT licensed (see `LICENSE`), and the data this repository adds,
in `results/`, is licensed CC BY 4.0 (see `results/LICENSE.md`). Third-party
data keep their own licence; `examples/demos/README.md` and `results/README.md`
give the attribution. pm4py, used by the experiments only, is AGPL-3.0 and is
installed separately.

### Using the data without the code

The results are plain files that other tools read directly.

- **Event logs.** `results/ocel/*.ocel.json` holds one OCEL 2.0 log per corpus,
  which `pm4py.read_ocel2_json` reads; `results/logs/` holds the τ²-bench airline
  log flat, as CSV and XES for ProM, Disco or pm4py. Every report folder in
  `results/` has the same three files for its own runs.
- **Recorded runs.** The harness corpora in `results/local/`,
  `results/local_otel_live/` and `results/local_subagents/runs/` keep one JSON
  file per run, with every message, tool call, argument, result and reasoning
  text; `agentpm/ingest/local_magentic.py` documents the format. OpenTelemetry
  recordings are in `capture.jsonl` files, one OTLP/JSON payload per line.
- **Verdicts.** The engine's verdicts per run are in `verdicts.json`,
  `task_type_conformance*.json` and `cross_agent.json`; the Judges' verdicts are
  in `results/judge/verdicts_*.json`, with every prompt and raw response in
  `results/judge/raw_*.jsonl`.

## Conventions

- `agentpm/` stays standard library only; pm4py and pandas live behind the `pm`
  extra and inside `experiments/`.
- Deterministic output: sorted iteration, no wall clock in results. The OCEL
  output is byte-identical under different hash seeds (a test checks it), and
  the task-type verdicts hash identically under ten seeds
  (`results/pm4py/noise_sweep_seeds.txt`).
- A new ingester documents the schema it was verified against and gets a
  fixture-based test. Fixtures are small.
- Every reported number is produced by a script in `experiments/`. Negative
  results are reported in full.

## Citation

See [`CITATION.cff`](CITATION.cff), or cite the thesis.
