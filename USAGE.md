# How to use agent-pm-pipeline

Turn an agent's execution log into a process-mining event log, then check it.
Three commands, one flag that names the input format.

Everything below assumes the environment from the README is active
(`pip install -e ".[pm,dev]"`) and the repository root is the working folder.

## In a browser

```bash
python -m agentpm.cli web
```

opens http://127.0.0.1:8765. Three demo links at the top fill the form in, one
per corpus of the thesis, all bundled with the repository: τ²-bench airline
(`examples/demos/tau2_airline/`), AgentHallu's Magentic-One runs
(`examples/demos/agenthallu_magentic_one/`), and the harness runs with
sub-agents (`results/local_subagents/runs/`) with their reference model
(`experiments/local_agents/claim_check_reference.json`), checked with
activity names that include the agent. While a run goes, the run log shows each
step as the pipeline takes it; the same log is saved with the report.
Otherwise, drop a trace file on the page, or type a path on this machine for
the folder formats, and press *Run*. The format is
detected automatically. The result page shows what the source exposes, the
pipeline's pictures, the pm4py models, one run step by step, the conformance
verdicts, a preview of the event log, and downloads of every file. A reference
model and object rules can be pasted under *Reference model and options*. Reports are kept in `~/agentpm_reports` and listed on the start
page. The site listens on this machine only.

## In one command

```bash
python -m agentpm.cli report <file or folder> --format <format> -o <output folder>
```

That is the whole pipeline: it reads the source, checks what the source
exposes, writes the flat and object-centric event logs, runs conformance,
and draws the pm4py models and the pipeline's own pictures. Open
`<output folder>/README.md` for what each file is, and read `CHECKS.txt`
first. Lines marked `!!` there are the things to know before trusting a
result: no tool calls, no agent attribution, no arguments, real-time overlap
the order does not show, heavy variability.

Five options cover almost everything else:

| option | when |
|---|---|
| `--reference ref.json` | to check against your own reference model (see §4). Without it, a source that carries gold actions (τ²) is checked per task against that answer key, any other source is not checked. |
| `--task-types template.json` | to check every run against the model of its task type (see §4) |
| `--by-agent` | when the reference names agents: activities are checked as `Agent.activity`, and a sub-agent start as `Parent.spawn:Child`. The logs keep the plain tool names. |
| `--objects rules.json` | when the tool arguments name business objects (see §3). τ² has its rules built in. |
| `--no-pictures` | logs and checks only, fast, no pm4py or Graphviz needed |

`python -m agentpm.cli formats` lists the eight formats. For your own agents:
an AutoGen team exports OpenTelemetry with `run_magentic.py --otel-file
capture.jsonl` (§5), and any other framework that speaks the GenAI
conventions can post to `experiments/otel_capture/collector.py`; then `--format otel`.

For example, the sub-agent demo from the command line:

```bash
python -m agentpm.cli report results/local_subagents/runs --format local_magentic -o report_subagents \
    --reference experiments/local_agents/claim_check_reference.json --by-agent
```

The sections below are the separate steps, for when you need one of them alone.

## Setup, once

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

That is enough for the numbered steps below. `pip install -e ".[pm]"` adds
pm4py and pandas for `report`'s pm4py models, the website's pictures and the
experiments.

## 1. See what you have

```bash
python -m agentpm.cli formats
```

```
  generic          file  a trace already in the canonical JSON shape
  openai           file  OpenAI Agents SDK export JSONL (traces and spans)
  claude           file  Claude Code session transcript JSONL
  otel             file  OpenTelemetry: GenAI semantic-convention spans, or Claude Code's log dialect
  tau2             file  tau2-bench trajectory JSON, one file per model and domain
  traceelephant    dir   TraceElephant: a directory of run folders
  agenthallu       dir   AgentHallu: a directory of run files
  local_magentic   dir   runs recorded by experiments/local_agents/run_magentic.py
```

`file` formats take one file, `dir` formats take the folder that holds the runs.

## 2. Count what is in a source

```bash
python -m agentpm.cli inventory examples/demos/tau2_airline/claude-opus-4-5_high_airline_gpt-5.2_4trials.json --format tau2
```

```
tau2: 200 traces, 3990 events, 15 activities, 2 resources
  activities: assistant_message(1438), user_message(976), get_reservation_details(583), ...
  resources:  assistant(3014), user(976)
```

Do this first with any new source. If the activity list is only `orchestrate`
and `report`, the source is not exposing its tool calls and nothing downstream
will mean much.

## 3. Write the event log

```bash
python -m agentpm.cli ocel examples/demos/agenthallu_magentic_one --format agenthallu -o magentic.ocel.json
```

Produces OCEL 2.0 JSON. Open it with pm4py:

```python
import pm4py
log = pm4py.read_ocel2_json("magentic.ocel.json")
```

Every event relates to its `run`, its `agent` and its `tool`. Business objects
come from **object rules**, declared per source as data: which object type an
identifier in the recorded arguments or results names. τ²-bench ships with
rules for its airline domain (`user`, `reservation`, `flight`, `payment`,
`route`), so this is enough:

```bash
python -m agentpm.cli ocel examples/demos/tau2_airline/claude-opus-4-5_high_airline_gpt-5.2_4trials.json --format tau2 \
    --exclude assistant_message user_message -o airline.ocel.json
```

For another source, write the rules as a JSON list and pass `--objects`:

```json
[{"type": "ticket", "path": "arguments.ticket_id"},
 {"type": "ticket", "path": "result.id", "activity": "create_ticket"},
 {"type": "file",   "path": "arguments.files[].name"}]
```

`path` walks the event's attributes with dots, `[]` fans out over a list, and
a JSON string is decoded on the way. `--no-objects` keeps only run, agent and
tool; `--exclude` drops activities such as the communication turns. The rule
format is documented in `agentpm/logbuild/objects.py`.

**Time in the log.** Events the source issued together, such as the tool
calls of one model turn, carry an identical timestamp in the OCEL file; the
event id carries the serialisation position. The flat XES view serialises
instead, because XES is a total order by definition. Note that pm4py's
discovery sorts by time and takes consecutive pairs, so it serialises too;
`experiments/partial_order.py` measures that, and `agentpm.analysis.discovery`
has a directly-follows graph that respects the ties.

## 4. Check conformance

With a reference model you wrote:

```bash
python -m agentpm.cli run runs.json --format generic --reference examples/reference_model.json -v
```

With the per-task gold actions a tau2 file carries:

```bash
python -m agentpm.cli run examples/demos/tau2_airline/claude-opus-4-5_high_airline_gpt-5.2_4trials.json --format tau2 --gold -v -o verdicts.json
```

```
172 traces checked, 147 compliant, mean fitness 0.9513
  ...
  VIOLATION 52772308-70ea-4b22-8c24- fitness 0.71
       required(calculate)
       precedence(calculate,update_reservation_baggages) at event 11
```

Every violation names its constraint and, where it is localisable, the event.

With one model per task type, all built from one template declared as data. A run
that states `task_type` on its outcome needs nothing else; otherwise the type is read
from the changes on the task's solution list. `examples/` holds the templates of the
two τ² domains, and a new domain brings its own file:

```bash
python -m agentpm.cli run examples/demos/tau2_airline/claude-opus-4-5_high_airline_gpt-5.2_4trials.json --format tau2 --task-types examples/task_types_tau2_airline.json
```

### Writing a reference model

A JSON file with any of these keys. All are optional.

```json
{
  "name": "cancellation",
  "required":        ["get_reservation_details", "cancel_reservation"],
  "precedence":      [["get_reservation_details", "cancel_reservation"]],
  "directly_before": [["user_message", "cancel_reservation"]],
  "transparent":     ["cancel_reservation", "assistant_message"],
  "forbidden_after": [["transfer_to_human_agents", "cancel_reservation"]],
  "allowed":         ["get_user_details", "get_reservation_details", "cancel_reservation"],
  "end_activities":  ["assistant_message"]
}
```

| key | meaning |
|---|---|
| `required` | each must occur at least once |
| `precedence` | `[a, b]`: if b occurs, a occurred before it |
| `directly_before` | `[a, b]`: every b has a right before it, looking through `transparent` |
| `transparent` | activities `directly_before` skips over (a batch of writes after one confirmation) |
| `forbidden_after` | `[a, b]`: b must not occur after a |
| `allowed` | closed world: nothing else may occur |
| `end_activities` | the run must end with one of these |

Fitness is the share of constraints satisfied, one unit each, so it does not
depend on run length.

## 5. Feeding in OpenTelemetry

Point the agent at the bundled collector, then ingest what it wrote:

```bash
python experiments/otel_capture/collector.py --out capture.jsonl        # terminal 1
# run the agent with OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 ...
python -m agentpm.cli inventory capture.jsonl --format otel        # terminal 2, afterwards
```

Two dialects are recognised automatically:

- **GenAI semantic-convention spans** (`execute_tool`, `invoke_agent`,
  `invoke_workflow`, `chat`). Tool spans are children of the agent that ran
  them, so every tool call is attributed to its agent, nested sub-agents come
  out with their own resource, and overlapping sub-agent lifetimes are
  preserved from the real span times. This is the full-fidelity path.
- **Claude Code's log dialect** (`tool_result`, `subagent_completed`, ...).
  Tool names, durations and cost, ordered by `event.sequence`. Sub-agent spawns
  and returns are events, but this dialect names no agent on tool events, so a
  sub-agent's tool calls cannot be assigned to it. That is a property of the
  source.

The conventions are still marked *Development* upstream and may change; the
attribute names relied on are listed in `agentpm/ingest/otel.py`.

### A framework that speaks the conventions: AutoGen

AutoGen 0.7 instruments itself against the GenAI conventions: one
`invoke_agent` span per agent turn, one `execute_tool` span per tool call
nested in it. The Magentic-One harness can export them:

```bash
# in the harness environment: pip install -r experiments/local_agents/requirements.txt
python experiments/local_agents/run_magentic.py --model gpt-oss:120b --otel-file otel_capture.jsonl
python -m agentpm.cli inventory otel_capture.jsonl --format otel
```

`--otel-endpoint http://127.0.0.1:4318` posts the same spans to the bundled
collector instead of, or as well as, the file. The exporter is
`experiments/local_agents/otel_export.py`, OTLP/JSON over the official
protobuf encoder; it is off unless one of the two flags is given.

To see the route work without a model endpoint:

```bash
python experiments/local_agents/smoke_otel.py
```

runs two AutoGen agents on scripted model responses through the collector
and the ingester and prints the same run twice, once from the harness's
message stream and once from the spans. Tool names, order and acting agents
agree. **Arguments and results do not arrive over this route**: the
convention makes `gen_ai.tool.call.arguments` opt-in and AutoGen 0.7.5 does
not set it. Keep the harness recording when you need them.

### Try a random run and see every output

```bash
# in the harness environment
python experiments/local_agents/random_otel_run.py --runs 6 --seed 1 --out results/otel_demo
# in the pipeline environment
python -m agentpm.cli report results/otel_demo/capture.jsonl --format otel -o results/otel_demo \
    --reference experiments/otel_demo_reference.json
```

The first command runs four AutoGen agents in a Swarm (an orchestrator that
hands work to a file surfer, a coder and an analyst with real tools) on a plan
drawn from the seed, with the framework's own OpenTelemetry spans captured.
Change `--seed` for different runs. The second turns the capture into the
flat event log (CSV and XES), the OCEL 2.0 log, a conformance check against
`experiments/otel_demo_reference.json`, pm4py's DFG, Petri net, process tree,
BPMN, object-centric DFG and Petri net, and the pipeline's own pictures: the
handover network, the lane model and one picture per run. `README.md` in the
folder says what each file is. The same report works on any `otel` capture,
including a live `run_magentic.py --otel-file` run.

## 6. Adding a new source

One file in `agentpm/ingest/`, about 150 lines, following `tau2.py`:

1. parse the raw format
2. every executed tool call -> `Event(activity=tool name, timestamp=position, resource=who, attributes={arguments, result, ...})`
3. every run -> `Trace(case_id, events, outcome)`
4. a hand-built fixture test in `tests/`
5. one line in `FORMATS` in `agentpm/cli.py`

Rules that every ingester keeps: reasoning text is an attribute, never an
activity; tool results attach to their call rather than becoming events; the
self-reported outcome is a claim to check, not ground truth; and the schema the
ingester was verified against is written in its docstring.

## Where results come from

Everything under `results/` is produced by a script in `experiments/`; the
command is in each script's docstring. `experiments/README.md` lists every
script with what it writes, and `results/README.md` says what each result file
is.
