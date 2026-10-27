# Capturing OpenTelemetry: a collector and a Claude Code run suite

Two tools for recording a live agent as OpenTelemetry.

- `collector.py` is a minimal OTLP/JSON collector. Anything that exports
  OTLP over HTTP with `OTEL_EXPORTER_OTLP_PROTOCOL=http/json` can post to it,
  and it appends every payload verbatim to a JSONL file that
  `python -m agentpm.cli ... --format otel` reads. The AutoGen route in
  `experiments/local_agents/` (`smoke_otel.py`, `run_magentic.py
  --otel-endpoint`) uses it too.
- `run_suite.py` with `tasks.json` drives Claude Code (`claude -p`) through ten
  short tasks in throwaway sandboxes, to capture nested and concurrent
  sub-agent execution from a real agent. Two of its runs, with personal fields
  removed, are in `results/otel_probe/`; `claude_code_capture.txt` there is the
  readable version.

## What this does and does not touch

Telemetry is enabled **only for the agent subprocesses this driver starts**. The
environment variables are set on the child process, never exported into your
shell, so no other Claude Code session is recorded. Each run executes inside a
freshly created sandbox directory seeded with three small throwaway Python
files, so the capture contains no private project content and the suite is
reproducible by anyone with this repository.

Run it from somewhere outside the thesis working tree if you want the sandboxes
out of the way:

```bash
python experiments/otel_capture/run_suite.py --out ~/claude_code_runs
```

## How to run it

Two terminals.

```bash
# 1. start the collector
python experiments/otel_capture/collector.py --out ~/claude_code_capture.jsonl

# 2. run the suite
python experiments/otel_capture/run_suite.py --out ~/claude_code_runs
```

Stop the collector with ctrl-c when the suite finishes. Roughly twenty minutes
for ten runs. Use `--only t04_fanout_review t06_chain_research` to run a subset,
and `--dry-run` to seed the sandboxes and print the plan without calling the
agent.

No API key is needed: `claude -p` uses your existing Claude Code authentication.

## What comes out

- `~/claude_code_capture.jsonl` — one line per OTLP payload, with the signal name and
  receive time. Raw and re-parsable, so the mapping can change without
  recapturing.
- `~/claude_code_runs/runs.json` — per run: task id, the control-flow **shape** the
  task was designed to elicit, the start and end timestamps, exit code and final
  text. The timestamps are what attribute a captured span to a run.
- `~/claude_code_runs/<task_id>/` — the sandbox as the agent left it.

## The task suite

Ten tasks across four shapes, declared in `tasks.json` so a later claim about
the log can be checked against what the task asked for:

| shape | tasks | what it should produce |
|---|---|---|
| `linear` | t01–t03 | one agent, a sequence of tool calls |
| `fanout` | t04, t05, t10 | several sub-agents launched in one turn, running concurrently |
| `chain` | t06, t07 | sub-agents spawned one after another, each using the previous result |
| `mixed` | t08, t09 | a fan-out followed by a step that depends on all of it |

The fan-out tasks are the point. If the capture shows overlapping sub-agent
spans, that is a recorded fact about how the agent schedules work, and it is the
evidence that a single-case event log cannot represent an agent run faithfully.

## What the capture showed

Claude Code exports OpenTelemetry **logs and metrics, not spans** (version
2.1.278, October 2026). Tool events carry the tool name, duration and size, but
no arguments and no agent attribution, so a sub-agent's tool calls cannot be
assigned to it. `agentpm/ingest/otel.py` reads this dialect next to the GenAI
semantic-convention spans; its docstring lists the attributes relied on, and
`tests/fixtures/claude_code_otel_subagents.jsonl` holds the log records of run A.
