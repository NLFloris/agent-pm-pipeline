# The Magentic-One harness: a self-instrumented multi-agent corpus

Magentic-One, the same framework as the TraceElephant magentic subset, driven by
a model running on your own machine. Because the harness is ours, the recording
keeps everything the other sources drop.

| | τ²-bench | TraceElephant | Claude Code telemetry | **here** |
|---|---|---|---|---|
| which agent acted | single agent | yes | **no** | yes |
| tool arguments | yes | yes | **byte sizes only** | yes |
| reasoning text | no | no | **redacted** | yes |
| real durations | no | no | yes | yes |
| cost per run | API cost | n/a | yes | free |
| runs available | fixed | 220, failures only | metered | unlimited |

## Two ways to serve the model

**BFH-TI MLMP inference API** (recommended). The same model line-up, running on
their hardware instead of your laptop, so the 120b and 27b models are usable and
your machine stays free. OpenAI-compatible, so the runner needs no code change.

```bash
export LLM_BASE_URL=https://inference.mlmp.ti.bfh.ch/api/v1
export LLM_API_KEY=<Settings, Account, API keys, press Show>
```

From outside the BFH network use `https://inference.proxy.ti.bfh.ch/api/v1`
instead. Both hosts answer from a normal connection; without a key they return
`401 Not authenticated`, which is the quickest way to check reachability. Keys
expire after 180 days.

Two things to respect: the service asks that no sensitive or personal data be
uploaded, which the sandbox task suite satisfies since every file is synthetic;
and the key belongs in the environment, never in a command line that lands in
your shell history.

**LM Studio**, locally. Developer tab, load a model, Start Server, default
`http://localhost:1234/v1`. Useful when offline or when you want unlimited runs
without touching a shared service. Slower, and the larger models may not fit.

## Setup, once

```bash
python3 -m venv ~/venvs/localagents
source ~/venvs/localagents/bin/activate
pip install -r experiments/local_agents/requirements.txt
```

Then start a server in **LM Studio**: Developer tab, load a model, Start Server.
The default endpoint is `http://localhost:1234/v1`. Note the model id the server
reports; that string is what `--model` needs.

Any served model works as long as it does tool calling. On the hosted endpoint,
start with `gpt-oss:120b`: tool-calling discipline is what makes or breaks a
multi-agent run, and the larger model is markedly steadier at it. Running
locally, prefer `qwen3.8-flash-next`, since many turns make throughput matter
more than raw capability. `apertus-v1.5-8b` will not work at all, it has no tool
use.

Running the same task suite on two different models is worth doing on its own:
it gives a controlled comparison of process structure, which is exactly the
question process mining is for.

## Running

```bash
source ~/venvs/localagents/bin/activate
python experiments/local_agents/run_magentic.py --model gpt-oss:120b
```

The endpoint and key come from `LLM_BASE_URL` and `LLM_API_KEY`, or pass
`--base-url` and `--api-key` explicitly.

Eight tasks, each in a fresh sandbox seeded with `sales.csv`, `notes.md` and
`calc.py`. No network, no private content. Use `--only l01_read_csv` for a single
task, `--base-url` for a different server, `--max-turns` to cap a run that will
not terminate.

Output lands in `~/local_agent_runs/<task_id>.json`, one file per run, plus the
sandbox as the team left it.

## The team

Three `AssistantAgent`s under the Magentic-One orchestrator, each with explicit
Python tools:

| agent | tools |
|---|---|
| FileSurfer | `list_files`, `read_file` |
| Coder | `write_file`, `run_python` |
| Analyst | `read_file`, `run_python` |

The overlap is deliberate: two agents can read and two can run code, so the
orchestrator has a real choice to make and the discovered model says something
about how it chooses.

AutoGen's own `FileSurfer` and `CodeExecutorAgent` were tried first and
rejected. They resolve their tool use internally and surface only a
`TextMessage`, so a full eight-task run came back as nothing but `orchestrate`
and `report`: 165 events, zero named actions, an activity alphabet of two.
`AssistantAgent` emits `ToolCallRequestEvent` and `ToolCallExecutionEvent`,
which is what gives the log something to mine.

`run_python` executes in the sandbox with a 30 second cap, and every path is
checked against the sandbox root before it is opened. The web surfer is
deliberately absent: it needs a browser stack, and a local model driving a
browser fails for reasons that say nothing about process structure.

## Task shapes

`tasks.json` declares the collaboration each task is meant to elicit, so a claim
about the discovered model can be checked against what was asked rather than
read off the result.

| shape | tasks | intent |
|---|---|---|
| `single` | l01 | one specialist should suffice |
| `relay` | l02, l03, l07 | one agent finds something, another acts on it |
| `verify` | l04, l06 | work is produced, then independently checked |
| `loop` | l05, l08 | the answer needs more than one pass |

## Sub-agents: ten runs of one task type

With `--subagents` the Analyst has no tools of its own but two sub-agents, each
wrapped as a tool (AutoGen's `AgentTool`) and each with its own tools:

| sub-agent | started by | tools |
|---|---|---|
| DataChecker | Analyst | `list_files`, `read_file` |
| Recalculator | Analyst | `run_python` |

`tasks_claims.json` holds ten tasks of one type, `claim_check`: check a claim
about `sales.csv` and end with `VERDICT: TRUE` or `VERDICT: FALSE`. One
reference model covers all ten (`claim_check_reference.json`), written and
hashed before the first run (`results/frozen/claim_check_reference.sha256`).

```bash
source ~/venvs/localagents/bin/activate
export LLM_BASE_URL=https://inference.proxy.ti.bfh.ch/v1 LLM_API_KEY=$(cat ~/.bfh_api_key)
python experiments/local_agents/run_magentic.py --model gpt-oss:120b --subagents --no-cache \
    --tasks experiments/local_agents/tasks_claims.json --out ~/local_agent_runs_subagents
cp ~/local_agent_runs_subagents/c*.json results/local_subagents/runs/
python experiments/local_agents/check_claims.py > results/local_subagents/conformance.txt
python -m agentpm.cli report results/local_subagents/runs --format local_magentic -o results/local_subagents/report
```

`--no-cache` matters on the BFH proxy, which otherwise answers an identical
request from its cache. A sub-agent's start becomes `spawn` by the parent with
the sub-agent on `spawns`; its own events carry `spawned_by`.

## Ingesting

```python
from agentpm.ingest import local_magentic
traces = local_magentic.ingest_dir("~/local_agent_runs")
```

A tool call becomes its function name, a speaking turn becomes `orchestrate` or
`report`, and a speaker selection becomes `delegate` with the chosen agent on
the `to` attribute. Reasoning is attached to the next event from that agent and
is never an activity, following decision L6.

## Expect failures

A local model will produce malformed tool calls, stall, and sometimes never
terminate. The harness records the error and moves on rather than dropping the
run. For this thesis that is useful: the evaluation needs runs that went wrong,
and these are genuine failures rather than seeded ones.
