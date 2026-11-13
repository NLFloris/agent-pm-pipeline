# Process mining results over three agent-trace corpora

Written 2026-10-01. Every number below is produced by a script in `experiments/`
and regenerable with the commands in [§8](#8-how-to-regenerate-everything).
Negative results are reported in the same detail as positive ones; three of the
findings here are negative and two of those change what the thesis can claim.

> **Note 2026-10-26.** Parts of this report are older than the validation round of
> 23 to 26 October; `results/VALIDATION.md` supersedes them. In particular, the
> judge's "byte-identical repeats" in §10 were cache copies of the endpoint, cost
> is now reported per verdict (about 4,400 tokens for the required-actions judge,
> not 22,221 per run), TraceElephant and AgentHallu Camel are no longer used in
> the thesis.
> pm4py's inductive miner at noise 0.2 builds a different net under different
> Python hash seeds (`results/pm4py/noise_sweep_seeds.txt`), so `noise_sweep.py`
> now fixes PYTHONHASHSEED=0; the confirmation constraint of the task-type models
> reads the listing of calls issued together and changes a few verdicts under
> shuffling (`results/task_type_details.txt`, section 4).
> Chapters 4 to 6 of the thesis were rewritten against the current results.

> **Note 2026-11-06.** This is a dated working report, kept as a record of how the
> results came about; the thesis and the output files named here have the final
> numbers. Chapter *n* of the drafts is Section *n* of the thesis. Since
> `noise_sweep.py` fixes PYTHONHASHSEED=0, its noise-0.2 net lacks four
> activities (`results/pm4py/noise_sweep.txt`).

> **Note 2026-11-08.** The three statements below that still contradicted the
> thesis are corrected in place and marked with this date: the noise-0.2 net in
> §10.1b, the reasons for the 22 misses in §10.1b, and the counts of §10.2.

Environment: Python 3.12, venv at `~/venvs/agentpm`, pm4py 2.7.23.8, pandas
2.3.3, graphviz 0.21 with Graphviz 16.1.0. Test suite 32 passed (79 on 8 November 2026).

---

## 1. What is in each corpus

Four sources, each answering a different question. They are not interchangeable
and no result below mixes them.

**τ²-bench airline** is a customer-service benchmark. One run is one
conversation between a simulated customer and a tool-using agent, over an
airline booking database. It is here because it ships a **reference model**: a
written policy, and per task a list of the actions a correct run must contain.
That is what makes conformance checking possible without the researcher writing
the rules.

**TraceElephant** is a failure-attribution benchmark (Chen et al., ACL 2026). One
run is one GAIA or SWE-bench task solved by a **team of agents**. Three expert
annotators marked, per run, which agent and which step made the failure
inevitable. It is here for that human ground truth and for the multi-agent
structure, and because it is the only peer-reviewed source of the four. Only the
220 failed runs of the paper's 380 executions are released.

**AgentHallu** is a hallucination corpus. One run is one question answered by a
multi-agent framework, with a human label for the step that hallucinated. It is
here as a second multi-agent topology to compare against, since Camel and
Magentic-One coordinate in structurally opposite ways.

**Local Magentic-One** is ours. Eight runs of AutoGen's Magentic-One over a
seeded sandbox, driven by `gpt-oss:120b` on the BFH-TI inference API, recorded by
our own harness. It is here because it keeps what the public corpora drop: the
acting agent, the full tool arguments and the model's reasoning text, all at
once.

| | τ²-bench airline | TraceElephant | AgentHallu | Local Magentic-One |
|---|---|---|---|---|
| what one run is | a conversation | a task solved by a team | a question answered | a task solved by a team |
| runs used | 200 | 191 | 187 | 8 |
| events | 3,990 | 5,446 | 1,800 | 109 |
| reference model shipped | **yes**, policy + gold actions | no | no | no |
| outcome label | `reward` 0/1 | none, failures only | none | none |
| process label | gold action checks | **expert agent + step** | human hallucination step | declared task shape |
| multi-agent | no | yes | yes | yes |
| tool arguments | yes | yes | yes | yes |
| reasoning text | no | no | no | **yes** |
| real durations | no | no | no | **yes** |
| peer reviewed | no | **yes, ACL 2026** | no | n/a |
| licence | MIT / public bucket | CC BY 4.0 | CC BY 4.0 | ours |

Provenance:

- **τ²-bench** — `sierra-tau-bench-public` S3 bucket, submission
  `claude-opus-4-5_sierra_2026-02-26`, file
  `claude-opus-4-5_high_airline_gpt-5.2_4trials.json` (19.9 MB). Mirrored
  2026-09-22 with sha256 manifest; 79 trajectory files / 6.19 GB across 35 text
  submissions. The bucket is leaderboard infrastructure, not a versioned
  release, so the mirror is the citable copy.
- **TraceElephant** — `huggingface.co/datasets/TraceElephant/TraceElephant`,
  `data.zip` (597 MB), 220 runs. Chen, Wang, Mu, Wang, Liu, Feng, Wang,
  *Seeing the Whole Elephant*, ACL 2026, arXiv 2604.22708.
- **AgentHallu** — `github.com/liuxuannan/AgentHallu`, the `Magentic_One` (94)
  and `Camel` (93) subsets.
- **Local Magentic-One** — generated 2026-10-13 by
  `experiments/local_agents/run_magentic.py` against
  `https://inference.mlmp.ti.bfh.ch/api/v1`, model `gpt-oss:120b`, 16 turn cap.
  Runs committed under `results/local/`.

---

## 2. Ingestion (R1: one canonical schema, several sources)

Three ingesters, each mapping a different raw format onto the same `Trace` /
`Event` model: `agentpm/ingest/tau2.py`, `traceelephant.py`, `agenthallu.py`.
Schemas were verified by reading the real files, not the papers. Fixture-based
tests assert determinism per ingester.

### 2.1 τ²-bench airline

```
runs 200   events 3990   mean 19.9/run   activity alphabet 15
top: assistant_message(1438) user_message(976) get_reservation_details(583)
     search_direct_flight(312) get_user_details(191) search_onestop_flight(100)
     get_flight_status(94) update_reservation_flights(69)
resources: assistant 3014, user 976
labels: 168 pass / 32 fail  (base pass rate 84%)
```

### 2.2 TraceElephant

| subset | runs | events | mean/run | alphabet |
|---|---|---|---|---|
| magentic-runs-gaia | 74 | 2,288 | 30.9 | 15 |
| captain-runs-gaia | 73 | 1,559 | 21.4 | **3** |
| swe-agent-runs-swe-bench | 44 | 1,599 | 36.3 | 5 |

Agents observed: WebSurfer (1,240 events), Orchestrator (772), FileSurfer (193),
Coder (73) in magentic; Orchestrator (884), WebServing_Expert (290),
InformationExtraction_Expert (279), Verification_Expert (61),
DataVerification_Expert (27), CSVHandling_Expert (18) in captain, whose expert
team is assembled per task.

In the SWE-Agent subset `agent_name` names the *tool* (`str_replace_editor`
1,114, `bash` 443, `submit` 32), because that system is single-agent. Any
cross-subset claim about resources has to account for that.

### 2.3 AgentHallu

| subset | runs | events | alphabet | agents |
|---|---|---|---|---|
| Magentic_One | 94 | 975 | 12 | 4 |
| Camel | 93 | 825 | 20 | 6 |

Magentic-One activities include the control actions this ingester derives from
the orchestrator's text: `delegate` (386), `orchestrate` (197), `report` (144),
alongside real tool calls `click` (60), `PythonInterpreterTool` (45),
`scroll_down` (43), `web_search` (28).

### 2.4 Local Magentic-One

```
runs 8   events 109   mean 13.6/run   activity alphabet 6
orchestrate 46, report 36, read_file 13, list_files 8, run_python 5, write_file 1
agents: MagenticOneOrchestrator 46, FileSurfer 42, Coder 12, user 8, Analyst 1
46 to 132 seconds per run, no harness errors
```

An earlier eight-run attempt produced 165 events and **zero** tool calls. AutoGen's
own `FileSurfer` and `CodeExecutorAgent` resolve their tool use internally and
surface only a text message, so the whole corpus came out as `orchestrate` and
`report`, an alphabet of two. Replacing them with `AssistantAgent`s carrying
explicit Python tools fixed it. The failed attempt is kept, because it is a
concrete instance of a general hazard: an agent framework can do real tool work
and still emit nothing tool-shaped, and a pipeline that trusts the stream will
silently record a process that does not exist.

---

## 3. Conformance (R3, R4, R5) — τ²-bench airline

Conformance needs a reference model. Two were tried on the same 200 runs with
the same engine on the same day. **Where the reference comes from turned out to
matter more than anything else measured in this document.**

### 3.1 Hand-transcribed policy reference — null result

Twelve constraints transcribed by hand from
`data/tau2/domains/airline/policy.md` into the DECLARE-style `Reference`
vocabulary (`allowed`, `precedence`, `forbidden_after`).

| | mean fitness |
|---|---|
| reward = 1 (n=168) | 1.0000 |
| reward = 0 (n=32) | 1.0000 |
| **separation** | **0.0000** |

Fully compliant: 200 / 200; no constraint ever fires. Until 2026-10-30 the
reference had a thirteenth constraint, `end_activities` (the conversation ends
with the agent), which the policy does not state. It fired on all 200 runs,
because every simulation terminates with `termination_reason: user_stop` so the
user simulator always has the last word, and it only lowered every run to
0.9231. It was removed; the separation is 0 either way.

Two things follow. The transcription carries no signal whatsoever, and nothing
in the output warns you: a wrongly chosen constraint produces a *constant*,
which is indistinguishable from a well-behaved metric until you split it by
label. This reproduces the same null result found on τ-bench in July with a
different model, dataset and transcription.

### 3.2 Gold-action reference — real signal

Built mechanically per task from `evaluation_criteria.actions`: `required` is
the set of agent-side gold action names, `precedence` their consecutive pairs.
No interpretation by the author.

| | mean fitness | n |
|---|---|---|
| reward = 1 | 0.9730 | 142 |
| reward = 0 | 0.8486 | 30 |

Runs with gold actions: 172 / 200 (the other 28 are refusal tasks with no
reference actions). As a failure predictor: **precision 64%, recall 53%, against
a 17% base rate** — roughly a 3× lift from control flow alone, no LLM involved.
(Until 2026-10-20 this read 53%/53% with 30 runs flagged: the engine then read
precedence off the serialised order, and five runs were flagged for issuing two
lookups or two writes in one turn in the opposite order from the gold list. See §3.5.)

### 3.2b Does it beat features that need no reference model?

A signal is only worth its machinery if something cheaper does not match it. On
the same 172 runs:

| predictor of failure | precision | recall | F1 | runs flagged |
|---|---|---|---|---|
| **gold-action reference not satisfied** | **64%** | **53%** | **0.58** | **25** |
| more tool calls than the median (7) | 29% | 80% | 0.42 | 84 |
| repeated tool use above the median (38%) | 26% | 73% | 0.38 | 85 |
| any tool returned an error | 75% | 10% | 0.18 | 4 |
| always predict failure | 17% | 100% | 0.30 | 172 |

Conformance wins on F1 and wins decisively on precision, and it flags 30 runs
where the best trivial feature flags 84. A length threshold reaches high recall
by suspecting half the corpus, which is not a usable verdict.

One further check, because a reviewer will ask whether this is process mining or
a checklist. Of the 44 violations raised, 28 are missing required activities and
**16 are ordering violations**, so 36% of the signal comes from constraints a
set-membership check cannot express.

**Reading:** a reference the dataset ships beats a reference the researcher
writes, decisively. This is the empirical justification for choosing τ² over
datasets that ship only a prose policy, and it is the strongest argument in this
document for the dataset-selection criterion "a formal reference you did not
author."

### 3.3 Worked example

Case `6abdb100-84f6-4ed3-b984-ccac652020c6`, task 39 trial 3, reward 0. Gold
actions: `get_user_details`, seven `get_reservation_details`, three
`cancel_reservation`. What the trace contains:

```
 0 [assistant] assistant_message       "Hi! How can I help you today?"
 1 [user     ] user_message            "Bonjour, I'm Amelia Davis. I'd like to cancel…"
 2 [assistant] get_user_details        {'user_id': 'amelia_davis_8890'}
 3 [assistant] get_reservation_details {'reservation_id': '8C8K4E'}
 … seven lookups …
13 [assistant] assistant_message       "based on the cancellation policy…"
14 [user     ] user_message            "Yes, please cancel LU15PA and 8C8K4E."
```

`cancel_reservation` never occurs. The required-activity constraint catches it
and localises the verdict to the absence, which is exactly the R5 claim: the
verdict names a constraint, not an opinion.

### 3.4 Determinism (R4), measured

The full set of 200 verdicts was computed three times from the source file and
hashed. All three repetitions give the identical SHA-256, so the signal is
deterministic by measurement and not only by construction. Section 3.5 qualifies
what that determinism is relative to.

### 3.5 Concurrency, and what flattening it costs

An event log is a set of sequences, and a sequence assumes that of any two
events one preceded the other. Agent traces do not always satisfy that. A model
may issue several tool calls in one turn, and those calls carry no order with
respect to one another. Process mining distinguishes this case, concurrency,
from interleaving, and a directly-follows graph cannot express it.

**How common it is**, counted per turn:

```
tool calls per turn: 1:441  2:134  3:43  4:54  5:24  6:11  7:18  8:3  9:6  10:4  11:4  12:4
turns with more than one call:        305 / 746
runs containing a parallel turn:      170 / 200   (85%)
```

It is not universal, and the variation is itself a finding: Captain-Agent never
issues more than one call per step, the local corpus is strictly sequential
because its orchestrator selects one speaker per turn, and TraceElephant's
Magentic-One subset sits in between at 42 of 74 runs. A pipeline can assume
neither case.

**What it costs the model.** Classifying every directly-follows relation by
whether its two events came from the same turn:

```
across turns (real succession)        547  (40%)
within one turn (arbitrary order)     830  (60%)
distinct edges in the model            81
edges that exist ONLY because of within-turn serialisation   6
```

**Sixty percent of the succession relations in the discovered model are
artefacts of representation.** One of the six spurious edges runs from
cancelling a reservation to reading one, which no execution performed in that
order.

**What it costs the verdict.** Calls issued in one turn were decided together,
before any result was seen; the harness executes them in the listed order (all
305 turns), but the agent did not choose that order on information. An engine
that read precedence off the serialised order, as the first version did,
changed its verdict for **9 of 172 runs** when the within-turn order was
reshuffled. Since 2026-10-20 the engine reads order by step: *a before b* holds
when some *a* is in the same step as the first *b* or an earlier one, and
*b not after a* is violated only by a *b* in a later step. The same reshuffle
now changes **0 of 172** verdicts, by construction, and five answer-key false
alarms disappeared (κ 0.57 → 0.65 on that reference; the task-type model of
Chapter 6 is unchanged at κ 0.50). Determinism under a fixed serialisation is
not the same as being well defined; the partial order has to reach the engine,
not only the log.

**Limitation.** This is concurrency within one agent. Nested concurrency, one
agent delegating to several that then act simultaneously, was observed while
instrumenting Claude Code, where two delegated agents ran 16.3 and 15.2 seconds
with overlapping lifetimes and interleaved tool calls. That telemetry does not
attribute a tool call to the agent that issued it, so no corpus here combines
nested delegation, concurrency and attribution. The question is identified, not
answered.

---

## 4. Discovery and performance (R7)

### 4.1 Variants

τ²-bench airline: **111 distinct tool-call variants over 200 runs, 68 of them
singletons.** The most frequent variant covers 10 runs:
`get_user_details → get_reservation_details → transfer_to_human_agents`.

The unfiltered directly-follows graph has 81 edges over 13 activities and is
unreadable — requirement R9 (trace variability producing spaghetti) showing up
in practice on the first dataset tried. Every figure in `results/figures/` is
therefore frequency-filtered, with the threshold stated in the caption.

### 4.2 Repeated tool use and self-loops

Repeated tool use = 1 − (distinct activities / activities). Self-loop = share of
consecutive identical pairs.

**τ²-bench airline, split by harness label:**

| | n | tool calls | repeated tool use | self-loop |
|---|---|---|---|---|
| reward = 1 | 161 | 7.4 | 33.6% | 36.3% |
| reward = 0 | 32 | **11.7** | **53.0%** | **52.2%** |

Failing runs make 58% more tool calls and repeat over half of them. This is a
secondary-lens result (performance, not conformance) and it separates the labels
without any reference model at all.

**AgentHallu, split by the human hallucination label:**

| subset | label | n | calls | repeated tool use | self-loop |
|---|---|---|---|---|---|
| Magentic_One | no hallucination | 25 | 4.2 | 19.4% | 21.8% |
| Magentic_One | hallucination | 25 | 4.6 | 26.4% | 34.8% |
| Camel | no hallucination | 26 | 6.1 | 30.9% | 32.4% |
| Camel | hallucination | 43 | 7.4 | 34.4% | 29.0% |

Weak and inconsistent. On an earlier whole-corpus run the two groups were
identical to the first decimal on every structural feature. **Hallucination is a
property of what an agent says, not of the path it walks**, so control-flow
features should not be expected to predict it. This is a scope boundary for the
thesis, not a failure of the method.

---

## 5. Organisational perspective: handover of work

Counting transitions where the acting resource changes.

**AgentHallu Magentic-One** — a star, perfectly reciprocated:

| from → to | out | back |
|---|---|---|
| Orchestrator → WebSurfer | 215 | 215 |
| Orchestrator → Coder | 132 | 132 |
| Orchestrator → ComputerTerminal | 45 | 45 |

6 of 6 edges reciprocated: every delegation is answered, none left open.

**AgentHallu Camel** — a one-way pipeline: Planner → Coordinator (95),
Coordinator → Search Agent (58), Coordinator → Developer Agent (35), Search
Agent → Developer Agent (11). **0 of 10 edges reciprocated.** No work flows back.

**TraceElephant Captain-Agent** — also reciprocated: Orchestrator ↔
WebServing_Expert 289/289, Orchestrator ↔ InformationExtraction_Expert 279/279.

**Local Magentic-One** — a reciprocated star again, and one routing result:

| from → to | out | back |
|---|---|---|
| Orchestrator → FileSurfer | 21 | 21 |
| Orchestrator → Coder | 6 | 6 |
| Orchestrator → Analyst | 1 | 1 |

The task suite declares, per task, the collaboration it was written to elicit.
Four tasks declare `verify` or `loop`, meaning they ask for a second opinion or a
second pass. **The Analyst, the agent whose only purpose is to re-check claims,
was used in one of those four.** The orchestrator sends 21 of 28 handovers to
FileSurfer.

This is a property of how the system routes work, visible in the log, and
invisible to any check that reads only the final answer, which in most of these
runs was correct. It is also only measurable because the declared shape was
recorded before the run, not inferred from it afterwards.

Three frameworks solving comparable tasks with structurally different
coordination topologies, measured by one metric from one log. This is the
clearest demonstration in the set of process mining answering a question about
agent systems that output-level evaluation cannot pose.

---

## 6. Failure localisation — negative result

**Question.** Can a deterministic process-mining signal localise the decisive
failure step as well as an LLM, on data where experts have marked it?

**Method.** Mine a reference from the corpus itself: a directly-follows edge
occurring in fewer than 2% of runs counts as structurally rare, and the first
such edge in a trace is the predicted decisive step. Scored against
TraceElephant's `mistake_step`, resolved to a canonical event index by the
ingester. No LLM involved.

| subset | n | step exact | within ±1 | agent |
|---|---|---|---|---|
| captain-runs-gaia | 73 | 0.0% | 0.0% | 0.0% |
| magentic-runs-gaia | 74 | 0.0% | 1.4% | 9.5% |
| swe-agent-runs-swe-bench | 44 | 2.3% | 2.3% | 2.3% |

**Trivial baselines on the same runs:**

| subset | random step | always-last | always-majority-agent |
|---|---|---|---|
| captain-runs-gaia | 11.0% | 0.0% | 37.0% (WebServing_Expert) |
| magentic-runs-gaia | 4.1% | 5.4% | 56.8% (WebSurfer) |
| swe-agent-runs-swe-bench | 2.3% | 0.0% | 79.5% (str_replace_editor) |

**Published LLM baselines on this dataset** (Chen et al. 2026, best of five
methods): agent-level 65.9%, step-level 30.3%.

### Three conclusions, all of them uncomfortable

1. **First-divergence does not work.** At or below random on every subset. The
   plan to beat 30.3% this way is dead as stated.
2. **A decisive failure is a content error, not a structural anomaly.** The agent
   does something perfectly ordinary and *wrong*. Converges with the AgentHallu
   result in §4.2. Process structure finds protocol violations, which is what §3
   demonstrates on τ²; it does not find wrong answers.
3. **The published baseline is weaker than it reads.** A constant prior — always
   name the most common agent in the subset — scores 37%, 57% and 80%. The
   paper's 65.9% agent-level figure is a dataset-wide average, and on two of
   three subsets a constant is competitive with it. Worth a paragraph in Ch. 2;
   it is a fair observation about the benchmark, not about the authors.

### 6.1 Why captain-gaia cannot support this analysis at all

| subset | runs | runs with **no tool calls whatsoever** |
|---|---|---|
| magentic-runs-gaia | 74 | 7 |
| **captain-runs-gaia** | 73 | **32 (44%)** |
| swe-agent-runs-swe-bench | 44 | 0 |

In captain-gaia the activity alphabet is three symbols (`orchestrate` 842,
`report` 675, `seek_experts_help` 42) and almost half the runs are agents
talking to each other with no tool use at all. There is no control-flow
structure for a deviation to appear in. The recorded `output` is frequently the
raw LLM API response envelope rather than clean text.

**Usable TraceElephant is magentic-gaia plus swe-agent: 118 runs, not 191.**

---

## 7. Object-centric logs (R6)

A flat log forces one case notion — run = case — and each event belongs to
exactly one case. OCEL 2.0 drops that: one event belongs to several objects at
once. Object identifiers are scoped per run, because two runs touching the same
`user_id` are independent replays against the same seeded database, and merging
them would fabricate object lifecycles that never happened.

Since 2026-10-19 these logs are produced by the pipeline's own builder
(`agentpm.logbuild.ocel.build_ocel_from_traces`), not by an experiment script.
`python -m agentpm.cli ocel <file> --format tau2 --exclude assistant_message
user_message` writes a byte-identical file (sha256 checked). Every event
relates to its `run`, `agent` and `tool`; business objects come from object
rules declared as data (`agentpm/logbuild/objects.py`), which for τ² name
`user_id`, `reservation_id` (plus the id *returned* by `book_reservation`,
which creates one), `flight_number` including entries inside a `flights`
list, `payment_id` and `payment_methods`, and origin>destination>date as a
`route`. Output of the build is in `results/ocel/BUILD.txt`.

| log | events | objects | E2O relations | object types |
|---|---|---|---|---|
| `tau2_airline.ocel.json` | 1,576 | 1,777 | 6,627 | run 200, agent 1, tool 13, reservation 615, route 325, flight 295, user 189, payment 139 |
| `tau2_airline_with_messages.ocel.json` | 3,990 | 1,780 | 13,869 | as above, agent 2, tool 15 |
| `te_magentic.ocel.json` | 2,288 | 96 | 6,864 | run 74, agent 7, tool 15 |
| `te_swe.ocel.json` | 1,599 | 56 | 4,797 | run 44, agent 7, tool 5 |
| `local_magentic.ocel.json` | 109 | 19 | 327 | run 8, agent 5, tool 6 |

**The headline number: 1,454 of 1,576 τ² tool events (92%) touch at least one
business object beyond run, agent and tool.** That is the share a single-case
flat log cannot represent without duplicating events or discarding
information. (An earlier version of this section reported "events belonging
to more than one object type"; with agent and tool objects on every event
that number is trivially 100%, so the business-object share is the one that
means something.) The multi-agent corpora have no business objects: their
tools take free text, so the rules have nothing to name. Their object of
interest is the `agent`.

All logs are verified by reading them back with `pm4py.read_ocel2_json`, so
they open in pm4py and ProM.

### 7.1 A modelling consequence worth stating in Ch. 7

An earlier τ² build that omitted the `run` object dropped 122 of 1,576 events on
reload: exactly `calculate` (52), `transfer_to_human_agents` (63) and
`list_all_airports` (7). Those activities carry no business identifier in their
arguments, so in a purely object-centric view they belong to nothing and
disappear. That `transfer_to_human_agents` — the escalation a compliance
analyst most wants to see — is among them is the point. The logs above keep the
`run` object precisely so nothing is orphaned.

---

### 7.2 The partial order, and what discovery does with it

Tool calls a model issues in one turn carry an identical timestamp in the
canonical trace and in the OCEL file (the serialisation position is kept
separately as `sequence`). OCEL 2.0 permits that. The question is whether
discovery uses it. `experiments/partial_order.py` builds four directly-follows
graphs from the same 200 airline runs, tool events only
(`results/partial_order.txt`):

| graph | distinct edges | relations | within-turn pairs counted as succession |
|---|---|---|---|
| flat, pipeline (serialisation) | 81 | 1,377 | 830 |
| partial order, pipeline (step to next step) | 84 | 3,133 | 0 |
| `pm4py.discover_dfg`, timestamps with ties | 81 | 1,377 | 830 |
| `pm4py.discover_ocdfg`, OCEL with ties | 81 | 1,377 | 830 |

Both pm4py algorithms return exactly the flat graph: they sort by time and
count consecutive pairs, so an identical timestamp changes nothing. Five of
the 81 flat edges exist only because of serialisation
(`book_reservation→book_reservation`, `cancel_reservation→cancel_reservation`,
`cancel_reservation→get_reservation_details`, `get_reservation_details→list_all_airports`,
`get_user_details→list_all_airports`). Respecting the partial order needs a
discovery that reads steps, which `agentpm.analysis.discovery.directly_follows`
does. The reshuffle flips in §3.5 are the conformance-side cost of the same
serialisation.

This corrects a claim made earlier in this project and in the Chapter 5
draft: the object-centric log *holds* the partial order, but it does not by
itself make discovery concurrency-aware. The reason for OCEL that survives is
the multi-object case notion of §7; concurrency is handled in the canonical
model and the pipeline's own discovery.

---

## 8. How to regenerate everything

```bash
source ~/venvs/agentpm/bin/activate
cd agent-pm-pipeline

pytest -q                                              # 48 passed

python experiments/run_three_corpora.py \
    --data ~/Downloads --out results                   # results/three_corpora.txt

python experiments/build_ocel.py \
    --data ~/Downloads --out results/ocel              # the three OCEL logs

PATH=/opt/homebrew/bin:$PATH \
python experiments/figures.py \
    --data ~/Downloads --out results/figures           # ten process maps
```

Data lives outside the repo: `~/Downloads/tau-data` (6.19 GB, with
`MANIFEST.json` carrying sha256 and source URLs), `~/Downloads/traceelephant`,
`~/Downloads/agenthallu`.

Figures in `results/figures/`: `tau2_airline_dfg`, `te_magentic_dfg`,
`te_captain_dfg`, `agenthallu_magentic_dfg`, `agenthallu_camel_dfg` and a
`_handover` counterpart for each.

---

## 9. Threats to validity

- **One τ² submission, one domain.** All τ² numbers come from
  `claude-opus-4-5` on airline. The mirror holds 34 other text submissions and
  four domains; none of this generalises until that is run.
- **The policy transcription is mine.** §3.1 reports its failure honestly, but a
  better transcription might carry signal. The claim is that the gold-action
  reference beats *this* transcription, not that prose policies are unusable.
- **No negative class in TraceElephant.** Only the 220 failed runs of the
  paper's 380 executions are released, so precision and false-positive rate
  cannot be measured there at all.
- **TraceElephant step-level annotator agreement is Krippendorff's α = 0.64.**
  That is moderate, and it caps what any method can score against these labels.
  Report it alongside any number scored against them.
- **The local corpus is eight runs on one model.** It demonstrates ingestion and
  structure, not a statistical result, and nothing in it should be read as a
  rate. Running the same suite on a second model would turn it into a
  comparison.
- **The local corpus has no independent labels.** The declared task shape is
  what was asked for, not a judgement of what happened, and the reference model
  would have to be authored by us. It therefore carries no conformance claim.
- **Claude Code telemetry was probed and set aside.** It emits OpenTelemetry
  logs with tool names, real durations and cost, but no tool arguments, no text,
  and no agent attribution on tool events, so sub-agent work cannot be assigned
  to a sub-agent. The capture is kept in `results/otel_probe/`.
- **Logical timestamps.** τ² and TraceElephant events are ordered by position,
  not wall clock, because parallel tool calls in one turn share an identical
  timestamp. No duration or throughput claim is made anywhere in this document.

---

## 10. Process mining against LLM-as-a-Judge (Chapter 6)

Design in `experiments/judge/README.md`. Ground truth is τ²'s per-action check, independent of both
methods. Judge model `gpt-oss:120b` on the BFH-TI endpoint, five repeats per
run, every prompt and raw response logged.

### 10.1 Alignment

| method | n | prec | rec | F1 | κ | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| process mining (per-task answer key) | 172 | 0.88 | 0.59 | 0.71 | 0.65 | [0.48, 0.79] |
| judge, full policy + transcript | 170 | 0.26 | 0.51 | 0.35 | 0.09 | [−0.05, 0.23] |
| **judge, required actions + transcript** | 172 | 0.93 | **1.00** | 0.96 | **0.95** | [0.89, 1.00] |
| judge, output only | 170 | 0.21 | 0.89 | 0.34 | −0.02 | [−0.07, 0.03] |

McNemar, process mining against each judge on their shared runs:

| versus | both right | both wrong | only PM right | only judge right | p |
|---|---:|---:|---:|---:|---|
| full judge (170) | 85 | 9 | 62 | 14 | < 0.0001 |
| **required-actions judge (172)** | 146 | 0 | 3 | **23** | 0.0001 |
| output-only judge (170) | 29 | 4 | 123 | 14 | < 0.0001 |

### 10.1b The same comparison against the task-type model (2026-10-19)

`evaluate.py --pm-verdicts results/task_type_conformance.json` scores the
judges against the per-task-type reference instead of the per-task gold
reference. This is the comparison Chapter 6 now reports, because the task-type
model is the only reference not derived from the gold list the label is
computed from. Output in `results/judge/evaluation_per_type.txt`.

| method | n | prec | rec | κ | 95% CI | only engine right / only judge right | McNemar p |
|---|---|---|---|---|---|---|---|
| declarative, per task type | 172 | 0.94 | 0.41 | 0.50 | [0.31, 0.67] | | |
| judge, policy | 170 | 0.26 | 0.51 | 0.09 | [−0.05, 0.23] | 62 / 14 | <0.001 |
| judge, required actions | 172 | 0.93 | 1.00 | 0.95 | [0.89, 1.00] | 3 / 23 | 0.0001 |
| judge, output only | 170 | 0.21 | 0.89 | −0.02 | [−0.07, 0.03] | 124 / 20 | <0.001 |

Of the 22 violations the required-actions judge finds and the task-type engine
does not, the judge's stated reason is a wrong argument in 10, a missing action
in 11, and both in 1. Eight of the eleven missing actions are gold actions
outside the type's required set (searches and calculations), the price of a
model that is not an answer key; in the other three a required action occurred
too few times (corrected 2026-11-08, `results/miss_reasons.txt`).

The procedural result is now a sweep over the inductive miner's noise
threshold (`experiments/noise_sweep.py`, `results/pm4py/noise_sweep.txt`):
κ 0.02 at noise 0.2 (net lacks four activities, flags 156/172; corrected 2026-11-08), 0.07 at 0.1,
0.34 at noise 0 (all activities, flags 9/172 at precision 1.00; 8 of those 9
the task-type engine marks compliant, since nothing in it forbids repetition).
The Chapter 6 walk-through case, task 29 trial 0, has alignment fitness 1.00 at
noise 0: the run that skipped the cancellation fits the discovered model
perfectly. That, not the dropped transition, is the argument against
discover-then-check.

### 10.2 Reading these three results together

> **Corrected 2026-11-08.** The counts in this section belong to the per-task
> model of the engine before it compared steps (20 October). With the current
> engine the per-task model flags 25 runs with 3 false alarms, not 8, misses 15
> violations, and reaches κ 0.65, 0.30 below the required-actions judge
> (Section 6.2 of the thesis, `results/judge/evaluation.txt`).

**The full judge is not a fair comparison, and its number is not a verdict on
judges.** It reads the whole policy and flags rules the ground truth does not
measure. On runs the truth calls compliant it cites "one tool call at a time"
85 times, which the policy states and the agent breaks in 85% of runs, the
transfer protocol 90 times, and unsupported information 60 times. Its low κ
against this truth is mostly a scope mismatch: it answers a broader question
than the truth asks. The 62-to-14 split is real, but what it shows is that the
truth and the policy disagree about what compliance means.

**The required-actions judge is the fair comparison, and it wins.** Asked the
same question process mining answers, and given the same reference, it reaches
κ 0.95 with recall 1.00. Of the 23 runs only the judge gets right, 15 are
violations process mining missed; the judge's own stated reason is wrong
arguments in 10, a missing action in 4, both in 1. The remaining 8 are runs
process mining flagged that are in fact compliant. Process mining's κ is 0.38
below the judge's.

The reason is the one Chapter 5 already named. The constraint language sees
activity names and not arguments; the judge sees both. The gap is a property of
the language, not of the approach, and it is testable: extend `Reference` with
argument predicates, using τ²'s `compare_args`, and re-run. If process mining
then matches the judge, the claim becomes "a deterministic check matches an LLM
judge at zero cost once it can see arguments", which is stronger and more
honest than anything the full-judge number supports.

**What survives unconditionally.** Cost: the required-actions judge spends
19,717 prompt and 2,504 completion tokens per run over five repeats; process
mining spends none. And the output-only judge is at chance,
κ −0.02, with recall 0.89 only because it flags nearly everything (156 of 170
runs). The 79 runs the endpoint first
refused were re-judged on 2026-10-21, and two runs gave no parseable answer.

### 10.3 Reproducibility could not be tested here

Every judge condition was unanimous on every run, Fleiss' κ 1.00. That is not a
property of the judge. The raw response texts are **byte-identical** across all
five repeats for all 172 runs, and remain byte-identical for all 50 runs
re-judged with an explicit temperature of 0.7. The endpoint samples
deterministically and ignores the temperature parameter. Testing reproducibility
needs an endpoint that honours sampling, or a local server
where the setting is under our control.

A consequence worth drawing out: on this endpoint the judge is exactly as
reproducible as process mining. The reproducibility argument of the thesis
therefore cannot rest on this measurement. It rests on the fact that process
mining is reproducible by construction regardless of serving infrastructure,
whereas the judge's reproducibility here is an accident of configuration that a
different endpoint would not preserve. That is a weaker claim than "the judge
varies and we do not", and it is the one the data supports.

### 10.4 Threats specific to this comparison

- The required-actions judge is handed the same reference the truth was built
  from, so its 0.95 shows that a language model can replicate a programmatic
  check when given the specification. That is a real result about the judge's
  capability, not evidence it would find violations without the list.
- One judge model, one prompt per condition. Prompt wording moved the full
  judge from 0.09 to 0.95 by changing only what it was asked; the sensitivity
  runs both ways.
- 37 violations in 172 runs. The confidence intervals are reported for that
  reason and should be quoted with the point estimates.
- The output-only condition covers 170 of 172 runs after the rerun of
  2026-10-21. It sees only the customer's first message and the agent's last
  reply, so it flags 156 of 170 runs: a floor, not a competitor.


---

## 11. The OpenTelemetry route on a real framework (2026-10-19)

Until this date the semantic-convention ingester had been tested only on a
fixture written by hand and on Claude Code's log dialect. `experiments/
local_agents/smoke_otel.py` closes that gap without a model endpoint: two
AutoGen 0.7.5 `AssistantAgent`s with real Python tools take turns in a
`RoundRobinGroupChat`, their model being a `ReplayChatCompletionClient` with
scripted tool calls. AutoGen's own GenAI spans go through the OpenTelemetry
SDK 1.45 and OTLP/JSON over HTTP into `experiments/otel_capture/collector.py`, and
the pipeline's `otel` ingester reads the capture.

| | |
|---|---|
| spans received by the collector | 109, in 109 payloads, 0 failed |
| of which GenAI spans | 13: 1 `invoke_workflow`, 2 `create_agent`, 6 `invoke_agent`, 4 `execute_tool` |
| cases after ingestion | 1 (`gen_ai.conversation.id` on the root span) |
| tool calls, acting agent, order | identical to the harness's message stream, 4/4 |
| tool arguments | harness 4/4, OpenTelemetry 0/4 |

The capture is `results/otel_probe/autogen_roundrobin_replay.jsonl`, the
harness view `…harness.json`, and the capture is also
`tests/fixtures/autogen_otel_roundrobin.jsonl` with a test over it. The
Magentic-One harness exports the same spans on a live run with
`--otel-file` or `--otel-endpoint` (`experiments/local_agents/otel_export.py`).

What the route delivers: names, order, nesting, attribution of every tool to
the agent that ran it, real start and end times. What it does not: tool
arguments and results, because `gen_ai.tool.call.arguments` and `.result` are
opt-in in the convention and AutoGen does not set them. The thesis's claim
that a log of executed actions decides names, order and presence but not
arguments is therefore also the exact boundary of what the standard route
carries from this framework; the harness recording exists to keep the rest.

### 11.1 The live run (2026-10-19, 12:27–12:35)

Floris ran the eight Magentic-One tasks on gpt-oss:120b via the BFH endpoint
with `--otel-file results/otel_live/capture.jsonl`, then
`experiments/otel_report.py` on the capture. Everything is in
`results/otel_live/`; the harness recording of the same runs is in
`results/local_otel_live/` (the earlier corpus of 2026-10-13, the one the
thesis numbers come from, is untouched in `results/local/`; note that
`~/local_agent_runs` now holds the new runs, so scripts that read it by
default would regenerate from today's corpus).

| | OpenTelemetry route | harness recording |
|---|---|---|
| runs | 8 | 8 |
| spans / events | 1,120 spans, 92 GenAI events | 115 events |
| tool calls | 28 | 28 |
| tool name, order, acting agent | identical in all 8 runs | |
| tool arguments and results | 0 of 28 | 28 of 28 |
| orchestrator turns | none (the Magentic-One orchestrator is not a ChatAgent, so AutoGen opens no `invoke_agent` span for it) | 48 `orchestrate`, 39 `report` |
| model calls, tokens | none (no `chat` spans from the OpenAI client in 0.7.5) | tokens per message |
| reasoning text | none | 4 events |
| agents visible | FileSurfer, Coder, and Analyst only as an `invoke_agent` target in l04 and l06, it made no tool call | all four |

The route is confirmed on a real model: what the standard spans carry from
this framework is the specialists' tool calls with attribution and timing,
and nothing of the orchestrator, the model calls or the content. The
discovered model over the 28 tool events is small and clean (`list_files`
first in all 8 runs, then reads, then code; 7 variants), which is the
process the tools expose, not the process the team ran. The harness view of
the same eight runs has 115 events because it sees the orchestrator.
