# Validation round: other agents, a second domain, a second Judge

Runs of 23 to 26 October 2026. This document reports what was run to test
whether the Chapter 6 results hold beyond one agent, one domain and one Judge
model, what came out, and what it changes. Every number below is produced by a
script in `experiments/` and can be found in the output file named at the start
of each section.

## Summary

1. **The engine's precision holds outside the runs it was written on.** Applied
   unchanged to fifteen agents per domain, it is precise when it flags a run:
   pooled precision 0.89 on retail against the per-action label, and 0.94 on
   airline once that label also counts writes not on the gold list (0.74 without).
   Its recall against the process labels stays between 0.3 and 0.5, and its
   misses are mostly listed actions that occur but do not match.
2. **The second Judge confirms the first where the first was good.** The Judge
   handed the required actions reaches κ 0.95 (gpt-oss:120b) and 0.97
   (glm-5.3-flash) on airline, 0.91 and 0.86 on a retail sample, and the two
   give the same verdict on 99% and 96% of runs.
3. **The policy Judge is unreliable in every way that was measured.** It barely
   aligns with any label (κ between −0.08 and 0.15), two models given the same
   prompt agree with each other little more than chance (κ 0.03 and 0.13), and
   one model asked five times is unanimous on only 8 of 30 runs.
4. **Chapter 6's reproducibility measurement was wrong.** The endpoint caches
   responses, so the "five repeats" were one answer and four copies.
5. **Which label is used decides who wins.** Against the per-action label the
   required-actions Judge beats the engine by a wide margin; against the
   benchmark's reward the two are level on airline and the engine is ahead on
   retail.

## 1. What was run

| | Agent runs judged | Methods |
|---|---|---|
| Engine, 15 agents | airline 2,576 and retail 6,645 scorable runs, published by Sierra | task-type models, unchanged |
| Engine and Judges, one agent | claude-opus-4-5: airline 172, retail 448 scorable runs | engine, gpt-oss:120b, glm-5.3-flash |
| glm-5.3-flash on retail | a stratified sample of 120 of the 448 retail runs | |

Two roles must not be confused. The **agent** is the model that served the
customer, here claude-opus-4-5 (and fourteen others for the engine). Sierra ran
those conversations and published them; nothing was run by us. The **Judges**
are gpt-oss:120b and glm-5.3-flash, open models served by BFH, which read a
finished conversation and return a verdict. Both Judges come from a different
vendor than claude-opus-4-5. Claude did not judge anything.

Each Judge is run in the three conditions of Chapter 6. The *required-actions*
Judge receives the task's gold actions and the transcript. The *policy* Judge
receives the domain policy and the transcript. The *output-only* Judge receives
the customer's first message and the agent's last. glm-5.3-flash was run in the
first two only (section 5.5 says why).

### 1.1 The three labels

τ²-bench evaluates every run in several ways, and the ingester carries those
evaluations along unchanged. The engine never reads them; they serve only as
labels to score the engine and the Judges against.

- **action**: every gold action of the task was performed with matching
  arguments. This is the label of Chapter 6, fixed beforehand.
- **action-or-db**: as *action*, and also the run's final database equals the
  expected one. The benchmark computes the expected database by replaying the
  gold actions on a clean copy of the starting database (its `db_check`). This
  label was introduced after the cross-agent results, for the reason given in
  section 3, and is always reported next to *action*, never instead of it.
- **reward**: the benchmark's reward. It measures the outcome, not the process:
  the final database plus checks on what the agent told the customer.

## 2. The proxy cache

Source: `results/judge/two_judges.txt` (last section), `results/judge/evaluation_per_type.txt`.

The BFH inference endpoint answers an identical request from a cache, with the
response id and timestamp of the first answer. The current endpoint is a LiteLLM
proxy and marks such answers with an `x-litellm-cache-key` header; the endpoint
used for Chapter 6 shows the same behaviour in its logs.

The airline Judge runs of Chapter 6 asked every question five times with an
identical request. In the required-actions and in the policy condition
(`results/judge/raw_gold-actions_gpt-oss-120b.jsonl`, `raw_full_gpt-oss-120b.jsonl`),
172 of the 860 calls each carry a new response id, always the first repeat, and
688 repeat an earlier one. In the run at temperature 0.7 the first answer differs
from the default run's on 50 of 50 runs, so the temperature was not ignored; it
changed the request and with it the cache key.

Consequences:

- **Each Chapter 6 verdict in those two conditions is one sample.** That is also
  how all new verdicts were produced (one fresh sample per run), so the verdicts
  remain comparable.
- **The statement that the Judge is "as reproducible as the engine" is not
  supported.** The unanimity it rests on was the cache. Section 5.3 reports the
  measurement with the cache switched off.
- **Cost per run was overstated by counting copies.** Cost is now reported per
  verdict over the calls the model actually answered (section 5.4).

`run_judge.py --no-cache` asks the proxy to sample afresh, and every call now
records whether it was a cache hit.

## 3. The engine on fifteen agents

Source: `results/cross_agent.txt`, `experiments/cross_agent.py`.

The airline task-type models were written while looking at claude-opus-4-5's
runs. The public results bucket holds the same tasks run by fifteen agent
models, claude-opus-4-5 among them, under the same harness and checks. The models were applied to all of them
unchanged; no model was called. `toolorchestra_nvidia` is excluded, an
orchestration system with one trial per task.

**Pooled over 15 agents, confirmation rule as frozen:**

| Domain | Label | Runs | Violations | Flags | Precision [95% CI] | Recall | κ [95% CI] |
|---|---|---|---|---|---|---|---|
| airline | action | 2,576 | 909 | 541 | 0.74 [0.70, 0.77] | 0.44 | 0.39 [0.35, 0.43] |
| airline | action-or-db | 2,576 | 1,076 | 541 | 0.94 [0.92, 0.96] | 0.47 | 0.48 [0.45, 0.52] |
| airline | reward | 2,576 | 856 | 541 | 0.89 [0.86, 0.92] | 0.56 | 0.58 [0.55, 0.62] |
| retail | action | 6,645 | 2,785 | 979 | 0.89 [0.87, 0.91] | 0.31 | 0.32 [0.30, 0.33] |
| retail | action-or-db | 6,645 | 2,878 | 979 | 0.92 [0.90, 0.93] | 0.31 | 0.31 [0.30, 0.33] |
| retail | reward | 6,645 | 1,565 | 979 | 0.77 [0.74, 0.79] | 0.48 | 0.50 [0.47, 0.52] |

Without claude-opus-4-5, the agent the models were written on, the airline
figures are practically the same: precision 0.73 (action) and 0.94
(action-or-db), recall 0.44 and 0.47.

**The label's blind spot.** On airline, 107 of the engine's 143 false alarms
against *action* come from the rule that a task requiring no write may contain
none. The per-action check only asks whether the listed actions occurred, so a
write not on the gold list leaves it satisfied. Against *action-or-db*, 106 of those
107 runs are violations, because that write changed the database.
This is why the second label exists. For claude-opus-4-5 alone it hardly
matters (37 against 39 violations), because only two of its runs pass the
per-action check with a wrong database.

**The confirmation rule was fitted to one agent.** The other 36 false alarms
come from the rule that a user message must precede every write. In every one of
them the customer's message is found once read-only tool calls are looked
through: the agent looked something up between the customer's last message and
the write. The frozen rule does not look through reads, and claude-opus-4-5 never
did this on airline. A variant that also looks through read-only tools, declared
in `retail_task_types.py` before the retail runs were scored, removes all of
these alarms: pooled precision against *action-or-db* becomes 1.00 [0.99, 1.00]
on airline and 0.98 [0.97, 0.99] on retail, at a recall of 0.44 and 0.30.

**What the engine misses.** Of the 511 airline violations against *action*
that the frozen engine misses, 427 contain every failed action by name: 333
with a wrong value and 94 with too few repetitions. In retail the figure is
1,240 of 1,910, 712 and 528. The other 84 and 670 lack a gold action that the
type models do not require, such as `calculate`. Source:
`results/miss_reasons.txt`.

## 4. Retail as a second domain

Source: `experiments/retail_task_types.py`, `results/frozen/retail_task_types.sha256`,
the output of `python experiments/task_type_conformance.py --domain retail`.

The retail models were written from the retail policy alone, with the same
template as airline, and their hash and time were recorded (2026-10-23 13:56 UTC)
before any retail run was scored. A docstring correction after scoring is
recorded in the same file; the code is unchanged.

Task types are read from the write actions a task's gold list requires, as for
airline. Retail has 23 types, against 11 for airline, because customers
combine requests; `task_type_conformance.py` lists them with the model each
gets. Every type gets its model from one template:

- the defining write(s) are required;
- the user is authenticated before any write, by email or by name and zip
  (both are relabelled to one activity, `authenticate_user`, before checking);
- `get_order_details` precedes every write on an order;
- a user message precedes every write, one confirmation covering a batch;
- a task requiring no write contains none.

"Exchange or modify an order only once" is not encoded. It concerns one order
object and needs the order id of each call, which a check over activity names
cannot express.

On claude-opus-4-5's 448 retail runs the engine reaches precision 0.92, recall
0.39 and κ 0.43 against *action*, close to airline (0.94, 0.41, 0.50).

## 5. Two Judges

Source: `results/judge/two_judges.txt`, `experiments/judge/compare_judges.py`.
All runs are claude-opus-4-5's. Retail figures are on the stratified sample of
120 runs, so that every method is scored on the same runs.

### 5.1 Alignment with the labels

κ with bootstrap 95% intervals. n is the number of runs a method returned a
verdict for.

**Label *action***

| Method | Airline n | Airline κ [95% CI] | Retail sample n | Retail κ [95% CI] |
|---|---|---|---|---|
| Engine, task type | 172 | 0.50 [0.31, 0.67] | 120 | 0.49 [0.35, 0.63] |
| Required actions, gpt-oss:120b | 172 | 0.95 [0.89, 1.00] | 120 | 0.91 [0.83, 0.98] |
| Required actions, glm-5.3-flash | 172 | 0.97 [0.91, 1.00] | 120 | 0.86 [0.76, 0.95] |
| Policy, gpt-oss:120b | 170 | 0.09 [−0.05, 0.23] | 118 | −0.03 [−0.21, 0.15] |
| Policy, glm-5.3-flash | 164 | 0.10 [−0.02, 0.23] | 120 | 0.05 [−0.14, 0.22] |
| Output only, gpt-oss:120b | 170 | −0.02 [−0.07, 0.03] | 118 | 0.00 [−0.08, 0.07] |

**Label *reward***

| Method | Airline κ [95% CI] | Retail sample κ [95% CI] |
|---|---|---|
| Engine, task type | 0.60 [0.41, 0.76] | 0.65 [0.47, 0.80] |
| Required actions, gpt-oss:120b | 0.68 [0.53, 0.81] | 0.50 [0.34, 0.66] |
| Required actions, glm-5.3-flash | 0.69 [0.54, 0.82] | 0.54 [0.39, 0.69] |
| Policy, gpt-oss:120b | 0.06 [−0.06, 0.20] | 0.02 [−0.16, 0.19] |
| Policy, glm-5.3-flash | 0.14 [0.03, 0.26] | 0.15 [−0.03, 0.33] |

Against *action*, both required-actions Judges beat the engine on both domains
(McNemar p ≤ 0.0001). Against *reward*, the engine and the required-actions
Judges do not differ on airline (p 0.85 and 1.00); on the retail sample the
engine is ahead of gpt-oss:120b (p 0.019) and of glm-5.3-flash, though the
latter difference is not significant (p 0.064). On all 448 retail runs,
where only gpt-oss:120b was run, the engine's 0.61 against the Judge's 0.44 is
significant (p < 0.0001). A likely reason is that the required-actions Judge
reproduces the per-action check, including its penalty for a skipped
`calculate` or an argument that does not change the outcome. Reward
is an outcome label, so this shows that the engine's flags are mostly
violations with consequences; it does not show that the engine judges the
process better.

The *action-or-db* label gives the same picture as *action*
(`two_judges.txt` has all three labels for every method).

### 5.2 The two Judges against each other

| Condition | Airline | Retail sample |
|---|---|---|
| Required actions | same verdict on 171 of 172 runs (99%), κ 0.98 | 115 of 120 (96%), κ 0.91 |
| Policy | same verdict on 84 of 162 runs (52%), κ 0.03 | 69 of 118 (58%), κ 0.13 |

This measurement needs no label. Two models given the same policy prompt agree
little more than chance would predict. They flag about as many runs that the
*action* label calls compliant (airline: 53 and 58; retail sample: 24 and 28,
gpt-oss:120b first), but largely different ones: 24 of those airline runs and
13 of those retail runs are flagged by both. The rules they cite most often
differ too. On airline, gpt-oss:120b cites "no information not provided by the
user or the tools" and "one tool call at a time" most (7 times each), and
glm-5.3-flash the transfer rule (6 times). `two_judges.txt` lists the six most
cited rules per Judge and domain.

### 5.3 Each Judge against itself

gpt-oss:120b, five fresh samples per run (cache off), on the first 30 airline
runs.

| Condition | Unanimous | Fleiss κ | Fresh majority equals the original verdict |
|---|---|---|---|
| Policy | 8 of 30 runs | 0.17 | 22 of 30 |
| Required actions | 29 of 30 runs | 0.97 | 30 of 30 |

Four policy runs have one unusable answer, so Fleiss κ is computed over the
first four answers of every run; over the 26 runs with five answers it is 0.26.

The policy Judge gives a different verdict on most runs when asked again; the
required-actions Judge almost never does.

The same measurement for glm-5.3-flash was started on 26 October and is still
running.

### 5.4 Cost and failed verdicts

Per verdict, over the calls the model actually answered, cache copies left out.

| Judge, condition | Airline | Retail |
|---|---|---|
| Required actions, gpt-oss:120b | 3,943 + 501 tokens, 19 s | 3,460 + 566 tokens, 21 s |
| Required actions, glm-5.3-flash | 3,960 + 662 tokens, 45 s | 3,543 + 1,002 tokens, 29 s |
| Policy, gpt-oss:120b | 5,371 + 883 tokens, 19 s | 4,679 + 888 tokens, 28 s |
| Policy, glm-5.3-flash | 5,376 + 6,346 tokens, 169 s | 4,743 + 6,707 tokens, 182 s |
| Engine | 0 tokens | 0 tokens |

Tokens are prompt + completion. Retail figures for gpt-oss:120b are over all 448
runs, for glm-5.3-flash over the 120 in the sample. Chapter 6's "19,717 prompt
and 2,504 completion tokens per run over five repeats" becomes 4,444 tokens per
verdict for the required-actions Judge.

A Judge does not always return a verdict. The policy condition lost 2 of 172
airline runs for gpt-oss:120b and 8 of 172 for glm-5.3-flash (five malformed
answers, one empty, two proxy errors), and 12 of 448 retail runs for
gpt-oss:120b. The engine always returns one.

### 5.5 Why glm-5.3-flash judged a sample

The proxy serves glm-5.3-flash from one host and allows it one request at a time
for all its users (`max_parallel_requests: 1`). A policy verdict takes about
three minutes, and the slot is shared. glm-5.3-flash therefore judged all 172
airline runs in two conditions, but in retail a sample of 120 runs, proportional
per task type, at least one run per type, drawn with seed 0 without looking at
any label (`experiments/judge/sample_runs.py`). The output-only condition was
not run for glm-5.3-flash, since gpt-oss:120b already places it at chance.

The sample happens to hold more violations than the full set (51 of 120, 43%,
against 158 of 448, 35%), one reason the engine's κ on the sample (0.49)
differs from its κ on all runs (0.43). Methods are therefore compared within the
sample only. Its intervals are wide: the two required-actions Judges (0.91 and
0.86) cannot be told apart.

## 6. What this changes in Chapter 6

- Rewrite the reproducibility paragraph: the cache explains the old result,
  reproducibility is now tested (section 5.3).
- Report Judge cost per verdict (section 5.4).
- Add glm-5.3-flash rows and the inter-Judge agreement to the Judge table.
- Add retail as a second domain, and the engine on fifteen agents.
- Report all three labels, stating that *action-or-db* was introduced afterwards
  and why.
- Update the threats: "one agent model, one domain, one Judge model" no longer
  holds; what remains is listed below.

## 7. Limitations that remain

- **The Judges were run on one agent.** Every Judge verdict concerns
  claude-opus-4-5, a strong agent whose errors are subtle. Only the engine was
  run on fifteen agents.
- **glm-5.3-flash judged a retail sample of 120 runs.** Differences between the
  two Judges in retail cannot be resolved (section 5.5).
- **glm-5.3-flash's agreement with itself is not yet measured.**
- **The *action-or-db* label was introduced after seeing results.** It raises
  the engine's pooled airline precision from 0.74 to 0.94, which is exactly why
  it must be reported next to *action* and with its reason.
- **The confirmation variant was chosen after the airline cross-agent run.** It
  was declared before the retail runs were scored, but on airline it is a
  post-hoc fix and must be presented as one.
- **The airline verdicts of gpt-oss:120b came from the previous endpoint**;
  all other verdicts, and the self-agreement runs, from the current one.

## 8. How to regenerate

```bash
python experiments/task_type_conformance.py --domain retail
python experiments/cross_agent.py > results/cross_agent.txt
python experiments/judge/sample_runs.py --domain retail --n 120
bash experiments/judge/run_second_judge.sh gptoss    # needs LLM_API_KEY
bash experiments/judge/run_second_judge.sh glm
# glm-5.3-flash self-agreement, as for gpt-oss:120b in run_second_judge.sh:
#   run_judge.py --model glm-5.3-flash --mode full --repeats 5 --limit 30 --no-cache --stream
#   --out results/judge/self_agreement   (then the same with --mode gold-actions)
python experiments/judge/compare_judges.py > results/judge/two_judges.txt
python experiments/task_type_details.py > results/task_type_details.txt   # confirmation rule, when a violation is visible, transfers
python experiments/thesis_figures.py                                      # the printed figure (results/figures/thesis)
python experiments/noise_sweep.py                                         # fixes PYTHONHASHSEED=0 itself
python experiments/noise_sweep_seeds.py > results/pm4py/noise_sweep_seeds.txt   # the same under ten seeds
```
