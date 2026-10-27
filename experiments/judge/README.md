# Judge baseline and the comparison

This is the evaluation of Chapter 6: the process-mining signal against an
LLM-as-a-Judge, on the same runs, against the same ground truth.

## The three roles

| role | what it does | here |
|---|---|---|
| the agent | performed the task and produced the trace | claude-opus-4-5, run by Sierra; fixed by the dataset |
| the judge | reads a trace afterwards and gives a verdict | gpt-oss:120b, the one choice we make |
| process mining | checks the trace against a reference | no model |

The judge must not be the model that produced the traces, otherwise it is
grading its own work. gpt-oss:120b is a different vendor from the agent.

## Ground truth

tau2-bench's own `action_checks`: for every gold action of a task, did the run
perform it with matching arguments. All matched means the protocol was
followed. This is computed by the benchmark harness, so it is independent of
both methods, and it describes the **process**, not the outcome. On the airline
file it labels 37 of 172 scorable runs as violations.

Process mining checks activity names only, so it will miss argument errors by
construction; the judge reads the transcript and may catch them. That asymmetry
is part of the result, not a flaw in the design.

## Running it

```bash
export LLM_BASE_URL=https://inference.mlmp.ti.bfh.ch/api/v1
export LLM_API_KEY=...
source ~/venvs/agentpm/bin/activate

python experiments/judge/run_judge.py --model gpt-oss:120b --mode full
python experiments/judge/run_judge.py --model gpt-oss:120b --mode output-only
python experiments/judge/evaluate.py
```

`--limit 10` judges ten runs first, to check the model answers in parsable
JSON before spending an hour. Both judge runs resume where they stopped. Every
prompt and every raw response is appended to `results/judge/raw_*.jsonl`.

Budget: 172 runs x 5 repeats x roughly 6k prompt tokens for the full judge is
about five million tokens. Expect one to two hours on the hosted endpoint.

## The proxy cache, and the second Judge (2026-10-23)

The endpoint is a LiteLLM proxy with a response cache. An identical request is
answered from the cache with the same response id and timestamp, so the five
repeats of the first gpt-oss:120b verdicts were one sample and four copies. The
"deterministic endpoint" read of those runs was wrong: at temperature 0.7 the
answers differ from the default ones on 50 of 50 runs. `--no-cache` makes the
proxy sample afresh, and every repeat now records `cache_hit`.

`run_second_judge.sh` runs glm-5.3-flash on airline, both Judges on retail
(one fresh sample per run, as the first verdicts effectively were), and five
fresh samples on the first 30 airline runs per Judge to measure self-agreement:

```bash
bash experiments/judge/run_second_judge.sh glm      # results/judge, results/judge/retail, results/judge/self_agreement
bash experiments/judge/run_second_judge.sh gptoss   # results/judge/retail, results/judge/self_agreement
```

## What the comparison reports

**Alignment.** Accuracy, precision, recall, F1 and Cohen's kappa for each method
against ground truth, each kappa with a bootstrap 95% confidence interval from
1,000 resamples. With 37 positives the intervals are wide; report them.

**Paired comparison, McNemar's exact test.** Both methods judge the same 172
runs, so comparing their accuracies as if they were independent samples throws
away information. McNemar looks only at the runs where the two methods disagree
on who is right:

```
                     judge right   judge wrong
process mining right     a             b
process mining wrong     c             d
```

Cells a and d say nothing about which method is better. If the methods are
equally good, b and c should be about equal; McNemar asks whether the observed
imbalance between b and c could be chance. With 172 pairs the test can detect a
difference of roughly ten percentage points in error rate. A p above 0.05 means
"no difference found on this sample", not "they are the same".

**Reproducibility.** Share of runs on which all five judge repeats agree, and
Fleiss' kappa across repeats. Process mining's three repetitions hash
identically, measured in `run_three_corpora.py`.

**Cost.** Tokens and seconds per run. Process mining is zero tokens.

## Pre-registered expectations

Written before the judge ran, so the comparison is a test rather than a search.

- H1, alignment: process mining's kappa is not more than 0.10 below the full
  judge's. Expected to hold on precision and fail on recall, because of
  argument errors.
- H2, reproducibility: the full judge is not unanimous on every run.
- H3, cost: the token ratio exceeds 100.
- H4, the output-only judge scores below both, because it cannot see the path.
