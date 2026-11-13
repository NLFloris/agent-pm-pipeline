"""First divergence against the step labels of AgentHallu's Magentic-One runs.

    python experiments/first_divergence.py > results/first_divergence.txt

First divergence takes the first transition of a run that occurs in no run without
a hallucination as the step where the run went wrong. It is scored against the
step that people labeled, on the runs that contain a hallucination. A second rule,
the first transition that occurs in no other run of the corpus, needs no labels.

The labels count steps from 1 and the event timestamps count from 0, so the step
of an event is its timestamp plus one. The script checks this against the raw
files. A random step is given as its expected value per run.

The same idea on TraceElephant is in results/three_corpora.txt. Standard library
only.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

from agentpm.ingest import agenthallu
from data_dir import DATA   # noqa: E402


def step_of(event):
    return event.timestamp + 1


def label(trace):
    return int(trace.outcome["hallucination_step"])


def transitions(trace):
    acts = trace.activities
    return set(zip(acts, acts[1:]))


def first_unknown(trace, known):
    """Index of the first event whose incoming transition is not in known."""
    acts = trace.activities
    for i in range(1, len(acts)):
        if (acts[i - 1], acts[i]) not in known:
            return i
    return None


def score(name, predict, runs):
    exact = near = silent = 0
    offsets = collections.Counter()
    for t in runs:
        i = predict(t)
        if i is None:
            silent += 1
            continue
        offset = step_of(t.events[i]) - label(t)
        offsets[offset] += 1
        exact += offset == 0
        near += abs(offset) <= 1
    n = len(runs)
    print(f"  {name:48} exact {exact:2} of {n} ({exact / n:5.1%})   within one step {near:2} ({near / n:5.1%})"
          f"   no prediction {silent}")
    print(f"    predicted minus labeled step: {sorted(offsets.items())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()
    root = os.path.join(a.data, "agenthallu", "Magentic_One")
    traces = agenthallu.ingest_dir(root)

    for t in traces:
        with open(os.path.join(root, t.case_id.split(":")[1] + ".json"), encoding="utf-8") as f:
            raw = {i: h["step"] for i, h in enumerate(json.load(f)["history"])}
        assert all(raw[e.timestamp] == step_of(e) for e in t.events)

    runs = [t for t in traces if str(t.outcome.get("is_hallucination")) == "True"]
    clean = [t for t in traces if str(t.outcome.get("is_hallucination")) != "True"]
    print(f"AgentHallu Magentic-One: {len(traces)} runs, {len(runs)} with a hallucination, {len(clean)} without\n")

    known = set().union(*(transitions(t) for t in clean))
    print(f"  transitions in the runs without a hallucination: {len(known)}")
    score("first transition not in the runs without one", lambda t: first_unknown(t, known), runs)
    score("first transition in no other run of the corpus",
          lambda t: first_unknown(t, set().union(*(transitions(o) for o in traces if o is not t))), runs)
    fires = sum(first_unknown(t, set().union(*(transitions(o) for o in clean if o is not t))) is not None for t in clean)
    print(f"  runs without a hallucination, each against the other {len(clean) - 1}: {fires} of {len(clean)} diverge")

    steps = lambda t: sorted({step_of(e) for e in t.events})
    n = len(runs)
    print("\n  baselines")
    print(f"    random step, expected: exact {sum(1 / len(steps(t)) for t in runs) / n:.1%}, within one step "
          f"{sum(sum(abs(s - label(t)) <= 1 for s in steps(t)) / len(steps(t)) for t in runs) / n:.1%}")
    common = collections.Counter(label(t) for t in runs).most_common(1)[0]
    print(f"    always step {common[0]}: exact {common[1] / n:.1%}")

    print("\n  the labeled steps")
    at_step = collections.Counter()
    print(f"    categories: {collections.Counter(t.outcome['hallucination_category'] for t in runs).most_common()}")
    for t in runs:
        events = [e for e in t.events if step_of(e) == label(t)]
        at_step[events[0].activity if events else None] += 1
    print(f"    activity: {at_step.most_common()}")
    no_tool = 0
    for t in runs:
        with open(os.path.join(root, t.case_id.split(":")[1] + ".json"), encoding="utf-8") as f:
            step = next(h for h in json.load(f)["history"] if h["step"] == label(t))
        no_tool += not step.get("tool_calls")
    print(f"    labeled steps without a tool call: {no_tool} of {n}")


if __name__ == "__main__":
    main()
