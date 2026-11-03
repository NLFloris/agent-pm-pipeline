"""Process-mining results over the three validation corpora.

    python experiments/run_three_corpora.py --data ~/Downloads --out results

Sections, per corpus where the data supports them:
  1. ingestion inventory
  2. conformance against a reference derived from the data (tau2 gold actions)
  3. discovery: variants, directly-follows structure
  4. performance proxy: rework and self-loops, split by outcome
  5. organizational: handover of work between agents
  6. localization: first structural divergence vs the expert mistake step

Every number printed here is regenerable by re-running this command.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys

from agentpm.analysis.conformance import check
from agentpm.ingest import agenthallu, tau2, traceelephant
from agentpm.reference import Reference

MESSAGES = {"assistant_message", "user_message"}
CONTROL = {"delegate", "report", "orchestrate"}


# --------------------------------------------------------------- helpers

def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def tool_seq(trace, drop=MESSAGES | CONTROL):
    return [a for a in trace.activities if a not in drop]


def inventory(name, traces, drop=MESSAGES):
    acts = collections.Counter(a for t in traces for a in t.activities)
    res = collections.Counter(e.resource for t in traces for e in t.events)
    n_events = sum(len(t.events) for t in traces)
    print(f"\n[{name}]")
    print(f"  runs {len(traces)}   events {n_events}   mean {n_events / max(1, len(traces)):.1f}/run")
    print(f"  activity alphabet {len(acts)}")
    print(f"  top activities   {', '.join(f'{a}({n})' for a, n in acts.most_common(8))}")
    print(f"  resources        {dict(res.most_common(8))}")
    return acts


def rework(traces, label_fn):
    """Rework rate and self-loop rate, split by a binary label."""
    groups = collections.defaultdict(list)
    for t in traces:
        s = tool_seq(t)
        if len(s) < 2:
            continue
        groups[label_fn(t)].append((
            len(s),
            1 - len(set(s)) / len(s),
            sum(1 for a, b in zip(s, s[1:]) if a == b) / (len(s) - 1),
        ))
    for key in sorted(groups, key=str):
        g = groups[key]
        print(f"    {str(key):22} n={len(g):4}  calls {statistics.mean(x[0] for x in g):5.1f}"
              f"  rework {statistics.mean(x[1] for x in g):6.1%}"
              f"  self-loop {statistics.mean(x[2] for x in g):6.1%}")


def handover(traces):
    edges = collections.Counter()
    for t in traces:
        res = [e.resource for e in t.ordered()]
        for a, b in zip(res, res[1:]):
            if a != b:
                edges[(a, b)] += 1
    return edges


# ------------------------------------------------------------------ tau2

AIRLINE_TOOLS = [
    "get_user_details", "get_reservation_details", "search_direct_flight",
    "search_onestop_flight", "get_flight_status", "list_all_airports", "calculate",
    "book_reservation", "update_reservation_flights", "update_reservation_baggages",
    "update_reservation_passengers", "cancel_reservation", "send_certificate",
    "transfer_to_human_agents",
]
WRITES = ["book_reservation", "update_reservation_flights", "update_reservation_baggages",
          "update_reservation_passengers", "cancel_reservation", "send_certificate"]

# Hand-transcribed from data/tau2/domains/airline/policy.md (accessed 2026-09-22).
# Kept in the experiment because its failure is itself a reported result: see
# the policy-vs-gold comparison below.
# 2026-10-30: dropped end_activities=["assistant_message"]. The policy has no
# such rule, and every tau2-bench conversation ends with the simulated customer,
# so it failed on all 200 runs alike and only lowered every fitness to 0.9231.
POLICY_REFERENCE = Reference(
    name="airline_policy",
    allowed=AIRLINE_TOOLS + ["assistant_message", "user_message"],
    precedence=([["get_user_details", "book_reservation"]]
                + [["get_reservation_details", w] for w in
                   ["update_reservation_flights", "update_reservation_baggages",
                    "update_reservation_passengers", "cancel_reservation"]]),
    forbidden_after=[["transfer_to_human_agents", w] for w in WRITES],
)


def gold_reference(trace):
    names = [a["name"] for a in trace.outcome.get("gold_actions", [])
             if a.get("requestor") == "assistant"]
    pairs = [[a, b] for a, b in zip(names, names[1:]) if a != b]
    return Reference(name="gold", required=sorted(set(names)), precedence=pairs)


def tau2_section(path):
    rule("CORPUS 1  tau2-bench airline (public, carries its own reference model)")
    traces = tau2.ingest(path)
    inventory("ingestion", traces)

    passed = sum(1 for t in traces if t.outcome.get("reward") == 1.0)
    print(f"\n  harness labels: {passed} pass / {len(traces) - passed} fail "
          f"(base rate {passed / len(traces):.0%})")

    print("\n  conformance vs hand-transcribed policy reference (authored by us)")
    pol = [(t, check(t, POLICY_REFERENCE)) for t in traces]
    by_pol = collections.defaultdict(list)
    for t, r in pol:
        by_pol[t.outcome.get("reward")].append(r.fitness)
    print(f"    constraints {pol[0][1].n_constraints}   "
          f"fully compliant {sum(1 for _, r in pol if r.compliant)}/{len(pol)}")
    for k in sorted(by_pol, key=lambda x: (x is None, x)):
        print(f"    mean fitness | reward={k}: {statistics.mean(by_pol[k]):.4f}  (n={len(by_pol[k])})")
    fired = collections.Counter(v.constraint for _, r in pol for v in r.violations)
    print(f"    constraints that ever fire: {dict(fired.most_common(5)) or 'none'}")
    spread = (max(statistics.mean(v) for v in by_pol.values())
              - min(statistics.mean(v) for v in by_pol.values()))
    print(f"    separation between labels: {spread:.4f}  <- the signal this reference carries")

    print("\n  conformance vs per-task gold-action reference (derived from the data)")
    results = [(t, check(t, gold_reference(t))) for t in traces]
    usable = [(t, r) for t, r in results if r.n_constraints]
    by = collections.defaultdict(list)
    for t, r in usable:
        by[t.outcome.get("reward")].append(r.fitness)
    print(f"    runs with gold actions {len(usable)}/{len(traces)}")
    for k in sorted(by, key=lambda x: (x is None, x)):
        print(f"    mean fitness | reward={k}: {statistics.mean(by[k]):.4f}  (n={len(by[k])})")

    tp = sum(1 for t, r in usable if not r.compliant and t.outcome["reward"] == 0.0)
    fp = sum(1 for t, r in usable if not r.compliant and t.outcome["reward"] == 1.0)
    fn = sum(1 for t, r in usable if r.compliant and t.outcome["reward"] == 0.0)
    base = sum(1 for t, _ in usable if t.outcome["reward"] == 0.0) / len(usable)
    print(f"    as a failure predictor: precision {tp / max(1, tp + fp):.0%}"
          f"  recall {tp / max(1, tp + fn):.0%}  (base rate {base:.0%})")

    print("\n  discovery")
    variants = collections.Counter(tuple(tool_seq(t)) for t in traces)
    print(f"    {len(variants)} distinct tool-call variants over {len(traces)} runs; "
          f"{sum(1 for v in variants.values() if v == 1)} occur once")
    print(f"    most frequent: {' -> '.join(variants.most_common(1)[0][0]) or '(no tool calls)'}"
          f"  ({variants.most_common(1)[0][1]} runs)")

    print("\n  performance proxy, split by harness label")
    rework(traces, lambda t: f"reward={t.outcome.get('reward')}")
    return traces


# ---------------------------------------------------------- traceelephant

def first_divergence(seq, edge_freq, threshold):
    """Index of the first event whose incoming edge is structurally rare."""
    for i in range(1, len(seq)):
        if edge_freq.get((seq[i - 1], seq[i]), 0) < threshold:
            return i
    return None


def te_section(root, subsets, threshold_pct=0.02):
    rule("CORPUS 2  TraceElephant (peer-reviewed, expert step-level ground truth)")
    all_traces = []
    for sub in subsets:
        path = os.path.join(root, sub)
        if not os.path.isdir(path):
            print(f"  [skip] {sub} not found")
            continue
        traces = traceelephant.ingest_dir(path)
        for t in traces:
            t.outcome["subset"] = sub
        all_traces.extend(traces)
        inventory(sub, traces)

    if not all_traces:
        return []

    multi = [t for t in all_traces if len(t.outcome["agents"]) > 1]
    print(f"\n  multi-agent runs: {len(multi)}/{len(all_traces)}")
    print("\n  handover of work (top edges, multi-agent runs)")
    for (a, b), n in handover(multi).most_common(6):
        print(f"    {n:5}  {a} -> {b}")

    rule("LOCALIZATION  first structural divergence vs the expert mistake step")
    print("  Reference is mined from the corpus itself: a directly-follows edge seen")
    print("  in fewer than {:.0%} of runs counts as structurally rare. The first such".format(threshold_pct))
    print("  edge is the predicted decisive step. No LLM is involved.\n")

    for sub in sorted({t.outcome["subset"] for t in all_traces}):
        traces = [t for t in all_traces if t.outcome["subset"] == sub]
        scored = [t for t in traces if t.outcome.get("decisive_event_index") is not None]
        if len(scored) < 5:
            print(f"  [{sub}] only {len(scored)} scorable runs, skipped")
            continue

        edge_runs = collections.Counter()
        for t in traces:
            seq = t.activities
            for pair in {(a, b) for a, b in zip(seq, seq[1:])}:
                edge_runs[pair] += 1
        threshold = max(2, int(threshold_pct * len(traces)))

        exact = near = agent_hit = 0
        for t in scored:
            pred = first_divergence(t.activities, edge_runs, threshold)
            truth = t.outcome["decisive_event_index"]
            if pred is None:
                continue
            if pred == truth:
                exact += 1
            if abs(pred - truth) <= 1:
                near += 1
            if t.events[pred].resource == t.outcome.get("mistake_agent"):
                agent_hit += 1

        # trivial baselines on the same runs
        rng = random.Random(0)
        rand_exact = sum(1 for t in scored
                         if rng.randrange(len(t.events)) == t.outcome["decisive_event_index"])
        last_exact = sum(1 for t in scored
                         if len(t.events) - 1 == t.outcome["decisive_event_index"])
        agent_counts = collections.Counter(t.outcome.get("mistake_agent") for t in scored)
        major_agent, major_n = agent_counts.most_common(1)[0]

        n = len(scored)
        print(f"  [{sub}]  n={n}")
        print(f"    first-divergence   step exact {exact / n:5.1%}   within +/-1 {near / n:5.1%}"
              f"   agent {agent_hit / n:5.1%}")
        print(f"    baselines          random step {rand_exact / n:5.1%}   always-last {last_exact / n:5.1%}"
              f"   always-{major_agent} {major_n / n:5.1%}")
    print("\n  published LLM baselines on this dataset (Chen et al. 2026, best method):")
    print("    agent-level 65.9%   step-level 30.3%")
    return all_traces


# ------------------------------------------------------------- agenthallu

def ah_section(root, subsets):
    rule("CORPUS 3  AgentHallu (public, multi-agent delegation and reporting back)")
    out = []
    for sub in subsets:
        path = os.path.join(root, sub)
        if not os.path.isdir(path):
            print(f"  [skip] {sub} not found")
            continue
        traces = agenthallu.ingest_dir(path)
        inventory(sub, traces, drop=set())
        edges = handover(traces)
        print("  handover of work")
        for (a, b), n in edges.most_common(6):
            print(f"    {n:5}  {a} -> {b}")
        closed = sum(1 for (a, b), n in edges.items()
                     if edges.get((b, a), 0) == n and n)
        print(f"    reciprocated edges (every handover answered): {closed}/{len(edges)}")
        print("  performance proxy, split by the human hallucination label")
        rework(traces, lambda t: f"hallucination={t.outcome.get('is_hallucination')}")
        # does using a tool at all go with hallucinating? a structural feature against the label
        talk = {"delegate", "orchestrate", "report"}
        print(f"  hallucination label: {sum(bool(t.outcome.get('is_hallucination')) for t in traces)} "
              f"of {len(traces)} runs")
        for uses in (True, False):
            g = [t for t in traces if any(e.activity not in talk for e in t.events) == uses]
            h = sum(bool(t.outcome.get("is_hallucination")) for t in g)
            print(f"    runs {'with' if uses else 'without'} a tool call: {len(g):3}, hallucinated "
                  f"{h} ({h / len(g):.1%})" if g else "")
        out.extend(traces)
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads"))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tau2_path = os.path.join(
        a.data, "tau-data/tau2-bench/submissions/claude-opus-4-5_sierra_2026-02-26",
        "trajectories/claude-opus-4-5_high_airline_gpt-5.2_4trials.json")

    t1 = tau2_section(tau2_path) if os.path.exists(tau2_path) else []
    if os.path.exists(tau2_path):
        baselines_section(tau2_path)
        concurrency_section(tau2_path)
    t2 = te_section(os.path.join(a.data, "traceelephant/data"),
                    ["magentic-runs-gaia", "captain-runs-gaia", "swe-agent-runs-swe-bench"])
    t3 = ah_section(os.path.join(a.data, "agenthallu"), ["Magentic_One", "Camel"])

    rule("SUMMARY  one canonical schema, three sources (R1)")
    rows = [("tau2-bench airline", t1, "policy + gold actions", "reward 0/1"),
            ("TraceElephant", t2, "none shipped", "expert agent + step"),
            ("AgentHallu", t3, "none shipped", "human hallucination step")]
    print(f"  {'corpus':22} {'runs':>5} {'events':>7}  {'reference':22} labels")
    for name, traces, ref, lab in rows:
        print(f"  {name:22} {len(traces):5} {sum(len(t.events) for t in traces):7}  {ref:22} {lab}")

    summary = {name: {"runs": len(tr), "events": sum(len(t.events) for t in tr)}
               for name, tr, _, _ in rows}
    with open(os.path.join(a.out, "three_corpora_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print(f"\n  wrote {a.out}/three_corpora_summary.json")



# ------------------------------------------------- concurrency and baselines

def concurrency_section(path):
    """How often are tool calls concurrent, and what does flattening them cost?"""
    import hashlib
    rule("CONCURRENCY  parallel tool calls and what flattening them costs")
    raw = json.load(open(path))
    turns = collections.Counter()
    runs_par = 0
    for sim in raw.get("simulations") or []:
        par = False
        for m in sim.get("messages") or []:
            k = len(m.get("tool_calls") or [])
            if k:
                turns[k] += 1
            par = par or k > 1
        runs_par += par
    multi = sum(v for k, v in turns.items() if k > 1)
    print(f"  tool calls per turn: {dict(sorted(turns.items()))}")
    print(f"  turns with more than one call: {multi}/{sum(turns.values())}")
    print(f"  runs containing a parallel turn: {runs_par}/{len(raw.get('simulations') or [])}")

    traces = tau2.ingest(path)
    within = across = 0
    ew, ea = collections.Counter(), collections.Counter()
    for t in traces:
        ev = [e for e in t.ordered() if e.activity not in MESSAGES]
        for a, b in zip(ev, ev[1:]):
            ta, tb = a.attributes.get("turn_idx"), b.attributes.get("turn_idx")
            if ta is not None and ta == tb:
                within += 1
                ew[(a.activity, b.activity)] += 1
            else:
                across += 1
                ea[(a.activity, b.activity)] += 1
    tot = within + across
    print(f"\n  directly-follows relations: {across} across turns ({across/tot:.0%}), "
          f"{within} within a turn ({within/tot:.0%})")
    print(f"  distinct edges {len(set(ew) | set(ea))}, of which "
          f"{len(set(ew) - set(ea))} exist only because of within-turn serialisation")

    scored = [(t, check(t, gold_reference(t))) for t in traces]
    scored = [(t, r) for t, r in scored if r.n_constraints]
    flips = 0
    for t, base in scored:
        ref = gold_reference(t)
        for seed in range(5):
            if _reshuffled_verdict(t, ref, seed) != (base.compliant, base.fitness):
                flips += 1
                break
    print(f"  verdicts that change when within-turn order is reshuffled: "
          f"{flips}/{len(scored)}")

    def fingerprint():
        out = [(t.case_id, r.fitness, r.compliant,
                tuple((v.constraint, v.at) for v in r.violations))
               for t, r in ((x, check(x, gold_reference(x))) for x in tau2.ingest(path))]
        return hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()
    hashes = {fingerprint() for _ in range(3)}
    print(f"  R4: three full repetitions give {'one' if len(hashes) == 1 else len(hashes)} "
          f"distinct verdict hash(es) -> deterministic: {len(hashes) == 1}")


def _reshuffled_verdict(trace, ref, seed):
    from agentpm.model import Event, Trace
    rng = random.Random(seed)
    ev = trace.ordered()
    out, i = [], 0
    while i < len(ev):
        turn = ev[i].attributes.get("turn_idx")
        j = i
        while j < len(ev) and turn is not None and ev[j].attributes.get("turn_idx") == turn:
            j += 1
        block = ev[i:j] if j > i else [ev[i]]
        rng.shuffle(block)
        out.extend(block)
        i = max(j, i + 1)
    # the shuffle permutes only the serialisation: events keep their step (timestamp),
    # so a verdict that changes here depends on an order the source does not carry
    alt = Trace(case_id=trace.case_id, outcome=trace.outcome,
                events=[Event(e.activity, e.timestamp, e.resource, e.attributes, sequence=k)
                        for k, e in enumerate(out)])
    r = check(alt, ref)
    return r.compliant, r.fitness


def baselines_section(path):
    """Does the compliance signal beat features that need no reference model?"""
    rule("BASELINES  conformance against trivial features")
    traces = tau2.ingest(path)
    scored = [(t, r) for t, r in ((x, check(x, gold_reference(x))) for x in traces)
              if r.n_constraints]
    n = len(scored)
    base = sum(1 for t, _ in scored if t.outcome["reward"] == 0.0)
    print(f"  n={n}, failures={base} (base rate {base/n:.0%})\n")

    def calls(t):
        return sum(1 for e in t.events if e.activity not in MESSAGES)

    def rework_of(t):
        s = [e.activity for e in t.ordered() if e.activity not in MESSAGES]
        return 1 - len(set(s)) / len(s) if s else 0.0

    med = sorted(calls(t) for t, _ in scored)[n // 2]
    medr = sorted(rework_of(t) for t, _ in scored)[n // 2]

    def report(label, predicate):
        tp = sum(1 for t, r in scored if predicate(t, r) and t.outcome["reward"] == 0.0)
        fp = sum(1 for t, r in scored if predicate(t, r) and t.outcome["reward"] == 1.0)
        fn = sum(1 for t, r in scored if not predicate(t, r) and t.outcome["reward"] == 0.0)
        p = tp / max(1, tp + fp)
        rc = tp / max(1, tp + fn)
        print(f"    {label:44} P {p:3.0%}  R {rc:3.0%}  F1 {2*p*rc/max(1e-9, p+rc):.2f}"
              f"   flags {tp+fp:3}")

    report("gold-action reference not satisfied", lambda t, r: not r.compliant)
    report(f"more tool calls than the median ({med})", lambda t, r: calls(t) > med)
    report("any tool returned an error",
           lambda t, r: any(e.attributes.get("error") for e in t.events))
    report(f"rework above the median ({medr:.0%})", lambda t, r: rework_of(t) > medr)
    report("always predict failure", lambda t, r: True)

    fired = collections.Counter()
    for t, r in scored:
        for v in r.violations:
            fired[v.constraint.split("(")[0]] += 1
    print(f"\n  violations by constraint kind: {dict(fired)}")
    print(f"  ordering constraints account for "
          f"{fired.get('precedence', 0)/max(1, sum(fired.values())):.0%} of them, so the "
          f"signal is not only a presence check")


if __name__ == "__main__":
    sys.exit(main())
