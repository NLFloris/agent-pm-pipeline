"""Task types assigned without the solution list, and the engine run on them.

    set -a; . ./.env; set +a
    python experiments/inferred_task_types.py classify --domain airline --input customer
    python experiments/inferred_task_types.py classify --domain airline --input customer+lookups
    python experiments/inferred_task_types.py classify --domain retail --input customer+lookups
    python experiments/inferred_task_types.py score --domain airline
    python experiments/inferred_task_types.py score --domain retail

In task_type_conformance.py the task type of a run is read from the write
actions on the task's solution list, the same list the per-action label is
computed from. A flag for a missing required write is then a violation of that
label by construction. Here an LLM assigns the type instead, from what the
run's customer says and the domain policy, without the solution list, the
agent's messages or the agent's actions. Two inputs:

  customer           every message of the simulated customer, nothing else
  customer+lookups   the same, plus the results of the agent's read-only tool
                     calls (user, reservation, order and product details), so
                     that the classifier can tell whether the policy allows a
                     request

The classifier answers with the requests it sees and, for each, whether the
policy lets the agent carry it out, refuse it or hand it to a human. The
actions it would carry out are mapped to a type key exactly as task_type()
maps the solution list, so the same type models apply. `score` compares the
inferred types with the ones from the solution list and runs the engine with
each.

Prompt versions. v1 (2026-11-10) asked for the writes and a transfer flag
directly. Its two main errors on airline were general ones, read from the
answers' reasons: a request the policy forbids was called a transfer (the
policy says to transfer only if a request "cannot be handled within the scope
of your actions", and refusing is within scope), and a change was left out
because a step such as a lookup or the customer's yes had not happened yet at
some point in the conversation. v2 (2026-11-10) states the policy's transfer
rule, asks for the changes by the end of the conversation, assumes the
required steps, and asks for a decision per request before the actions. It is
domain-generic and was frozen before its first run; retail was never run with
v1, so retail is an unseen check of v2.

TRANSFER. In v2 the main mapping reads only the actions the agent must carry
out, and a transfer counts as no write. Whether a refused request ends in a
transfer is not settled by the benchmark itself: airline task 13 (a change of
origin, which the policy forbids) has a transfer on its solution list, while
task 36 (a basic-economy flight that has already flown, for which the policy
says "transfer is needed") has none. A transfer is allowed in the no-write
models, so the only cost is that the transfer type (task 13 on airline) cannot
be inferred. The mapping that keeps the transfer is scored next to it. Both
were fixed after four test answers of v2 and before its full run.

The classifier is gpt-oss:120b, the model of the Judges, at temperature 0 and
without the proxy's response cache. Every prompt and answer is logged.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "judge"))

from agentpm.analysis.conformance import check   # noqa: E402
from agentpm.ingest import tau2                  # noqa: E402
from run_judge import call_model                 # noqa: E402
import retail_task_types as rt                    # noqa: E402
import task_type_conformance as at                # noqa: E402

TRAJ = os.path.join(HERE, "..", "..", "Thesis", "data", "tau-data", "tau2-bench", "submissions",
                    "claude-opus-4-5_sierra_2026-02-26", "trajectories")
DATA = {"airline": os.path.join(TRAJ, "claude-opus-4-5_high_airline_gpt-5.2_4trials.json"),
        "retail": os.path.join(TRAJ, "claude-opus-4-5_high_retail_gpt-5.2_4trials.json")}
OUT = os.path.join(HERE, "..", "results", "inferred_task_types")
MAX_RESULT = 2500

DOMAIN = {
    "airline": {
        "writes": at.WRITES,
        "reads": ["get_user_details", "get_reservation_details", "search_direct_flight",
                  "search_onestop_flight", "get_flight_status", "list_all_airports", "calculate"],
        "task_type": at.task_type, "type_model": at.type_model, "prepare": lambda t: t,
        "describe": {
            "book_reservation": "book a new reservation",
            "cancel_reservation": "cancel an existing reservation",
            "update_reservation_flights": "change the flights or the cabin of an existing reservation",
            "update_reservation_baggages": "change the number of checked bags of a reservation",
            "update_reservation_passengers": "change the passenger details of a reservation",
            "send_certificate": "send the customer a travel certificate as compensation",
        },
    },
    "retail": {
        "writes": rt.WRITES, "reads": rt.READS,
        "task_type": rt.task_type, "type_model": rt.type_model, "prepare": rt.abstract,
        "describe": {
            "cancel_pending_order": "cancel a pending order",
            "modify_pending_order_address": "change the shipping address of a pending order",
            "modify_pending_order_items": "change items of a pending order to other options of the same product",
            "modify_pending_order_payment": "change the payment method of a pending order",
            "modify_user_address": "change the user's default address",
            "return_delivered_order_items": "return items of a delivered order",
            "exchange_delivered_order_items": "exchange items of a delivered order for other options of the same product",
        },
    },
}

# ---------------------------------------------------------------- prompt v1
SYSTEM_V1 = (
    "You classify customer-service requests for an airline. You receive the airline's "
    "policy and what one customer said during a conversation with an agent. Decide which "
    "changes the agent should make for this customer under the policy. "
    "Answer with a single JSON object and nothing else."
)
SCHEMA_V1 = ('{"writes": [zero or more of the action names above], '
             '"transfer": true or false, "reason": "one sentence"}')

# ---------------------------------------------------------------- prompt v2
SYSTEM_V2 = (
    "You read what happened in one {domain} customer-service conversation and decide which "
    "changes to the database the agent must have made by the end of the conversation if it "
    "follows its policy exactly. You judge the task, not the agent: you do not see what the "
    "agent did. Answer with a single JSON object and nothing else."
)
RULES_V2 = """HOW TO DECIDE
1. Read the whole conversation. A customer can make several requests, add one late, or drop one. Use the requests the customer still wants at the end.
2. For each request, decide from the policy{facts} what the agent must do:
   - "do": the policy allows it, so the agent carries it out with one of the actions above.
   - "refuse": the policy does not allow it, or it needs no change to the database (a question, a complaint, a request for information). Refusing is handled by the agent itself.
   - "transfer": the policy says the request cannot be handled within the scope of the agent's actions. The policy says: transfer "if and only if the request cannot be handled within the scope of your actions". A request that the policy forbids is refused, not transferred, even if the customer asks for a human or insists.
3. Assume the agent performs every step the policy requires before a change, such as identifying the customer, looking up the record, and getting the customer's explicit yes. Never mark a request as "refuse" because such a step had not happened yet at some moment in the conversation.
4. A request the customer withdraws, or no longer wants after hearing the details, is left out."""
SCHEMA_V2 = ('{"requests": [{"request": "short description", "decision": "do" or "refuse" or '
             '"transfer", "action": one action name above if decision is "do", otherwise null, '
             '"policy": "the policy rule that decides it, short"}]}')


def customer_lines(messages, writes, reads, with_lookups):
    """Customer messages, and optionally the results of read-only tool calls, in order."""
    calls = {}
    for m in messages:
        for c in m.get("tool_calls") or []:
            if c.get("name") in reads:
                calls[c.get("id")] = c
    lines, i = [], 0
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
            i += 1
            text = m["content"].replace("###STOP###", "").replace("###TRANSFER###", "").strip()
            lines.append(f"[{i}] customer: {text}")
        elif with_lookups and m.get("role") == "tool" and m.get("id") in calls:
            c = calls[m["id"]]
            res = str(m.get("content") or "")
            if len(res) > MAX_RESULT:
                res = res[:MAX_RESULT] + " ...[truncated]"
            args = json.dumps(c.get("arguments"), ensure_ascii=False)
            lines.append(f"      lookup {c['name']}({args}) -> {res}")
    return "\n".join(lines)


def build_prompt(version, kind, domain, policy, sim):
    cfg = DOMAIN[domain]
    lookups = kind == "customer+lookups"
    body = customer_lines(sim.get("messages") or [], cfg["writes"], cfg["reads"], lookups)
    what = ("WHAT THE CUSTOMER SAID, WITH THE RESULTS OF THE AGENT'S LOOKUPS" if lookups
            else "WHAT THE CUSTOMER SAID")
    actions = "\n".join(f"  {k}: {v}" for k, v in cfg["describe"].items())
    head = (f"POLICY\n------\n{policy}\n\n{what}\n{'-' * len(what)}\n{body}\n\n"
            f"ACTIONS THAT CHANGE THE DATABASE\n--------------------------------\n{actions}\n\n")
    if version == "v1":
        user = head + ("Which of these actions should the agent perform, given the policy? List an "
                       "action only if the customer asks for it and the policy allows it; list none if "
                       "the request must be refused or needs no change. Set transfer to true only if "
                       f"the policy requires handing the customer to a human agent. Reply with JSON: {SCHEMA_V1}")
        return SYSTEM_V1, user
    facts = (" and the facts in the lookups" if lookups
             else " and the facts the customer states; you do not see the agent's lookups")
    user = head + RULES_V2.format(facts=facts) + f"\n\nReply with JSON: {SCHEMA_V2}"
    return SYSTEM_V2.format(domain=domain), user


def parse(text, version):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(v, dict):
        return None
    if version == "v1":
        return v if isinstance(v.get("writes"), list) else None
    reqs = v.get("requests")
    if not isinstance(reqs, list) or not all(isinstance(r, dict) for r in reqs):
        return None
    return v


def to_key(v, version, writes_all, keep_transfer=None):
    """The same mapping as task_type(), applied to the classifier's answer.

    v2 maps a transfer to no-write unless keep_transfer is set: see TRANSFER."""
    if keep_transfer is None:
        keep_transfer = version == "v1"
    if version == "v1":
        writes, transfer = v.get("writes", []), v.get("transfer")
    else:
        reqs = v["requests"]
        writes = [r.get("action") for r in reqs if r.get("decision") == "do"]
        transfer = any(r.get("decision") == "transfer" for r in reqs)
    transfer = transfer and keep_transfer
    writes = sorted(set(w for w in writes if w in writes_all))
    if writes:
        return "+".join(writes)
    return "transfer" if transfer else "no-write"


def out_path(domain, kind, version):
    return os.path.join(OUT, f"{domain}_{kind.replace('+', '_')}_{version}.jsonl")


def classify(a):
    data = json.load(open(DATA[a.domain]))
    policy = data["info"]["environment_info"]["policy"]
    scorable = {t.case_id for t in tau2.ingest(DATA[a.domain]) if t.outcome.get("action_checks")}
    sims = [s for s in data["simulations"] if s["id"] in scorable]
    os.makedirs(OUT, exist_ok=True)
    path = out_path(a.domain, a.input, a.prompt)
    done = set(load(path)) if os.path.exists(path) else set()
    todo = [s for s in sims if s["id"] not in done][: a.limit]
    print(f"{len(sims)} scorable runs, {len(done)} done, {len(todo)} to classify")
    lock = threading.Lock()
    writes_all = DOMAIN[a.domain]["writes"]

    def one(sim):
        system, user = build_prompt(a.prompt, a.input, a.domain, policy, sim)
        row = {"case_id": sim["id"], "task_id": sim["task_id"], "trial": sim.get("trial"),
               "domain": a.domain, "input": a.input, "prompt_version": a.prompt, "model": a.model,
               "system": system, "prompt": user, "attempts": []}
        for attempt in range(5):
            try:
                resp, secs, hit = call_model(a.base_url, a.api_key, a.model, system, user, 300,
                                             temperature=0, no_cache=True)
            except Exception as exc:
                row["attempts"].append({"error": str(exc)[:300]})
                time.sleep(10 * (attempt + 1))   # the endpoint allows two requests at a time
                continue
            text = resp["choices"][0]["message"].get("content")
            v = parse(text, a.prompt)
            row["attempts"].append({"answer": text, "usage": resp.get("usage"), "seconds": secs,
                                    "cache_hit": hit, "parsed": v is not None})
            if v is not None:
                row.update({"verdict": v, "type": to_key(v, a.prompt, writes_all)})
                break
        with lock, open(path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row.get("type")

    with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"wrote {path}")


def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("type"):
            rows[r["case_id"]] = r
    return rows


def metrics(flags, truth):
    """flags/truth: lists of booleans, True = violation."""
    tp = sum(f and t for f, t in zip(flags, truth))
    fp = sum(f and not t for f, t in zip(flags, truth))
    fn = sum(t and not f for f, t in zip(flags, truth))
    n = len(flags)
    tn = n - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return {"n": n, "flags": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(p, 2), "recall": round(r, 2), "kappa": round(k, 2)}


def tokens(row):
    if "attempts" not in row:   # v1 rows keep only the last answer
        return (row.get("usage") or {}).get("total_tokens", 0)
    return sum((x.get("usage") or {}).get("total_tokens", 0) for x in row["attempts"])


def score(a):
    cfg = DOMAIN[a.domain]
    traces = [t for t in tau2.ingest(DATA[a.domain]) if t.outcome.get("action_checks")]
    tr = {t.case_id: cfg["prepare"](t) for t in traces}
    given = {t.case_id: cfg["task_type"](t) for t in traces}
    sources = {"solution list": (given, None)}
    for kind in ("customer", "customer+lookups"):
        for version in ("v1", "v2"):
            p = out_path(a.domain, kind, version)
            if os.path.exists(p):
                rows = load(p)
                sources[f"{kind}, {version}"] = (
                    {c: to_key(r["verdict"], version, cfg["writes"]) for c, r in rows.items()}, rows)
                if version == "v2":
                    sources[f"{kind}, {version}, transfer kept"] = (
                        {c: to_key(r["verdict"], "v2", cfg["writes"], keep_transfer=True)
                         for c, r in rows.items()}, rows)
    viol = {t.case_id: not all(c.get("action_match") for c in t.outcome["action_checks"])
            for t in traces}
    reward_fail = {t.case_id: t.outcome.get("reward") != 1.0 for t in traces}
    verdict = {}

    def flagged(c, key):
        if (c, key) not in verdict:
            verdict[(c, key)] = not check(tr[c], cfg["type_model"](key)).compliant
        return verdict[(c, key)]

    report = {}
    lines = [f"{a.domain}: {len(traces)} scorable runs, {sum(viol.values())} violations "
             f"(per-action label)\n"]
    for name, (types, rows) in sources.items():
        ids = [t.case_id for t in traces if t.case_id in types]
        fl = [flagged(c, types[c]) for c in ids]
        agree = sum(types[c] == given[c] for c in ids)
        m_label = metrics(fl, [viol[c] for c in ids])
        m_reward = metrics(fl, [reward_fail[c] for c in ids])
        entry = {"runs": len(ids), "type_agrees": agree,
                 "vs_per_action_label": m_label, "vs_reward": m_reward}
        lines.append(f"== task type from {name}: {len(ids)} runs, type equals the solution-list "
                     f"type on {agree} ({agree / len(ids):.0%})")
        lines.append(f"   engine vs per-action label: {m_label}")
        lines.append(f"   engine vs reward:           {m_reward}")
        if rows is not None:
            tok = [tokens(rows[c]) for c in ids]
            per_task = collections.defaultdict(set)
            for c in ids:
                per_task[rows[c]["task_id"]].add(types[c])
            stable = sum(len(v) == 1 for v in per_task.values())
            same = [c for c in ids if types[c] == given[c]]
            changed = sum(flagged(c, types[c]) != flagged(c, given[c]) for c in ids)
            entry.update({"tokens_per_run": round(sum(tok) / len(tok)), "tasks_stable": stable,
                          "tasks": len(per_task), "verdict_changed": changed})
            lines.append(f"   tokens per run {sum(tok) / len(tok):.0f}; tasks with one type over all "
                         f"trials {stable}/{len(per_task)}; engine verdict changes on {changed} runs")
            if same:
                ms = metrics([flagged(c, given[c]) for c in same], [viol[c] for c in same])
                lines.append(f"   on the {len(same)} runs with the right type: {ms}")
            conf = collections.Counter((given[c], types[c]) for c in ids if types[c] != given[c])
            lines.append("   disagreements (solution-list type -> inferred type: runs)")
            for (g, i), n in conf.most_common(15):
                lines.append(f"     {g:50} -> {i:50} {n}")
            entry["disagreements"] = [{"given": g, "inferred": i, "runs": n}
                                      for (g, i), n in conf.most_common()]
        report[name] = entry
        lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, f"score_{a.domain}.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT, f"score_{a.domain}.json"), "w") as f:
        json.dump(report, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("--domain", choices=sorted(DATA), default="airline")
    c.add_argument("--input", choices=["customer", "customer+lookups"], required=True)
    c.add_argument("--prompt", choices=["v1", "v2"], default="v2")
    c.add_argument("--model", default="gpt-oss:120b")
    c.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    c.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"))
    c.add_argument("--workers", type=int, default=2)
    c.add_argument("--limit", type=int)
    s = sub.add_parser("score")
    s.add_argument("--domain", choices=sorted(DATA), default="airline")
    a = ap.parse_args()
    return classify(a) if a.cmd == "classify" else score(a)


if __name__ == "__main__":
    sys.exit(main())
