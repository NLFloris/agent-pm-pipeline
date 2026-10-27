"""LLM-as-a-Judge baseline over tau2-bench runs.

    export LLM_BASE_URL=https://inference.proxy.ti.bfh.ch/v1
    export LLM_API_KEY=$(cat ~/.bfh_api_key)
    python experiments/judge/run_judge.py --model gpt-oss:120b --mode full
    python experiments/judge/run_judge.py --model gpt-oss:120b --mode output-only
    python experiments/judge/run_judge.py --model glm-5.3-flash --mode full --workers 4 \
        --data <retail trajectory file> --out results/judge/retail

The endpoint moved on 2026-10-23 from inference.mlmp.ti.bfh.ch/api/v1 to the
proxy above. The airline gpt-oss:120b verdicts were produced on the old one.

Both endpoints are LiteLLM proxies with a response cache. An identical request
is answered from the cache, with the same response id, so the first verdicts'
five repeats were one sample and four copies, not a deterministic model
(found 2026-10-23; the answers at temperature 0.7 differ from the default ones
on 50 of 50 runs). `--no-cache` asks the proxy to sample afresh, and every
repeat records whether it was a cache hit.

Two judges, same model, same runs:

  full         sees the domain policy and the complete transcript, including
               every tool call and its arguments, and is asked whether the
               agent followed the policy, which rule it broke and at which step.
               This is the compliance-checklist judge of Gritta et al. and the
               strongest judge-side method for this task.
  output-only  sees only the customer's request and the agent's final message.
               The lower bound: what an evaluation that reads outcomes can see.
  gold-actions sees the task's required actions and the transcript, and is asked
               whether every required action was performed. This is the same
               question the process-mining reference answers, so it is the fair
               comparison: it tests whether a judge can do that job, with the
               advantage that it can also read the arguments.

Each run is judged `--repeats` times at the model's default temperature, so the
judge's agreement with itself can be measured. Every request and every raw
response is appended to a JSONL log, which is what makes the baseline auditable.

The judge model must not be the model that produced the traces. The tau2 airline
and retail files used here were produced by claude-opus-4-5; gpt-oss:120b and
glm-5.3-flash are from other vendors, so self-preference bias does not apply.

The domain is read from the file. Only the output-only prompt names it ("normal
airline policy"); for airline the prompts are byte-identical to the ones the
first verdicts were produced with.

Standard library only, so it runs in the pipeline's own environment.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TOOL_RESULT = 600          # characters of a tool result shown to the judge

SYSTEM_FULL = (
    "You are an auditor checking whether a customer-service agent followed its "
    "operating policy. You will receive the policy and a numbered transcript of one "
    "conversation, including every tool call the agent made with its arguments. "
    "Judge the agent's PROCESS, not whether the customer ended up happy. "
    "Answer with a single JSON object and nothing else."
)
SYSTEM_OUTPUT_ONLY = (
    "You are evaluating a customer-service agent. You will receive only the "
    "customer's request and the agent's final reply. Judge whether the agent "
    "handled the request correctly and in line with normal airline policy. "
    "Answer with a single JSON object and nothing else."
)
SYSTEM_GOLD = (
    "You are checking a customer-service transcript against a list of REQUIRED ACTIONS "
    "for this task. For each required action, decide whether the agent performed it, "
    "with the arguments shown. Order matters only where one action logically depends "
    "on another. Ignore anything else the agent did; judge only the required list. "
    "Answer with a single JSON object and nothing else."
)
SCHEMA_GOLD = (
    '{"compliant": true if every required action was performed with matching arguments else false, '
    '"missing_actions": ["name of each required action not performed"], '
    '"wrong_arguments": ["name of each required action performed with different arguments"], '
    '"first_violation_step": integer step number of the first problem or null, '
    '"confidence": number between 0 and 1, '
    '"reasoning": "two or three sentences"}'
)
SCHEMA = (
    '{"compliant": true or false, '
    '"violated_rules": ["short quote or paraphrase of each policy rule broken"], '
    '"first_violation_step": integer step number or null, '
    '"confidence": number between 0 and 1, '
    '"reasoning": "two or three sentences"}'
)


def render_transcript(messages):
    """Number every turn so the judge can cite a step."""
    lines = []
    step = 0
    for m in messages:
        role = m.get("role")
        if role == "tool":
            content = str(m.get("content") or "")
            if len(content) > MAX_TOOL_RESULT:
                content = content[:MAX_TOOL_RESULT] + " ...[truncated]"
            lines.append(f"      -> tool result ({m.get('id')}): {content}")
            continue
        for call in m.get("tool_calls") or []:
            step += 1
            args = json.dumps(call.get("arguments"), ensure_ascii=False)
            lines.append(f"[{step}] {role} CALLS {call.get('name')}({args})  id={call.get('id')}")
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            step += 1
            lines.append(f"[{step}] {role}: {content.strip()}")
    return "\n".join(lines), step


def build_prompt(mode, policy, sim, gold=None, domain="airline"):
    messages = sim.get("messages") or []
    if mode == "gold-actions":
        transcript, n = render_transcript(messages)
        req = "\n".join(f"  {i+1}. {a.get('name')}({json.dumps(a.get('arguments'), ensure_ascii=False)})"
                         for i, a in enumerate(gold or []) if a.get("requestor") == "assistant")
        user = (f"REQUIRED ACTIONS for this task\n------------------------------\n{req}\n\n"
                f"TRANSCRIPT ({n} numbered steps)\n----------\n{transcript}\n\n"
                f"Were all required actions performed? Reply with JSON: {SCHEMA_GOLD}")
        return SYSTEM_GOLD, user
    if mode == "full":
        transcript, n = render_transcript(messages)
        user = (f"POLICY\n------\n{policy}\n\nTRANSCRIPT ({n} numbered steps)\n----------\n"
                f"{transcript}\n\nDid the agent follow the policy? Reply with JSON: {SCHEMA}")
        return SYSTEM_FULL, user
    # output-only: first customer message and last agent message
    first = next((m.get("content") for m in messages
                  if m.get("role") == "user" and m.get("content")), "")
    last = next((m.get("content") for m in reversed(messages)
                 if m.get("role") == "assistant" and m.get("content")), "")
    user = (f"CUSTOMER REQUEST\n----------------\n{first}\n\nAGENT'S FINAL REPLY\n"
            f"-------------------\n{last}\n\nWas this handled correctly? Reply with JSON: {SCHEMA}")
    return SYSTEM_OUTPUT_ONLY.replace("airline", domain), user


def read_stream(resp):
    """Assemble a streamed chat completion into the shape of a non-streamed one."""
    out = {"id": None, "created": None, "model": None, "usage": None}
    content, reasoning, finish = [], [], None
    for raw in resp:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        for k in ("id", "created", "model"):
            out[k] = out[k] or chunk.get(k)
        out["usage"] = chunk.get("usage") or out["usage"]
        for ch in chunk.get("choices") or []:
            delta = ch.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            finish = ch.get("finish_reason") or finish
    message = {"role": "assistant", "content": "".join(content)}
    if any(reasoning):
        message["reasoning_content"] = "".join(reasoning)
    out["choices"] = [{"index": 0, "message": message, "finish_reason": finish}]
    out["streamed"] = True
    return out


def call_model(base_url, api_key, model, system, user, timeout, temperature=None, no_cache=False,
               stream=False):
    payload = {"model": model,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    if temperature is not None:
        payload["temperature"] = temperature
    if no_cache:
        payload["cache"] = {"no-cache": True}   # LiteLLM: do not answer from the cache
    if stream:
        # a long answer arrives in pieces, so the gateway's timeout (about five
        # minutes, which cut off the longest glm answers with a 504) never fires
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = read_stream(resp) if stream else json.loads(resp.read().decode("utf-8"))
            # LiteLLM sets this header only when the answer came from its cache
            cache_hit = resp.headers.get("x-litellm-cache-key") is not None
    except urllib.error.HTTPError as exc:
        # keep the server's explanation; a bare status code is undiagnosable
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise urllib.error.HTTPError(exc.url, exc.code, f"{exc.reason}: {detail}",
                                     exc.headers, None) from None
    return payload, time.monotonic() - t0, cache_hit


def parse_verdict(text):
    """The model is asked for bare JSON; tolerate a fenced block or trailing prose."""
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(v, dict) or "compliant" not in v:
        return None
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["full", "output-only", "gold-actions"], default="full")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1"))
    ap.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", "not-needed"))
    ap.add_argument("--data", default=os.path.expanduser(
        "~/Downloads/tau-data/tau2-bench/submissions/claude-opus-4-5_sierra_2026-02-26/"
        "trajectories/claude-opus-4-5_high_airline_gpt-5.2_4trials.json"))
    ap.add_argument("--out", default="results/judge")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--limit", type=int, help="judge only the first N scorable runs")
    ap.add_argument("--cases", help="judge only the case ids listed in this JSON file, "
                    "e.g. a stratified sample from sample_runs.py")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--temperature", type=float, default=None,
                    help="set explicitly to test reproducibility; the endpoint default "
                         "turned out to be deterministic, which makes repeats uninformative")
    ap.add_argument("--no-cache", action="store_true",
                    help="ask the proxy to sample afresh instead of answering from its cache")
    ap.add_argument("--stream", action="store_true",
                    help="receive the answer as a stream; same answer, no gateway timeout")
    ap.add_argument("--workers", type=int, default=1,
                    help="runs judged in parallel; repeats of one run stay sequential")
    ap.add_argument("--dry-run", action="store_true", help="build prompts, call nothing")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(a.data, encoding="utf-8") as f:
        data = json.load(f)
    policy = data["info"]["environment_info"]["policy"]
    domain = data["info"]["environment_info"].get("domain_name") or "airline"
    gold_by_task = {str(t.get("id")): (t.get("evaluation_criteria") or {}).get("actions") or []
                    for t in data.get("tasks") or []}
    sims = [s for s in data["simulations"]
            if (s.get("reward_info") or {}).get("action_checks")]   # scorable only
    if a.cases:
        with open(a.cases, encoding="utf-8") as f:
            keep = set(json.load(f)["case_ids"])
        sims = [s for s in sims if s["id"] in keep]
    if a.limit:
        sims = sims[:a.limit]

    tag = f"{a.mode}_{re.sub(r'[^A-Za-z0-9]+', '-', a.model)}"
    if a.temperature is not None:
        tag += f"_t{a.temperature:g}"
    verdict_path = os.path.join(a.out, f"verdicts_{tag}.json")
    log_path = os.path.join(a.out, f"raw_{tag}.jsonl")

    done = {}
    if os.path.exists(verdict_path):
        with open(verdict_path, encoding="utf-8") as f:
            for v in json.load(f):
                # a record whose repeats all failed is not done; redo it
                if any(r.get("parsed") for r in v.get("repeats") or []):
                    done[v["case_id"]] = v
        print(f"resuming: {len(done)} runs already judged with at least one parsed verdict")

    print(f"{a.mode} judge, model {a.model}, {len(sims)} runs x {a.repeats} repeats")
    if a.dry_run:
        system, user = build_prompt(a.mode, policy, sims[0], gold_by_task.get(str(sims[0].get("task_id"))),
                                     domain)
        print(f"\n--- system ---\n{system}\n--- example prompt ({len(user)} chars) ---\n{user[:1500]}\n...")
        return 0

    results = list(done.values())
    lock = threading.Lock()

    def judge(sim):
        system, user = build_prompt(a.mode, policy, sim, gold_by_task.get(str(sim.get("task_id"))), domain)
        repeats = []
        for k in range(a.repeats):
            payload, text, usage, seconds, err, cache_hit = None, None, {}, None, None, None
            attempt, busy = 0, 0
            while attempt < 4:
                try:
                    payload, seconds, cache_hit = call_model(a.base_url, a.api_key, a.model, system,
                                                             user, a.timeout, a.temperature, a.no_cache,
                                                             a.stream)
                    text = payload["choices"][0]["message"]["content"]
                    usage = payload.get("usage") or {}
                    err = None
                    break
                except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    # 429: the proxy's parallel slots for this model are taken, by us or
                    # by other users; wait for one instead of spending an attempt
                    if getattr(exc, "code", None) == 429 and busy < 360:
                        busy += 1
                        time.sleep(5)
                        continue
                    # the hosted endpoint answered a burst of ~80 consecutive requests
                    # with 400 once; back off instead of burning through the run set
                    attempt += 1
                    if attempt < 4:
                        time.sleep(5 * (2 ** (attempt - 1)))
            verdict = parse_verdict(text)
            repeats.append({
                "repeat": k, "compliant": verdict.get("compliant") if verdict else None,
                "violated_rules": verdict.get("violated_rules") if verdict else None,
                "missing_actions": verdict.get("missing_actions") if verdict else None,
                "wrong_arguments": verdict.get("wrong_arguments") if verdict else None,
                "first_violation_step": verdict.get("first_violation_step") if verdict else None,
                "confidence": verdict.get("confidence") if verdict else None,
                "parsed": verdict is not None, "error": err, "seconds": seconds, "cache_hit": cache_hit,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            })
            line = json.dumps({
                "case_id": sim["id"], "repeat": k, "mode": a.mode, "model": a.model,
                "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "system": system, "user": user, "response": payload, "error": err,
                "cache_hit": cache_hit, "no_cache": a.no_cache,
            }) + "\n"
            with lock, open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
        record = {"case_id": sim["id"], "task_id": sim.get("task_id"), "trial": sim.get("trial"),
                  "mode": a.mode, "model": a.model, "repeats": repeats}
        with lock:
            results.append(record)
            with open(verdict_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=1)
            votes = [r["compliant"] for r in repeats if r["compliant"] is not None]
            print(f"[{len(results)}/{len(sims)}] {sim['id'][:8]}  votes compliant="
                  f"{sum(1 for v in votes if v)}/{len(votes)}  "
                  f"{sum(r['seconds'] or 0 for r in repeats):.0f}s", flush=True)

    todo = [sim for sim in sims if sim["id"] not in done]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        for fut in [pool.submit(judge, sim) for sim in todo]:
            fut.result()
    print(f"\nwrote {verdict_path} and {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
