"""One command, every output: source check, event logs, conformance, pictures.

    python -m agentpm.cli report <path> --format <f> -o <folder> [--reference ref.json]

This is the whole pipeline in one call. It writes into the folder:

  CHECKS.txt          what the source exposes, checked automatically: tool calls,
                      agent attribution, arguments, results, reasoning, concurrency,
                      business objects, variability, determinism. Read this first.
  inventory.txt       runs, events, activities, agents, and every run on one line
  event_log.csv       the flat event log, one row per event
  event_log.xes       the same in XES (needs pm4py)
  runs.ocel.json      the object-centric log, OCEL 2.0
  conformance.txt     every run against the reference, with the event responsible
  verdicts.json       the same, machine-readable
  pm4py_*.png         DFG, Petri net, process tree, BPMN from the inductive miner
                      (needs pm4py), and pm4py_summary.txt
  handover.png, lane_model.png, dfg_by_agent.png, run_<id>.png
                      the pipeline's own pictures (needs graphviz)
  README.md           what each file is

Conformance: with --reference the given model is used, and --by-agent checks it
against Agent.activity names (agent_qualified below); with --task-types every run
is checked against the model of its task type; without either, a source that
carries gold actions (tau2) is checked per task against them, and any other
source is not checked. Pictures and pm4py models leave out communication
turns and framework bookkeeping (messages, agent creation, the group chat's
turn-taking); the logs keep everything. The core of this module needs the
standard library only; pm4py and graphviz are used when installed.
"""
from __future__ import annotations

import collections
import csv
import dataclasses
import hashlib
import json
import os
from typing import Any, Optional

from .analysis.conformance import check
from .analysis.discovery import concurrency_summary
from .logbuild.objects import ObjectRule
from .logbuild.ocel import build_ocel_from_traces, object_sharing
from .model import Event, Trace
from .reference import Reference, TaskTypes

# activities that are not an executed tool call: communication, coordination,
# framework bookkeeping
NON_ACTION = {"assistant_message", "user_message", "orchestrate", "report", "delegate",
              "invoke_agent", "invoke_workflow", "create_agent", "llm_call", "user_prompt",
              "subagent_completed", "handoff", "guardrail"}
PICTURE_DROP = {"assistant_message", "user_message", "invoke_workflow", "create_agent",
                "invoke_agent", "llm_call", "user_prompt"}
_CELL = 2000


def _w(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def _cell(v: Any) -> Any:
    if v is None or isinstance(v, (int, float, bool)):
        return v
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return s if len(s) <= _CELL else s[:_CELL] + f"... [{len(s)} chars]"


def agent_qualified(trace: Trace) -> Trace:
    """The run with every activity named Agent.activity, and Parent.spawn:Child for a sub-agent start.

    For a reference model that says which agent does what. Only the conformance
    check reads these names; the logs and pictures keep the plain tool names.
    experiments/local_agents/check_claims.py relabels the same way.
    """
    def name(e: Event) -> str:
        if e.activity == "spawn":
            return f"{e.resource}.spawn:{e.attributes.get('spawns')}"
        return f"{e.resource}.{e.activity}"
    return Trace(trace.case_id, [dataclasses.replace(e, activity=name(e)) for e in trace.events], trace.outcome)


def gold_reference(trace: Trace) -> Reference:
    names = [a["name"] for a in trace.outcome.get("gold_actions", []) if a.get("requestor") == "assistant"]
    pairs = [[a, b] for a, b in zip(names, names[1:]) if a != b]
    return Reference(name="gold", required=sorted(set(names)), precedence=pairs)


# ------------------------------------------------------------------ checks

def source_checks(traces: list[Trace], ocel: dict[str, Any], ocel_again: dict[str, Any]) -> list[tuple[str, str]]:
    """(status, line) pairs; status is 'ok', 'warn' or 'info'."""
    out: list[tuple[str, str]] = []
    actions = [e for t in traces for e in t.events if e.activity not in NON_ACTION]
    n = len(actions)
    if n == 0:
        out.append(("warn", "no tool calls at all. The framework may resolve tool use internally and emit only "
                            "text (AutoGen's stock agents do). Conformance can only see messages."))
        return out
    out.append(("ok", f"{n} tool calls over {len({e.activity for e in actions})} distinct tools"))

    actors = collections.Counter(e.resource for e in actions)
    if len(actors) == 1:
        who = next(iter(actors))
        out.append(("warn" if who in {"main", "unknown", "root"} else "info",
                    f"one actor for every tool call ({who}). Either a single-agent source, or the source does not "
                    "say which agent ran a tool (Claude Code's OpenTelemetry does not)."))
    else:
        out.append(("ok", "tool calls attributed to " + ", ".join(f"{a} ({c})" for a, c in actors.most_common())))

    with_args = sum(1 for e in actions if e.attributes.get("arguments") not in (None, "", {}, "{}"))
    with_res = sum(1 for e in actions if e.attributes.get("result") not in (None, ""))
    if with_args == 0:
        out.append(("warn", "no tool arguments. The log says which tools ran, by whom and in what order, but not "
                            "what they were called with, so a call with a wrong argument looks correct. "
                            "(OpenTelemetry makes arguments opt-in, and many frameworks leave them out.)"))
    else:
        out.append(("ok" if with_args == n else "info", f"arguments on {with_args}/{n} tool calls, results on {with_res}/{n}"))

    reasoning = sum(1 for t in traces for e in t.events if e.attributes.get("reasoning"))
    out.append(("info", f"reasoning text on {reasoning} events (kept as an attribute, never an activity)"))

    tool_only = [Trace(t.case_id, [e for e in t.events if e.activity not in NON_ACTION], t.outcome) for t in traces]
    s = concurrency_summary(tool_only)
    if s["steps_with_more_than_one_event"]:
        out.append(("info", f"{s['steps_with_more_than_one_event']} turns hold several tool calls issued together "
                            f"(largest {s['largest_step']}, in {s['traces_with_concurrency']} runs). They share a "
                            "timestamp; flat logs and pm4py serialise them anyway."))
    else:
        overlap = 0
        for t in traces:
            spans = [(e.attributes.get("started_ns"), e.attributes.get("ended_ns")) for e in t.events
                     if e.activity not in NON_ACTION and e.attributes.get("started_ns") and e.attributes.get("ended_ns")]
            spans.sort()
            overlap += sum(1 for (a0, a1), (b0, b1) in zip(spans, spans[1:]) if b0 < a1)
        if overlap:
            out.append(("warn", f"{overlap} tool calls overlap in real time with the previous one, but each has its "
                                "own logical timestamp. Real concurrency is in started_ns/ended_ns, not in the order."))
        else:
            out.append(("ok", "strictly sequential: no calls issued together, no overlap in real time"))

    share = object_sharing(ocel)
    otype = {o["id"]: o["type"] for o in ocel["objects"]}
    tool_events = [e for e in ocel["events"] if e["type"] not in NON_ACTION]
    b = sum(1 for e in tool_events if {otype[r["objectId"]] for r in e["relationships"]} - {"run", "agent", "tool"})
    if b == 0:
        out.append(("info", "no business objects in the OCEL log, only run, agent and tool. Give --objects "
                            "rules.json if the tool arguments name things (a ticket, a file, a reservation)."))
    else:
        out.append(("ok", f"{b}/{len(tool_events)} tool calls touch a business object: " +
                    ", ".join(f"{k} {v}" for k, v in share["objects_by_type"].items() if k not in {'run', 'agent', 'tool'})))

    variants = collections.Counter(tuple(e.activity for e in t.ordered() if e.activity not in NON_ACTION) for t in traces)
    once = sum(1 for v in variants.values() if v == 1)
    out.append(("warn" if len(traces) >= 10 and len(variants) > len(traces) / 2 else "info",
                f"{len(variants)} variants over {len(traces)} runs, {once} of them once" +
                (". Expect spaghetti; the pictures are frequency-filtered." if len(traces) >= 10 and len(variants) > len(traces) / 2 else "")))

    same = hashlib.sha256(json.dumps(ocel, sort_keys=True).encode()).hexdigest() == \
        hashlib.sha256(json.dumps(ocel_again, sort_keys=True).encode()).hexdigest()
    out.append(("ok" if same else "warn", "OCEL log built twice: " + ("byte-identical" if same else "DIFFERENT")))
    return out


# ------------------------------------------------------------------ writers

def write_event_log(traces: list[Trace], path: str) -> int:
    extra = ["to", "call_id", "duration_ms", "error"]
    if any(e.attributes.get("spawns") or e.attributes.get("spawned_by") for t in traces for e in t.events):
        extra += ["spawns", "spawned_by"]          # only for sources with sub-agents
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "position", "step", "logical_time", "sequence", "activity", "agent",
                    "arguments", "result", "reasoning"] + extra)
        for t in traces:
            labels = t.step_labels()
            for pos, e in enumerate(t.ordered()):
                a = e.attributes
                w.writerow([t.case_id, pos, labels[pos], e.timestamp, e.sequence, e.activity, e.resource,
                            _cell(a.get("arguments")), _cell(a.get("result")), _cell(a.get("reasoning"))]
                           + [_cell(a.get(k)) for k in extra])
                rows += 1
    return rows


def inventory_text(traces: list[Trace]) -> str:
    acts = collections.Counter(e.activity for t in traces for e in t.events)
    res = collections.Counter(e.resource for t in traces for e in t.events)
    lines = [f"{len(traces)} runs, {sum(len(t.events) for t in traces)} events, {len(acts)} activities, {len(res)} agents",
             "activities: " + ", ".join(f"{k}({v})" for k, v in acts.most_common()),
             "agents:     " + ", ".join(f"{k}({v})" for k, v in res.most_common()), ""]
    for t in traces[:500]:
        lines.append(f"{t.case_id}: " + " > ".join(f"{e.resource}:{e.activity}" for e in t.ordered()
                                                   if e.activity not in PICTURE_DROP))
    if len(traces) > 500:
        lines.append(f"... {len(traces) - 500} more runs")
    return "\n".join(lines)


def conformance_text(traces: list[Trace], refs: list[Reference], out: str) -> str:
    verdicts, lines = [], []
    for t, ref in zip(traces, refs):
        r = check(t, ref)
        ordered = t.ordered()
        labels = t.step_labels()
        verdicts.append({"case_id": t.case_id, "reference": ref.name, "compliant": r.compliant, "fitness": r.fitness,
                         "violations": [{"constraint": v.constraint, "at": v.at,
                                         "at_label": labels[v.at] if v.at is not None and v.at < len(labels) else None,
                                         "description": v.description}
                                        for v in r.violations]})
        lines.append(f"{'ok       ' if r.compliant else 'VIOLATION'} {t.case_id}  fitness {r.fitness:.2f}")
        for v in r.violations:
            where = ""
            if v.at is not None and v.at < len(ordered):
                where = f" at step {labels[v.at]} ({ordered[v.at].resource}:{ordered[v.at].activity})"
            lines.append(f"      {v.constraint}{where}")
    with open(os.path.join(out, "verdicts.json"), "w", encoding="utf-8") as f:
        json.dump(verdicts, f, indent=1)
    ok = sum(1 for v in verdicts if v["compliant"])
    head = [f"reference: {refs[0].name if len(set(r.name for r in refs)) == 1 else 'per run'}",
            f"{ok}/{len(verdicts)} runs compliant, mean fitness "
            f"{sum(v['fitness'] for v in verdicts) / max(1, len(verdicts)):.3f}", ""]
    return "\n".join(head + lines)


def _visible(traces: list[Trace]) -> list[Trace]:
    return [Trace(t.case_id, [e for e in t.ordered() if e.activity not in PICTURE_DROP], t.outcome) for t in traces]


def pm4py_outputs(traces: list[Trace], out: str, noise: float, say=print) -> Optional[str]:
    say("pm4py: loading the library")
    try:
        import pandas as pd
        import pm4py
    except ImportError:
        say("pm4py is not installed: no XES file and no mined models")
        return None
    from datetime import datetime, timedelta, timezone
    base = datetime(2000, 1, 1, tzinfo=timezone.utc)
    rows = [{"case:concept:name": t.case_id, "concept:name": e.activity,
             "time:timestamp": base + timedelta(seconds=i), "org:resource": e.resource}
            for t in traces for i, e in enumerate(t.ordered())]
    if not rows:
        return "pm4py: nothing to mine"
    log = pm4py.format_dataframe(pd.DataFrame(rows), case_id="case:concept:name",
                                 activity_key="concept:name", timestamp_key="time:timestamp")
    pm4py.write_xes(log, os.path.join(out, "event_log.xes"))
    say(f"event_log.xes written ({len(log)} events)")
    vis = _visible(traces)
    vrows = [r for r in rows if r["concept:name"] not in PICTURE_DROP]
    vlog = pm4py.format_dataframe(pd.DataFrame(vrows), case_id="case:concept:name",
                                  activity_key="concept:name", timestamp_key="time:timestamp")
    lines = [f"pm4py {pm4py.__version__}, inductive miner noise threshold {noise}"]
    try:
        say(f"pm4py.discover_dfg on {len(vlog)} events")
        dfg, start, end = pm4py.discover_dfg(vlog)
        pm4py.save_vis_dfg(dfg, start, end, os.path.join(out, "pm4py_dfg.png"), rankdir="TB")
        say(f"pm4py inductive miner, noise threshold {noise}")
        net, im, fm = pm4py.discover_petri_net_inductive(vlog, noise_threshold=noise)
        pm4py.save_vis_petri_net(net, im, fm, os.path.join(out, "pm4py_petri_net.png"), rankdir="TB")
        tree = pm4py.discover_process_tree_inductive(vlog, noise_threshold=noise)
        pm4py.save_vis_process_tree(tree, os.path.join(out, "pm4py_process_tree.png"))
        bpmn = pm4py.discover_bpmn_inductive(vlog, noise_threshold=noise)
        pm4py.save_vis_bpmn(bpmn, os.path.join(out, "pm4py_bpmn.png"), rankdir="TB")
        say(f"Petri net {len(net.places)} places, {len(net.transitions)} transitions; process tree and BPMN drawn")
        variants = pm4py.get_variants(vlog)
        lines += [f"{len(vis)} runs, {len(vlog)} events shown, {vlog['concept:name'].nunique()} activities",
                  f"start activities: {dict(pm4py.get_start_activities(vlog))}",
                  f"end activities:   {dict(pm4py.get_end_activities(vlog))}",
                  f"directly-follows edges: {len(dfg)}, variants: {len(variants)}",
                  f"Petri net: {len(net.places)} places, {len(net.transitions)} transitions",
                  f"process tree: {tree}"]
        for v, c in sorted(variants.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"  {c}x  " + " > ".join(v))
    except Exception as exc:  # a picture failing must not lose the logs
        lines.append(f"pm4py models skipped: {type(exc).__name__}: {exc}")
        say(lines[-1])
    text = "\n".join(lines)
    _w(os.path.join(out, "pm4py_summary.txt"), text)
    return text


def safe_id(case_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in case_id)[:60]


def picture_outputs(traces: list[Trace], out: str, say=print, failing: frozenset = frozenset(),
                    limit: int = 24) -> Optional[list[str]]:
    from . import viz
    if not viz.available():
        say("graphviz is not installed: no pipeline pictures")
        return None
    say("drawing the pipeline's pictures with graphviz")
    vis = [t for t in _visible(traces) if t.events]
    if not vis:
        return []
    min_edge = max(1, len(vis) // 20)
    colors = viz.agent_colors(vis)
    made = []
    edges = viz.handover(vis)
    if edges:
        viz.render_handover(edges, colors, os.path.join(out, "handover"), "Handover of work")
        made.append("handover.png")
    viz.render_lane_model(vis, colors, os.path.join(out, "lane_model"),
                          f"One lane per agent ({len(vis)} runs, edges seen at least {min_edge}x)", min_edge=min_edge)
    viz.render_dfg_by_agent(vis, colors, os.path.join(out, "dfg_by_agent"),
                            f"Directly-follows graph by agent ({len(vis)} runs)", min_edge=min_edge)
    made += ["lane_model.png", "dfg_by_agent.png"]
    # single runs: failing runs first, then the rest, up to the limit; drawn from the
    # full trace so that box numbers are the event positions the verdicts point at
    shown = {t.case_id for t in vis}
    pick = sorted((t for t in traces if t.case_id in shown), key=lambda t: (t.case_id not in failing, t.case_id))[:limit]
    for t in pick:
        safe = safe_id(t.case_id)
        viz.render_single_run(t, colors, os.path.join(out, f"run_{safe}"), f"Run {t.case_id}", drop=PICTURE_DROP)
        made.append(f"run_{safe}.png")
    return made


# ------------------------------------------------------------------ transcripts

TEXT_ATTRS = ["content", "instruction", "input", "arguments", "result", "reasoning"]
TASK_KEYS = ["prompt", "task", "question", "instruction", "description"]
OUTCOME_KEYS = ["task_id", "trial", "shape", "model", "reward", "termination_reason", "ground_truth", "decisive_event_index",
                "mistake_agent", "mistake_step", "mistake_reason", "is_hallucination", "hallucination_step",
                "hallucination_reason", "true_answer", "agent_answer", "harness_error", "gold_actions", "action_checks"]
_TEXT_CAP = 8000


def _text(v: Any) -> Any:
    if v is None or v == "" or v == {} or v == []:
        return None
    s = v if isinstance(v, str) else json.dumps(v, indent=1, sort_keys=True, default=str)
    return s if len(s) <= _TEXT_CAP else s[:_TEXT_CAP] + f"\n... [{len(s)} characters, cut]"


def write_transcripts(traces: list[Trace], out: str) -> int:
    """One JSON file per run: what was asked, how it ended, and every event with its text."""
    d = os.path.join(out, "transcripts")
    os.makedirs(d, exist_ok=True)
    for t in traces:
        task = next((t.outcome[k] for k in TASK_KEYS if t.outcome.get(k)), None)
        if task is None:   # tau2 keeps no task text; the customer's first message is the request
            first = next((e for e in t.ordered() if e.activity == "user_message"), None)
            task = first.attributes.get("content") if first else None
        events = []
        labels = t.step_labels()
        for pos, e in enumerate(t.ordered()):
            row = {"pos": pos, "label": labels[pos], "t": e.timestamp, "agent": e.resource, "activity": e.activity}
            if e.attributes.get("to"):
                row["to"] = e.attributes["to"]
            if e.attributes.get("error") is True:
                row["error"] = True
            for k in TEXT_ATTRS:
                v = _text(e.attributes.get(k))
                if v is not None:
                    row[k] = v
            events.append(row)
        doc = {"case_id": t.case_id, "task": _text(task),
               "outcome": {k: t.outcome[k] for k in OUTCOME_KEYS if t.outcome.get(k) not in (None, "", [])},
               "events": events}
        with open(os.path.join(d, safe_id(t.case_id) + ".json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, default=str)
    return len(traces)


# ------------------------------------------------------------------ entry

def report(traces: list[Trace], out: str, source: str = "", reference: Optional[Reference] = None,
           gold: bool = False, object_rules: Optional[list[ObjectRule]] = None,
           pictures: bool = True, noise: float = 0.0, by_agent: bool = False, say=None,
           task_types: Optional[TaskTypes] = None) -> list[tuple[str, str]]:
    say = say or (lambda *_: None)
    os.makedirs(out, exist_ok=True)
    rules = list(object_rules or [])
    say(f"building the OCEL 2.0 log ({len(rules)} object rules)")
    ocel = build_ocel_from_traces(traces, object_rules=rules)
    say(f"runs.ocel.json: {len(ocel['events'])} events, {len(ocel['objects'])} objects, "
        f"{len(ocel['objectTypes'])} object types")
    say("checking what the source exposes")
    with open(os.path.join(out, "runs.ocel.json"), "w", encoding="utf-8") as f:
        json.dump(ocel, f, indent=1, sort_keys=True)
    checks = source_checks(traces, ocel, build_ocel_from_traces(traces, object_rules=rules))
    mark = {"ok": "  ok  ", "warn": "  !!  ", "info": "  --  "}
    _w(os.path.join(out, "CHECKS.txt"),
       f"What {source or 'this source'} exposes ({len(traces)} runs). !! = read before trusting a result.\n\n" +
       "\n".join(mark[s] + line for s, line in checks))
    with open(os.path.join(out, "checks.json"), "w", encoding="utf-8") as f:
        json.dump([{"status": st, "text": line} for st, line in checks], f, indent=1)
    for st, line in checks:
        say(f"[{ {'ok': 'ok', 'warn': '!!', 'info': '--'}[st] }] {line}")
    _w(os.path.join(out, "inventory.txt"), inventory_text(traces))
    n_rows = write_event_log(traces, os.path.join(out, "event_log.csv"))
    say(f"event_log.csv: {n_rows} rows")
    write_transcripts(traces, out)
    say(f"transcripts/: one file per run, {len(traces)} runs")

    conf, answer_key = None, False
    if reference is not None:
        if by_agent:
            say("conformance reads activities as Agent.activity, and a sub-agent start as Parent.spawn:Child")
        checked = [agent_qualified(t) for t in traces] if by_agent else traces
        conf = conformance_text(checked, [reference] * len(traces), out)
    elif task_types is not None:
        pairs = task_types.pairs(traces)
        if pairs:
            conf = conformance_text([t for t, _ in pairs], [r for _, r in pairs], out)
    elif gold or any(t.outcome.get("gold_actions") for t in traces):
        pairs = [(t, gold_reference(t)) for t in traces]
        pairs = [(t, r) for t, r in pairs if r.required or r.precedence]
        if pairs:
            conf = conformance_text([t for t, _ in pairs], [r for _, r in pairs], out)
            answer_key = True
    if conf is not None:
        _w(os.path.join(out, "conformance.txt"), conf)
        say("conformance: " + conf.splitlines()[1])
    else:
        say("conformance: skipped, no reference model and no gold actions in the source")

    pm = pm4py_outputs(traces, out, noise, say) if pictures else None
    failing = frozenset()
    if conf is not None:
        with open(os.path.join(out, "verdicts.json"), encoding="utf-8") as f:
            failing = frozenset(v["case_id"] for v in json.load(f) if not v["compliant"])
    pics = picture_outputs(traces, out, say, failing) if pictures else None

    files = [("CHECKS.txt", "what the source exposes, checked automatically; read this first"),
             ("inventory.txt", "counts, and every run on one line as agent:activity"),
             ("event_log.csv", "the flat event log, one row per event, with arguments, results and reasoning"),
             ("runs.ocel.json", "the object-centric log, OCEL 2.0")]
    if pm:
        files.append(("event_log.xes", "the flat log in XES, for ProM, Disco and pm4py"))
    if conf is not None:
        files += [("conformance.txt", "each run against the reference, with the event responsible"
                   + (", activities named Agent.activity" if by_agent and reference is not None else "")),
                  ("verdicts.json", "the same, machine-readable")]
    if pm:
        files += [("pm4py_dfg.png, pm4py_petri_net.png, pm4py_process_tree.png, pm4py_bpmn.png",
                   f"pm4py's directly-follows graph and inductive-miner model (noise {noise})"),
                  ("pm4py_summary.txt", "variants, start and end activities, model sizes")]
    if pics:
        files += [("handover.png", "who hands work to whom"), ("lane_model.png", "aggregate model, one lane per agent"),
                  ("dfg_by_agent.png", "directly-follows graph, nodes coloured by agent"),
                  ("run_<id>.png", "single runs drawn literally, failing runs first, up to 24")]
    readme = [f"# Report: {source or 'agent traces'}", "", f"{len(traces)} runs. Produced by `agentpm report`.", "",
              "| file | what it is |", "|---|---|"] + [f"| `{a}` | {b} |" for a, b in files]
    readme += ["", "Pictures and pm4py models leave out messages and framework bookkeeping: "
               + ", ".join(sorted(PICTURE_DROP)) + ". The logs keep everything."]
    if answer_key:
        readme += ["", "Conformance used each task's own solution list, because no reference model was given. "
                   "That list is an answer key, so these verdicts show what the engine reaches when it is handed "
                   "the solution. `--task-types` checks every run against the model of its task type instead.", ""]
    if pictures and pm is None:
        readme.append("pm4py is not installed, so there is no XES file and no pm4py model.")
    if pictures and pics is None:
        readme.append("graphviz is not installed, so there are no pipeline pictures.")
    _w(os.path.join(out, "README.md"), "\n".join(readme))
    return checks
