"""Command-line entry point: any supported source -> event log -> analysis.

    python -m agentpm.cli web                                         # the same, in a browser
    python -m agentpm.cli report     <path> --format otel -o out/      # everything at once
    python -m agentpm.cli formats
    python -m agentpm.cli inventory  <path> --format tau2
    python -m agentpm.cli ocel       <path> --format agenthallu -o out.ocel.json
    python -m agentpm.cli run        <path> --format tau2 --reference ref.json
    python -m agentpm.cli run        <path> --format tau2 --gold
    python -m agentpm.cli run        <path> --format tau2 --task-types models.json

<path> is a file for file-based formats and a directory for formats that store
one run per file or per folder (traceelephant, agenthallu, local_magentic).
The core needs nothing beyond the standard library; `ocel` writes OCEL 2.0 JSON
that pm4py reads back with pm4py.read_ocel2_json.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Callable

from .analysis.conformance import check
from .ingest import agenthallu, local_magentic, otel, tau2, traceelephant
from .ingest.claude_code import ClaudeCodeIngester
from .ingest.generic_json import GenericJsonIngester
from .ingest.openai_agents import OpenAIAgentsIngester
from .logbuild.objects import BUILTIN as BUILTIN_OBJECT_RULES, load_rules
from .logbuild.ocel import build_ocel_from_traces, object_sharing
from .model import Trace
from .report import report as build_report
from .reference import Reference, load_reference, load_task_types

# name -> (loader, accepts directory, one-line description)
FORMATS: dict[str, tuple[Callable[[str], list[Trace]], bool, str]] = {
    "generic": (lambda p: GenericJsonIngester().ingest(p), False,
                "a trace already in the canonical JSON shape"),
    "openai": (lambda p: OpenAIAgentsIngester().ingest(p), False,
               "OpenAI Agents SDK export JSONL (traces and spans)"),
    "claude": (lambda p: ClaudeCodeIngester().ingest(p), False,
               "Claude Code session transcript JSONL"),
    "otel": (otel.ingest, False,
             "OpenTelemetry: GenAI semantic-convention spans, or Claude Code's log dialect"),
    "tau2": (tau2.ingest, False,
             "tau2-bench trajectory JSON, one file per model and domain"),
    "traceelephant": (traceelephant.ingest_dir, True,
                      "TraceElephant: a directory of run folders"),
    "agenthallu": (agenthallu.ingest_dir, True,
                   "AgentHallu: a directory of run files"),
    "local_magentic": (local_magentic.ingest_dir, True,
                       "runs recorded by experiments/local_agents/run_magentic.py"),
}


def load(path: str, fmt: str) -> list[Trace]:
    loader, wants_dir, _ = FORMATS[fmt]
    if wants_dir and not os.path.isdir(path):
        sys.exit(f"--format {fmt} expects a directory, got a file: {path}")
    if not wants_dir and os.path.isdir(path):
        sys.exit(f"--format {fmt} expects a file, got a directory: {path}")
    return loader(path)


def gold_reference(trace: Trace) -> Reference:
    """Per-task reference from gold actions carried on the outcome (tau2)."""
    names = [a["name"] for a in trace.outcome.get("gold_actions", [])
             if a.get("requestor") == "assistant"]
    pairs = [[a, b] for a, b in zip(names, names[1:]) if a != b]
    return Reference(name="gold", required=sorted(set(names)), precedence=pairs)


# ---------------------------------------------------------------- commands

def cmd_formats(_: argparse.Namespace) -> int:
    for name, (_, wants_dir, desc) in FORMATS.items():
        print(f"  {name:16} {'dir ' if wants_dir else 'file'}  {desc}")
    return 0


def cmd_inventory(a: argparse.Namespace) -> int:
    traces = load(a.path, a.format)
    acts = collections.Counter(e.activity for t in traces for e in t.events)
    res = collections.Counter(e.resource for t in traces for e in t.events)
    n = sum(len(t.events) for t in traces)
    print(f"{a.format}: {len(traces)} traces, {n} events, {len(acts)} activities, {len(res)} resources")
    print("  activities:", ", ".join(f"{k}({v})" for k, v in acts.most_common(12)))
    print("  resources: ", ", ".join(f"{k}({v})" for k, v in res.most_common(8)))
    return 0


def cmd_ocel(a: argparse.Namespace) -> int:
    traces = load(a.path, a.format)
    if a.objects:
        rules = load_rules(a.objects)
    elif a.no_objects:
        rules = []
    else:
        rules = BUILTIN_OBJECT_RULES.get(a.format, [])
    ocel = build_ocel_from_traces(traces, object_rules=rules, tool_objects=not a.no_tool_objects,
                                  exclude=set(a.exclude or ()))
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(ocel, f, indent=1, sort_keys=True)
    share = object_sharing(ocel)
    print(f"wrote {a.output}: {len(ocel['events'])} events, {len(ocel['objects'])} objects, "
          f"{len(ocel['objectTypes'])} object types")
    print("  objects by type: " + ", ".join(f"{t}({n})" for t, n in share["objects_by_type"].items()))
    print(f"  events touching a business object (beyond run, agent, tool): "
          f"{share['events_touching_a_business_object']}/{share['events']}")
    if rules:
        print(f"  object rules: {len(rules)} ({'file ' + a.objects if a.objects else 'built in for ' + a.format})")
    return 0


def pairs_to_check(a: argparse.Namespace, traces: list[Trace]) -> list[tuple[Trace, Reference]]:
    """(trace, reference) for every run the chosen reference model can check."""
    if a.task_types:
        pairs = load_task_types(a.task_types).pairs(traces)
        if not pairs:
            sys.exit("--task-types needs traces that state a task_type or carry gold_actions on their outcome")
    elif a.gold:
        pairs = [(t, gold_reference(t)) for t in traces]
        pairs = [(t, r) for t, r in pairs if r.required or r.precedence]
        if not pairs:
            sys.exit("--gold needs traces that carry gold_actions on their outcome (tau2)")
    elif a.reference:
        ref = load_reference(a.reference)
        pairs = [(t, ref) for t in traces]
    else:
        sys.exit("give --reference <file.json>, --task-types <file.json> or --gold")
    return pairs


def cmd_run(a: argparse.Namespace) -> int:
    traces = load(a.path, a.format)
    pairs = pairs_to_check(a, traces)

    results = [(t, check(t, r)) for t, r in pairs]
    compliant = sum(1 for _, r in results if r.compliant)
    print(f"{len(results)} traces checked, {compliant} compliant, "
          f"mean fitness {sum(r.fitness for _, r in results) / len(results):.4f}")
    if a.verbose:
        for t, r in results:
            flag = "ok " if r.compliant else "VIOLATION"
            print(f"  {flag} {t.case_id[:24]:24} fitness {r.fitness:.2f}")
            for v in r.violations:
                print(f"       {v.constraint}" + (f" at event {v.at}" if v.at is not None else ""))
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump([{"case_id": t.case_id, "compliant": r.compliant, "fitness": r.fitness,
                        "violations": [{"constraint": v.constraint, "at": v.at,
                                        "description": v.description} for v in r.violations]}
                       for t, r in results], f, indent=1)
        print(f"wrote {a.output}")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    traces = load(a.path, a.format)
    if a.objects:
        rules = load_rules(a.objects)
    else:
        rules = BUILTIN_OBJECT_RULES.get(a.format, [])
    ref = load_reference(a.reference) if a.reference else None
    checks = build_report(traces, a.output, source=f"{os.path.basename(a.path.rstrip('/'))} ({a.format})",
                          reference=ref, gold=a.gold, object_rules=rules,
                          pictures=not a.no_pictures, noise=a.noise, by_agent=a.by_agent,
                          task_types=load_task_types(a.task_types) if a.task_types else None)
    print(f"{len(traces)} runs from {a.path} -> {a.output}/\n")
    mark = {"ok": "  ok  ", "warn": "  !!  ", "info": "  --  "}
    for status, line in checks:
        print(mark[status] + line)
    print(f"\nopen {a.output}/README.md for what each file is")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentpm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("formats", help="list the supported input formats").set_defaults(fn=cmd_formats)

    w = sub.add_parser("web", help="a local website: drop in a trace file, see every output")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--out", default="~/agentpm_reports", help="where reports are kept")
    w.add_argument("--no-browser", action="store_true")
    w.set_defaults(fn=lambda a: (__import__("agentpm.web", fromlist=["serve"]).serve(a.port, a.out, not a.no_browser), 0)[1])

    r = sub.add_parser("report", help="everything at once: checks, logs, conformance, pictures")
    r.add_argument("path", help="input file, or directory for per-run formats")
    r.add_argument("--format", choices=list(FORMATS), required=True)
    r.add_argument("-o", "--output", default="report")
    r.add_argument("--reference", help="reference model JSON; default: gold actions if the source has them")
    r.add_argument("--task-types", metavar="TEMPLATE.json",
                   help="check every run against the model of its task type, e.g. examples/task_types_tau2_airline.json")
    r.add_argument("--gold", action="store_true", help="check against gold actions carried by the source (tau2)")
    r.add_argument("--by-agent", action="store_true",
                   help="check the reference against Agent.activity names (Parent.spawn:Child for a sub-agent start)")
    r.add_argument("--objects", metavar="RULES.json", help="object rules; default: built in for the format")
    r.add_argument("--noise", type=float, default=0.0, help="inductive miner noise threshold (default 0)")
    r.add_argument("--no-pictures", action="store_true", help="logs and checks only, no pm4py or graphviz")
    r.set_defaults(fn=cmd_report)

    for name, help_text, fn in (("inventory", "count traces, events, activities, resources", cmd_inventory),
                                ("ocel", "write an OCEL 2.0 JSON event log", cmd_ocel),
                                ("run", "conformance-check every trace against a reference", cmd_run)):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("path", help="input file, or directory for per-run formats")
        p.add_argument("--format", choices=list(FORMATS), required=True)
        p.set_defaults(fn=fn)
        if name == "ocel":
            p.add_argument("-o", "--output", default="out.ocel.json")
            p.add_argument("--objects", metavar="RULES.json",
                           help="object extraction rules (see agentpm/logbuild/objects.py); "
                                "default: the built-in rules for the format, if any")
            p.add_argument("--no-objects", action="store_true",
                           help="emit only run, agent and tool objects")
            p.add_argument("--no-tool-objects", action="store_true",
                           help="do not emit one tool object per activity")
            p.add_argument("--exclude", nargs="*", metavar="ACTIVITY",
                           help="drop these activities, e.g. assistant_message user_message")
        if name == "run":
            p.add_argument("--reference", help="reference model JSON")
            p.add_argument("--task-types", metavar="TEMPLATE.json",
                           help="check every run against the model of its task type")
            p.add_argument("--gold", action="store_true",
                           help="build a per-task reference from gold actions (tau2)")
            p.add_argument("-o", "--output", help="write per-trace verdicts as JSON")
            p.add_argument("-v", "--verbose", action="store_true")

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
