"""Pictures of canonical traces, drawn with Graphviz (optional dependency).

Four views, used by the report command and by experiments/figures.py:

  render_dfg_by_agent   directly-follows graph, each activity coloured by the
                        agent that performs it most
  render_handover       handover-of-work network between agents
  render_lane_model     aggregate model, one lane per agent
  render_single_run     one run drawn literally, every event in its agent's lane

Needs the graphviz Python package and a `dot` binary on PATH; the core
pipeline does not.
"""
from __future__ import annotations

import collections
import os
import re

from .analysis.performance import handovers as handover

try:
    from graphviz import Digraph
except ImportError:  # pragma: no cover - optional
    Digraph = None


def available() -> bool:
    import shutil
    return Digraph is not None and shutil.which("dot") is not None


PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#7f7f7f"]
MESSAGES = {"assistant_message", "user_message"}


def agent_colors(traces):
    agents = sorted({e.resource for t in traces for e in t.events})
    return {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(agents)}


def _base(title):
    g = Digraph(format="png")
    g.attr(rankdir="TB", label=title, labelloc="t", fontsize="18", fontname="Helvetica")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="11")
    return g


def render_handover(edges, colors, path, title):
    g = _base(title)
    g.attr("node", fontcolor="white")
    seen = {a for e in edges for a in e}
    for a in sorted(seen):
        g.node(a, a, fillcolor=colors.get(a, "#666666"))
    top = max(edges.values()) if edges else 1
    for (a, b), n in sorted(edges.items(), key=lambda x: -x[1]):
        c = colors.get(a, "#666666")
        g.edge(a, b, label=str(n), color=c, fontcolor=c, penwidth=f"{0.8 + 4.0 * n / top:.2f}")
    g.render(path, cleanup=True)
    print(f"  {os.path.basename(path)}.png  {len(seen)} agents, {len(edges)} edges")


def render_dfg_by_agent(traces, colors, path, title, min_edge=3, drop=frozenset()):
    owner = collections.defaultdict(collections.Counter)
    acts, edges = collections.Counter(), collections.Counter()
    starts, ends = collections.Counter(), collections.Counter()
    for t in traces:
        ev = [e for e in t.ordered() if e.activity not in drop]
        if not ev:
            continue
        starts[ev[0].activity] += 1
        ends[ev[-1].activity] += 1
        for e in ev:
            acts[e.activity] += 1
            owner[e.activity][e.resource] += 1
        for a, b in zip(ev, ev[1:]):
            edges[(a.activity, b.activity)] += 1

    kept = {e: n for e, n in edges.items() if n >= min_edge}
    live = {a for e in kept for a in e} | set(starts) | set(ends)
    g = _base(title)
    g.attr("node", fontcolor="white")
    for a in sorted(live):
        main = owner[a].most_common(1)[0][0] if owner[a] else "unknown"
        share = owner[a][main] / max(1, sum(owner[a].values()))
        label = f"{a}\\n{acts[a]}\\n[{main}]" + ("" if share > 0.95 else f" {share:.0%}")
        g.node(a, label, fillcolor=colors.get(main, "#666666"))
    g.node("__start", "", shape="circle", fillcolor="#2e7d32", style="filled", width="0.3")
    g.node("__end", "", shape="doublecircle", fillcolor="#b71c1c", style="filled", width="0.3")
    for a, n in starts.items():
        if a in live:
            g.edge("__start", a, label=str(n), color="#2e7d32")
    for a, n in ends.items():
        if a in live:
            g.edge(a, "__end", label=str(n), color="#b71c1c")
    top = max(kept.values()) if kept else 1
    for (a, b), n in sorted(kept.items(), key=lambda x: -x[1]):
        g.edge(a, b, label=str(n), penwidth=f"{0.6 + 3.0 * n / top:.2f}")
    g.render(path, cleanup=True)
    print(f"  {os.path.basename(path)}.png  {len(live)} activities, {len(kept)} edges")


def _lane_cluster(g, i, agent, color, nodes):
    with g.subgraph(name=f"cluster_{i}") as c:
        c.attr(label=f"  {agent}  ", labeljust="l", labelloc="t", style="rounded",
               color=color, penwidth="1.6", fontname="Helvetica-Bold", fontsize="13",
               fontcolor=color, bgcolor="transparent", margin="10")
        for nid, label in nodes:
            c.node(nid, label, fillcolor=color)


def render_lane_model(traces, colors, path, title, min_edge=5, drop=frozenset()):
    """One lane per agent, Start on the left, End on the right.

    An aggregate model: every activity an agent performs appears once in that
    agent's lane, with directly-follows edges between them. Edges that stay
    inside a lane are grey; edges that cross lanes are a handover and take the
    colour of the agent handing over.
    """
    acts = collections.Counter()
    lane_of = {}
    edges = collections.Counter()
    starts, ends = collections.Counter(), collections.Counter()
    for t in traces:
        ev = [e for e in t.ordered() if e.activity not in drop]
        if not ev:
            continue
        # a sub-agent start names the sub-agent, so two sub-agents are two nodes
        keys = [(e.resource, f"spawn {e.attributes['spawns']}" if e.attributes.get("spawns") else e.activity)
                for e in ev]
        starts[keys[0]] += 1
        ends[keys[-1]] += 1
        for key in keys:
            acts[key] += 1
            lane_of.setdefault(key, key[0])
        for a, b in zip(keys, keys[1:]):
            edges[(a, b)] += 1

    kept = {e: n for e, n in edges.items() if n >= min_edge}
    live = {k for e in kept for k in e} | set(starts) | set(ends)
    if not live:
        print(f"  [skip] {os.path.basename(path)}: nothing above the threshold")
        return

    g = Digraph(format="png")
    g.attr(rankdir="LR", label=title, labelloc="t", fontsize="19",
           fontname="Helvetica", compound="true", nodesep="0.25", ranksep="0.7")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
           fontsize="11", fontcolor="white", margin="0.14,0.07")

    def nid(key):
        return "n_" + re.sub(r"\W+", "_", f"{key[0]}__{key[1]}")

    by_lane = collections.defaultdict(list)
    for key in sorted(live):
        by_lane[key[0]].append((nid(key), f"{key[1]}\\n{acts.get(key, 0)}"))
    for i, agent in enumerate(sorted(by_lane)):
        _lane_cluster(g, i, agent, colors.get(agent, "#666666"), by_lane[agent])

    g.node("__start", "START", shape="oval", fillcolor="#2e7d32", fontsize="12")
    g.node("__end", "END", shape="oval", fillcolor="#b71c1c", fontsize="12")
    for key, n in starts.items():
        if key in live:
            g.edge("__start", nid(key), label=str(n), color="#2e7d32", fontcolor="#2e7d32")
    for key, n in ends.items():
        if key in live:
            g.edge(nid(key), "__end", label=str(n), color="#b71c1c", fontcolor="#b71c1c")

    top = max(kept.values())
    for (a, b), n in sorted(kept.items(), key=lambda x: -x[1]):
        crosses = a[0] != b[0]
        color = colors.get(a[0], "#666666") if crosses else "#8c99a3"
        g.edge(nid(a), nid(b), label=str(n), color=color,
               fontcolor=color if crosses else "#6a767f",
               penwidth=f"{0.6 + 3.0 * n / top:.2f}",
               style="solid" if crosses else "dashed")
    g.render(path, cleanup=True)
    print(f"  {os.path.basename(path)}.png  {len(by_lane)} lanes, {len(live)} nodes, "
          f"{len(kept)} edges")


def _handover_label(a, b) -> str:
    """A sub-agent is started and returns; every other change of agent hands over."""
    if a.attributes.get("spawns") == b.resource:
        return "starts"
    if a.attributes.get("spawned_by") == b.resource:
        return "returns"
    return "hands over"


def _note_text(text: str, width: int = 46, max_lines: int = 7) -> str:
    """Reasoning as a left-aligned Graphviz label: wrapped, cut after a few lines."""
    import textwrap
    flat = " ".join(str(text).split())
    lines = textwrap.wrap(flat, width) or [""]
    cut = len(lines) > max_lines
    lines = lines[:max_lines]
    if cut:
        lines[-1] = lines[-1][: width - 2] + " …"
    return "".join(l.replace("\\", "\\\\") + "\\l" for l in lines)


def render_single_run(trace, colors, path, title, drop=frozenset(), reasoning=True):
    """One run drawn literally: every event in its agent's lane, in order.

    Events the source issued together share a step. They are drawn side by side
    in a dashed box labelled "issued together", with one arrow from the step
    before and one to the step after, and no arrow between them, because the
    source gives them no order. Box labels are the step labels of the trace
    ("4a", "4b"), the same labels the verdicts and the transcript use.

    Activities in ``drop`` (messages, framework bookkeeping) are not drawn as
    boxes, but they are not silently skipped either: a run of hidden steps
    becomes one grey line ("0-3  4 messages"), and a hidden event inside a
    shown step becomes a grey line in that step's box, so the numbering reads
    continuously and matches the transcript.

    With ``reasoning`` set, an event that carries reasoning text gets a note
    beside it holding the first lines of that text. The note is an annotation of
    the step, never a step of its own (decision L6).
    """
    all_ordered = trace.ordered()
    labels = trace.step_labels()
    ev = dict(enumerate(all_ordered))
    shown = {i for i, e in ev.items() if e.activity not in drop}
    # all steps, in order: lists of positions sharing a timestamp
    steps = []
    for i in range(len(all_ordered)):
        if steps and ev[steps[-1][0]].timestamp == ev[i].timestamp:
            steps[-1].append(i)
        else:
            steps.append([i])

    g = Digraph(format="png")
    g.attr(rankdir="TB", label=title, labelloc="t", fontsize="17", fontname="Helvetica",
           nodesep="0.3", ranksep="0.38", newrank="true", compound="true")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
           fontsize="11", fontcolor="white", margin="0.14,0.07")

    def ghost_label(positions):
        names = collections.Counter(ev[i].activity for i in positions)
        what = ", ".join(f"{n} {a}" if n > 1 else a for a, n in names.most_common())
        if len(positions) == 1:
            return f"{labels[positions[0]]}  {what}"
        return f"{labels[positions[0]]}-{labels[positions[-1]]}  {what}"

    # the visible chain: visible steps, plus one ghost per run of fully hidden steps
    chain = []          # entries: ("step", [positions]) or ("ghost", [positions])
    for st in steps:
        vis = [i for i in st if i in shown]
        if vis:
            chain.append(("step", st))
        elif chain and chain[-1][0] == "ghost":
            chain[-1][1].extend(st)
        else:
            chain.append(("ghost", list(st)))

    by_lane = collections.defaultdict(list)
    for i in shown:
        by_lane[ev[i].resource].append(i)
    ghost_lane = {}
    for k, (kind, positions) in enumerate(chain):
        if kind != "ghost":
            continue
        near = next((chain[j][1] for j in range(k + 1, len(chain)) if chain[j][0] == "step"), None) \
            or next((chain[j][1] for j in range(k - 1, -1, -1) if chain[j][0] == "step"), None)
        vis = [i for i in (near or []) if i in shown]
        ghost_lane[k] = ev[vis[0]].resource if vis else None
    group_of = {}        # position -> cluster name of its "issued together" box, if any
    n_notes = 0
    note_on = set()
    for st in steps:
        seen = set()
        for i in st:
            if i not in shown:
                continue
            thought = ev[i].attributes.get("reasoning") if reasoning else None
            if thought and thought not in seen:
                seen.add(thought)
                note_on.add(i)

    def node(c, i, color):
        nonlocal n_notes
        e = ev[i]
        thought = e.attributes.get("reasoning") if reasoning else None
        spawned = e.attributes.get("spawns")
        c.node(f"n{i}", f"{labels[i]}. {e.activity}" + (f" {spawned}" if spawned else ""), fillcolor=color,
               penwidth="2.4" if thought else "1", color="#222222" if thought else "black")
        if i in note_on:
            n_notes += 1
            with c.subgraph() as r:
                r.attr(rank="same")
                r.node(f"n{i}")
                r.node(f"r{i}", _note_text(thought), shape="note", style="filled",
                       fillcolor="#f5f3ee", color="#b9b4a8", fontcolor="#3a3833",
                       fontname="Helvetica", fontsize="9.5", margin="0.12,0.06")
            c.edge(f"n{i}", f"r{i}", style="dotted", color="#9a958a", arrowhead="none")

    def ghost(c, name, text):
        c.node(name, text, shape="plaintext", style="", fontcolor="#8a8a8a", fontsize="9.5",
               fontname="Helvetica", margin="0.05,0.02")

    for li, agent in enumerate(sorted(by_lane)):
        color = colors.get(agent, "#666666")
        with g.subgraph(name=f"cluster_{li}") as c:
            c.attr(label=f"  {agent}  ", labeljust="l", labelloc="t", style="rounded",
                   color=color, penwidth="1.6", fontname="Helvetica-Bold", fontsize="13",
                   fontcolor=color, bgcolor="transparent", margin="10")
            for k, (kind, positions) in enumerate(chain):
                if kind == "ghost" and ghost_lane.get(k) == agent:
                    ghost(c, f"g{k}", ghost_label(positions))
            for k, st in enumerate(steps):
                mine = [i for i in st if i in shown and ev[i].resource == agent]
                hidden = [i for i in st if i not in shown and ev[i].resource == agent and any(j in shown for j in st)]
                if not mine:
                    continue
                if len(mine) == 1 and not hidden:
                    node(c, mine[0], color)
                    continue
                name = f"cluster_s{k}_{li}"
                with c.subgraph(name=name) as box:
                    box.attr(label="issued together", labeljust="l", style="dashed", color="#8a8a8a",
                             fontcolor="#6a6a6a", fontname="Helvetica", fontsize="9.5", margin="8")
                    with box.subgraph() as row:
                        row.attr(rank="same")
                        for i in mine:
                            node(row, i, color)
                        if hidden:
                            ghost(row, f"h{k}_{li}", ghost_label(hidden))
                for i in mine:
                    group_of[i] = name

    g.node("__start", "START", shape="oval", fillcolor="#2e7d32", fontsize="12")
    g.node("__end", "END", shape="oval", fillcolor="#b71c1c", fontsize="12")

    for k, (kind, positions) in enumerate(chain):
        if kind == "ghost" and ghost_lane.get(k) is None:
            ghost(g, f"g{k}", ghost_label(positions))

    def anchor(k):
        kind, positions = chain[k]
        if kind == "ghost":
            return f"g{k}", None
        vis = [i for i in positions if i in shown]
        return f"n{vis[0]}", group_of.get(vis[0])

    def link(a, b, color, label="", crosses=False):
        na, ca = anchor(a)
        nb, cb = anchor(b)
        attrs = {"color": color, "penwidth": "1.8" if crosses else "1.1",
                 "style": "solid" if crosses else "dashed", "label": label}
        if ca:
            attrs["ltail"] = ca
        if cb:
            attrs["lhead"] = cb
        g.edge(na, nb, **attrs)

    def agent_of(k):
        kind, positions = chain[k]
        vis = [i for i in positions if i in shown] or positions
        return ev[vis[0]].resource

    if chain:
        n0, c0 = anchor(0)
        g.edge("__start", n0, color="#2e7d32", **({"lhead": c0} if c0 else {}))
        nl, cl = anchor(len(chain) - 1)
        g.edge(nl, "__end", color="#b71c1c", **({"ltail": cl} if cl else {}))
    for k in range(len(chain) - 1):
        a_kind, b_kind = chain[k][0], chain[k + 1][0]
        if a_kind == "step" and b_kind == "step":
            cur = [i for i in chain[k][1] if i in shown]
            nxt = [i for i in chain[k + 1][1] if i in shown]
            pairs = {}
            for a in cur:
                for b in nxt:
                    pairs.setdefault((group_of.get(a, a), group_of.get(b, b)), (a, b))
            for a, b in pairs.values():
                crosses = ev[a].resource != ev[b].resource
                attrs = {"color": colors.get(ev[a].resource, "#666666") if crosses else "#8c99a3",
                         "penwidth": "1.8" if crosses else "1.1", "style": "solid" if crosses else "dashed",
                         "label": _handover_label(ev[a], ev[b]) if crosses else ""}
                if a in group_of:
                    attrs["ltail"] = group_of[a]
                if b in group_of:
                    attrs["lhead"] = group_of[b]
                g.edge(f"n{a}", f"n{b}", **attrs)
        else:
            crosses = agent_of(k) != agent_of(k + 1) and a_kind == "step" and b_kind == "step"
            link(k, k + 1, "#b0b0b0", "", False)
    g.render(path, cleanup=True)
    boxed = sum(1 for st in steps if len([i for i in st if i in shown]) > 1)
    hidden_total = len(all_ordered) - len(shown)
    print(f"  {os.path.basename(path)}.png  {len(by_lane)} lanes, {len(shown)} events"
          + (f", {hidden_total} hidden as grey lines" if hidden_total else "")
          + (f", {boxed} steps issued together" if boxed else "")
          + (f", {n_notes} with reasoning" if n_notes else ""))
