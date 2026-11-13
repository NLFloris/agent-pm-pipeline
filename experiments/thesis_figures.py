"""The figures the thesis prints, rendered as vector PDF.

    python experiments/thesis_figures.py --out results/figures/thesis

The pipeline's own pictures (agentpm/viz.py) render PNG for the report and the
website. A printed page needs vector graphics, so this script draws the same
views with Graphviz's PDF output and without a title inside the picture, since
the thesis caption carries it.

  subagent_lanes.pdf   the ten claim_check runs with sub-agents, one lane per
                       agent, every edge shown (results/local_subagents/runs)
  tau2_airline_dfg.pdf the unfiltered directly-follows graph of the 200 airline
                       runs of Claude Opus 4.5, as pm4py draws it (needs pm4py
                       and the benchmark's trajectory file)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agentpm import viz                          # noqa: E402
from agentpm.ingest import local_magentic, tau2  # noqa: E402
from data_dir import OPUS_AIRLINE   # noqa: E402

TAU2_AIRLINE = OPUS_AIRLINE


class _PdfDigraph(viz.Digraph):
    """The pipeline's Digraph, writing PDF instead of PNG, with larger text and
    tighter spacing so that the lane figure stays readable at the width of one
    column of the thesis."""

    def __init__(self, *args, **kwargs):
        kwargs["format"] = "pdf"
        super().__init__(*args, **kwargs)

    def attr(self, kw=None, **attrs):
        if kw is None and "rankdir" in attrs:   # the graph
            attrs.update(nodesep="0.12", ranksep="0.3")
        elif kw is None:                        # a lane
            attrs.update(fontsize="20", margin="6")
        elif kw == "node":
            attrs.update(fontsize="20", margin="0.08,0.04")
        return super().attr(kw, **attrs)

    def node(self, *args, **attrs):
        if "fontsize" in attrs:                 # START and END
            attrs["fontsize"] = "18"
        return super().node(*args, **attrs)

    def edge(self, *args, **attrs):
        attrs["fontsize"] = "18"
        return super().edge(*args, **attrs)


def tau2_dfg(path, out):
    """pm4py's directly-follows graph over the tool calls, as in experiments/discover.py."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pm4py
    from discover import flat_log
    dfg, start, end = pm4py.discover_dfg(flat_log(tau2.ingest(path)))
    print(f"tau2 airline: {len(dfg)} edges")
    # as pm4py.save_vis_dfg draws it, with a larger font so that the names stay readable in print
    from pm4py.visualization.dfg import visualizer
    gviz = visualizer.apply(dfg, variant=visualizer.Variants.FREQUENCY, parameters={
        "format": "pdf", "rankdir": "TB", "bgcolor": "white", "font_size": 24,
        "start_activities": start, "end_activities": end})
    visualizer.save(gviz, os.path.join(out, "tau2_airline_dfg.pdf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subagents", default="results/local_subagents/runs")
    ap.add_argument("--tau2-airline", default=TAU2_AIRLINE)
    ap.add_argument("--out", default="results/figures/thesis")
    a = ap.parse_args()
    if not viz.available():
        sys.exit("needs the graphviz package and a dot binary")
    os.makedirs(a.out, exist_ok=True)
    viz.Digraph = _PdfDigraph
    traces = local_magentic.ingest_dir(a.subagents)
    viz.render_lane_model(traces, viz.agent_colors(traces), os.path.join(a.out, "subagent_lanes"),
                          "", min_edge=1)
    tau2_dfg(a.tau2_airline, a.out)


if __name__ == "__main__":
    sys.exit(main())
