"""The figures the thesis prints, rendered as vector PDF.

    python experiments/thesis_figures.py --out results/figures/thesis

The pipeline's own pictures (agentpm/viz.py) render PNG for the report and the
website. A printed page needs vector graphics, so this script draws the same
views with Graphviz's PDF output and without a title inside the picture, since
the thesis caption carries it.

  subagent_lanes.pdf   the ten claim_check runs with sub-agents, one lane per
                       agent, every edge shown (results/local_subagents/runs)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agentpm import viz                          # noqa: E402
from agentpm.ingest import local_magentic        # noqa: E402


class _PdfDigraph(viz.Digraph):
    """The pipeline's Digraph, writing PDF instead of PNG."""

    def __init__(self, *args, **kwargs):
        kwargs["format"] = "pdf"
        super().__init__(*args, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subagents", default="results/local_subagents/runs")
    ap.add_argument("--out", default="results/figures/thesis")
    a = ap.parse_args()
    if not viz.available():
        sys.exit("needs the graphviz package and a dot binary")
    os.makedirs(a.out, exist_ok=True)
    viz.Digraph = _PdfDigraph
    traces = local_magentic.ingest_dir(a.subagents)
    viz.render_lane_model(traces, viz.agent_colors(traces), os.path.join(a.out, "subagent_lanes"),
                          "", min_edge=1)


if __name__ == "__main__":
    sys.exit(main())
