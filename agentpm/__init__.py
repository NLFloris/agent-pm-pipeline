"""agentpm - a trace to event log to process-mining pipeline for LLM-agent evaluation.

The package implements the four-layer pipeline from THESIS_PLAN.md:
  L1 ingest      raw telemetry (OpenTelemetry GenAI spans, benchmark JSON) -> traces
  L2 model       canonical Trace / Event schema (this is agentpm.model)
  L3 logbuild    traces -> event log (tabular now, XES / OCEL 2.0 later)
  L4 analysis    conformance checking (primary), discovery and performance (secondary)

The core runs on the standard library only. A pm4py backend is optional.
"""
__version__ = "0.1.0"
