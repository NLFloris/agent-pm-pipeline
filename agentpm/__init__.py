"""agentpm - a trace to event log to process-mining pipeline for LLM-agent evaluation.

The package implements the four layers of the pipeline described in the README:
  L1 ingest      raw telemetry (OpenTelemetry GenAI spans, benchmark JSON) -> traces
  L2 model       canonical Trace / Event schema (this is agentpm.model)
  L3 logbuild    traces -> event log (CSV, XES, OCEL 2.0)
  L4 analysis    conformance checking (primary); discovery, handovers and rework (secondary lenses)
After the verdict, agentpm.evaluation scores the verdicts against labels.

The core runs on the standard library only. A pm4py backend is optional.
"""
__version__ = "0.1.0"
