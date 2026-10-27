"""Event-log construction (pipeline layer L3).

Turns canonical traces into a tabular event log. A dependency-free path returns a
list of row dicts using XES-style column names. A pm4py path (XES export,
alignment-based conformance) is available when pm4py and pandas are installed.

A flat log is a total order by definition, so this view serialises the partial
order of the canonical trace: ``time:timestamp`` is the serialisation position
(unique, so that no importer reorders ties), and the logical time that marks
concurrent events is carried alongside as ``logical_time``. The object-centric
log in ``ocel.py`` keeps the ties instead.
"""
from __future__ import annotations

from typing import Any

from ..model import Trace

CASE = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"
RESOURCE = "org:resource"
LOGICAL_TIME = "logical_time"


def to_rows(traces: list[Trace]) -> list[dict[str, Any]]:
    """Flatten traces into XES-style event rows, serialised within each case."""
    rows: list[dict[str, Any]] = []
    for t in traces:
        for pos, e in enumerate(t.ordered()):
            rows.append({
                CASE: t.case_id,
                ACTIVITY: e.activity,
                TIMESTAMP: pos,
                LOGICAL_TIME: e.timestamp,
                RESOURCE: e.resource,
            })
    return rows


def to_pm4py_eventlog(traces: list[Trace]):
    """Build a pm4py event log. Requires the optional pm4py and pandas extras."""
    try:
        import pandas as pd
        import pm4py
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pm4py and pandas are required for the pm4py path. "
            "Install with 'pip install agent-pm-pipeline[pm]'."
        ) from exc
    df = pd.DataFrame(to_rows(traces))
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], unit="s", errors="coerce")
    return pm4py.format_dataframe(
        df, case_id=CASE, activity_key=ACTIVITY, timestamp_key=TIMESTAMP
    )
