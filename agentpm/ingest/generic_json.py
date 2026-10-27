"""Ingest a generic JSON trace dump into canonical traces (L1 to L2).

Accepts either the canonical schema (a single trace object with 'case_id' and
'events') or a raw list of tool-call records, which it normalises. This shows the
normalisation step; production sources get their own ingester.
"""
from __future__ import annotations

import json
from typing import Any

from ..model import Event, Trace, trace_from_dict

_RAW_META = {"tool", "activity", "action", "ts", "timestamp", "resource"}


def _normalise_raw(records: list[dict[str, Any]], case_id: str) -> Trace:
    events = []
    for i, r in enumerate(records):
        events.append(
            Event(
                activity=r.get("tool") or r.get("activity") or r.get("action") or "unknown",
                timestamp=int(r.get("ts", r.get("timestamp", i))),
                resource=r.get("resource", "unknown"),
                attributes={k: v for k, v in r.items() if k not in _RAW_META},
            )
        )
    return Trace(case_id=case_id, events=events)


class GenericJsonIngester:
    def ingest(self, path: str) -> list[Trace]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "events" in data:
            return [trace_from_dict(data)]
        if isinstance(data, list):
            return [_normalise_raw(data, case_id=path)]
        raise ValueError(f"Unrecognised trace format in {path}")
