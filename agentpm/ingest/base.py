"""Ingestion interface (pipeline layer L1).

An ingester maps a raw telemetry source to canonical Trace objects. Different
sources (OpenTelemetry GenAI spans, benchmark trajectory dumps) implement the same
interface so that everything downstream is source-agnostic (R1).
"""
from __future__ import annotations

from typing import Protocol

from ..model import Trace


class Ingester(Protocol):
    def ingest(self, path: str) -> list[Trace]:
        ...
