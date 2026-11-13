"""Where the experiment scripts find the public corpora.

The data folder holds tau-data/, agenthallu/ and optionally traceelephant/, laid
out as README.md describes. It is ~/Downloads unless the environment variable
AGENTPM_DATA names another folder, for example

    export AGENTPM_DATA=/path/to/data

A script's --data option, where it has one, overrides both.
"""
from __future__ import annotations

import os

DATA = os.path.expanduser(os.environ.get("AGENTPM_DATA") or "~/Downloads")
TAU2 = os.path.join(DATA, "tau-data", "tau2-bench")
TAU2_SUBMISSIONS = os.path.join(TAU2, "submissions")
TAU2_DOMAINS = os.path.join(TAU2, "data", "tau2", "domains")
OPUS_AIRLINE = os.path.join(TAU2_SUBMISSIONS, "claude-opus-4-5_sierra_2026-02-26", "trajectories",
                            "claude-opus-4-5_high_airline_gpt-5.2_4trials.json")
