"""Kept for old commands; the report is now part of the pipeline.

    python experiments/otel_report.py <capture.jsonl> [--out folder] [--reference ref.json]

is the same as

    python -m agentpm.cli report <capture.jsonl> --format otel -o <folder> [--reference ref.json]
"""
import os
import sys

from agentpm.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        sys.exit(__doc__)
    capture, rest = args[0], args[1:]
    out = os.path.dirname(os.path.abspath(capture))
    if "--out" in rest:
        i = rest.index("--out")
        out = rest[i + 1]
        rest = rest[:i] + rest[i + 2:]
    sys.exit(main(["report", capture, "--format", "otel", "-o", out] + rest))
