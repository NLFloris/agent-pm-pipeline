"""Drive the self-instrumented corpus: N short agent runs, telemetry captured.

    # terminal 1
    python experiments/otel_capture/collector.py --out claude_code_capture.jsonl
    # terminal 2
    python experiments/otel_capture/run_suite.py --out claude_code_runs

Each task runs in its own freshly seeded sandbox directory, so runs are
independent and nothing outside the sandbox is read or written. The sandbox is
created by this script, never pointed at an existing project.

The driver records, per run, the task id, the control-flow shape the task was
designed to elicit, the wall-clock window, the exit code and the agent's final
text. The window is what lets a captured span be attributed to a run, since the
collector receives one undifferentiated stream.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

SEED_FILES = {
    "calc.py": '''"""Small calculator used by the task suite."""


def add(a, b):
    x = a + b
    return x


def divide(a, b):
    # TODO: decide what dividing by zero should do
    return a / b


def average(values):
    total = 0
    for v in values:
        total = total + v
    return total / len(values)


if __name__ == "__main__":
    print(add(2, 3))
    print(divide(10, 0))
''',
    "strings.py": '''"""String helpers used by the task suite."""


def titlecase(s):
    x = " ".join(w.capitalize() for w in s.split())
    return x


def truncate(s, n):
    # TODO: should this add an ellipsis?
    if len(s) <= n:
        return s
    return s[:n]


def word_count(s):
    total = 0
    for w in s.split():
        total = total + 1
    return total
''',
    "dates.py": '''"""Date helpers used by the task suite."""
import datetime


def days_between(a, b):
    x = (b - a).days
    return x


def is_weekend(d):
    return d.weekday() >= 5


def month_name(d):
    names = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]
    return names[d.month - 1]
''',
}


def seed(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    for name, body in SEED_FILES.items():
        with open(os.path.join(path, name), "w", encoding="utf-8") as f:
            f.write(body)


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="claude_code_runs", help="where run metadata and sandboxes go")
    ap.add_argument("--tasks", default=os.path.join(HERE, "tasks.json"))
    ap.add_argument("--endpoint", default="http://127.0.0.1:4318")
    ap.add_argument("--only", nargs="*", help="run only these task ids")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--dry-run", action="store_true", help="seed sandboxes, print commands, run nothing")
    a = ap.parse_args()

    with open(a.tasks, encoding="utf-8") as f:
        tasks = json.load(f)["tasks"]
    if a.only:
        tasks = [t for t in tasks if t["id"] in set(a.only)]
    if not tasks:
        print("no tasks selected")
        return 1

    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
        "OTEL_EXPORTER_OTLP_ENDPOINT": a.endpoint,
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_METRICS_EXPORTER": "otlp",
        "OTEL_METRIC_EXPORT_INTERVAL": "5000",
        "OTEL_BSP_SCHEDULE_DELAY": "1000",
    })

    manifest = []
    for i, task in enumerate(tasks, 1):
        sandbox = os.path.abspath(os.path.join(a.out, task["id"]))
        seed(sandbox)
        cmd = ["claude", "-p", task["prompt"], "--allowed-tools",
               "Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent"]
        print(f"[{i}/{len(tasks)}] {task['id']}  ({task['shape']})")
        if a.dry_run:
            print("   " + " ".join(cmd[:2]) + " ...")
            continue

        started = now()
        t0 = time.monotonic()
        try:
            proc = subprocess.run(cmd, cwd=sandbox, env=env, timeout=a.timeout,
                                  capture_output=True, text=True)
            code, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            code, out, err = -1, "", f"timed out after {a.timeout}s"
        elapsed = time.monotonic() - t0
        finished = now()

        manifest.append({
            "task_id": task["id"], "shape": task["shape"], "prompt": task["prompt"],
            "sandbox": sandbox, "started_at": started, "finished_at": finished,
            "seconds": round(elapsed, 2), "exit_code": code,
            "final_text": (out or "").strip()[:4000],
            "stderr": (err or "").strip()[:2000],
        })
        print(f"   exit {code} in {elapsed:.0f}s")
        # let the exporter flush before the next run starts
        time.sleep(3)

    path = os.path.join(a.out, "runs.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nwrote {path}  ({len(manifest)} runs)")
    print("stop the collector, then ingest its capture alongside this manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
