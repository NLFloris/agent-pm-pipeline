#!/usr/bin/env python3
"""Mirror the tau2-bench and tau-bench data needed for the pipeline.

Usage:
  python mirror_tau.py DEST [--all-submissions] [--submission NAME ...]

Default: text submissions only (skips the voice/realtime ones, which use a
different per-simulation layout). Writes DEST/MANIFEST.json with sha256 + size.
"""
import argparse, collections, hashlib, json, os, sys, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

BUCKET = "https://sierra-tau-bench-public.s3.amazonaws.com/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
RAW2 = "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/"
RAW1 = "https://raw.githubusercontent.com/sierra-research/tau-bench/main/"

TAU1 = ["historical_trajectories/gpt-4o-airline.json",
        "historical_trajectories/gpt-4o-retail.json",
        "historical_trajectories/sonnet-35-new-airline.json",
        "historical_trajectories/sonnet-35-new-retail.json",
        "tau_bench/envs/airline/wiki.md",
        "tau_bench/envs/retail/wiki.md"]

TAU2_REF = ["data/tau2/domains/airline/policy.md", "data/tau2/domains/airline/tasks.json",
            "data/tau2/domains/retail/policy.md",  "data/tau2/domains/retail/tasks.json",
            "data/tau2/domains/telecom/main_policy.md",
            "data/tau2/domains/telecom/tech_support_manual.md",
            "data/tau2/domains/telecom/tech_support_workflow.md",
            "data/tau2/domains/telecom/tasks_small.json",
            "data/tau2/domains/telecom/workflows/tech_support_path1_no_service.dot",
            "data/tau2/domains/telecom/workflows/tech_support_path2_mobile_data.dot",
            "data/tau2/domains/telecom/workflows/tech_support_path3_mms.dot"]


def list_bucket(prefix="submissions/"):
    tok, out = None, []
    while True:
        u = f"{BUCKET}?list-type=2&prefix={prefix}"
        if tok:
            u += "&continuation-token=" + urllib.parse.quote(tok)
        root = ET.fromstring(urllib.request.urlopen(u).read())
        for c in root.findall(NS + "Contents"):
            out.append((c.find(NS + "Key").text, int(c.find(NS + "Size").text)))
        nxt = root.find(NS + "NextContinuationToken")
        if nxt is None:
            return out
        tok = nxt.text


def fetch(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return os.path.getsize(dest), "cached"
    tmp = dest + ".part"
    h = hashlib.sha256()
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk); h.update(chunk)
    os.replace(tmp, dest)
    return os.path.getsize(dest), h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--all-submissions", action="store_true",
                    help="include voice/realtime submissions (35.6 GB instead of 6.1 GB)")
    ap.add_argument("--submission", action="append", default=[],
                    help="mirror only these submission names")
    a = ap.parse_args()

    traj = [(k, s) for k, s in list_bucket() if "/trajectories/" in k and k.endswith(".json")]
    by = collections.defaultdict(list)
    for k, s in traj:
        by[k.split("/")[1]].append((k, s))

    if a.submission:
        chosen = {n: by[n] for n in a.submission if n in by}
        missing = [n for n in a.submission if n not in by]
        if missing:
            sys.exit(f"unknown submission(s): {missing}")
    elif a.all_submissions:
        chosen = dict(by)
    else:  # text submissions ship one file per domain; voice ones ship 100+
        chosen = {n: v for n, v in by.items() if len(v) <= 4}

    plan = [(BUCKET + urllib.parse.quote(k), os.path.join(a.dest, "tau2-bench", k), s)
            for v in chosen.values() for k, s in v]
    plan += [(RAW2 + p, os.path.join(a.dest, "tau2-bench", p), None) for p in TAU2_REF]
    plan += [(RAW1 + p, os.path.join(a.dest, "tau-bench", p), None) for p in TAU1]

    known = sum(s for _, _, s in plan if s)
    print(f"{len(plan)} files, {known/1e9:.2f} GB of trajectories from "
          f"{len(chosen)} submissions -> {a.dest}")

    manifest = {}
    for i, (url, dest, _) in enumerate(plan, 1):
        size, digest = fetch(url, dest)
        if digest == "cached":
            digest = sha256(dest)
        rel = os.path.relpath(dest, a.dest)
        manifest[rel] = {"url": url, "bytes": size, "sha256": digest}
        print(f"[{i}/{len(plan)}] {size/1e6:8.1f} MB  {rel}")

    with open(os.path.join(a.dest, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f"\nwrote MANIFEST.json ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
