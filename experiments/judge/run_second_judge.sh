#!/usr/bin/env bash
# The second Judge (glm-5.3-flash) on airline, both Judges on retail, and the
# Judges' agreement with themselves with the proxy cache off. 2026-10-23.
#
#   bash experiments/judge/run_second_judge.sh glm      # lane 1
#   bash experiments/judge/run_second_judge.sh gptoss   # lane 2, may run at the same time
#
# Every verdict is one fresh sample (--no-cache, one repeat). The first airline
# verdicts were one sample too: their four other repeats were cache hits.
# Self-agreement is measured separately for gpt-oss, five fresh samples on the
# first 30 airline runs.
# All commands resume where they stopped.
set -euo pipefail
cd "$(dirname "$0")/../.."
export LLM_BASE_URL=${LLM_BASE_URL:-https://inference.proxy.ti.bfh.ch/v1}
export LLM_API_KEY=${LLM_API_KEY:-$(tr -d '[:space:]' < ~/.bfh_api_key)}
PY=${PY:-python}
SUB=~/Downloads/tau-data/tau2-bench/submissions/claude-opus-4-5_sierra_2026-02-26/trajectories
AIRLINE=$SUB/claude-opus-4-5_high_airline_gpt-5.2_4trials.json
RETAIL=$SUB/claude-opus-4-5_high_retail_gpt-5.2_4trials.json
COMMON="--no-cache --timeout 900 --workers ${WORKERS:-4}"

judge() { $PY experiments/judge/run_judge.py --model "$1" --mode "$2" --data "$3" --out "$4" $COMMON "${@:5}"; }

case "${1:?glm or gptoss}" in
  glm)
    # glm-5.3-flash is served from one machine, one request at a time for every
    # user of the proxy (max_parallel_requests=1), so one worker, no output-only
    # condition, no self-agreement, and a stratified sample of the retail runs.
    # Streamed, because answers over about five minutes hit the gateway timeout (504).
    export WORKERS=1 COMMON="--no-cache --stream --timeout 900 --workers 1"
    for mode in gold-actions full; do judge glm-5.3-flash $mode "$AIRLINE" results/judge --repeats 1; done
    $PY experiments/judge/sample_runs.py --domain retail --n 120
    for mode in gold-actions full; do
      judge glm-5.3-flash $mode "$RETAIL" results/judge/retail --repeats 1 --cases results/judge/retail/sample_120.json
    done
    ;;
  gptoss)
    for mode in gold-actions full output-only; do judge gpt-oss:120b $mode "$RETAIL" results/judge/retail --repeats 1; done
    for mode in gold-actions full; do judge gpt-oss:120b $mode "$AIRLINE" results/judge/self_agreement --repeats 5 --limit 30; done
    ;;
esac
