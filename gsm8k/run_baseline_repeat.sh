#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MAX_JOBS=8 CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_CACHE_ROOT=/work/cache/vllm TRITON_CACHE_DIR=/work/cache/triton
export PYTHONPATH=/work/baseline:/work
unset RAY_ADDRESS
cd /work/baseline
mkdir -p /work/gsm8k/baseline-repeat
date -u +%Y-%m-%dT%H:%M:%SZ > /work/gsm8k/baseline-repeat/started.txt
set +e
timeout --signal=TERM --kill-after=30s 3600s /work/.venv/bin/python -u /work/run_gsm8k.py --mode baseline --output /work/gsm8k/baseline-repeat > /work/gsm8k/baseline-repeat/run.log 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > /work/gsm8k/baseline-repeat/exit.txt
date -u +%Y-%m-%dT%H:%M:%SZ > /work/gsm8k/baseline-repeat/finished.txt
exit "$rc"
