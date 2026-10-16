#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MAX_JOBS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_CACHE_ROOT=/work/cache/vllm
export TRITON_CACHE_DIR=/work/cache/triton
unset RAY_ADDRESS
mkdir -p /work/gsm8k
for mode in baseline bf16 fp16; do
    source_dir=/work/source
    if [ "$mode" = baseline ]; then source_dir=/work/baseline; fi
    export PYTHONPATH="$source_dir:/work"
    cd "$source_dir"
    mkdir -p "/work/gsm8k/$mode"
    date -u +%Y-%m-%dT%H:%M:%SZ > "/work/gsm8k/$mode/started.txt"
    set +e
    timeout --signal=TERM --kill-after=30s 3600s /work/.venv/bin/python -u \
        /work/run_gsm8k.py --mode "$mode" --output "/work/gsm8k/$mode" \
        > "/work/gsm8k/$mode/run.log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" > "/work/gsm8k/$mode/exit.txt"
    date -u +%Y-%m-%dT%H:%M:%SZ > "/work/gsm8k/$mode/finished.txt"
    if [ "$rc" -ne 0 ]; then exit "$rc"; fi
    /work/.venv/bin/python - "$mode" <<'PY'
import json, sys
from pathlib import Path
mode = sys.argv[1]
r = json.loads((Path('/work/gsm8k') / mode / 'result.json').read_text())
m = r['metrics']
assert r['status'] == 'complete'
assert m['invalid_rate'] <= 0.01 and m['truncation_rate'] <= 0.01, m
if mode == 'baseline':
    assert m['accuracy'] >= 0.8, ('Baseline sanity floor failed; diagnose protocol/runtime before candidate evaluation', m)
print(mode, m, flush=True)
PY
done
/work/.venv/bin/python /work/compare_gsm8k.py /work/gsm8k
