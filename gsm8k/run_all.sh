#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MAX_JOBS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_CACHE_ROOT=/work/cache/vllm-gsm8k
export TRITON_CACHE_DIR=/work/cache/triton-gsm8k
unset RAY_ADDRESS

for mode in baseline bf16 fp16; do
    source_dir=/work/source
    if [ "$mode" = baseline ]; then source_dir=/work/baseline; fi
    export PYTHONPATH="$source_dir:/work/gsm8k"
    cd "$source_dir"
    mkdir -p "/work/gsm8k/$mode"
    args=(--mode "$mode" --output "/work/gsm8k/$mode"
          --cache-dir "/work/cache/gsm8k-flashinfer/$mode")
    if [ "$mode" != baseline ]; then
        args+=(--baseline-result /work/gsm8k/baseline/result.json)
    fi
    date -u +%Y-%m-%dT%H:%M:%SZ > "/work/gsm8k/$mode/started.txt"
    set +e
    timeout --signal=TERM --kill-after=30s 3600s /work/.venv/bin/python -u \
        /work/gsm8k/run_gsm8k.py "${args[@]}" \
        > "/work/gsm8k/$mode/run.log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" > "/work/gsm8k/$mode/exit.txt"
    date -u +%Y-%m-%dT%H:%M:%SZ > "/work/gsm8k/$mode/finished.txt"
    if [ "$rc" -ne 0 ]; then exit "$rc"; fi
    /work/.venv/bin/python /work/gsm8k/check_cache_log.py "/work/gsm8k/$mode"
    /work/.venv/bin/python - "$mode" <<'PY'
import json
from pathlib import Path
import sys
mode = sys.argv[1]
result = json.loads((Path('/work/gsm8k') / mode / 'result.json').read_text())
metrics = result['metrics']
assert result['status'] == 'complete' and result['shutdown_completed']
assert metrics['invalid_rate'] <= 0.01 and metrics['truncation_rate'] <= 0.01, metrics
if mode == 'baseline':
    assert metrics['accuracy'] >= 0.8, ('Baseline sanity floor failed', metrics)
print(mode, metrics, flush=True)
PY
done
/work/.venv/bin/python /work/gsm8k/compare_gsm8k.py /work/gsm8k
