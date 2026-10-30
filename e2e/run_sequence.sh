#!/usr/bin/env bash
# Run inside the allocated container after source/runtime/model verification.
set -euo pipefail

if [[ -f /work/ready/DEFER_TO_FRESH_ALLOCATION ]]; then
    printf "E2E deferred before any measurements to a fresh allocation with full time budget.\n" >&2
    exit 75
fi

: "${V4_DEADLINE_EPOCH:?Set the UTC epoch deadline before the allocation cleanup reserve}"
V4_PYTHON="${V4_PYTHON:-/work/.venv/bin/python}"
V4_ROOT="${V4_ROOT:-/work/ready}"
V4_ENGINE_CAP_SECONDS="${V4_ENGINE_CAP_SECONDS:-1200}"
V4_MIN_START_SECONDS="${V4_MIN_START_SECONDS:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/work/cache/ready-v4/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/work/cache/ready-v4/triton}"
mkdir -p "$V4_ROOT/logs" "$V4_ROOT/results" /work/cache/ready-v4/flashinfer

for arm in baseline default-a fp16-a fp16-b default-b cutlass; do
    remaining=$((V4_DEADLINE_EPOCH - $(date -u +%s)))
    if ((remaining < V4_MIN_START_SECONDS)); then
        printf 'Insufficient time for %s: %s seconds remain before deadline\n' "$arm" "$remaining" >&2
        exit 75
    fi
    duration=$V4_ENGINE_CAP_SECONDS
    if ((duration > remaining)); then duration=$remaining; fi
    source_root=/work/source
    if [[ "$arm" == baseline || "$arm" == cutlass ]]; then source_root=/work/baseline; fi
    command=("$V4_PYTHON" -u "$V4_ROOT/harness/run_e2e.py"
        --manifest "$V4_ROOT/manifest.json" --arm "$arm"
        --output "$V4_ROOT/results/$arm" --cache-dir "/work/cache/ready-v4/flashinfer/$arm")
    if [[ "$arm" != baseline ]]; then
        command+=(--baseline-result "$V4_ROOT/results/baseline/result.json")
    fi
    if [[ "$arm" == fp16-b ]]; then
        command+=(--reference-result "$V4_ROOT/results/fp16-a/result.json")
    fi
    export PYTHONPATH="$source_root:$V4_ROOT/harness"
    "$V4_PYTHON" - "$V4_ROOT/logs/$arm.invocation.json" "$duration" "${command[@]}" <<'PY'
import datetime, json, os, pathlib, sys
record = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "timeout_seconds": int(sys.argv[2]), "argv": sys.argv[3:],
          "environment": {key: os.environ.get(key) for key in (
              "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR")}}
pathlib.Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
PY
    set +e
    timeout --signal=TERM --kill-after=120s "${duration}s" "${command[@]}" >"$V4_ROOT/logs/$arm.log" 2>&1
    exit_status=$?
    set -e
    printf '%s\n' "$exit_status" >"$V4_ROOT/logs/$arm.exit.txt"
    if ((exit_status != 0)); then
        printf 'Arm %s failed with status %s; preserving evidence and stopping\n' "$arm" "$exit_status" >&2
        exit "$exit_status"
    fi
done

"$V4_PYTHON" "$V4_ROOT/harness/compare_e2e.py" --results-root "$V4_ROOT/results" --output "$V4_ROOT/comparison.json" >"$V4_ROOT/comparison.log" 2>&1
