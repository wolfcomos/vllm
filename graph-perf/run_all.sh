#!/usr/bin/env bash
set -euo pipefail

# The parent must supply this signal only after quality/operator jobs have exited.
if [[ "${M3_GRAPH_START_AUTHORIZED:-0}" != 1 ]]; then
    echo "Graph launch held: parent must explicitly set M3_GRAPH_START_AUTHORIZED=1." >&2
    exit 64
fi
stage="${1:-timing}"
if [[ "$stage" != timing && "$stage" != diagnostics ]]; then
    echo "Usage: run_all.sh [timing|diagnostics]" >&2
    exit 64
fi
graph_dir="${M3_GRAPH_DIR:-/work/graph-perf}"
graph_python="${M3_GRAPH_PYTHON:-/work/.venv/bin/python}"
model_dir="${M3_GRAPH_MODEL:-/work/models/MiniMax-M3-NVFP4}"
pin_file="${M3_GRAPH_SOURCE_PINS:-/work/source-pins.json}"
baseline_source="${M3_GRAPH_BASELINE_SOURCE:-/work/baseline}"
candidate_source="${M3_GRAPH_CANDIDATE_SOURCE:-/work/source}"
cache_dir="${M3_GRAPH_FI_CACHE:-/work/cache/graph-flashinfer}"
gsm8k_comparison="${M3_GSM8K_COMPARISON:-/work/gsm8k/comparison.json}"
# Explicit operator release and a completed quality pass are separate gates.
# Run this before tool discovery or any GPU queries, for both stages.
"$graph_python" - "$gsm8k_comparison" <<'PY'
import json
from pathlib import Path
import sys
try:
    comparison = json.loads(Path(sys.argv[1]).read_text())
    if comparison.get('passed') is not True:
        raise ValueError('passed must be JSON true')
    identical = comparison['comparisons']['bf16']['same_output_token_sequences']
    if identical != 1319:
        raise ValueError(f'BF16 identical token sequences must equal 1319, got {identical!r}')
except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
    print(f'Graph launch held: GSM8K quality gate failed for {sys.argv[1]}: {exc}', file=sys.stderr)
    raise SystemExit(78)
print(f'GSM8K quality gate passed: {sys.argv[1]} (passed=true, BF16 identical token sequences=1319)')
PY
results_dir="$graph_dir/results"
diagnostics_dir="$graph_dir/diagnostics"
baseline_result="$results_dir/baseline/result.json"
active_output=""
finalize_on_exit() {
    local rc=$?
    if [[ -n "$active_output" ]]; then
        printf '%s\n' "$rc" > "$active_output/runner-exit.txt"
        if [[ ! -e "$active_output/exit.txt" ]]; then printf '%s\n' "$rc" > "$active_output/exit.txt"; fi
        if [[ ! -e "$active_output/finished.txt" ]]; then date -u +%Y-%m-%dT%H:%M:%SZ > "$active_output/finished.txt"; fi
    fi
}
trap finalize_on_exit EXIT
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/work/cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/work/cache/triton}"
unset RAY_ADDRESS
for required in timeout nvidia-smi flock pgrep; do command -v "$required" >/dev/null; done
mkdir -p "$results_dir" "$diagnostics_dir"
exec 9>"$graph_dir/.runner.lock"
flock -n 9 || { echo "Another graph runner owns this validation directory." >&2; exit 75; }
"$graph_python" - "$graph_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
for name, expected in json.loads((root / 'SHA256SUMS.json').read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, ('Staged file hash differs', name)
PY

remaining_seconds() {
    "$graph_python" - <<'PY'
from datetime import datetime, timezone
deadline = datetime.fromisoformat('2026-10-29T00:04:28+00:00')
print(int((deadline - datetime.now(timezone.utc)).total_seconds()) - 15 * 60)
PY
}

snapshot() {
    local output="$1" label="$2"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$output/$label.timestamp.txt"
    timeout 15s nvidia-smi \
        --query-gpu=index,uuid,name,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used \
        --format=csv > "$output/$label.gpus.csv" 2> "$output/$label.gpus.stderr.txt"
    timeout 15s nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory \
        --format=csv,noheader > "$output/$label.processes.csv" 2> "$output/$label.processes.stderr.txt"
}

require_idle() {
    local output="$1"
    "$graph_python" - "$output/before.processes.csv" <<'PY'
from pathlib import Path
import sys
rows = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
assert not rows or rows == ['No running processes found'], ('GPU compute processes still present', rows)
PY
    if pgrep -x nsys > "$output/before.nsys-pids.txt" || pgrep -x ncu > "$output/before.ncu-pids.txt"; then
        echo "An existing Nsight profiler is present; refusing concurrent profiling/timing." >&2
        return 75
    fi
}

run_arm() {
    local phase="$1" mode="$2" use_nsys="$3"
    local output source_dir remaining limit rc
    source_dir="$candidate_source"
    if [[ "$mode" == baseline ]]; then source_dir="$baseline_source"; fi
    output="$results_dir/$mode"
    if [[ "$phase" == diagnostic ]]; then output="$diagnostics_dir/$mode"; fi
    [[ ! -e "$output/run.log" && ! -e "$output/result.json" ]] || {
        echo "Existing arm artifacts: $output. Use a new experiment directory; do not overwrite." >&2
        return 73
    }
    remaining="$(remaining_seconds)"
    if (( remaining < 600 )); then
        echo "Insufficient time before 23:49:28 UTC cleanup reserve; arm $mode held." >&2
        return 75
    fi
    mkdir -p "$output"
    active_output="$output"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$output/started.txt"
    snapshot "$output" before
    require_idle "$output"
    local -a command=("$graph_python" -u "$graph_dir/run_graph_perf.py" run
        --mode "$mode" --phase "$phase" --model "$model_dir" --source-pins "$pin_file"
        --prompts "$graph_dir/prompts.json" --cache-dir "$cache_dir" --repeats 5 --output "$output")
    if [[ "$phase" == diagnostic || "$mode" != baseline ]]; then
        command+=(--baseline-result "$baseline_result")
    fi
    if [[ "$use_nsys" == 1 ]]; then
        command+=(--profile-cuda)
        command=(nsys profile --trace=cuda,nvtx --trace-fork-before-exec=true
            --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=repeat
            --sample=none --cpuctxsw=none --output="$output/nsys" "${command[@]}")
    fi
    command=(env "PYTHONPATH=$source_dir:$graph_dir" "${command[@]}")
    printf '%q ' "${command[@]}" > "$output/command.txt"
    printf '\n' >> "$output/command.txt"
    # Keep 30s for TERM->KILL and 60s for final snapshots/reporting before cleanup.
    limit=$(( $(remaining_seconds) - 90 ))
    if (( limit <= 0 )); then return 75; fi
    set +e
    timeout --signal=TERM --kill-after=30s "${limit}s" "${command[@]}" > "$output/run.log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$output/exit.txt"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$output/finished.txt"
    snapshot "$output" after || true
    if (( rc != 0 )); then return "$rc"; fi
    local -a log_check=("$graph_python" "$graph_dir/check_cache_logs.py"
        --result "$output/result.json" --log "$output/run.log" --output "$output/cache-log-check.json")
    if [[ "$phase" == diagnostic || "$mode" != baseline ]]; then log_check+=(--seeded); fi
    if [[ "$phase" == diagnostic || "$mode" == fp16 || "$mode" == default-repeat ]]; then
        log_check+=(--reference "$results_dir/default/cache-log-check.json")
    fi
    set +e
    "${log_check[@]}" > "$output/cache-log-check.log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$output/cache-log-check.exit.txt"
    if (( rc != 0 )); then return "$rc"; fi
    printf '0\n' > "$output/runner-exit.txt"
    active_output=""
}

"$graph_python" "$graph_dir/run_graph_perf.py" prepare --prompts "$graph_dir/prompts.json"
if [[ "$stage" == timing ]]; then
    if (( $(remaining_seconds) < 3000 )); then
        echo "Fewer than 50 minutes remain before the cleanup reserve; timing sequence held." >&2
        exit 75
    fi
    "$graph_python" - "$cache_dir" <<'PY'
from pathlib import Path
import sys
assert not any(Path(sys.argv[1]).rglob('*.json')), 'Fresh graph baseline requires an unused FlashInfer cache root'
PY
    for mode in baseline default fp16 default-repeat; do run_arm timing "$mode" 0; done
    "$graph_python" "$graph_dir/compare_graph_perf.py" \
        "$results_dir/baseline/result.json" "$results_dir/default/result.json" \
        "$results_dir/fp16/result.json" "$results_dir/default-repeat/result.json" \
        --output "$results_dir/comparison.json" > "$results_dir/comparison.log" 2>&1
else
    "$graph_python" - "$results_dir/comparison.json" <<'PY'
import json
from pathlib import Path
import sys
assert json.loads(Path(sys.argv[1]).read_text())['status'] == 'complete', 'Timing comparison must complete first'
PY
    run_arm diagnostic baseline 0
    run_arm diagnostic default 0
    use_nsys=0
    if [[ "${M3_GRAPH_NSYS_FP16:-0}" == 1 ]] && command -v nsys >/dev/null && (( $(remaining_seconds) >= 1200 )); then
        use_nsys=1
    else
        printf '%s\n' 'FP16 uses native graph metrics; Nsight not requested, unavailable, or below 20-minute remaining budget.' \
            > "$diagnostics_dir/nsys-skipped.txt"
    fi
    run_arm diagnostic fp16 "$use_nsys"
fi
