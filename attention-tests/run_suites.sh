#!/usr/bin/env bash
# Run unchanged repository test suites; no selection filters or test adapters.
set -eu
checkout=${1:?source checkout required}
python_bin=${2:?venv Python required}
results_dir=${3:?result directory required}
mkdir -p "$results_dir"
cd "$checkout"
export PYTHONPATH="$checkout"
unset VLLM_MINIMAX_M3_SPARSE_DECODE_FP16_DOT VLLM_BATCH_INVARIANT
suite_failed=0
for suite in \
 tests/kernels/attention/test_minimax_m3.py \
 tests/kernels/test_minimax_m3_sparse_attn_fp8_scale.py \
 tests/kernels/attention/test_minimax_m3_fp8_triton_indexer.py \
 tests/kernels/attention/test_minimax_m3_msa_cutlass_sparse_decode.py
do
 name=$(basename "$suite" .py)
 date -u
 printf 'Running full suite: %s\n' "$suite"
 if "$python_bin" -m pytest -q -ra --durations=20 \
     --junitxml="$results_dir/$name.xml" "$suite" \
     > "$results_dir/$name.log" 2>&1
 then
   printf '%s\n' 0 > "$results_dir/$name.exitcode"
 else
   status=$?
   printf '%s\n' "$status" > "$results_dir/$name.exitcode"
   suite_failed=1
 fi
 tail -n 40 "$results_dir/$name.log"
done
exit "$suite_failed"
