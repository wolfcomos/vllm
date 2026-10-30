# Prospective current-main GB200 E2E v4

This new experiment is fixed before measurements. Earlier v3 results, its failed criteria and the later exploratory common-tail analysis remain unchanged. Discovering a potentially useful metric after that run is not evidence of a prespecified v3 benefit; v4 prospectively measures it on new sources, longer output and a current-default backend reference.

## Frozen treatment and controls

Source baseline: `a20280f30e3169d33a7e21d1d117e9719bace00b`; candidate: `25acf1953bf42857aec52d5f90cb7eac174d3189`. A manifest pins both source maps, common API hashes, model revision, input artifact and full engine settings. The baseline uses its matching native wheel. Record all package versions, Torch/CUDA and native binary hashes from baseline, and require identical runtime/binaries in every later engine; no obsolete runtime or tactic count is assumed.

The model is pretrained MiniMax-M3 NVFP4, TP4 on the same four GB200 GPUs, model activations BF16, main/index KV FP8, MRV1. Max sequences40, batched-token budget4096, model length8192, KV memory8GiB/rank, APC off, chunked prefill on. Compilation mode0, FULL_DECODE_ONLY, exact capture sizes [8,40], eager prefill. Same immutable synthetic 2048-token prompts, seed42, greedy, ignoreEOS, exactly2048 output tokens, detokenization disabled. The full model executes real generation with synthetic input; this is not a natural-traffic or HTTP serving benchmark.

Six fresh engines in this frozen order: baseline with explicit Triton, candidate default-A with explicit Triton, candidate FP16-A with explicit Triton, candidate FP16-B with explicit Triton, candidate default-B with explicit Triton, baseline current-default backend with native fallback. The final arm keeps the machine identifier `cutlass`, omits the backend override and must resolve `use_cutlass_decode=True`; that flag establishes static eligibility, not exclusive execution of a CUTLASS kernel. FP16 must resolve true on the live implementations only in the FP16 arms, with SM100/BF16/FP8/scalar-scale eligibility checked. The final engine's comparisons are descriptive references subject to process/order confounding, not paired backend superiority claims.

Current-main's SM100 native threshold is 16 requests for this FP8 geometry. Its B8 path therefore uses Triton fallback; the model still emits FP8 Q under the default backend and the fallback dequantizes Q to BF16. It is not numerically identical to selecting the explicit Triton backend, which emits BF16 Q directly. B40 is CUTLASS eligible, but a draining batch may select fallback. Before timing, record actual layer and metadata-builder geometry and call the native `should_prepare_decode_metadata` helper for every size 1..40, including B8/B40. Record both unpadded eligibility and eligibility for the graph-dispatched padded size. Graph captures [8,40] can pad a draining shape, so the raw threshold alone does not establish the actual runtime transition or mixture. These are untimed policy snapshots, not observed per-step dispatch counts. No environment threshold override or timing instrumentation is introduced. This clarification is fixed before any v4 performance measurement; workloads and gates are unchanged.

Every engine runs B8 followed by B40. Each has five untimed warmups and seven measured repetitions. Retain every observation, token array, native metric, timestamp and failure. No selective repeats or dropped outliers. No code repair and silent rerun under the same protocol.

## Fresh tactics and source integrity

The first baseline creates fresh FlashInfer tactics in an empty private directory using the current-main TP4 cache policy. Current TP4 selects legacy per-rank cache files naturally; the external resolver hook asserts that decision and does not change it. Each later engine copies baseline's exact rank-specific cache bytes into its own resolved directory before load. Record configuration counts and metadata without fixing them to the older160 count. Require source/API/runtime identity, exact tactic hashes and zero seeded-arm startup tuning; verify full tuner statistics and cache files remain unchanged through generation. The orchestration independently verifies model weights, current physical GPU UUIDs, clocks/telemetry, source archives and compute cleanup.

## Measurement intervals and prospective common tail

Pause scheduling with `sleep(level=0, mode="wait")` before each call, enqueue all requests, resume scheduling, then wait for completion. Level0 does not offload GPU memory. The wall interval includes enqueue, resume and completion; pause and validation stay outside. Fresh input lists and SamplingParams are used every call.

All arms enable identical native request statistics. Report full wall latency, output-token throughput, native frontend TTFT and per-request mean TPOT `(last_token_ts-first_token_ts)/2047`. Core *_ts are monotonic; arrival_time is wall clock and is never subtracted from them.

The prospective common-tail duration is `max(last_token_ts)-max(first_token_ts)`: the interval from the last request receiving its first output until the last request finishes. Require a request ID to be both last-first and last-finished; ties are allowed. Only then divide by that request's2047 remaining token intervals for common-tail TPOT. If the invariant fails, persist raw metrics and stop without inventing a divisor. This is a draining-batch interval: earlier requests can finish and batch size decreases. It is not a constant-B40 per-step metric, a pure GPU-kernel interval, or p99 ITL. Full wall and original mean-TPOT results remain visible alongside it.

No profiler or custom per-step recorder runs in measured processes. Pre-timing capture-entry/dispatcher checks establish availability and selection, not actual replay of every timed token. External low-rate GPU clock/power/temperature/utilization/throttle sampling runs throughout, with CPU affinity and launch metadata recorded. Telemetry alone cannot prove a cause.

## Integrity and decision criteria

Require exact output lengths, zero preemptions, unchanged inputs, exact measured repeats within each arm, candidate default-A/default-B exactness to the fresh Triton baseline, and FP16-A/FP16-B exactness. Do not require current-default/native-fallback tokens to match explicit Triton; record its hashes and whether they match. Quality testing is a separate requirement and is not inferred from performance repeatability.

For each shape, control-tail drift is the greater of absolute default-A/baseline and default-B/default-A common-tail median changes. Evidence for a positive decode-tail benefit requires all four FP16/default median comparisons to be positive, each benefit strictly greater than twice that control drift, and no >2% full-wall regression for any comparison. Report this separately for B8 and B40. Also report each arm's wall/common-tail relative MAD and whether <=2%; poor repeatability remains a limitation, not a hidden exclusion.

Keep the previous3% mean-TPOT requirement visible as a separate result: each FP16 median must improve original per-request mean TPOT by >=3% against both default controls, with no >2% full-wall regression. Passing the new prospective common-tail criterion never relabels a failed original-TPOT criterion or the earlier v3 run. These are local evidence criteria, not vLLM policy or statistical confidence intervals.

The allocation/orchestrator defines a hard timeout and cleanup reserve; no automatic allocation extension, protocol loosening, or follow-up run is authorized by this harness. Integrity failures stop the sequence and preserve artifacts.
