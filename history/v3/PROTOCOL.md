# GB200 E2E comparison, October 30, 2026

This is a new protocol, `gb200-e2e-abba-v3`. Earlier experiments stay unchanged. It evaluates fixed-work pretrained full-model latency and native per-request decode time on one allocated four-GB200 node. No production source edits or live flag mutation.

## Fixed sources and workload

Baseline: `9731f30a030c545e7f29a8a4ed96b8562be12c88`; candidate: `faf31b04c73400c2af770c280fb8d739a5997fcb`. Reuse the immutable model revision, runtime/package pins, synthetic input artifact, helper scripts and four rank-specific FlashInfer tactic files from the previous baseline. A supplied model path may relocate the same verified model. The orchestration owns whole-model hash validation, physical GPU UUID identity and resource cleanup.

TP4, BF16 model, FP8 main/index KV, Triton MSA, MRV1, 8 GiB KV per rank, 40 maximum sequences, token budget 4096, max model length 8192, APC off, chunked prefill on. Compilation mode 0, FULL_DECODE_ONLY, capture sizes [8,40]; eager prefill. Exactly the existing 40 synthetic 2048-token inputs, greedy seed42, ignore EOS, 256 output tokens, no detokenization. These are real full-model executions with synthetic inputs, not a natural traffic/serving benchmark.

Fresh sequential engines: baseline witness, candidate default-A, FP16-A, FP16-B, default-B (A/B/B/A around the two settings). Every engine runs B8 followed by B40, five untimed warmups then seven measured repetitions for each batch. No discarded measured observations, selective retries or post hoc gate changes. All warmups, raw outputs and timestamps are retained. Request input lists and SamplingParams are freshly constructed each call. No old-machine GPU UUID match is required; all new arms must use the same newly allocated physical GPUs.

## Intervals and metrics

Pause scheduling with `sleep(level=0, mode="wait")`, enqueue all requests, resume with `wake_up(tags=["scheduling"])`, then `wait_for_completion`. The measured wall interval includes enqueue, resume and completion; scheduling pause and output validation are outside. Level0 does not offload GPU memory.

All arms enable the identical native request statistics path (`disable_log_stats=False`). Persist each output's native frontend TTFT, core queue/prefill/decode durations and per-request mean TPOT = `(last_token_ts - first_token_ts) / (num_generation_tokens - 1)`. Core *_ts use one monotonic clock; arrival_time is wall clock and is never subtracted from those fields. Per-request mean TPOT includes scheduling and overlapping work, not just kernel execution. It is not per-token p99 ITL. Report median across seven trials of the trial mean TPOT, mean TTFT, total wall latency and output-token throughput. Keep every per-request metric so summaries can be recomputed.

No profiler, dispatch recorder or custom logging hook runs in measured processes. Capture-entry and dispatcher checks run before measurements; they establish graph availability/selection but do not independently prove actual replay for every timed token. An external low-rate telemetry sampler records GPU clock/power/temperature/utilization and throttle status throughout the same interval, with CPU affinity recorded. Telemetry cannot prove a performance cause by itself.

## Integrity and local decision gates

Require source/API/runtime hashes, prompt hashes, baseline-seeded tactic hashes and unchanged tuner statistics. Require exactly256 tokens, zero preemptions, unchanged inputs, measured within-arm output exactness, default output exactness to fresh baseline, and FP16-A/FP16-B output exactness. Different FP16/default outputs are allowed; prior GSM8K evaluation owns the separate quality evidence. Integrity failure stops orchestration and preserves logs, rather than repairing and silently retrying.

For both B8 and B40, every arm's wall-latency and mean-TPOT relative MAD must be <=2%; default-A/baseline and default-B/default-A medians must stay within2%. For a positive B40 decode-latency claim, both FP16 mean-TPOT medians must be at least3% below both default controls, with no >2% B40 total-wall regression versus either control. All B8 results and TTFT results remain visible. These are local conservative criteria, not an upstream acceptance requirement or statistical confidence interval. Any failed gate leaves the positive performance claim unsupported; no automatic follow-up GPU run.

The allocation/orchestrator provides a hard wall-time cap, cleanup reserve, command/environment manifest, external telemetry, and same-node GPU UUID validation. Do not begin another engine without sufficient budget for completion and cleanup. The standalone CPU comparator has no authority to allocate or extend compute.
