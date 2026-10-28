# Admission-controlled follow-up, 2026-10-28

This is a separate experiment. The original four-arm measurements and their failed strict comparison remain immutable and inconclusive. The standalone immutable-input probe stopped after its first warmup because its graph logging handler captured no rows, although its raw log does show FULL graph steps. The hook was attached before vLLM logging initialization; its exact failure mechanism is unresolved. Neither experiment is relabeled as passing.

## Question and treatment

Does an FP16 sparse-decode opt-in improve fixed-work full-model latency once every request is queued before scheduling resumes? Source inspection establishes that ordinary multiprocess offline generate can begin scheduling before all requests arrive. This is a plausible confound, not a measured explanation of the earlier variations.

Use the pinned public APIs: sleep(level=0, mode="wait"), enqueue fresh prompts and SamplingParams, wake_up(tags=["scheduling"]), wait_for_completion. Level 0 pauses scheduling without changing GPU memory. Pause is outside the interval; enqueue, resume, and completion are inside it. This interval differs from the original generate experiment, so timings are compared only within this follow-up.

## Frozen controls

- Sources: baseline 9731f30a030c545e7f29a8a4ed96b8562be12c88; candidate faf31b04c73400c2af770c280fb8d739a5997fcb. Same pinned MiniMax-M3 NVFP4 pretrained weights, native binaries, runtime, and ordered four GB200 GPUs as the completed quality and graph runs.
- TP4, MRV1, BF16 model, FP8 main/index KV, Triton MSA, 8 GiB KV per rank, chunked prefill, APC disabled. Max sequences 40, batched-token budget 4096, model length 8192. Compilation mode 0, FULL_DECODE_ONLY, capture sizes [8,40], one capture warmup. Prefill remains eager.
- Existing immutable 40 synthetic prompts, 2048 tokens each, random seed 42 and token IDs below 10000. Greedy, seed 42, ignore EOS, exactly 256 output tokens, detokenization disabled. B8 precedes B40 in every engine.
- Ordered independent engines: baseline witness, default-A, fp16-A, fp16-B, default-B. Three warmups and seven measured repetitions per batch. All observations retained, with no selective retries or outlier removal.
- Every arm gets a fresh private cache directory seeded from the completed original graph baseline's exact 160-configuration-per-rank FlashInfer caches. Verify hashes, zero startup tuning activity, and unchanged caches/statistics after generation. Shared already-warmed compilation caches are allowed and recorded.
- Construct fresh prompt lists and normalized sampling parameters for every call. Compare before/after inputs and returned prompts against an independent immutable tuple oracle. Validate request IDs/order, output counts and lengths, source hashes, runtime and GPU identity. Save all raw output token arrays privately and publish only hashes/counts.
- No profiler or dispatch recorder in measured processes. Worker capture/dispatch checks happen outside measurements. If timing gates pass, use a separate untimed diagnostic process to verify native FULL execution counts for B8/B40. Attach the metrics handler after vLLM logging initialization, verify it remains installed, and record actual LogRecord names. Both vllm.v1.metrics.loggers and vllm.compilation.cuda_graph are observed; a formatted caller filename alone does not identify the logger namespace. Native metrics support runtime mode; they are not a hardware trace.

## Gates fixed before execution

Integrity failures stop the sequence. Within each arm and workload, all seven measured output token arrays must match exactly. Baseline self-variation stops before default-A. Default-A must match baseline exactly before either FP16 arm. Default-B must match baseline/default-A; fp16-B must match fp16-A. Warmup observations are retained but excluded from both the timing summary and the repeatability gate.

Performance is evaluated only after all five arms and integrity gates complete. For each workload, report every measured latency, median, min/max and relative median absolute deviation (MAD/median). Require each arm's relative MAD <= 2%, absolute default-B/default-A median drift <= 2%, and absolute default-A/baseline median difference <= 2%. These are conservative local acceptance criteria, not vLLM standards or statistical equivalence tests.

For a positive B40 claim, both FP16 arm medians must be at least 3% lower than both default arm medians, in addition to the control/noise gates. Always report B8, including any regression; no all-workload speedup claim follows from B40 alone. Failure of any gate leaves the performance conclusion unsupported and the PR draft. No gate will be loosened after data are seen. Output changes under FP16 are expected; the completed separate GSM8K evaluation owns quality acceptance.

## Bound and stop conditions

Use only the existing allocation. Maximum wall time for this follow-up, including any untimed diagnostic, is 60 minutes from release. Do not start another engine unless at least 12 minutes remain before that cap or the cleanup reserve, whichever is earlier. The allocation ends at 2026-10-29 00:04:28 UTC; cleanup starts no later than 2026-10-28 23:49:28 UTC. On failure, preserve artifacts and stop; do not silently repair and rerun. Any new protocol needs a distinct label and pin. Root releases the exact reviewed harness/protocol SHA manifest before GPU launch.
