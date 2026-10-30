# Historical GB200 E2E v3: failed performance gates

This run completed five engines with **70 measured trials, 50 warmups and 2,880 native request observations**. Integrity checks passed; **9 of 36 frozen performance checks failed**: four control-drift checks, two B40 wall-regression checks and three B40 mean-TPOT-benefit checks. All 20 within-engine relative-MAD checks passed. These results do **not** establish an overall E2E speedup.

The run used baseline `9731f30a030c545e7f29a8a4ed96b8562be12c88` and candidate `faf31b04c73400c2af770c280fb8d739a5997fcb`. It predates the current-main rebase and prospective v4 experiment. It does not validate their sources or turn their pending results into successes.

MiniMax-M3-NVFP4 ran on four GB200 GPUs with TP4, BF16 activations, FP8 main/index KV, explicit Triton MSA, MRV1, chunked prefill, a 4,096-token scheduling budget and `FULL_DECODE_ONLY` graphs. Inputs were fixed synthetic 2,048-token sequences; outputs were exactly 256 tokens, greedy seed42 with EOS ignored and detokenization disabled. Each fresh engine ran B8 then B40, five warmups and seven measured repetitions each, in the order below. No observations were discarded.

| Arm | B8 wall median (s) | B8 post hoc tail (s) | B40 wall median (s) | B40 post hoc tail (s) |
|---|---:|---:|---:|---:|
| Baseline | 2.148204 | 1.414778 | 5.812892 | 2.653168 |
| Default-A | 2.255404 | 1.414858 | 5.824077 | 2.652942 |
| FP16-A | 2.377649 | 1.397710 | 6.396451 | 2.604151 |
| FP16-B | 2.130319 | 1.397256 | 5.735552 | 2.604038 |
| Default-B | 2.248478 | 1.414746 | 6.245982 | 2.653423 |

The tail diagnostic was selected **after inspecting v3 results**. It spans latest first token to latest last token. All 120 observations have a common request witnessing both endpoints, with 255 remaining intervals; the batch drains as other requests finish. FP16 tail medians are 1.20–1.24% shorter at B8 and 1.84–1.86% at B40 versus the two default controls. This is exploratory phase localization, not a prespecified result, fixed-batch kernel measurement, serving-QPS claim or a replacement for the failed E2E/mean-TPOT gates. The endpoints do not reveal other requests' remaining token counts or individual ITLs; they do not identify the CPU/GPU cause of the earlier-phase drift. Different FP16/default outputs may also change model routing work.

`frozen-comparison.json` preserves the exact original comparator result, including every gate, threshold, ratio and timing statistic. `observations.json` preserves every trial/warmup and anonymous per-request timing. Monotonic timestamps are shifted to the earliest queued request per trial; wall-clock arrivals are shifted independently. `posthoc-tail.json` contains separately labeled decompositions and all tail summaries. `provenance.json` records source/model hashes, full settings, cache identities and exact development-build runtime versions; public availability of those builds is not established.

The exact frozen protocol and three harness files are included. Two hash-verified dependencies remain in the [earlier immutable supplement](https://github.com/wolfcomos/vllm/tree/567b5d5ce2b5110bfae5bf5397fbe79c6902b852/graph-perf): `run_graph_perf.py` and `graph_perf_cache_seed.py`. The original run reused that baseline's rank-specific FlashInfer tactics. Reproduction also needs the original private input/output evidence and tactic witness; this public projection alone cannot rerun literal token comparisons. Portable `/work` paths in source are container mount conventions.

Protocol correction: the external sampler recorded approximately 1 Hz clocks/power/temperature/utilization, not a continuous throttle-reason field. The separate audit found zero cumulative throttle-counter increments across each engine. Graph policy/capture checks do not prove replay of every timed token. Neither observation establishes a performance cause.

Raw prompts, token arrays, model responses, request IDs and machine/infrastructure identifiers are excluded. `redaction-audit.json` records the projection and verification; `raw-evidence-sha256.json` identifies private inputs by digest. `SHA256SUMS.json` covers every other file in this supplement. Failed original gates remain unchanged.
