# MiniMax M3 sparse decode validation supplement

This bundle records validation of baseline `9731f30a030c545e7f29a8a4ed96b8562be12c88` and candidate `a450316f26fcff8f5715e51c444cd272b9cad12a`. The candidate changes SM100 decode split scheduling and adds an opt-in FP16 dot path. It does **not** contain scalar-scale hoisting.

**The primary quality gate failed.** Default BF16 scored 1246/1319 versus the baseline's 1259/1319, a drop of 0.9856 percentage points. The predeclared local limit was 0.5 percentage points. FP16 opt-in scored 1260/1319 and passed that limit; it does not erase the default-path failure. The separate baseline repeat reproduced all 1,319 prediction rows and token sequences byte-for-byte; it remains diagnostic and does not replace the primary baseline. These are validation findings for a blocked draft, not a merge-readiness claim.

## Files and provenance

- `performance/`: exact tested microbenchmark, helper scripts and complete baseline/candidate operator snapshots; sanitized per-shape measurements retain their timing samples and numerical checks.
- `attention-tests/`: exact full-suite runner, compact suite counts and original-test preservation checks.
- `gsm8k/`: exact evaluation, comparison, cache-seeding and launch scripts; public source/model/dataset pins; compact independent primary audit.
- `audited-results.json`: complete audited primary aggregates and the separate baseline-repeat diagnostic.
- `runtime-summary.json`: public runtime identifiers and reproducibility limitations.
- `exact-file-hashes.json`: hashes of unchanged scripts/source copies. `LICENSE` preserves the repository's Apache-2.0 license for the source snapshots. `SHA256SUMS` covers the public bundle's files except itself.

The source snapshots correspond to `vllm/models/minimax_m3/common/ops/sparse_attn.py` at the two commits above. The complete checkouts are still needed for imports, original tests and model evaluation. `gsm8k/source-pins.json` records the five production-file hashes checked at runtime; keys there are relative to the checkout's `vllm/` directory.

No model weights, tokenizer files, raw dataset, prompts, generated answers, prediction token IDs, private image location, cluster identifiers or full runtime logs are included. Published result JSONs omit machine-specific runtime fields; all measurement samples and source/harness hashes are retained. Scripts are byte-for-byte copies and can themselves generate detailed local evidence, including raw prompts and device identifiers. Filter those newly generated outputs before sharing them.

## Runtime and layout

The recorded runs used NVIDIA GB200 (SM100, 152 SM), one GPU for microbenchmarks and four for the pretrained GSM8K runs. See `runtime-summary.json` for the development package versions. The identical development binary may not be publicly available. Matching the version and source identifiers does not guarantee bitwise reproduction.

Provide a compatible vLLM CUDA/native runtime at `/work/.venv/bin/python`, with both Python source trees able to load its native extensions. The measured native wheel was built from `cc2e4fecf2fca603939445faccf20dd466f1502e`; its native build inputs matched the baseline, and no native API shim was used. This is a record of the tested environment, not a claim that an arbitrary wheel can be overlaid safely.

The scripts deliberately preserve their actual `/work` paths. Use a dedicated work directory or container and arrange the following layout. This placement is a portable setup instruction, not an additional tested installation procedure:

| Path | Content |
| --- | --- |
| `/work/baseline` | Complete baseline checkout at `9731f30a030c545e7f29a8a4ed96b8562be12c88` |
| `/work/source` | Complete candidate checkout at `a450316f26fcff8f5715e51c444cd272b9cad12a` |
| `/work/performance` | The bundle's `performance` directory, including `sources` |
| `/work/attention-tests` | The bundle's `attention-tests` directory |
| `/work/run_gsm8k.py` | Copy of `gsm8k/run_gsm8k.py` |
| `/work/gsm8k_cache_seed.py` | Copy of `gsm8k/gsm8k_cache_seed.py` |
| `/work/compare_gsm8k.py` | Copy of `gsm8k/compare_gsm8k.py` |
| `/work/source-pins.json` | Copy of `gsm8k/source-pins.json` |
| `/work/run_all.sh` | Copy of `gsm8k/run_all.sh` |
| `/work/run_baseline_repeat.sh` | Copy of `gsm8k/run_baseline_repeat.sh` |
| `/work/models/MiniMax-M3-NVFP4` | Complete public pretrained checkpoint pinned below |
| `/work/gsm8k` and `/work/cache` | Dedicated run output and cache directories |

Obtain `nvidia/MiniMax-M3-NVFP4` revision `901464083161bf8612a29ff7ad29914cd4ab4a85` from its public model repository, preserving its tokenizer and chat template. The measured download comprised 107 files, including 88 weight shards, totaling 250,137,296,832 bytes; weight hashes were checked against the pinned manifest. This bundle does not provide a replacement download or model implementation.

## Full original attention suites

`attention-tests/run_suites.sh` is the exact tested runner. It takes a checkout, Python interpreter and output directory, runs all cases in each of the following files, preserves failure exit codes, and writes JUnit/log evidence:

```
tests/kernels/attention/test_minimax_m3.py
tests/kernels/test_minimax_m3_sparse_attn_fp8_scale.py
tests/kernels/attention/test_minimax_m3_fp8_triton_indexer.py
tests/kernels/attention/test_minimax_m3_msa_cutlass_sparse_decode.py
```

The recorded full-suite invocations used the baseline on GPU0 and the final candidate on GPU1 after the operator benchmark finished:

```bash
CUDA_VISIBLE_DEVICES=0 bash /work/attention-tests/run_suites.sh /work/baseline /work/.venv/bin/python /work/attention-tests/baseline
CUDA_VISIBLE_DEVICES=1 bash /work/attention-tests/run_suites.sh /work/source /work/.venv/bin/python /work/attention-tests/candidate
```

Baseline: 181 passed, 46 skipped. Candidate: 199 passed, 46 skipped. There were no failures or removed original test cases. The 46 skips are the existing ROCm-only cases on CUDA. The candidate adds eight long-context cases and ten FP8 split/scale/range cases. Suite runtime is not a performance comparison.

## Synthetic operator comparison

These are the exact recorded benchmark commands:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/work/source /work/.venv/bin/python   /work/performance/bench_upstream.py --suite main --output /work/performance/results/main.json
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/work/source /work/.venv/bin/python   /work/performance/bench_upstream.py --suite edge --output /work/performance/results/edge.json
```

Both exited zero. Each main timing includes decode and merge in a warmed CUDA graph: 32 operator invocations per capture, four replays per sample, seven paired rounds with rotating arm order. Per-shape speedup is the median of paired baseline/candidate sample ratios; aggregate speedup is their geometric mean. This excludes host planning, launch and compilation time. Clocks were not locked; repeated inputs and cache residency are part of this synthetic protocol. There was no concurrent workload on the measured GPU.

The main sweep has 24 shapes: 21 scalar-scaled E4M3 cases and three controls (unscaled E4M3, per-token/head scales, BF16 cache). Q/output are BF16; main geometry is 16 Q heads, one KV head, dimension 128, context 4096 and top-K 16. The FP16 arm falls back in the per-token/head and BF16-cache controls. Its aggregate therefore describes the opt-in configuration, not FP16 arithmetic in every row.

| Arm | Geomean speedup | Minimum shape speedup | Maximum shape speedup |
| --- | ---: | ---: | ---: |
| Candidate BF16 | 1.12049x | 0.98887x | 1.60788x |
| Candidate FP16 opt-in | 1.13585x | 0.92837x | 1.63125x |

Some shapes regress. These timings do not establish serving throughput or model accuracy.

All 62 main/edge shapes passed both independent exact-dequantization FP32 and BF16-rounded-dequantization references with TF32 disabled (NRMSE <1%, maximum absolute error <0.017). There are 372 graph/eager comparisons, including changed-Q replay; 76 batch-invariant comparisons are bitwise equal to baseline; 16 dtype-fallback comparisons are bitwise equal across candidate flags. Edge checks include HND/NHD, two KV heads, short contexts, invalid selected blocks and padded requests; numerical assertions cover active outputs, not padded output values.

`performance/diagnose_fp16_range.py` and `performance/results/range-fixed.json` preserve the separate eight-case E4M3/E5M2 range diagnostic. All final outputs are finite and the default BF16 arm is bitwise baseline. FP16 opt-in saturates scaled K/V to ±65504, so those out-of-range values intentionally need not equal BF16. The four repository range regressions are also included in the full-suite counts above.

## Pretrained GSM8K evaluation

`gsm8k/protocol.json` records the model, dataset, prompt and label hashes, decoding settings and predeclared local quality criterion. The harness imports the repository's `_build_gsm8k_prompts(1319, 5)`, `_score_gsm8k` and answer parser, using the first five training examples and all 1,319 test examples. The public dataset URLs and SHA256 values are recorded there. The repository loader caches these files under `/tmp`; verify cached inputs match those hashes before attributing a new run to this protocol.

The actual run uses real pretrained weights, TP4, eager execution, the original model runner, BF16 compute, FP8 main/indexer KV caches, Triton sparse decode, block size 128, `max_num_seqs=40`, model length 8192, 4096 batched tokens, 8 GiB KV cache per rank, chunked prefill and no prefix caching. Generation is greedy with seed 42, up to 1024 tokens and `thinking_mode="disabled"`. An eight-question pilot runs before the complete evaluation. It is not the reported full score.

After arranging the paths above, launch the unchanged primary script:

```bash
bash /work/run_all.sh
```

`run_all.sh` sets the device/environment/cwd, executes baseline, BF16 and FP16 in that order, and invokes the exact comparator. Its underlying per-arm command is:

```bash
timeout --signal=TERM --kill-after=30s 3600s /work/.venv/bin/python -u   /work/run_gsm8k.py --mode "$mode" --output "/work/gsm8k/$mode"
```

The shell variable, source checkout, `PYTHONPATH`, cache variables, redirection and order are established by the launcher; do not treat the abbreviated per-arm command as an independent setup. The comparator returns a failure exit status when the quality gate fails, even if inference completed normally.

Candidate evaluation uses the included `gsm8k_cache_seed.py` worker extension to copy the completed baseline's four FlashInfer tuning-result JSON files into the exact namespace returned by the live cache resolver. It leaves the returned path, execution math and production configuration hashes unchanged; it does not copy compiled execution or CUDA graph caches. Baseline and candidate runs must retain the same GPUs, model path and shared baseline cache for its strict identity checks. All 12 primary rank files were byte-identical, each with 160 operation configurations, and remained unchanged from startup through generation. Both candidate modes loaded those configurations. This control isolates attention changes from independently selected non-attention tactics; it does not prove token-level determinism.

The telemetry records actual kernel launches and subtracts pilot counters. It confirms changed split shapes and flag selection on every rank, including 40-query batches: baseline four chunks versus candidate six chunks. The model evaluation is eager; the separate operator replay checks are the CUDA graph evidence.

| Primary arm | Correct / total | Accuracy | Change vs baseline | Local quality gate |
| --- | ---: | ---: | ---: | --- |
| Baseline | 1259 / 1319 | 95.4511% | — | Sanity floor passed |
| Candidate BF16 | 1246 / 1319 | 94.4655% | -0.9856 pp | **Failed** |
| Candidate FP16 opt-in | 1260 / 1319 | 95.5269% | +0.0758 pp | Passed |

All arms have zero invalid answers and zero truncated completions. The paired BF16 comparison loses 28 correct answers and gains 15; FP16 loses 16 and gains 17. `gsm8k/independent-primary-audit.json` contains the independently recomputed counts and raw-prediction digests. Statistical uncertainty and a later repeat do not override the predeclared primary gate.

The one separate diagnostic repeat used the unchanged `run_baseline_repeat.sh` launcher and wrote `/work/gsm8k/baseline-repeat`. It again scored 1259/1319 (95.4511%), with no invalid or truncated responses. Every prediction row and output token sequence was byte-identical to the original baseline; pilots, full dispatch histograms, source/settings, and cache bytes also matched. See `gsm8k/baseline-stability.json`. This repeat observed no baseline variation explaining the BF16 difference. It does not replace `/work/gsm8k/baseline` or change the primary comparison or its acceptance gate.
