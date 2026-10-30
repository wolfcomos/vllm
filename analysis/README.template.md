# MiniMax M3 sparse decode: current-main validation

Baseline: `@BASELINE@`. Candidate: `@CANDIDATE@`.

@AVAILABILITY@

This supplement contains exact pinned operator source, executed validation harnesses, protocols and selected public result fields. `SHA256SUMS.json` covers every other artifact. `raw-evidence-sha256.json` records hashes of the original result files; the projected JSON is deliberately not a byte-identical copy of those private files. Raw prompts, model responses, token arrays, machine identities, infrastructure addresses and original filesystem locations are excluded. `/work` paths inside harnesses describe a portable container mount layout, not the machine used for the experiment.

## What the change affects

The new configuration opts eligible SM100 BF16-query/FP8-KV Triton sparse decode into FP16 dot operands and a split schedule based on SM count. Default BF16 Triton math and split selection remain the original path. FP32 accumulation remains; the independent quality evaluation measures the numerical tradeoff.

The E2E arms called `baseline`, `default-a`, `fp16-a`, `fp16-b`, and `default-b` explicitly select the Triton backend. The final arm's identifier is `cutlass`, but its meaning is **current-main default backend with native Triton fallback**, not exclusive CUTLASS kernel execution. For the tested FP8 geometry, the native unpadded threshold is 16 requests: B8 falls back to Triton; B40 is CUTLASS eligible. Draining and graph padding can affect selection. Default-backend Q preparation is also different: it emits FP8 Q, and its Triton fallback dequantizes Q to BF16; explicit Triton receives BF16 Q directly. Static helper/dispatcher snapshots establish policy and eligibility, not observed per-step execution counts. This late single engine is a descriptive backend reference subject to order confounding.

## Reading the results

- `operator-validation/results.json` reports every collected JUnit test case and its pass/fail/error/skip status from the full four original attention suites. Collection errors without a repository test label retain anonymous error records. Failed diagnostics are retained privately; no raw tracebacks are republished. An absent final process exit code leaves the arm incomplete even when JUnit observations exist.
- `gsm8k/results.json` and `gsm8k/comparison.json`, when available, report all 1,319 questions, first-five training fewshots, nonthinking chat generation, greedy seed42, and max1024 output tokens. The local gate permits at most0.5 percentage points accuracy loss, invalid/capped rates at most1%, baseline accuracy at least80%, and exact default output tokens. These are local gates, not upstream policy or proof of quality equivalence. Instrumented evaluation elapsed time is not a speed claim.
- `e2e/results.json` retains every collected measured trial and warmup, including failed attempts' observations. Verified raw token hashes can still reveal within-arm token variation: this is an explicit integrity failure, not a passed timing comparison. Observed consistency covers only collected trials and does not establish full protocol integrity. Native per-request timing observations remain in array order with anonymous positions. Monotonic timestamps are shifted by that trial's earliest queued timestamp, while arrival wall timestamps are shifted independently; those clock domains are never subtracted from one another. Raw request identifiers are removed.
- `e2e/comparison.json`, when available, retains all ratios, repeatability checks, prospective common-tail gates and the previous3% mean-TPOT criterion, including failures. Passing one criterion does not relabel a failure in another. The common tail spans latest first token to latest last token with a verified common request witness; it is a draining-batch interval, not fixed-batch kernel time or serving p99 ITL.
- `availability.json` lists missing or incomplete evidence explicitly. An absent comparator supplies no verdict. `runtime-summary.json` and per-arm runtime fields contain the observed package versions; native-library hashes appear where recorded. No unrecorded driver version is inferred.
- `telemetry/gsm8k-gpu-samples.csv` belongs to the instrumented quality run; `telemetry/e2e-gpu-samples.csv` belongs to the separate performance run. Either can be unavailable. Optional throttle bitmasks are retained when recorded, including explicit unavailable values; missing values are never converted to zero. Naive sampler timestamps retain their original clock convention.

The frozen E2E harness records FlashInfer `configuration_count` as the payload key count minus one. Its value of 161 includes the reserved `_generation` key: each payload actually contains 160 tactics plus `_metadata` and `_generation`. The original count is retained here. Independent validation checked the actual payload hashes and the 160-tactic load/save log counts; this bookkeeping correction does not change the harness, observations or performance gates.

@HISTORY@

The E2E workload uses real pretrained `nvidia/MiniMax-M3-NVFP4` weights at revision `901464083161bf8612a29ff7ad29914cd4ab4a85`, with fixed synthetic 2048-token inputs and2048 output tokens, batches8/40, TP4 on four GB200 GPUs, FP8 main/index KV, chunked prefill, and `FULL_DECODE_ONLY` graphs with eager prefill. It is not a natural-traffic or HTTP serving benchmark. Graph availability/policy checks do not prove every timed token replayed a graph. The complete workload and predeclared local criteria are in `e2e/PROTOCOL.md`.

## Reproduction

Use the commits and source hashes in `source-pins.json` and the exact observed runtime versions. Public packages or matching source builds are required; this supplement does not redistribute native libraries, model weights or private container images. Obtain the model from its public repository subject to its terms. The independent baseline must generate fresh FlashInfer tactics, which later arms reuse exactly per rank; GSM8K and E2E use separate caches. Do not replace the current TP4 cache policy or mix a different runtime into one arm.

Arrange baseline and candidate checkouts at `/work/baseline` and `/work/source`, the model at `/work/models/MiniMax-M3-NVFP4`, and the Python environment at `/work/.venv`. Copy the public GSM8K harness to `/work/gsm8k`, obtain the standard train/test files under `/work/gsm8k/data`, and verify `gsm8k/dataset-manifest.json`. Run `bash /work/gsm8k/run_all.sh` on an allocated, otherwise idle four-GPU environment. The orchestration owner must enforce an allocation deadline and clean up owned resources.

For E2E, copy its harness to `/work/ready/harness`, reproduce the synthetic inputs using `analysis/prepare_inputs.py`, and create a fresh manifest with `make_manifest.py --baseline /work/baseline --candidate /work/source --model /work/models/MiniMax-M3-NVFP4 --prompts /work/ready/prompts.json --output /work/ready/manifest.json`. Set `V4_DEADLINE_EPOCH` to a real allocation deadline minus cleanup reserve, then run `run_sequence.sh`. Keep all arms, warmups, measurements and failures. The complete raw evidence is required to rerun the original token/answer comparisons; the public projection alone cannot perform them.

To project a newly collected evidence tree locally, without publishing it:

```bash
python3 analysis/build_bundle.py \
  --notes /path/to/validation-notes \
  --source-repo /path/to/vllm \
  --e2e-telemetry /path/to/e2e-gpu-telemetry.csv \
  --output /path/to/new-public-bundle
```

Expected raw subdirectories are `remote-results/results`, `remote-results/tests-ready`, `remote-results/ready`, and `remote-results/gsm8k`. Place the audited v3 supplement under `publication/historical-v3`; its fixed file list, pinned manifest and every checksum must match before inclusion. Missing or incomplete required records fail by default. `--allow-incomplete` instead publishes only available observations with explicit omissions; it never substitutes projected values or pending successes. The builder is bounded to15MiB, performs an allowlist projection and privacy scan, and never pushes to GitHub. Review the resulting bundle before publication.
