# MiniMax-M3 opt-in FP16 sparse decode validation

Validation for [vLLM PR #62098](https://github.com/vllm-project/vllm/pull/62098), candidate `faf31b04c73400c2af770c280fb8d739a5997fcb` against baseline `9731f30a030c545e7f29a8a4ed96b8562be12c88`. This supplement contains validation artifacts only; the implementation is in the PR.

The candidate keeps the original BF16 operand math and split policy by default and on unsupported inputs. Eligible FP16 opt-in changes both dot operands and split selection. It requires SM100-family CUDA, BF16 queries, FP8 KV, absent or scalar scales, and batch-invariant mode disabled. Prefill and CUTLASS decode are unchanged.

## Evidence status

Operator and full GSM8K validation pass. The original CUDA Graph E2E experiment is inconclusive: strict baseline/default output parity fails and the unchanged default anchor drifts materially. No validated E2E speedup is claimed. The separate admission-controlled follow-up passes integrity checks but fails its fixed performance gates; the PR remains draft.

Full GSM8K results are in [gsm8k/results.json](gsm8k/results.json):

| Arm | Correct / 1,319 | Accuracy | Invalid / capped |
|---|---:|---:|---:|
| Baseline | 1,263 | 95.7544% | 0 / 0 |
| Candidate default BF16 | 1,263 | 95.7544% | 0 / 0 |
| Candidate FP16 | 1,260 | 95.5269% | 0 / 1 |

Default BF16 reproduces all 1,319 baseline output token sequences; the complete prediction files are byte-identical. FP16 loses 19 correct answers and gains 16, a net decrease of 3 answers / 0.2274 percentage points, within the predeclared local 0.5-point limit. It retains 270 identical full output sequences. The sole capped FP16 answer is example 814, incorrect in both arms. Scores, prompts, labels, source hashes, dispatch and tactic-cache checks were independently recomputed from raw evidence. This quality result is not a numerical-equivalence claim or evidence for other tasks.

The original unfiltered operator suites pass on both source revisions:

| Suite | Baseline passed / skipped | Candidate passed / skipped |
|---|---:|---:|
| `attention/test_minimax_m3.py` | 125 / 46 | 133 / 46 |
| `test_minimax_m3_sparse_attn_fp8_scale.py` | 4 / 0 | 14 / 0 |
| `attention/test_minimax_m3_fp8_triton_indexer.py` | 3 / 0 | 3 / 0 |
| `attention/test_minimax_m3_msa_cutlass_sparse_decode.py` | 49 / 0 | 49 / 0 |
| Total | 181 / 46 | 199 / 46 |

All original test IDs are retained. Both revisions have the same ROCm-only skips and zero failures or errors. The added policy regression cases expose four expected numerical failures on earlier candidate `4df6ebd262af62f984626f7b9bb5c5179b7def28`; all six pass on the final candidate. Direct baseline/candidate comparisons at Q `[40,16,128]`, one KV head and top-k 16 are bitwise equal for default scalar-FP8, per-token-scale FP8 fallback, and BF16-cache fallback. See [operator results](operator-validation/results.json) and [direct comparison](operator-validation/direct-baseline.json).

## Runtime and source pins

Model measurements use four GB200 GPUs, 152 SMs each, TP4, with pairwise `NV6` reported by `nvidia-smi topo`. Operator suites and direct operator comparisons each use one GPU. See [runtime-summary.json](runtime-summary.json), [source-pins.json](source-pins.json), and [native-library-hashes.json](native-library-hashes.json). The runtime is a development build; identical binaries may not be publicly available. Public version identifiers and source hashes do not guarantee bitwise reproduction.

Model: [`nvidia/MiniMax-M3-NVFP4`](https://huggingface.co/nvidia/MiniMax-M3-NVFP4/tree/901464083161bf8612a29ff7ad29914cd4ab4a85), revision `901464083161bf8612a29ff7ad29914cd4ab4a85`. All 107 model files and all 88 weight-shard hashes were checked against the pinned model manifest. The [public manifest](model-manifest.json) records filenames, sizes and hashes; model weights are not included.

The matching native build inputs were verified against the source baseline and the native operator ABI was tested without a compatibility shim. Both arms use the same native libraries. Candidate Python source and test files were checked against exact Git objects before execution. Copies of the pinned sparse attention source are under `sources/`.

## Reproduction layout

The archived harnesses preserve the `/work` layout. Prepare an isolated Linux GPU environment with the dependencies above and sufficient memory for TP4. Place the baseline checkout at `/work/baseline`, candidate at `/work/source`, and the earlier candidate at `/work/old` if running the negative control. Use separate clean checkouts at the full commits in `source-pins.json`. Install matching native vLLM components and test dependencies into `/work/.venv`; make each source tree importable with `PYTHONPATH` as in the scripts.

Place the pinned model snapshot in `/work/models/MiniMax-M3-NVFP4`, this bundle's `source-pins.json` at `/work/source-pins.json`, and its `gsm8k/dataset-manifest.json` at `/work/dataset-manifest.json`. Copy the GSM8K Python scripts into `/work`, preserving their filenames. Use empty result directories and fresh FlashInfer tactic caches for a new comparison. The scripts refuse or detect inconsistent source, data and cache inputs; do not remove these checks to reuse unrelated results.

Original tests can also be run directly in each checkout:

```bash
/work/.venv/bin/python -m pytest -q tests/kernels/attention/test_minimax_m3.py
/work/.venv/bin/python -m pytest -q tests/kernels/test_minimax_m3_sparse_attn_fp8_scale.py
/work/.venv/bin/python -m pytest -q tests/kernels/attention/test_minimax_m3_fp8_triton_indexer.py
/work/.venv/bin/python -m pytest -q tests/kernels/attention/test_minimax_m3_msa_cutlass_sparse_decode.py
```

`operator-validation/run_operator_validation.py` additionally runs the old-head negative control and verifies case preservation. `check_default_baseline.py` performs the three direct comparisons; run baseline then candidate with the same physical GPU and corresponding `PYTHONPATH`.

## GSM8K quality protocol

Run `bash gsm8k/run_all.sh` after preparing the layout. It evaluates baseline, default BF16 and FP16 opt-in sequentially, then compares results. The dataset is the repository's public GSM8K train/test asset; [dataset-manifest.json](gsm8k/dataset-manifest.json) pins the train/test bytes, prompt order and labels. Existing `/tmp/train.jsonl` and `/tmp/test.jsonl` must match those hashes. The first five training examples are the few-shot context; all 1,319 test examples are evaluated with the repository's prompt builder and answer scorer.

Settings: pretrained weights, TP4, BF16 model dtype, FP8 main and indexer caches, Triton MSA decode, eager MRV1, 8 GiB KV cache per rank, maximum sequence length 8,192, batch limit 40, 4,096 batched tokens, chunked prefill enabled, prefix caching disabled. Chat thinking is disabled; decoding is greedy with seed 42, at most 1,024 output tokens and no extra stop strings. An eight-question pilot precedes the full evaluation.

The predeclared local quality gate permits at most a 0.5 percentage-point accuracy drop from the paired baseline, requires baseline accuracy at least 80%, invalid/truncated rates at most 1%, and no runtime errors. The default BF16 arm must additionally reproduce all 1,319 output token sequences exactly. This is a local validation criterion, not an upstream acceptance policy.

Candidates reuse the fresh baseline's FlashInfer tactics through a validation-only worker extension. Source-specific cache destinations are resolved by the runtime and seeded before candidate initialization; no production math or compiled cache is patched. Cache hashes and native autotuner statistics check that candidate tuning does not occur and that tactics remain stable through evaluation. Dispatch records verify the actual operand mode and split selection on every rank. These diagnostics make quality-run elapsed time unsuitable for a performance claim.

## CUDA Graph performance protocol

The independent graph benchmark uses the same pretrained model with fixed synthetic token prompts, not GSM8K prompts. It measures complete offline generation for batches 8 and 40, input length 2,048 and output length 256, with EOS ignored and detokenization disabled. Each arm runs two warmups and five measured trials per workload. Engine order is baseline, default, FP16, default-repeat. These are ordered workload-matched ratios; they are not randomized or independent paired experiments. The default-repeat arm checks temporal drift.

`enforce_eager=False`, compilation mode 0 and `FULL_DECODE_ONLY` with capture sizes `[8,40]` configure eager prefill and full decode graphs. All workers had the expected capture entries and dispatch choices, but these checks alone do not prove actual replay in each timed trial. This tests a specified offline model configuration, not HTTP serving QPS, TTFT, or the default fully compiled serving stack. Timed processes contain no launch recorder or profiler.

The exact `run_all.sh` retains this experiment's allocation deadline and is archived orchestration evidence, not a future-date generic launcher. After completing the operator/quality gates, the following Bash commands run the same timing driver on otherwise idle GPUs. Use fresh output and FlashInfer cache directories. The local worker RPC hooks are for this isolated offline evaluation.

```bash
M3_PY=/path/to/venv/bin/python
M3_GRAPH=/path/to/supplement/graph-perf
M3_PINS=/path/to/supplement/source-pins.json
M3_MODEL=/path/to/MiniMax-M3-NVFP4
M3_BASELINE=/path/to/baseline-checkout
M3_CANDIDATE=/path/to/candidate-checkout
M3_RUN=/path/to/new-results
M3_FI_CACHE=/path/to/new-empty-flashinfer-cache

export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4
export VLLM_CACHE_ROOT=/path/to/vllm-cache
export TRITON_CACHE_DIR=/path/to/triton-cache
unset RAY_ADDRESS

"$M3_PY" "$M3_GRAPH/run_graph_perf.py" prepare --prompts "$M3_RUN/prompts.json"
m3_common=(--model "$M3_MODEL" --source-pins "$M3_PINS"
  --prompts "$M3_RUN/prompts.json" --cache-dir "$M3_FI_CACHE" --repeats 5)

for m3_mode in baseline default fp16 default-repeat; do
  m3_source="$M3_CANDIDATE"
  m3_seed=()
  m3_check=()
  if [[ "$m3_mode" == baseline ]]; then
    m3_source="$M3_BASELINE"
  else
    m3_seed=(--baseline-result "$M3_RUN/baseline/result.json")
    m3_check=(--seeded)
  fi
  if [[ "$m3_mode" == fp16 || "$m3_mode" == default-repeat ]]; then
    m3_check+=(--reference "$M3_RUN/default/cache-log-check.json")
  fi
  mkdir -p "$M3_RUN/$m3_mode"
  PYTHONPATH="$m3_source:$M3_GRAPH" \
    "$M3_PY" "$M3_GRAPH/run_graph_perf.py" run \
    --mode "$m3_mode" --phase timing "${m3_common[@]}" \
    "${m3_seed[@]}" --output "$M3_RUN/$m3_mode" \
    > "$M3_RUN/$m3_mode/run.log" 2>&1 || exit
  "$M3_PY" "$M3_GRAPH/check_cache_logs.py" \
    --result "$M3_RUN/$m3_mode/result.json" \
    --log "$M3_RUN/$m3_mode/run.log" "${m3_check[@]}" \
    --output "$M3_RUN/$m3_mode/cache-log-check.json" || exit
done

"$M3_PY" "$M3_GRAPH/compare_graph_perf.py" \
  "$M3_RUN/baseline/result.json" "$M3_RUN/default/result.json" \
  "$M3_RUN/fp16/result.json" "$M3_RUN/default-repeat/result.json" \
  --output "$M3_RUN/comparison.json"
```

The [complete four-arm projection](graph-perf/results.json) and [independent summary](graph-perf/summary.json) retain all 40 measured trials. Median generation seconds:

| Arm | B8 | B40 |
|---|---:|---:|
| Baseline | 2.1750 | 6.1680 |
| Default | 2.1841 | 6.1487 |
| FP16 | 2.3108 | 5.8289 |
| Default repeat | 2.1185 | 5.7535 |

The strict comparator exits 1 on baseline/default token parity. Repeated baseline calls also vary, so that failure alone does not attribute a regression to the PR. The unchanged default anchor improves 3.00% at B8 and 6.43% at B40; FP16 B40 is slower than that final default. These observations cannot substantiate an FP16 E2E speedup. Inputs, source/runtime controls, all 40 trial files and exact tactic caches were independently checked. Emitted JIT warnings precede measurements, but warning suppression prevents proving zero later compilation. No timed clock/thermal samples or Nsight replay trace were collected.

A separate [immutable-input probe](graph-perf/immutable-input-probe.json) stopped after its first warmup: the logging hook captured no graph rows although its raw native log contains 251 FULL B8 steps. Its first-call input and sampling checks passed, but repeated calls never ran. The exact archived `check_prompt_stability.py` and original driver's diagnostic branch preserve this instrumentation defect; they are evidence, not validated diagnostic launchers. The handler was attached before vLLM logging initialization. A formatted caller filename does not identify the emitting logger namespace; the precise failure mechanism remains unresolved. The original timing comparator and failure are preserved without relaxation.

## Admission-controlled follow-up

A distinct [protocol](graph-perf-admission-controlled/PROTOCOL.md) and [release record](graph-perf-admission-controlled/release.json) were frozen before launch. The original experiment is retained without changing its failed comparator. Source inspection shows that ordinary multiprocess offline `generate` can schedule requests before the whole batch arrives; this is a plausible confound, not a proven explanation of the previous output variation.

The new driver uses public `sleep(level=0, mode="wait")`, `enqueue`, `wake_up(tags=["scheduling"])`, and `wait_for_completion`. Level 0 pauses scheduling without moving GPU memory. Pause and checks are outside the interval; enqueue, resume and completion are measured. Compare these timings only within this protocol.

All other model/runtime/workload settings are retained. Engines run baseline witness, default-A, FP16-A, FP16-B, default-B, each with three warmups and seven measured calls at B8 then B40. Every call receives fresh inputs and parameters checked against an immutable oracle. All arms reuse the exact original graph-baseline FlashInfer tactics. Measured outputs must repeat within each arm; defaults must match the witness, and both FP16 engines must match each other. Baseline variation or default-A mismatch stops the sequence early. Warmups are retained but excluded from timing and repeatability gates.

Local performance gates require relative MAD at most 2%, default-anchor drift at most 2%, default-A/baseline median difference at most 2%, and both FP16 B40 medians at least 3% below both default controls. B8 is reported even if slower. These are local criteria, not upstream standards. Passing timings permit one separate FP16 diagnostic engine for native FULL graph metrics, with logging hooks installed after LLM initialization; no profiler or logging hook is installed in measured processes.

The frozen [driver](graph-perf-admission-controlled/run_admission_controlled.py) reuses the included original `graph-perf` helpers. Its CLI requires `--arm`, `--phase`, `--baseline-v1-result`, `--prompts`, `--source-pins`, `--cache-dir`, and `--output`; later arms also receive `--witness-result` and `--previous-result`. Use Python >=3.10, absolute paths and new output/cache directories. The comparator accepts five ordered result paths plus `--prompts`, `--baseline-v1-result`, `--source-pins`, and `--output`. The exact orchestration script archives this run's fixed deadline; it is not a future-date launcher. CPU mock/state-machine checks are included and explicitly labeled CPU-only.

The follow-up completed all five engines. Its [frozen comparator report](graph-perf-admission-controlled/comparison.json) has `integrity_passed=true`, `passed=false`. The separate untimed diagnostic was held automatically; there is no native runtime graph-metric or hardware replay trace from this follow-up. The expected graph capture entries and dispatch settings were checked on every worker.

| Arm | B8 median seconds | B40 median seconds |
|---|---:|---:|
| Baseline | 2.250934 | 6.196644 |
| Default-A | 2.234938 | 6.115664 |
| FP16-A | 2.206171 | 6.083021 |
| FP16-B | 2.100160 | 5.637360 |
| Default-B | 2.152193 | 5.794458 |

All 70 measured calls are retained. Within each arm/workload all seven outputs match exactly; both default engines match baseline, and the two FP16 engines match each other. All caches and recorded tuner statistics remain unchanged. Per-arm relative MADs are below 0.73%, but unchanged-default medians drift by -3.70% at B8 and -5.25% at B40, failing the 2% control gate. Identical FP16 engines differ by -4.81% and -7.33% respectively. The cause of these process-to-process differences was not established; no clock, thermal, JIT or kernel-dispatch attribution is made.

FP16-A B40 is only 0.534% below Default-A and 4.980% above Default-B. FP16-B is 2.711% below Default-B. Only one of four required B40 comparisons meets the fixed 3% criterion. Therefore this experiment does not substantiate an FP16 E2E speedup or upstream readiness. The local gates were not relaxed, no observations were discarded, and no retry followed the failure. This does not establish that FP16 can never improve another workload.

`comparison.json` is the exact generated CPU comparator report. `results.json` is the independently audited public projection of all five runtime arms; it omits raw token arrays and private identifiers while retaining every measured time and output hash. The separate CPU projector [analysis/project_admission_results.py](analysis/project_admission_results.py) does not issue the performance verdict. The final independent audit checks all 70 measured and 30 warmup arrays, all 20 actual per-rank cache files, pinned Git source/API bytes and the raw cache logs; rerunning the frozen comparator reproduces its report byte-for-byte. Reproduce the selected public fields from a regenerated full evidence tree with:

```bash
python3 analysis/project_admission_results.py \
  --notes /path/to/regenerated-full-tree \
  --arms baseline default-a fp16-a fp16-b default-b \
  --source-repo /path/to/vllm-checkout-containing-both-pins \
  --output /path/to/new-public-projection.json
```

This requires the full raw result/token/log tree, `staging/source-pins.json`, the immutable graph prompts, frozen harnesses, and mirrored cache files under their recorded relative `/work` layout. The public projection omits raw outputs and cannot by itself rerun the literal-token checks.

## Artifact scope

Archived harnesses and pinned source copies are exact bytes; their hashes are recorded separately. Runtime result JSON files are deliberately selected projections of the full local evidence. The admission-controlled comparator report is an exact copy of its generated derived report, as labeled above. They retain measurements, aggregate correctness and provenance checks, while omitting raw model responses, prompt text, machine identifiers, private runtime locations and identity-bearing cache filenames. They are not byte-identical raw runtime files.

The CPU-only scripts under `analysis/` independently rescore and project a regenerated full evidence tree; they are separate from the GPU-executed harnesses. For example:

```bash
python3 /path/to/supplement/analysis/project_public_results.py \
  --notes /path/to/regenerated-full-tree \
  --output /path/to/regenerated-full-tree/gsm8k/public-results-projection.json
```

This requires the three arms' full `result.json` and `predictions.jsonl`, candidate cache-log verification files, the dataset manifest and pinned train/test tar archive, mirrored FlashInfer cache files, and operator source provenance. It reads these local inputs and writes the selected public fields; the public projection alone does not contain the raw responses needed to repeat scoring.

Earlier candidate results remain separately archived in the [historical supplement](https://github.com/wolfcomos/vllm/tree/2c5dfae05cb32d8e8329baa8d2be43aa2d4d4045). Its new BF16 split policy regressed the prior GSM8K run and was removed. Historical BF16/fallback timings do not describe this candidate and are not combined with this experiment.

Xin Li and Suyog Gupta are credited in the implementation commits. AI assistance was used for implementation, review and validation.
