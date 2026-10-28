"""Fixed-work pretrained MiniMax M3 CUDA Graph validation, outside production."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import re
import time
import traceback

REVISION = "901464083161bf8612a29ff7ad29914cd4ab4a85"
SOURCE_FILES = ["config/attention.py", "models/minimax_m3/common/ops/sparse_attn.py",
                "models/minimax_m3/common/sparse_attention.py",
                "models/minimax_m3/nvidia/model.py",
                "models/minimax_m3/nvidia/sparse_attention_msa.py"]
BATCHES = (8, 40)
INPUT_LEN = 2048
OUTPUT_LEN = 256
TUNING_ACTIVITY_FIELDS = (
    "tuned_op_successful_configs", "failed_profiling_count", "failed_tactics"
)
GRAPH_ROW = re.compile(
    r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(FULL|PIECEWISE|NONE)\s*\|\s*(\d+)\s*\|"
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def prepare_prompts(path):
    rng = random.Random(42)
    prompts = [[rng.randrange(10000) for _ in range(INPUT_LEN)] for _ in range(40)]
    data = {"seed": 42, "input_len": INPUT_LEN, "prompts": prompts,
            "prompts_sha256": digest(prompts)}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == data, "Existing prompt file differs"
    else:
        write_json(path, data)
    print(json.dumps({"path": str(path), "prompts_sha256": data["prompts_sha256"]}))


def canonical_settings(settings):
    settings = json.loads(json.dumps(settings))
    settings["attention_config"].pop("minimax_m3_msa_decode_fp16_dot", None)
    settings.pop("disable_log_stats", None)
    settings.pop("cudagraph_metrics", None)
    settings.pop("profiler_config", None)
    return settings


def read_tuner_statistics(worker):
    from flashinfer.autotuner import AutoTuner

    stats = AutoTuner.get().stats
    return {"rank": worker.rank, "statistics": {
        "statistics_object_id": id(stats),
        "cache_misses": int(stats.cache_misses),
        "tuned_op_total_configs": dict(stats.tuned_op_total_configs),
        "tuned_op_successful_configs": dict(stats.tuned_op_successful_configs),
        "failed_profiling_count": {op: len(keys) for op, keys in stats.failed_profiling_count.items()},
        "failed_tactics": {op: len(tactics) for op, tactics in stats.failed_tactics.items()},
    }}


def validate_tuner_startup(workers, seeded):
    assert sorted(worker["rank"] for worker in workers) == [0, 1, 2, 3]
    for worker in workers:
        stats = worker["flashinfer_statistics"]
        if seeded:
            for field in TUNING_ACTIVITY_FIELDS:
                assert all(value == 0 for value in stats[field].values()), (
                    "Seeded worker profiled FlashInfer tactics during startup", worker["rank"], field, stats[field]
                )


def validate_tuner_after_generation(workers, snapshots):
    assert sorted(row["rank"] for row in snapshots) == [0, 1, 2, 3]
    final = {row["rank"]: row["statistics"] for row in snapshots}
    for worker in workers:
        before = worker["flashinfer_statistics"]
        after = final[worker["rank"]]
        assert before["statistics_object_id"] == after["statistics_object_id"], (
            "FlashInfer statistics object changed; snapshot deltas are invalid", worker["rank"]
        )
        for field in TUNING_ACTIVITY_FIELDS:
            assert before[field] == after[field], (
                "FlashInfer tracked tuning activity during generation", worker["rank"], field, before[field], after[field]
            )


def inspect_worker(worker):
    import inspect
    import torch
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.models.minimax_m3.common.ops import sparse_attn

    runner = worker.model_runner
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    entries = []
    for wrapper in CUDAGraphWrapper._all_instances:
        for descriptor, entry in wrapper.concrete_cudagraph_entries.items():
            if entry.cudagraph is not None:
                entries.append({"mode": str(wrapper.runtime_mode),
                                "num_tokens": descriptor.num_tokens,
                                "descriptor": repr(descriptor)})
    dispatch = []
    for batch in BATCHES:
        mode, descriptor = runner.cudagraph_dispatcher.dispatch(batch, uniform_decode=True)
        dispatch.append({"batch": batch, "mode": str(mode),
                         "padded_tokens": descriptor.num_tokens})
    return {"rank": worker.rank, "gpu": str(props), "num_sms": props.multi_processor_count,
            "source": inspect.getsourcefile(sparse_attn),
            "source_sha256": file_digest(sparse_attn.__file__),
            "attention_config": str(worker.vllm_config.attention_config),
            "compilation_config": str(worker.vllm_config.compilation_config),
            "resolved_graph_mode": str(runner.cudagraph_dispatcher.cudagraph_mode),
            "captured_entries": entries, "dispatch": dispatch,
            "flashinfer_statistics": read_tuner_statistics(worker)["statistics"]}


class GraphLogCapture(logging.Handler):
    """Read native graph metrics in diagnostic processes only."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def emit(self, record):
        for unpadded, padded, paddings, mode, count in GRAPH_ROW.findall(record.getMessage()):
            self.rows.append({"unpadded": int(unpadded), "padded": int(padded),
                              "paddings": int(paddings), "mode": mode, "count": int(count)})


def cache_manifest(output):
    records = [json.loads(path.read_text()) for path in sorted(output.glob("cache-rank*.json"))]
    assert sorted(row["rank"] for row in records) == [0, 1, 2, 3]
    files = sorted({row["actual_file"] for row in records})
    assert len(files) == 4
    return [{"path": path, "sha256": file_digest(path)} for path in files]


def verify_cache_matches_baseline(current, baseline):
    expected = {Path(row["path"]).name: row["sha256"] for row in baseline["cache_after_generation"]}
    actual = {Path(row["path"]).name: row["sha256"] for row in current}
    assert actual == expected, "FlashInfer tactics differ from fresh graph baseline"


def run(args):
    assert args.output.is_absolute() and args.model.is_absolute()
    assert args.cache_dir.is_absolute()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "result.json").exists(), "Use a fresh result directory"
    if args.repeats == 3:
        assert args.repeat_budget_note, "Document the 3-repeat budget decision before timing"
    baseline = None
    if args.baseline_result:
        baseline = json.loads(args.baseline_result.read_text())
        assert baseline["status"] == "complete" and baseline["mode"] == "baseline"
        assert baseline["model_revision"] == REVISION
        if args.phase == "timing" and baseline["phase"] == "timing":
            assert baseline["repeats"] == args.repeats, "Use one repeat count for all timing arms"
    assert args.mode == "baseline" or baseline is not None
    for key, value in {"VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
                       "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_BATCH_INVARIANT": "0",
                       "VLLM_USE_V2_MODEL_RUNNER": "0",
                       "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": str(args.cache_dir),
                       "M3_GRAPH_OUTPUT": str(args.output)}.items():
        os.environ[key] = value
    if args.baseline_result:
        os.environ["M3_GRAPH_BASELINE_RESULT"] = str(args.baseline_result.resolve())
    else:
        os.environ.pop("M3_GRAPH_BASELINE_RESULT", None)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import torch
    import vllm

    prompt_data = json.loads(args.prompts.read_text())
    prompt_ids = prompt_data["prompts"]
    assert prompt_data["seed"] == 42 and prompt_data["input_len"] == INPUT_LEN
    assert len(prompt_ids) == 40 and all(len(p) == INPUT_LEN for p in prompt_ids)
    assert all(type(token) is int and 0 <= token < 10000 for p in prompt_ids for token in p)
    assert prompt_data["prompts_sha256"] == digest(prompt_ids)
    if baseline:
        assert baseline["prompts_sha256"] == digest(prompt_ids)

    attention = {"minimax_m3_msa_decode_backend": "triton", "indexer_kv_dtype": "fp8"}
    if args.mode == "fp16":
        attention["minimax_m3_msa_decode_fp16_dot"] = True
    settings = dict(model=str(args.model), language_model_only=True, revision=REVISION,
                    tokenizer_revision=REVISION, tensor_parallel_size=4,
                    distributed_executor_backend="mp", dtype="bfloat16", kv_cache_dtype="fp8",
                    block_size=128, seed=42, enforce_eager=False, max_model_len=8192,
                    max_num_seqs=40, max_num_batched_tokens=4096,
                    kv_cache_memory_bytes=8 * 2**30, enable_chunked_prefill=True,
                    enable_prefix_caching=False, generation_config="vllm",
                    attention_config=attention, disable_log_stats=args.phase == "timing",
                    cudagraph_metrics=args.phase == "diagnostic",
                    kernel_config={"enable_flashinfer_autotune": True},
                    compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                        "cudagraph_capture_sizes": [8, 40],
                                        "cudagraph_num_of_warmups": 1},
                    worker_extension_cls="graph_perf_cache_seed.GraphPerfCacheSeed")
    if args.profile_cuda:
        assert args.phase == "diagnostic", "Profiling is forbidden in timing processes"
        settings["profiler_config"] = {"profiler": "cuda", "delay_iterations": 64,
                                       "max_iterations": 16}
    if baseline:
        assert canonical_settings(settings) == canonical_settings(baseline["settings"])
    sampling = {"temperature": 0.0, "seed": 42, "ignore_eos": True,
                "max_tokens": OUTPUT_LEN, "detokenize": False, "stop": None}
    pins = json.loads(args.source_pins.read_text())
    pin = pins["baseline" if args.mode == "baseline" else "candidate"]
    package = Path(vllm.__file__).parent
    hashes = {name: file_digest(package / name) for name in SOURCE_FILES}
    assert hashes == pin["source_hashes"], "Source differs from pinned revision"
    result = {"status": "starting", "phase": args.phase, "mode": args.mode,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "model_revision": REVISION, "source_commit": pin["commit"], "source_hashes": hashes,
              "settings": settings, "sampling": sampling, "prompts_sha256": digest(prompt_ids),
              "input_len": INPUT_LEN, "output_len": OUTPUT_LEN, "batches": list(BATCHES),
              "warmups": 2, "repeats": args.repeats, "repeat_budget_note": args.repeat_budget_note,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "packages": {name: importlib.metadata.version(name) for name in
                           ["vllm", "triton", "flashinfer-python", "nvidia-cutlass-dsl"]},
              "harness_hashes": {name: file_digest(Path(__file__).parent / name) for name in
                                 ["run_graph_perf.py", "graph_perf_cache_seed.py"]},
              "trials": [], "diagnostics": [],
              "profiling_enabled": args.profile_cuda, "dispatch_recorder_installed": False}
    if baseline:
        for key in ("torch", "cuda", "packages", "harness_hashes"):
            assert result[key] == baseline[key], f"Runtime or harness differs: {key}"
    result_path = args.output / "result.json"
    write_json(result_path, result)
    llm = None
    graph_logs = GraphLogCapture() if args.phase == "diagnostic" else None
    metrics_logger = logging.getLogger("vllm.v1.metrics.loggers")
    if graph_logs:
        metrics_logger.addHandler(graph_logs)
    try:
        started = time.perf_counter()
        llm = LLM(**settings)
        result["startup_seconds"] = time.perf_counter() - started
        result["workers"] = llm.collective_rpc(inspect_worker, timeout=60)
        assert sorted(w["rank"] for w in result["workers"]) == [0, 1, 2, 3]
        for worker in result["workers"]:
            assert worker["source_sha256"] == hashes["models/minimax_m3/common/ops/sparse_attn.py"]
            assert worker["resolved_graph_mode"] == "FULL_DECODE_ONLY"
            assert all(d["mode"] == "FULL" and d["padded_tokens"] == d["batch"]
                       for d in worker["dispatch"])
            assert all(any(e["mode"] == "FULL" and e["num_tokens"] == batch
                           for e in worker["captured_entries"]) for batch in BATCHES)
        if baseline:
            assert [w["gpu"] for w in result["workers"]] == [w["gpu"] for w in baseline["workers"]]
        validate_tuner_startup(result["workers"], seeded=baseline is not None)
        result["cache_after_startup"] = cache_manifest(args.output)
        if baseline:
            verify_cache_matches_baseline(result["cache_after_startup"], baseline)
        params = SamplingParams(**sampling)
        for batch in BATCHES:
            prompts = [TokensPrompt(prompt_token_ids=p) for p in prompt_ids[:batch]]
            warmup_seconds = []
            for _ in range(2):
                started = time.perf_counter()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                warmup_seconds.append(time.perf_counter() - started)
                assert len(outputs) == batch
                assert all(len(o.outputs[0].token_ids) == OUTPUT_LEN for o in outputs)
            write_json(args.output / f"warmup-b{batch}.json", {"seconds": warmup_seconds})
            if args.phase == "diagnostic":
                llm.llm_engine.do_log_stats()
                offset = len(graph_logs.rows)
                if args.profile_cuda:
                    llm.start_profile()
                try:
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                finally:
                    if args.profile_cuda:
                        llm.stop_profile()
                assert len(outputs) == batch
                assert all(len(o.outputs[0].token_ids) == OUTPUT_LEN for o in outputs)
                llm.llm_engine.do_log_stats()
                rows = graph_logs.rows[offset:]
                full_steps = sum(row["count"] for row in rows if row["mode"] == "FULL"
                                 and row["unpadded"] == batch and row["padded"] == batch)
                assert full_steps >= 128, ("Too few unpadded full-graph decode steps", batch, rows)
                result["diagnostics"].append({"batch": batch, "native_graph_metric_rows": rows,
                                              "unpadded_full_graph_steps": full_steps})
            else:
                for repeat in range(args.repeats):
                    started = time.perf_counter()
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                    seconds = time.perf_counter() - started
                    assert len(outputs) == batch
                    token_ids = [list(o.outputs[0].token_ids) for o in outputs]
                    assert all(len(ids) == OUTPUT_LEN for ids in token_ids)
                    assert all(list(o.prompt_token_ids) == prompt_ids[i] for i, o in enumerate(outputs))
                    row = {"batch": batch, "repeat": repeat, "seed": 42, "seconds": seconds,
                           "output_tokens": batch * OUTPUT_LEN,
                           "output_tokens_per_second": batch * OUTPUT_LEN / seconds,
                           "output_token_ids_sha256": digest(token_ids)}
                    result["trials"].append(row)
                    write_json(args.output / f"tokens-b{batch}-r{repeat}.json", token_ids)
                    write_json(result_path, result)
                    print(json.dumps({"mode": args.mode, **row}), flush=True)
            write_json(result_path, result)
        result["tuner_after_generation"] = llm.collective_rpc(read_tuner_statistics, timeout=60)
        validate_tuner_after_generation(result["workers"], result["tuner_after_generation"])
        result["cache_after_generation"] = cache_manifest(args.output)
        assert result["cache_after_generation"] == result["cache_after_startup"], "Tactics changed during generation"
        if baseline:
            verify_cache_matches_baseline(result["cache_after_generation"], baseline)
        if args.phase == "timing":
            assert Counter(row["batch"] for row in result["trials"]) == {8: args.repeats, 40: args.repeats}
            result["measured_seconds"] = sum(row["seconds"] for row in result["trials"])
        result["status"] = "complete"
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(result_path, result)
    except Exception:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        write_json(result_path, result)
        raise
    finally:
        if graph_logs:
            metrics_logger.removeHandler(graph_logs)
        if llm is not None:
            llm.llm_engine.engine_core.shutdown(timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--prompts", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--mode", choices=["baseline", "default", "fp16", "default-repeat"], required=True)
    execute.add_argument("--phase", choices=["timing", "diagnostic"], required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--model", type=Path, required=True)
    execute.add_argument("--prompts", type=Path, required=True)
    execute.add_argument("--source-pins", type=Path, required=True)
    execute.add_argument("--cache-dir", type=Path, required=True)
    execute.add_argument("--baseline-result", type=Path)
    execute.add_argument("--repeats", type=int, choices=[3, 5], default=5)
    execute.add_argument("--repeat-budget-note")
    execute.add_argument("--profile-cuda", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_prompts(args.prompts)
    else:
        run(args)


if __name__ == "__main__":
    main()
