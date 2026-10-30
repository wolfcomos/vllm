"""Pinned pretrained MiniMax M3 GSM8K regression; external validation harness."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

from gsm8k_cache_seed import PROTOCOL, cache_identity, cache_manifest, file_digest

COMMITS = {
    "baseline": "a20280f30e3169d33a7e21d1d117e9719bace00b",
    "candidate": "25acf1953bf42857aec52d5f90cb7eac174d3189",
}
SOURCE_FILES = (
    "config/attention.py", "models/minimax_m3/common/ops/sparse_attn.py",
    "models/minimax_m3/common/sparse_attention.py", "models/minimax_m3/nvidia/model.py",
    "models/minimax_m3/nvidia/sparse_attention_msa.py",
)
API_FILES = (
    "entrypoints/llm.py", "sampling_params.py", "v1/engine/llm_engine.py",
    "v1/engine/core_client.py", "v1/engine/core.py", "v1/core/sched/scheduler.py",
    "v1/engine/output_processor.py", "v1/metrics/stats.py", "outputs.py",
    "model_executor/warmup/kernel_warmup.py", "model_executor/warmup/flashinfer_autotune_cache.py",
)


def prepare_dataset(directory, manifest):
    """Make the repository loader consume the supplied immutable local files."""
    import tempfile

    for name, expected in manifest["files"].items():
        source = directory / name
        assert file_digest(source) == expected, ("Dataset changed", source)
        target = Path(tempfile.gettempdir()) / name
        if target.exists():
            assert file_digest(target) == expected, ("Different cached dataset", target)
        else:
            temporary = target.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_bytes(source.read_bytes())
            temporary.replace(target)
    return Path(tempfile.gettempdir())


def install_dispatch_recorder(worker):
    import inspect
    import torch
    from vllm.models.minimax_m3.common.ops import sparse_attn
    from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseImpl

    counter = {"launches": 0, "variants": {}, "samples": []}
    worker.gsm8k_dispatch_counter = counter
    original = sparse_attn._gqa_sparse_decode_kernel

    class Recorder:
        def __getitem__(self, grid):
            run = original[grid]

            def launch(*args, **kwargs):
                result = run(*args, **kwargs)
                counter["launches"] += 1
                metadata = {
                    "grid": list(grid),
                    "q_shape": list(args[0].shape),
                    "q_dtype": str(args[0].dtype),
                    "q_device": str(args[0].device),
                    "kv_dtype": str(args[1].dtype),
                    "total_q": int(args[9]),
                    "num_kv_heads": int(grid[1]),
                    "topk": int(args[12]),
                    "USE_FP16_DOT": kwargs.get("USE_FP16_DOT", kwargs.get("FP16_DOT", False)),
                    "USE_FP8": kwargs["USE_FP8"],
                    "KV_SCALE_MODE": kwargs["KV_SCALE_MODE"],
                    "NUM_TOPK_CHUNKS": kwargs["NUM_TOPK_CHUNKS"],
                }
                key = json.dumps(metadata, sort_keys=True)
                counter["variants"][key] = counter["variants"].get(key, 0) + 1
                if len(counter["samples"]) < 3:
                    counter["samples"].append(metadata)
                return result

            return launch

    sparse_attn._gqa_sparse_decode_kernel = Recorder()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    implementations = []
    for name, module in worker.model_runner.model.named_modules():
        impl = getattr(module, "impl", None)
        if isinstance(impl, MiniMaxM3SparseImpl):
            implementations.append({"module": name, "class": type(impl).__name__,
                                    "fp16_dot": getattr(impl, "fp16_dot", False),
                                    "use_cutlass_decode": getattr(impl, "use_cutlass_decode", False),
                                    "kv_cache_dtype": impl.kv_cache_dtype})
    assert implementations and not any(row["use_cutlass_decode"] for row in implementations)
    return {
        "rank": worker.rank,
        "source": inspect.getsourcefile(sparse_attn),
        "source_sha256": hashlib.sha256(Path(sparse_attn.__file__).read_bytes()).hexdigest(),
        "gpu": str(props), "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
        "capability": [props.major, props.minor], "num_sms": props.multi_processor_count,
        "implementations": implementations,
        "attention_config": str(worker.vllm_config.attention_config),
    }


def read_dispatch_recorder(worker):
    return {"rank": worker.rank, **worker.gsm8k_dispatch_counter}



def read_tuner_statistics(worker):
    from flashinfer.autotuner import AutoTuner
    stats = AutoTuner.get().stats
    return {"rank": worker.rank, "statistics_identity": id(stats), "cache_misses": stats.cache_misses,
            "tuned_op_total_configs": dict(stats.tuned_op_total_configs),
            "tuned_op_successful_configs": dict(stats.tuned_op_successful_configs),
            "failed_profiling_count": {k: len(v) for k, v in stats.failed_profiling_count.items()},
            "failed_tactics": {k: len(v) for k, v in stats.failed_tactics.items()}}


def assert_no_tuning(statistics):
    assert sorted(s["rank"] for s in statistics) == list(range(4))
    for stats in statistics:
        assert not any(stats["tuned_op_successful_configs"].values()), stats
        assert not any(stats["failed_profiling_count"].values()), stats
        assert not any(stats["failed_tactics"].values()), stats


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str))
    temp.replace(path)


def full_dispatch_delta(before, after, workers, mode):
    summary = []
    for initial, final, worker in zip(before, after, workers, strict=True):
        variants = {k: n - initial['variants'].get(k, 0)
                    for k, n in final['variants'].items()
                    if n > initial['variants'].get(k, 0)}
        changed_split_launches = 0
        for key, count in variants.items():
            m = json.loads(key)
            assert m['USE_FP8'] and m['q_dtype'] == 'torch.bfloat16'
            assert m['USE_FP16_DOT'] == (mode == 'fp16')
            denom = max(1, m['total_q'] * m['num_kv_heads'])
            target = max(1, min(m['topk'], 256 // denom))
            old = 1 << (target.bit_length() - 1)
            target = max(1, min(m['topk'], 2 * worker['num_sms'] // denom))
            blocks = (m['topk'] + target - 1) // target
            new = (m['topk'] + blocks - 1) // blocks
            assert m['NUM_TOPK_CHUNKS'] == (new if mode == 'fp16' else old)
            changed_split_launches += count * (old != new)
        assert changed_split_launches > 0, 'Full evaluation never exercised shapes that distinguish original and new split policies'
        summary.append({'rank': final['rank'], 'launches': final['launches'] - initial['launches'],
                        'split_sensitive_shape_launches': changed_split_launches,
                        'expected_policy': 'sm_count' if mode == 'fp16' else 'original_power_of_two',
                        'max_total_q': max(json.loads(k)['total_q'] for k in variants),
                        'variants': variants})
    return summary


def main():
    if not __debug__:
        raise RuntimeError("Assertions must remain enabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "bf16", "fp16"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=1319)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--source-pins", type=Path, default=Path("/work/source-pins.json"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("/work/gsm8k/data"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path(__file__).with_name("dataset-manifest.json"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "result.json").exists(), "Do not overwrite an earlier attempt"
    assert not args.cache_dir.exists(), "Each arm requires a fresh cache directory"
    args.cache_dir.mkdir(parents=True)
    dataset_pin = json.loads(args.dataset_manifest.read_text())
    dataset_cache = prepare_dataset(args.dataset_dir, dataset_pin)
    witness = json.loads(args.baseline_result.read_text()) if args.baseline_result else None
    if args.mode == "baseline":
        assert witness is None
        os.environ.pop("GSM8K_BASELINE_RESULT", None)
    else:
        assert witness and witness["status"] == "complete" and witness["mode"] == "baseline"
        assert witness["protocol"] == PROTOCOL
        os.environ["GSM8K_BASELINE_RESULT"] = str(args.baseline_result)
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"] = str(args.cache_dir)
    os.environ["GSM8K_RESULT_DIR"] = str(args.output)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    from vllm import LLM, SamplingParams
    import torch
    import vllm
    from tests.evals.gsm8k.gsm8k_eval import _build_gsm8k_prompts, _score_gsm8k, get_answer_value

    attention_config = {"minimax_m3_msa_decode_backend": "triton", "indexer_kv_dtype": "fp8"}
    if args.mode != "baseline":
        attention_config["minimax_m3_msa_decode_fp16_dot"] = args.mode == "fp16"
    settings = dict(
        model="/work/models/MiniMax-M3-NVFP4", language_model_only=True,
        revision="901464083161bf8612a29ff7ad29914cd4ab4a85",
        tokenizer_revision="901464083161bf8612a29ff7ad29914cd4ab4a85",
        tensor_parallel_size=4, distributed_executor_backend="mp",
        dtype="bfloat16", kv_cache_dtype="fp8", block_size=128,
        seed=42, enforce_eager=True, max_model_len=8192,
        max_num_seqs=40, max_num_batched_tokens=4096,
        kv_cache_memory_bytes=8 * 2**30, enable_chunked_prefill=True,
        enable_prefix_caching=False, generation_config="vllm",
        attention_config=attention_config, disable_log_stats=True,
        worker_extension_cls="gsm8k_cache_seed.FlashInferCacheSeed",
    )
    params = dict(temperature=0.0, seed=42, max_tokens=1024, stop=None)
    result = {
        "mode": args.mode, "status": "starting", "settings": settings,
        "sampling": params, "chat_template_kwargs": {"thinking_mode": "disabled"},
        "protocol": PROTOCOL,
        "prompt_protocol": "repository GSM8K first5 train fewshots, complete1319 test, chat, nonthinking",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": vllm.__file__, "torch": torch.__version__, "cuda": torch.version.cuda,
        "packages": {p: importlib.metadata.version(p) for p in ["vllm", "torch", "triton", "nvidia-cutlass-dsl", "flashinfer-python", "transformers", "tokenizers"]},
        "criterion": {"baseline_sanity_floor": 0.80, "max_accuracy_drop": 0.005, "max_invalid_rate": 0.01, "max_truncation_rate": 0.01, "default_exact_output_tokens": True},
        "autotune_cache_before": [], "autotune_cache_root": str(args.cache_dir),
        "harness_hashes": {name: file_digest(Path(__file__).with_name(name))
                           for name in ("run_gsm8k.py", "gsm8k_cache_seed.py", "compare_gsm8k.py", "check_cache_log.py", "PROTOCOL.md")},
        "dataset_manifest_sha256": file_digest(args.dataset_manifest),
        "source_pins_sha256": file_digest(args.source_pins),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR")},
    }
    package_root = Path(vllm.__file__).parent
    result["source_hashes"] = {f: file_digest(package_root / f) for f in SOURCE_FILES}
    result["api_hashes"] = {f: file_digest(package_root / f) for f in API_FILES}
    import inspect
    result["repository_eval_sha256"] = file_digest(inspect.getsourcefile(_build_gsm8k_prompts))
    pins = json.loads(args.source_pins.read_text())
    assert all(pins[key]["commit"] == value for key, value in COMMITS.items())
    pin = pins['baseline' if args.mode == 'baseline' else 'candidate']
    assert result['source_hashes'] == pin['source_hashes'], 'Source differs from pinned revision'
    result['source_commit'] = pin['commit']
    if witness:
        result["baseline_result_sha256"] = file_digest(args.baseline_result)
        for key in ("torch", "cuda", "packages", "api_hashes", "repository_eval_sha256", "harness_hashes", "source_pins_sha256", "dataset_manifest_sha256"):
            assert result[key] == witness[key], ("Baseline pairing changed", key)
        normalized = json.loads(json.dumps(settings))
        normalized["attention_config"].pop("minimax_m3_msa_decode_fp16_dot", None)
        assert normalized == witness["settings"]
    out = args.output / "result.json"
    write_json(out, result)
    llm = None
    try:
        start = time.perf_counter()
        llm = LLM(**settings)
        result["startup_seconds"] = time.perf_counter() - start
        result["workers"] = llm.collective_rpc(install_dispatch_recorder, timeout=60)
        assert sorted(w["rank"] for w in result["workers"]) == list(range(4))
        assert all(w["source_sha256"] == pin["source_hashes"]["models/minimax_m3/common/ops/sparse_attn.py"] for w in result["workers"])
        for worker in result["workers"]:
            assert worker["capability"][0] == 10 and "GB200" in worker["gpu"]
            assert all(impl["fp16_dot"] == (args.mode == "fp16") and impl["kv_cache_dtype"] == "fp8"
                       for impl in worker["implementations"])
        result["autotuner_statistics_after_startup"] = llm.collective_rpc(read_tuner_statistics, timeout=60)
        if args.mode != 'baseline':
            assert_no_tuning(result["autotuner_statistics_after_startup"])
        result["autotune_cache_after_startup"] = cache_manifest(args.output)
        result["validation_cache_seed"] = [json.loads(p.read_text())
            for p in sorted(args.output.glob("cache-rank*.json"))]
        if args.mode != 'baseline':
            assert cache_identity(result["autotune_cache_after_startup"]) == cache_identity(witness["autotune_cache_after_generation"])
            assert all(row["seeded_from"] for row in result["validation_cache_seed"])
        prompts, labels = _build_gsm8k_prompts(args.questions, 5)
        assert len(prompts) == args.questions == 1319, 'Full GSM8K test split required'
        from vllm.assets.base import VLLM_S3_BUCKET_URL
        result['dataset'] = {'train_examples': 7473, 'test_examples': len(prompts),
            'train_fewshot_indices': list(range(5)),
            'files': {f: {'url': f'{VLLM_S3_BUCKET_URL}/ci-datasets/gsm8k/{f}',
                          'sha256': file_digest(dataset_cache / f)}
                      for f in ['train.jsonl', 'test.jsonl']}}
        train_rows = [json.loads(s) for s in (dataset_cache / 'train.jsonl').read_text().splitlines()]
        assert len(train_rows) == 7473
        result['dataset']['fewshot_examples'] = train_rows[:5]
        tokenizer = llm.get_tokenizer()
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        rendered = tokenizer.apply_chat_template(conversations[0], tokenize=False, add_generation_prompt=True, thinking_mode="disabled")
        result["first_rendered_prompt"] = rendered
        assert rendered.endswith("</mm:think>"), repr(rendered[-100:])
        result["prompts_sha256"] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
        result["labels_sha256"] = hashlib.sha256(json.dumps(labels).encode()).hexdigest()
        assert {f: d['sha256'] for f, d in result['dataset']['files'].items()} == dataset_pin['files']
        assert result['prompts_sha256'] == dataset_pin['prompts_sha256']
        assert result['labels_sha256'] == dataset_pin['labels_sha256']
        sampling = SamplingParams(**params)
        start = time.perf_counter()
        smoke = llm.chat(conversations[:8], sampling, chat_template_kwargs={"thinking_mode": "disabled"}, use_tqdm=False)
        result["smoke_seconds"] = time.perf_counter() - start
        result["smoke"] = [{"text": o.outputs[0].text, "tokens": len(o.outputs[0].token_ids), "finish_reason": o.outputs[0].finish_reason, "answer": get_answer_value(o.outputs[0].text), "label": labels[i]} for i, o in enumerate(smoke)]
        result["smoke_dispatch"] = llm.collective_rpc(read_dispatch_recorder, timeout=60)
        assert all(w["launches"] > 0 for w in result["smoke_dispatch"])
        for w in result["smoke_dispatch"]:
            for key in w["variants"]:
                meta = json.loads(key)
                assert meta["USE_FP8"] and meta["q_dtype"] == "torch.bfloat16"
                assert meta["USE_FP16_DOT"] == (args.mode == "fp16"), meta
        assert all(s["finish_reason"] != "length" for s in result["smoke"]), result["smoke"]
        result["status"] = "smoke_passed"
        write_json(out, result)
        print(json.dumps({"mode": args.mode, "status": "smoke_passed", "startup_seconds": result["startup_seconds"], "smoke_seconds": result["smoke_seconds"], "smoke_correct": sum(s["answer"] == s["label"] for s in result["smoke"])}), flush=True)
        if args.smoke_only:
            return
        start = time.perf_counter()
        outputs = llm.chat(conversations, sampling, chat_template_kwargs={"thinking_mode": "disabled"}, use_tqdm=True)
        elapsed = time.perf_counter() - start
        assert len(outputs) == len(prompts)
        states, lengths = [], []
        with (args.output / "predictions.jsonl").open("w") as f:
            for i, output in enumerate(outputs):
                completion = output.outputs[0]
                pred = get_answer_value(completion.text)
                row = {"id": i, "prompt": prompts[i], "prompt_token_ids": list(output.prompt_token_ids), "label": labels[i], "prediction": pred, "correct": pred == labels[i], "text": completion.text, "token_ids": list(completion.token_ids), "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason}
                f.write(json.dumps(row) + "\n")
                states.append(completion.text)
                lengths.append(len(completion.token_ids))
        result["metrics"] = _score_gsm8k(states, lengths, labels, 5, 1024, elapsed)
        result["metrics"]["truncation_rate"] = sum(o.outputs[0].finish_reason == "length" for o in outputs) / len(outputs)
        result["dispatch"] = llm.collective_rpc(read_dispatch_recorder, timeout=60)
        result['full_dispatch'] = full_dispatch_delta(result['smoke_dispatch'], result['dispatch'], result['workers'], args.mode)
        result["autotuner_statistics_after_generation"] = llm.collective_rpc(read_tuner_statistics, timeout=60)
        for initial, final in zip(result["autotuner_statistics_after_startup"], result["autotuner_statistics_after_generation"], strict=True):
            assert final["statistics_identity"] == initial["statistics_identity"], "Tuner statistics were reset"
            for key in ["cache_misses", "tuned_op_total_configs", "tuned_op_successful_configs", "failed_profiling_count", "failed_tactics"]:
                assert final[key] == initial[key], ('Unexpected generation-time tuning', key, initial, final)
        if args.mode != 'baseline':
            assert_no_tuning(result["autotuner_statistics_after_generation"])
        result["autotune_cache_after_generation"] = cache_manifest(args.output)
        if args.mode != 'baseline':
            assert cache_identity(result["autotune_cache_after_generation"]) == cache_identity(witness["autotune_cache_after_generation"])
        assert result['autotune_cache_after_generation'] == result['autotune_cache_after_startup'], 'FlashInfer cache changed during generation'
        result["status"] = "complete"
        result["predictions_sha256"] = file_digest(args.output / "predictions.jsonl")
        result["finished_generation_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out, result)
        print(json.dumps({"mode": args.mode, "status": result["status"], "metrics": result["metrics"]}), flush=True)
    except Exception:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        write_json(out, result)
        raise
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown(timeout=60)
                result["shutdown_completed"] = True
            except Exception:
                result["shutdown_completed"] = False
                result["shutdown_traceback"] = traceback.format_exc()
                result["status"] = "failed"
                raise
            finally:
                result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                write_json(out, result)


if __name__ == "__main__":
    main()
