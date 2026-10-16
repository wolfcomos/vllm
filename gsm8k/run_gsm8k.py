"""Pinned pretrained MiniMax M3 GSM8K regression; external validation harness."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback


def install_dispatch_recorder(worker):
    import inspect
    import torch
    from vllm.models.minimax_m3.common.ops import sparse_attn

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
                    "FP16_DOT": kwargs.get("FP16_DOT", False),
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
    return {
        "rank": worker.rank,
        "source": inspect.getsourcefile(sparse_attn),
        "source_sha256": hashlib.sha256(Path(sparse_attn.__file__).read_bytes()).hexdigest(),
        "gpu": str(torch.cuda.get_device_properties(torch.cuda.current_device())),
        "num_sms": torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
        "attention_config": str(worker.vllm_config.attention_config),
    }


def read_dispatch_recorder(worker):
    return {"rank": worker.rank, **worker.gsm8k_dispatch_counter}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str))
    temp.replace(path)


def cache_manifest():
    root = Path('/work/cache/flashinfer')
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


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
            assert m['FP16_DOT'] == (mode == 'fp16')
            denom = max(1, m['total_q'] * m['num_kv_heads'])
            target = max(1, min(m['topk'], 256 // denom))
            old = 1 << (target.bit_length() - 1)
            target = max(1, min(m['topk'], 2 * worker['num_sms'] // denom))
            blocks = (m['topk'] + target - 1) // target
            new = (m['topk'] + blocks - 1) // blocks
            assert m['NUM_TOPK_CHUNKS'] == (old if mode == 'baseline' else new)
            changed_split_launches += count * (old != new)
        assert changed_split_launches > 0, 'Full evaluation never exercised changed split shapes'
        summary.append({'rank': final['rank'], 'launches': final['launches'] - initial['launches'],
                        'changed_split_shape_launches': changed_split_launches,
                        'max_total_q': max(json.loads(k)['total_q'] for k in variants),
                        'variants': variants})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "bf16", "fp16"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=1319)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"] = "/work/cache/flashinfer"
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
    )
    if args.mode != 'baseline':
        settings['worker_extension_cls'] = 'gsm8k_cache_seed.FlashInferCacheSeed'
        os.environ['GSM8K_RESULT_DIR'] = str(args.output)
    params = dict(temperature=0.0, seed=42, max_tokens=1024, stop=None)
    result = {
        "mode": args.mode, "status": "starting", "settings": settings,
        "sampling": params, "chat_template_kwargs": {"thinking_mode": "disabled"},
        "protocol": "repository GSM8K first5 train fewshots, complete1319 test, chat, nonthinking",
        "source": vllm.__file__, "torch": torch.__version__, "cuda": torch.version.cuda,
        "packages": {p: importlib.metadata.version(p) for p in ["vllm", "triton", "nvidia-cutlass-dsl", "flashinfer-python"]},
        "criterion": {"baseline_sanity_floor": 0.80, "max_accuracy_drop": 0.005, "max_invalid_rate": 0.01, "max_truncation_rate": 0.01},
        "autotune_cache_before": cache_manifest(),
    }
    package_root = Path(vllm.__file__).parent
    result["source_hashes"] = {f: hashlib.sha256((package_root/f).read_bytes()).hexdigest()
        for f in ['config/attention.py', 'models/minimax_m3/common/ops/sparse_attn.py',
                  'models/minimax_m3/common/sparse_attention.py',
                  'models/minimax_m3/nvidia/model.py',
                  'models/minimax_m3/nvidia/sparse_attention_msa.py']}
    pin = json.loads(Path('/work/source-pins.json').read_text())['baseline' if args.mode == 'baseline' else 'candidate']
    assert result['source_hashes'] == pin['source_hashes'], 'Source differs from pinned revision'
    result['source_commit'] = pin['commit']
    out = args.output / "result.json"
    write_json(out, result)
    llm = None
    try:
        start = time.perf_counter()
        llm = LLM(**settings)
        result["startup_seconds"] = time.perf_counter() - start
        result["workers"] = llm.collective_rpc(install_dispatch_recorder, timeout=60)
        result["autotune_cache_after_startup"] = cache_manifest()
        if args.mode != 'baseline':
            result['validation_cache_seed'] = [json.loads(p.read_text())
                for p in sorted(args.output.glob('cache-seed-rank*.json'))]
            assert len(result['validation_cache_seed']) == 4
            for seed in result['validation_cache_seed']:
                for item in seed['files']:
                    source = Path(item['source']).read_bytes()
                    destination = Path(item['destination']).read_bytes()
                    assert json.loads(source) == json.loads(destination), 'FlashInfer tactics changed'
                    item['after_startup_sha256'] = hashlib.sha256(destination).hexdigest()
        prompts, labels = _build_gsm8k_prompts(args.questions, 5)
        assert len(prompts) == args.questions == 1319, 'Full GSM8K test split required'
        from vllm.assets.base import VLLM_S3_BUCKET_URL
        result['dataset'] = {'train_examples': 7473, 'test_examples': len(prompts),
            'train_fewshot_indices': list(range(5)),
            'files': {f: {'url': f'{VLLM_S3_BUCKET_URL}/ci-datasets/gsm8k/{f}',
                          'sha256': hashlib.sha256((Path('/tmp')/f).read_bytes()).hexdigest()}
                      for f in ['train.jsonl', 'test.jsonl']}}
        train_rows = [json.loads(s) for s in (Path('/tmp')/'train.jsonl').read_text().splitlines()]
        result['dataset']['fewshot_examples'] = train_rows[:5]
        tokenizer = llm.get_tokenizer()
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        rendered = tokenizer.apply_chat_template(conversations[0], tokenize=False, add_generation_prompt=True, thinking_mode="disabled")
        result["first_rendered_prompt"] = rendered
        assert rendered.endswith("</mm:think>"), repr(rendered[-100:])
        result["prompts_sha256"] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
        result["labels_sha256"] = hashlib.sha256(json.dumps(labels).encode()).hexdigest()
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
                assert meta["FP16_DOT"] == (args.mode == "fp16"), meta
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
        result["autotune_cache_after_generation"] = cache_manifest()
        result["status"] = "complete"
        write_json(out, result)
        print(json.dumps({"mode": args.mode, "status": result["status"], "metrics": result["metrics"]}), flush=True)
    except Exception:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        write_json(out, result)
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown(timeout=60)


if __name__ == "__main__":
    main()
