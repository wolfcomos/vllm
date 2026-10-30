"""Run one prospective v4 full-model arm from a frozen manifest."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import time
import traceback

from e2e_support import (
    API_FILES, ARMS, BATCHES, COMMITS, INPUT_LEN, OUTPUT_LEN, PACKAGES, PROTOCOL,
    SOURCE_FILES, cache_identity, cache_manifest, digest, file_digest, inspect_worker,
    load, metric_record, normalized_settings, read_tuner, stats_unchanged, tail_metrics,
    write_json,
)


def run(args):
    if not __debug__:
        raise RuntimeError("Assertions must remain enabled")
    manifest = load(args.manifest)
    assert manifest["protocol"] == PROTOCOL
    pins = manifest["source_pins"]
    assert all(pins[key]["commit"] == commit for key, commit in COMMITS.items())
    source_kind = "baseline" if args.arm in ("baseline", "cutlass") else "candidate"
    pin = pins[source_kind]
    witness = load(args.baseline_result) if args.baseline_result else None
    reference = load(args.reference_result) if args.reference_result else None
    reference_path = args.reference_result
    if args.arm != "baseline":
        assert witness and witness["status"] == "complete" and witness["mode"] == "baseline"
    if args.arm.startswith("default"):
        reference, reference_path = witness, args.baseline_result
    if args.arm == "fp16-b":
        assert reference and reference["mode"] == "fp16-a"
    if reference:
        assert reference["protocol"] == PROTOCOL and reference["status"] == "complete"
    prompt_data = load(manifest["prompts_path"])
    immutable = tuple(tuple(row) for row in prompt_data["prompts"])
    assert prompt_data["seed"] == 42 and prompt_data["input_len"] == INPUT_LEN
    assert len(immutable) == 40 and all(len(row) == INPUT_LEN for row in immutable)
    assert all(type(token) is int and 0 <= token < 10000 for row in immutable for token in row)
    assert digest(immutable) == prompt_data["prompts_sha256"] == manifest["prompts_sha256"]
    settings = json.loads(json.dumps(manifest["settings"]))
    assert settings["revision"] == settings["tokenizer_revision"] == manifest["model_revision"]
    assert settings["dtype"] == "bfloat16" and settings["kv_cache_dtype"] == "fp8"
    assert settings["tensor_parallel_size"] == 4 and settings["max_num_seqs"] == 40
    assert settings["max_model_len"] >= INPUT_LEN + OUTPUT_LEN
    assert settings["disable_log_stats"] is False and settings["cudagraph_metrics"] is False
    assert settings["enforce_eager"] is False and settings["enable_prefix_caching"] is False
    assert settings["compilation_config"]["mode"] == 0
    assert settings["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert settings["compilation_config"]["cudagraph_capture_sizes"] == list(BATCHES)
    assert settings["kernel_config"]["enable_flashinfer_autotune"] is True
    assert settings["worker_extension_cls"] == "e2e_support.CurrentCacheSeed"
    assert "profiler_config" not in settings
    attention = settings["attention_config"]
    assert attention["indexer_kv_dtype"] == "fp8"
    assert "minimax_m3_msa_decode_backend" not in attention
    assert "minimax_m3_msa_decode_fp16_dot" not in attention
    if args.arm != "cutlass":
        attention["minimax_m3_msa_decode_backend"] = "triton"
    if args.arm.startswith("fp16"):
        attention["minimax_m3_msa_decode_fp16_dot"] = True
    sampling = {"temperature": 0.0, "seed": 42, "ignore_eos": True,
                "max_tokens": OUTPUT_LEN, "detokenize": False, "stop": None}
    assert not args.output.exists() and not args.cache_dir.exists(), "Use fresh output/cache paths"
    args.output.mkdir(parents=True)
    args.cache_dir.mkdir(parents=True)
    environment = {
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "1", "VLLM_BATCH_INVARIANT": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": str(args.cache_dir),
        "M3_V4_OUTPUT": str(args.output),
    }
    os.environ.update(environment)
    if witness:
        os.environ["M3_V4_BASELINE_RESULT"] = str(args.baseline_result)
    else:
        os.environ.pop("M3_V4_BASELINE_RESULT", None)
    harness_dir = Path(__file__).parent
    result = {
        "status": "starting", "protocol": PROTOCOL, "mode": args.arm, "phase": "timing",
        "source_commit": pin["commit"], "source_kind": source_kind,
        "manifest_sha256": file_digest(args.manifest), "model_revision": manifest["model_revision"],
        "harness_hashes": {name: file_digest(harness_dir / name)
                           for name in ("e2e_support.py", "run_e2e.py", "compare_e2e.py", "PROTOCOL.md")},
        "settings": settings, "sampling": sampling, "prompts_sha256": digest(immutable),
        "batches": list(BATCHES), "input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
        "warmup_count": 5, "repeats": 7, "trials": [], "warmups": [], "references": {},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "environment": {key: os.environ.get(key) for key in (*environment, "CUDA_VISIBLE_DEVICES",
                        "OMP_NUM_THREADS", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR")},
        "interval": "enqueue + resume scheduling + completion; pause/validation excluded",
        "metrics_scope": "Native frontend TTFT, request mean TPOT and prospective draining-batch common tail",
        "backend_scope": ("Current-main default backend with native Triton fallback; not a pure CUTLASS kernel arm"
                          if args.arm == "cutlass" else "Explicit Triton backend"),
        "profiling_enabled": False, "dispatch_recorder_installed": False,
    }
    if witness:
        result["baseline_result_sha256"] = file_digest(args.baseline_result)
        assert witness["manifest_sha256"] == result["manifest_sha256"]
        assert normalized_settings(settings) == normalized_settings(witness["settings"])
        assert result["harness_hashes"] == witness["harness_hashes"]
    if reference_path:
        result["reference_result_sha256"] = file_digest(reference_path)
    result_path = args.output / "result.json"
    write_json(result_path, result)
    llm = None
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        package = Path(vllm.__file__).resolve().parent
        assert package == (Path(pin["root"]) / "vllm").resolve(), "Wrong source tree imported"
        result["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                             "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
                             "native_binaries": {path.name: file_digest(path) for path in sorted(package.glob("*.so"))}}
        assert result["runtime"]["native_binaries"], "Native vLLM binaries missing"
        result["source_hashes"] = {name: file_digest(package / name) for name in SOURCE_FILES}
        result["api_hashes"] = {name: file_digest(package / name) for name in API_FILES}
        assert result["source_hashes"] == pin["source_hashes"]
        assert result["api_hashes"] == manifest["api_hashes"]
        if witness:
            assert result["runtime"] == witness["runtime"], "Runtime/native binaries differ"
        llm = LLM(**settings)
        workers = result["workers"] = llm.collective_rpc(inspect_worker, timeout=90)
        assert sorted(worker["rank"] for worker in workers) == [0, 1, 2, 3]
        for worker in workers:
            assert "GB200" in worker["gpu"] and worker["capability"] == [10, 0]
            assert worker["implementations"]
            assert all(impl["fp16_dot"] == args.arm.startswith("fp16") for impl in worker["implementations"])
            assert all(impl["use_cutlass_decode"] == (args.arm == "cutlass") for impl in worker["implementations"])
            assert all(impl["kv_cache_dtype"] == "fp8" for impl in worker["implementations"])
            assert all(impl[field] in (None, 1) for impl in worker["implementations"]
                       for field in ("k_scale_numel", "v_scale_numel")), "Per-token scale makes FP16 ineligible"
            assert worker["native_backend_policies"], "No live MSA metadata builders found"
            for policy in worker["native_backend_policies"]:
                shapes = {row["batch"]: row for row in policy["shapes"]}
                assert sorted(shapes) == list(range(1, 41))
                assert policy["min_batch_override"] is None
                if args.arm == "cutlass":
                    assert policy["geometry"]["decode_backend"] == "cutlass"
                    assert shapes[8]["should_prepare_at_unpadded_batch"] is False
                    assert shapes[40]["should_prepare_at_unpadded_batch"] is True
                    assert shapes[8]["should_prepare_at_graph_shape"] is False
                    assert shapes[40]["should_prepare_at_graph_shape"] is True
                else:
                    assert policy["geometry"]["decode_backend"] == "triton"
                    assert not any(row["should_prepare_at_unpadded_batch"] or row["should_prepare_at_graph_shape"]
                                   for row in policy["shapes"])
            assert worker["resolved_graph_mode"] == "FULL_DECODE_ONLY"
            assert all(row["mode"] == "FULL" and row["padded_tokens"] == row["batch"] for row in worker["dispatch"])
            assert all(any(entry["mode"] == "FULL" and entry["num_tokens"] == batch
                           for entry in worker["captured_entries"]) for batch in BATCHES)
            if witness:
                for field in ("tuned_op_successful_configs", "failed_profiling_count", "failed_tactics"):
                    assert all(value == 0 for value in worker["statistics"][field].values()), "Seeded arm tuned"
        if witness:
            identity = lambda group: [(row["rank"], row["gpu_uuid"], row["gpu"], row["num_sms"]) for row in group]
            assert identity(workers) == identity(witness["workers"])
        result["cache_after_startup"] = cache_manifest(args.output)
        if witness:
            assert cache_identity(result["cache_after_startup"]) == cache_identity(witness["cache_after_generation"])
        seen_ids = set()
        for batch in BATCHES:
            expected_prompts = immutable[:batch]
            measured_reference = None
            external_reference = None
            if reference:
                external_reference = load(reference_path.parent / f"reference-b{batch}.json")
                assert digest(external_reference) == reference["references"][str(batch)]
            for stage, count in (("warmup", 5), ("measured", 7)):
                for repeat in range(count):
                    llm.sleep(level=0, mode="wait")
                    prompts = [TokensPrompt(prompt_token_ids=list(row)) for row in expected_prompts]
                    params = SamplingParams(**sampling)
                    assert params.n == 1 and params.stop == []
                    params_hash = digest({key: getattr(params, key) for key in sampling})
                    started_utc = datetime.now(timezone.utc).isoformat()
                    started = time.perf_counter()
                    queued = llm.enqueue(prompts, sampling_params=params, use_tqdm=False)
                    awake = llm.wake_up(tags=["scheduling"])
                    outputs = llm.wait_for_completion(use_tqdm=False)
                    seconds = time.perf_counter() - started
                    finished_utc = datetime.now(timezone.utc).isoformat()
                    assert awake is True and len(queued) == len(outputs) == batch
                    assert tuple(tuple(row["prompt_token_ids"]) for row in prompts) == expected_prompts
                    assert tuple(tuple(row.prompt_token_ids) for row in outputs) == expected_prompts
                    assert digest({key: getattr(params, key) for key in sampling}) == params_hash
                    ids = [str(request_id).partition("-")[0] for request_id in queued]
                    assert all(request_id.isdigit() for request_id in ids)
                    assert [output.request_id for output in outputs] == ids
                    assert len(set(ids)) == batch and not seen_ids.intersection(ids)
                    seen_ids.update(ids)
                    assert all(output.finished and len(output.outputs) == 1 for output in outputs)
                    tokens = [list(output.outputs[0].token_ids) for output in outputs]
                    metrics = [metric_record(output) for output in outputs]
                    row = {
                        "batch": batch, "repeat": repeat, "stage": stage, "seconds": seconds,
                        "started_at_utc": started_utc, "finished_at_utc": finished_utc,
                        "output_tokens": batch * OUTPUT_LEN, "output_tokens_per_second": batch * OUTPUT_LEN / seconds,
                        "input_before_sha256": digest(expected_prompts),
                        "input_after_sha256": digest([p["prompt_token_ids"] for p in prompts]),
                        "returned_prompts_sha256": digest([o.prompt_token_ids for o in outputs]),
                        "normalized_sampling_sha256": params_hash, "request_ids": ids,
                        "queued_request_ids": queued, "request_metrics": metrics,
                        "output_token_ids_sha256": digest(tokens),
                        "mean_tpot_seconds": statistics.mean(metric["mean_tpot_seconds"] for metric in metrics),
                        "mean_ttft_seconds": statistics.mean(metric["ttft_seconds"] for metric in metrics),
                    }
                    row["tokens_file"] = f"tokens-{stage}-b{batch}-r{repeat}.json"
                    write_json(args.output / row["tokens_file"], tokens)
                    result["warmups" if stage == "warmup" else "trials"].append(row)
                    write_json(result_path, result)
                    # Persist raw metrics/tokens before validating the tail invariant.
                    row.update(tail_metrics(metrics))
                    if stage == "measured":
                        if measured_reference is None:
                            measured_reference = tokens
                            result["references"][str(batch)] = digest(tokens)
                            write_json(args.output / f"reference-b{batch}.json", tokens)
                        assert tokens == measured_reference, ("Within-arm token variation", args.arm, batch, repeat)
                        if external_reference is not None:
                            assert tokens == external_reference, ("Reference tokens differ", args.arm, batch, repeat)
                    write_json(result_path, result)
                    print(json.dumps({"mode": args.arm, **{key: row[key] for key in (
                        "batch", "repeat", "stage", "seconds", "mean_tpot_seconds", "mean_ttft_seconds",
                        "common_tail_seconds", "common_tail_tpot_seconds", "started_at_utc", "finished_at_utc")}}), flush=True)
        result["tuner_after_generation"] = llm.collective_rpc(read_tuner, timeout=90)
        stats_unchanged(workers, result["tuner_after_generation"])
        result["cache_after_generation"] = cache_manifest(args.output)
        assert result["cache_after_startup"] == result["cache_after_generation"]
        if witness:
            assert cache_identity(result["cache_after_generation"]) == cache_identity(witness["cache_after_generation"])
        result["status"] = "complete"
    except BaseException:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        try:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown(timeout=60)
                result["shutdown_completed"] = True
        except BaseException:
            result["status"] = "failed"
            result["shutdown_traceback"] = traceback.format_exc()
            raise
        finally:
            result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(result_path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    for name in ("manifest", "output", "cache-dir"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve(), required=True)
    for name in ("baseline-result", "reference-result"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve())
    args = parser.parse_args()

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}; preserve evidence and shut down")

    signal.signal(signal.SIGTERM, terminate)
    run(args)


if __name__ == "__main__":
    main()
