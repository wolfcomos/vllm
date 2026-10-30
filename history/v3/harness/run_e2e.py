"""Pinned MiniMax-M3 GB200 full-model latency comparison, one engine per arm."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback

PROTOCOL = "gb200-e2e-abba-v3"
ARMS = ("baseline", "default-a", "fp16-a", "fp16-b", "default-b")
COMMITS = {
    "baseline": "9731f30a030c545e7f29a8a4ed96b8562be12c88",
    "candidate": "faf31b04c73400c2af770c280fb8d739a5997fcb",
}
API_FILES = (
    "entrypoints/common/offline.py", "entrypoints/llm.py", "sampling_params.py",
    "v1/engine/llm_engine.py", "v1/engine/core_client.py", "v1/engine/core.py",
    "v1/core/sched/scheduler.py", "v1/engine/output_processor.py",
    "v1/metrics/stats.py", "outputs.py", "compilation/cuda_graph.py",
)


def load(path):
    return json.loads(Path(path).read_text())


def metric_record(output):
    """Core *_ts are monotonic; arrival_time is wall clock and stays separate."""
    stats = output.metrics
    assert stats is not None, "Request metrics require disable_log_stats=False"
    count = len(output.outputs[0].token_ids)
    assert stats.num_generation_tokens == count == 256
    assert stats.num_preemptions == 0, "Preemption changes the intended workload"
    assert 0 < stats.queued_ts <= stats.scheduled_ts <= stats.first_token_ts < stats.last_token_ts
    assert stats.first_token_latency > 0
    decode = stats.last_token_ts - stats.first_token_ts
    return {
        "request_id": output.request_id,
        "num_generation_tokens": count, "num_preemptions": stats.num_preemptions,
        "arrival_time_wall": stats.arrival_time,
        "queued_ts_monotonic": stats.queued_ts,
        "scheduled_ts_monotonic": stats.scheduled_ts,
        "first_token_ts_monotonic": stats.first_token_ts,
        "last_token_ts_monotonic": stats.last_token_ts,
        "ttft_seconds": stats.first_token_latency,
        "queue_seconds": stats.scheduled_ts - stats.queued_ts,
        "prefill_seconds": stats.first_token_ts - stats.scheduled_ts,
        "decode_seconds": decode,
        "mean_tpot_seconds": decode / (count - 1),
    }


def normalized_settings(settings):
    result = json.loads(json.dumps(settings))
    result["attention_config"].pop("minimax_m3_msa_decode_fp16_dot", None)
    return result


def inspect_backend(worker):
    from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseImpl

    found = []
    for name, module in worker.model_runner.model.named_modules():
        impl = getattr(module, "impl", None)
        if isinstance(impl, MiniMaxM3SparseImpl):
            found.append({"module": name, "class": type(impl).__name__,
                          "fp16_dot": getattr(impl, "fp16_dot", False)})
    return {"rank": worker.rank, "implementations": found}


def run(args):
    if not __debug__:
        raise RuntimeError("Assertions must remain enabled")
    sys.path.insert(0, str(args.helper_dir))
    from run_graph_perf import (
        BATCHES, INPUT_LEN, OUTPUT_LEN, REVISION, SOURCE_FILES, cache_manifest,
        digest, file_digest, inspect_worker, read_tuner_statistics,
        validate_tuner_after_generation, validate_tuner_startup,
        verify_cache_matches_baseline, write_json,
    )

    old = load(args.tactic_witness)
    assert (old["status"], old["mode"], old["phase"]) == ("complete", "baseline", "timing")
    assert old["source_commit"] == COMMITS["baseline"] and old["model_revision"] == REVISION
    helpers = {name: file_digest(args.helper_dir / name)
               for name in ("run_graph_perf.py", "graph_perf_cache_seed.py")}
    assert helpers == old["harness_hashes"], "Frozen helper changed"
    pins = load(args.source_pins)
    assert all(pins[key]["commit"] == value for key, value in COMMITS.items())
    assert pins["baseline"]["source_hashes"] == old["source_hashes"]
    pin = pins["baseline" if args.arm == "baseline" else "candidate"]
    witness = load(args.baseline_result) if args.baseline_result else None
    reference = load(args.reference_result) if args.reference_result else None
    if args.arm != "baseline":
        assert witness and witness["status"] == "complete" and witness["mode"] == "baseline"
    if args.arm.startswith("default"):
        reference, reference_path = witness, args.baseline_result
    else:
        reference_path = args.reference_result
    if args.arm == "fp16-b":
        assert reference and reference["mode"] == "fp16-a", "FP16 repeat needs its first arm"
    if reference:
        assert reference["protocol"] == PROTOCOL and reference["status"] == "complete"

    prompts_data = load(args.prompts)
    immutable = tuple(tuple(row) for row in prompts_data["prompts"])
    assert prompts_data["seed"] == 42 and prompts_data["input_len"] == INPUT_LEN
    assert len(immutable) == 40 and all(len(row) == INPUT_LEN for row in immutable)
    assert all(type(t) is int and 0 <= t < 10000 for row in immutable for t in row)
    assert digest(immutable) == prompts_data["prompts_sha256"] == old["prompts_sha256"]
    settings = json.loads(json.dumps(old["settings"]))
    settings.update(model=str(args.model), disable_log_stats=False, cudagraph_metrics=False)
    settings["attention_config"].pop("minimax_m3_msa_decode_fp16_dot", None)
    if args.arm.startswith("fp16"):
        settings["attention_config"]["minimax_m3_msa_decode_fp16_dot"] = True
    assert settings["compilation_config"]["mode"] == 0
    assert settings["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert settings["compilation_config"]["cudagraph_capture_sizes"] == list(BATCHES)
    assert settings["worker_extension_cls"] == "graph_perf_cache_seed.GraphPerfCacheSeed"
    assert settings["enable_prefix_caching"] is False and settings["enforce_eager"] is False
    assert "profiler_config" not in settings
    sampling = old["sampling"]
    assert sampling == {"temperature": 0.0, "seed": 42, "ignore_eos": True,
                        "max_tokens": OUTPUT_LEN, "detokenize": False, "stop": None}
    assert not args.output.exists() and not args.cache_dir.exists(), "Use fresh directories"
    args.output.mkdir(parents=True)
    args.cache_dir.mkdir(parents=True)
    for key, value in {
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "1", "VLLM_BATCH_INVARIANT": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": str(args.cache_dir),
        "M3_GRAPH_OUTPUT": str(args.output), "M3_GRAPH_BASELINE_RESULT": str(args.tactic_witness),
    }.items():
        os.environ[key] = value
    result = {
        "status": "starting", "protocol": PROTOCOL, "mode": args.arm, "phase": "timing",
        "source_commit": pin["commit"], "model_revision": REVISION,
        "source_pins_sha256": file_digest(args.source_pins),
        "tactic_witness_sha256": file_digest(args.tactic_witness),
        "harness_hashes": {**helpers, Path(__file__).name: file_digest(__file__),
                           "PROTOCOL.md": file_digest(Path(__file__).with_name("PROTOCOL.md"))},
        "settings": settings, "sampling": sampling, "prompts_sha256": digest(immutable),
        "batches": list(BATCHES), "input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
        "warmup_count": 5, "repeats": 7, "trials": [], "warmups": [], "references": {},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR")},
        "interval": "enqueue + wake_up(scheduling) + completion; pause and validation excluded",
        "metrics_scope": "Native frontend TTFT and core per-request mean TPOT; not individual-token p99 ITL",
        "profiling_enabled": False, "dispatch_recorder_installed": False,
    }
    if witness:
        result["baseline_result_sha256"] = file_digest(args.baseline_result)
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

        result.update(torch=torch.__version__, cuda=torch.version.cuda,
                      packages={name: importlib.metadata.version(name) for name in old["packages"]})
        for key in ("torch", "cuda", "packages"):
            assert result[key] == old[key], ("Runtime changed", key)
        package = Path(vllm.__file__).parent
        result["source_hashes"] = {name: file_digest(package / name) for name in SOURCE_FILES}
        result["api_hashes"] = {name: file_digest(package / name) for name in API_FILES}
        assert result["source_hashes"] == pin["source_hashes"]
        if witness:
            assert result["api_hashes"] == witness["api_hashes"]
        llm = LLM(**settings)
        workers = result["workers"] = llm.collective_rpc(inspect_worker, timeout=90)
        validate_tuner_startup(workers, seeded=True)
        implementations = result["backend_implementations"] = llm.collective_rpc(inspect_backend, timeout=90)
        assert sorted(row["rank"] for row in implementations) == [0, 1, 2, 3]
        for worker in implementations:
            assert worker["implementations"], "No MiniMax attention implementations found"
            assert all(impl["fp16_dot"] == args.arm.startswith("fp16") for impl in worker["implementations"])
        assert all("GB200" in worker["gpu"] for worker in workers)
        if witness:
            assert [(w["rank"], w["gpu"], w["num_sms"]) for w in workers] == [
                (w["rank"], w["gpu"], w["num_sms"]) for w in witness["workers"]]
        for worker in workers:
            assert worker["source_sha256"] == pin["source_hashes"][SOURCE_FILES[1]]
            assert worker["resolved_graph_mode"] == "FULL_DECODE_ONLY"
            assert all(d["mode"] == "FULL" and d["padded_tokens"] == d["batch"] for d in worker["dispatch"])
            assert all(any(e["mode"] == "FULL" and e["num_tokens"] == batch
                           for e in worker["captured_entries"]) for batch in BATCHES)
        result["cache_after_startup"] = cache_manifest(args.output)
        verify_cache_matches_baseline(result["cache_after_startup"], old)
        seen_ids = set()
        for batch in BATCHES:
            expected_prompts = immutable[:batch]
            measured_reference = None
            external_reference = None
            if reference:
                external_reference = tuple(tuple(row) for row in load(reference_path.parent / f"reference-b{batch}.json"))
                assert digest(external_reference) == reference["references"][str(batch)]
            for stage, count in (("warmup", 5), ("measured", 7)):
                for repeat in range(count):
                    llm.sleep(level=0, mode="wait")
                    prompts = [TokensPrompt(prompt_token_ids=list(row)) for row in expected_prompts]
                    params = SamplingParams(**sampling)
                    assert params.n == 1 and params.stop == []
                    params_hash = digest({key: getattr(params, key) for key in sampling})
                    started_utc = datetime.now(timezone.utc).isoformat()
                    start = time.perf_counter()
                    queued_ids = llm.enqueue(prompts, sampling_params=params, use_tqdm=False)
                    awake = llm.wake_up(tags=["scheduling"])
                    outputs = llm.wait_for_completion(use_tqdm=False)
                    seconds = time.perf_counter() - start
                    finished_utc = datetime.now(timezone.utc).isoformat()
                    assert awake is True and len(outputs) == len(queued_ids) == batch
                    assert tuple(tuple(row["prompt_token_ids"]) for row in prompts) == expected_prompts
                    assert tuple(tuple(row.prompt_token_ids) for row in outputs) == expected_prompts
                    assert digest({key: getattr(params, key) for key in sampling}) == params_hash
                    external_ids = [str(request_id).partition("-")[0] for request_id in queued_ids]
                    assert all(request_id.isdigit() for request_id in external_ids)
                    assert [o.request_id for o in outputs] == external_ids
                    assert len(set(external_ids)) == batch and not seen_ids.intersection(external_ids)
                    seen_ids.update(external_ids)
                    assert all(o.finished and len(o.outputs) == 1 for o in outputs)
                    tokens = tuple(tuple(o.outputs[0].token_ids) for o in outputs)
                    assert all(len(row) == OUTPUT_LEN for row in tokens)
                    metrics = [metric_record(output) for output in outputs]
                    row = {
                        "batch": batch, "repeat": repeat, "stage": stage, "seconds": seconds,
                        "started_at_utc": started_utc, "finished_at_utc": finished_utc,
                        "output_tokens": batch * OUTPUT_LEN,
                        "output_tokens_per_second": batch * OUTPUT_LEN / seconds,
                        "output_token_ids_sha256": digest(tokens),
                        "input_before_sha256": digest(expected_prompts),
                        "input_after_sha256": digest([p["prompt_token_ids"] for p in prompts]),
                        "returned_prompts_sha256": digest([o.prompt_token_ids for o in outputs]),
                        "normalized_sampling_sha256": params_hash, "inputs_unchanged": True,
                        "request_ids": external_ids, "queued_request_ids": queued_ids,
                        "request_metrics": metrics,
                        "mean_tpot_seconds": statistics.mean(m["mean_tpot_seconds"] for m in metrics),
                        "median_tpot_seconds": statistics.median(m["mean_tpot_seconds"] for m in metrics),
                        "mean_ttft_seconds": statistics.mean(m["ttft_seconds"] for m in metrics),
                    }
                    token_name = f"tokens-{stage}-b{batch}-r{repeat}.json"
                    row["tokens_file"] = token_name
                    write_json(args.output / token_name, tokens)
                    result["warmups" if stage == "warmup" else "trials"].append(row)
                    write_json(result_path, result)
                    if stage == "measured":
                        if measured_reference is None:
                            measured_reference = tokens
                            result["references"][str(batch)] = digest(tokens)
                            write_json(args.output / f"reference-b{batch}.json", tokens)
                        assert tokens == measured_reference, ("Within-arm outputs vary", batch, repeat)
                        if external_reference is not None:
                            assert tokens == external_reference, ("Reference outputs differ", batch, repeat)
                    write_json(result_path, result)
                    print(json.dumps({"mode": args.arm, **{key: row[key] for key in (
                        "batch", "repeat", "stage", "seconds", "mean_tpot_seconds", "mean_ttft_seconds",
                        "started_at_utc", "finished_at_utc")}}), flush=True)
        result["tuner_after_generation"] = llm.collective_rpc(read_tuner_statistics, timeout=90)
        validate_tuner_after_generation(workers, result["tuner_after_generation"])
        for before, after in zip(sorted(workers, key=lambda row: row["rank"]),
                                 sorted(result["tuner_after_generation"], key=lambda row: row["rank"])):
            assert before["flashinfer_statistics"] == after["statistics"], "Tuner state changed"
        result["cache_after_generation"] = cache_manifest(args.output)
        assert result["cache_after_startup"] == result["cache_after_generation"]
        verify_cache_matches_baseline(result["cache_after_generation"], old)
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
    for name in ("helper-dir", "tactic-witness", "source-pins", "prompts", "model", "cache-dir", "output"):
        parser.add_argument(f"--{name}", type=lambda x: Path(x).resolve(), required=True)
    for name in ("baseline-result", "reference-result"):
        parser.add_argument(f"--{name}", type=lambda x: Path(x).resolve())
    args = parser.parse_args()

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}; shutting down")

    signal.signal(signal.SIGTERM, terminate)
    run(args)


if __name__ == "__main__":
    main()
