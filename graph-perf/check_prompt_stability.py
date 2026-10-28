"""Baseline input-alias and first-token diagnostic; never a performance result."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import signal
import traceback

from run_graph_perf import (
    BATCHES, GraphLogCapture, INPUT_LEN, OUTPUT_LEN, REVISION, SOURCE_FILES,
    cache_manifest, canonical_settings, digest, file_digest, inspect_worker,
    read_tuner_statistics, validate_tuner_after_generation,
    validate_tuner_startup, verify_cache_matches_baseline, write_json,
)

BASELINE_COMMIT = "9731f30a030c545e7f29a8a4ed96b8562be12c88"
BATCH = 8


def summarize(calls):
    return {
        phase: {
            "calls": len(rows),
            "distinct_output_hashes": len({r["output_token_ids_sha256"] for r in rows}),
            "distinct_first_token_hashes": len({r["first_token_ids_sha256"] for r in rows}),
            "distinct_first_tokens_per_prompt": [
                len({r["first_token_ids"][i] for r in rows}) for i in range(BATCH)
            ],
        }
        for phase in ("warmup-256", "full-256", "prefill-only-1")
        if (rows := [r for r in calls if r["phase"] == phase])
    }


def run(args):
    if not __debug__:
        raise RuntimeError("Assertions must remain enabled")
    assert all(p.is_absolute() for p in vars(args).values())
    baseline = json.loads(args.baseline_result.read_text())
    assert (baseline["status"], baseline["mode"], baseline["phase"]) == (
        "complete", "baseline", "timing"
    )
    assert baseline["source_commit"] == BASELINE_COMMIT
    assert baseline["model_revision"] == REVISION
    assert baseline["cache_after_startup"] == baseline["cache_after_generation"]
    pins = json.loads(args.source_pins.read_text())["baseline"]
    assert pins["commit"] == BASELINE_COMMIT
    assert pins["source_hashes"] == baseline["source_hashes"]
    helper_hashes = {name: file_digest(Path(__file__).parent / name)
                     for name in ("run_graph_perf.py", "graph_perf_cache_seed.py")}
    assert helper_hashes == baseline["harness_hashes"], "Frozen helper differs"

    prompt_data = json.loads(args.prompts.read_text())
    immutable_prompts = tuple(tuple(p) for p in prompt_data["prompts"])
    assert prompt_data["seed"] == 42 and prompt_data["input_len"] == INPUT_LEN
    assert len(immutable_prompts) == 40
    assert all(len(p) == INPUT_LEN for p in immutable_prompts)
    assert all(type(t) is int and 0 <= t < 10000 for p in immutable_prompts for t in p)
    assert digest(immutable_prompts) == prompt_data["prompts_sha256"] == baseline["prompts_sha256"]
    expected_prompts = immutable_prompts[:BATCH]
    sampling = baseline["sampling"]
    assert sampling == {"temperature": 0.0, "seed": 42, "ignore_eos": True,
                        "max_tokens": OUTPUT_LEN, "detokenize": False, "stop": None}
    settings = json.loads(json.dumps(baseline["settings"]))
    assert "profiler_config" not in settings
    assert settings["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert settings["enforce_eager"] is False
    assert settings["worker_extension_cls"] == "graph_perf_cache_seed.GraphPerfCacheSeed"
    settings.update(disable_log_stats=False, cudagraph_metrics=True)
    assert canonical_settings(settings) == canonical_settings(baseline["settings"])

    # Require new directories so this diagnostic cannot overwrite a frozen arm/cache.
    assert not args.output.exists() and not args.cache_dir.exists(), "Use fresh directories"
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache_dir.mkdir(parents=True, exist_ok=False)
    for key, value in {
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_BATCH_INVARIANT": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": str(args.cache_dir),
        "M3_GRAPH_OUTPUT": str(args.output),
        "M3_GRAPH_BASELINE_RESULT": str(args.baseline_result),
    }.items():
        os.environ[key] = value

    result = {
        "status": "starting", "mode": "baseline", "phase": "input-stability-diagnostic",
        "source_commit": BASELINE_COMMIT, "model_revision": REVISION,
        "baseline_result_sha256": file_digest(args.baseline_result),
        "source_pins_sha256": file_digest(args.source_pins),
        "prompts_sha256": digest(immutable_prompts),
        "batch_prompts_sha256": digest(expected_prompts), "batch": BATCH,
        "settings": settings, "sampling": sampling,
        "settings_changes": {"disable_log_stats": False, "cudagraph_metrics": True},
        "harness_hashes": {**helper_hashes, Path(__file__).name: file_digest(__file__)},
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "calls": [],
        "interpretation": "Input/parameter alias diagnostic only; no timing or speedup claim. "
                          "The single-output-token phase changes scheduling and does not "
                          "establish equivalence to the full 256-token workload.",
    }
    result_path = args.output / "result.json"
    write_json(result_path, result)
    llm = None
    graph_logs = GraphLogCapture()
    metrics_logger = logging.getLogger("vllm.v1.metrics.loggers")
    metrics_logger.addHandler(graph_logs)
    try:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        import torch
        import vllm

        result.update(torch=torch.__version__, cuda=torch.version.cuda,
                      packages={name: importlib.metadata.version(name)
                                for name in baseline["packages"]})
        for key in ("torch", "cuda", "packages"):
            assert result[key] == baseline[key], ("Runtime differs", key)
        result["source_hashes"] = {
            name: file_digest(Path(vllm.__file__).parent / name) for name in SOURCE_FILES
        }
        assert result["source_hashes"] == pins["source_hashes"]
        llm = LLM(**settings)
        workers = result["workers"] = llm.collective_rpc(inspect_worker, timeout=60)
        validate_tuner_startup(workers, seeded=True)
        assert [(w["rank"], w["gpu"], w["num_sms"]) for w in workers] == [
            (w["rank"], w["gpu"], w["num_sms"]) for w in baseline["workers"]
        ], "GPU/rank identities differ"
        for worker in workers:
            assert worker["source_sha256"] == pins["source_hashes"][SOURCE_FILES[1]]
            assert worker["resolved_graph_mode"] == "FULL_DECODE_ONLY"
            assert all(d["mode"] == "FULL" and d["padded_tokens"] == d["batch"]
                       for d in worker["dispatch"])
            assert all(any(e["mode"] == "FULL" and e["num_tokens"] == batch
                           for e in worker["captured_entries"]) for batch in BATCHES)
        result["cache_after_startup"] = cache_manifest(args.output)
        verify_cache_matches_baseline(result["cache_after_startup"], baseline)
        seen_request_ids = set()

        for phase, count, max_tokens in (("warmup-256", 2, 256), ("full-256", 4, 256),
                                         ("prefill-only-1", 4, 1)):
            for repeat in range(count):
                prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in expected_prompts]
                call_sampling = {**sampling, "max_tokens": max_tokens}
                params = SamplingParams(**call_sampling)
                assert params.n == 1 and params.stop == []
                sampling_before = digest({k: getattr(params, k) for k in call_sampling})
                assert tuple(tuple(p["prompt_token_ids"]) for p in prompts) == expected_prompts
                llm.llm_engine.do_log_stats()
                offset = len(graph_logs.rows)
                try:
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                finally:
                    # Compare against tuples, never the mutable input lists themselves.
                    assert tuple(tuple(p) for p in prompt_data["prompts"]) == immutable_prompts
                    assert tuple(tuple(p["prompt_token_ids"]) for p in prompts) == expected_prompts
                    assert digest({k: getattr(params, k) for k in call_sampling}) == sampling_before
                llm.llm_engine.do_log_stats()
                assert len(outputs) == BATCH
                assert tuple(tuple(o.prompt_token_ids) for o in outputs) == expected_prompts
                request_ids = [int(o.request_id) for o in outputs]
                assert request_ids == sorted(request_ids) and len(set(request_ids)) == BATCH
                assert not seen_request_ids.intersection(request_ids)
                seen_request_ids.update(request_ids)
                assert all(o.finished and len(o.outputs) == 1 for o in outputs)
                token_ids = [list(o.outputs[0].token_ids) for o in outputs]
                assert all(len(ids) == max_tokens for ids in token_ids)
                rows = graph_logs.rows[offset:]
                full_steps = sum(r["count"] for r in rows if r["mode"] == "FULL"
                                 and r["unpadded"] == BATCH and r["padded"] == BATCH)
                row = {"phase": phase, "repeat": repeat, "max_tokens": max_tokens,
                       "request_ids": request_ids, "inputs_unchanged": True,
                       "sampling_unchanged": True, "returned_prompts_match": True,
                       "normalized_sampling_sha256": sampling_before,
                       "first_token_ids": [ids[0] for ids in token_ids],
                       "first_token_ids_sha256": digest([ids[0] for ids in token_ids]),
                       "output_token_ids_sha256": digest(token_ids),
                       "input_before_sha256": digest(expected_prompts),
                       "input_after_sha256": digest([p["prompt_token_ids"] for p in prompts]),
                       "returned_prompts_sha256": digest([o.prompt_token_ids for o in outputs]),
                       "finish_reasons": [o.outputs[0].finish_reason for o in outputs],
                       "native_graph_metric_rows": rows, "unpadded_full_graph_steps": full_steps}
                write_json(args.output / f"tokens-{phase}-r{repeat}.json", token_ids)
                result["calls"].append(row)
                write_json(result_path, result)
                if max_tokens == 256:
                    assert full_steps >= 128, ("Full runtime graph mode not observed", phase, repeat, rows)
                print(json.dumps(row), flush=True)

        result["tuner_after_generation"] = llm.collective_rpc(read_tuner_statistics, timeout=60)
        validate_tuner_after_generation(workers, result["tuner_after_generation"])
        result["cache_after_generation"] = cache_manifest(args.output)
        assert result["cache_after_generation"] == result["cache_after_startup"]
        verify_cache_matches_baseline(result["cache_after_generation"], baseline)
        result["summary"] = summarize(result["calls"])
        result["status"] = "complete"
    except BaseException:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        metrics_logger.removeHandler(graph_logs)
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
    for name in ("baseline-result", "prompts", "source-pins", "cache-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}; shutting down the diagnostic")

    signal.signal(signal.SIGTERM, terminate)
    run(args)


if __name__ == "__main__":
    main()
