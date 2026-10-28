"""Separate admission-controlled MiniMax M3 experiment; frozen v1 is untouched."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
import traceback

FROZEN_DIR = Path(__file__).resolve().parent.parent / "graph-perf"
sys.path.insert(0, str(FROZEN_DIR))
from run_graph_perf import (  # noqa: E402
    BATCHES, GraphLogCapture, INPUT_LEN, OUTPUT_LEN, REVISION, SOURCE_FILES,
    cache_manifest, canonical_settings, digest, file_digest, inspect_worker,
    read_tuner_statistics, validate_tuner_after_generation,
    validate_tuner_startup, verify_cache_matches_baseline, write_json,
)

PROTOCOL = "admission-controlled-v2"
ARMS = ("baseline", "default-a", "fp16-a", "fp16-b", "default-b")
COMMITS = {"baseline": "9731f30a030c545e7f29a8a4ed96b8562be12c88",
           "candidate": "faf31b04c73400c2af770c280fb8d739a5997fcb"}
API_FILES = ("entrypoints/common/offline.py", "entrypoints/llm.py", "sampling_params.py",
             "v1/engine/llm_engine.py", "v1/engine/core_client.py", "v1/engine/core.py",
             "v1/core/sched/scheduler.py", "logger.py", "v1/metrics/loggers.py",
             "compilation/cuda_graph.py")


class NamedGraphCapture(GraphLogCapture):
    def emit(self, record):
        start = len(self.rows)
        super().emit(record)
        for row in self.rows[start:]:
            row["logger_name"] = record.name


def load(path):
    return json.loads(path.read_text())


def literal_tokens(path, expected_hash):
    tokens = tuple(tuple(row) for row in load(path))
    assert digest(tokens) == expected_hash, ("Token artifact hash differs", path)
    return tokens


def run(args):
    if not __debug__:
        raise RuntimeError("Assertions must remain enabled")
    for name in ("baseline_v1_result", "prompts", "source_pins", "cache_dir", "output"):
        assert getattr(args, name).is_absolute(), name
    v1 = load(args.baseline_v1_result)
    assert (v1["status"], v1["mode"], v1["phase"]) == ("complete", "baseline", "timing")
    assert v1["source_commit"] == COMMITS["baseline"] and v1["model_revision"] == REVISION
    assert v1["cache_after_startup"] == v1["cache_after_generation"]
    helpers = {name: file_digest(FROZEN_DIR / name)
               for name in ("run_graph_perf.py", "graph_perf_cache_seed.py")}
    assert helpers == v1["harness_hashes"], "Frozen dependency changed"
    harness_hashes = {**helpers, Path(__file__).name: file_digest(__file__),
                      "PROTOCOL.md": file_digest(Path(__file__).parent / "PROTOCOL.md")}
    pins = load(args.source_pins)
    pin = pins["baseline" if args.arm == "baseline" else "candidate"]
    assert all(pins[key]["commit"] == commit for key, commit in COMMITS.items())
    assert pins["baseline"]["source_hashes"] == v1["source_hashes"]
    witness = load(args.witness_result) if args.witness_result else None
    previous = load(args.previous_result) if args.previous_result else None
    timing = load(args.timing_result) if args.timing_result else None
    for reference in (witness, previous, timing):
        if reference:
            assert reference["status"] == "complete" and reference["protocol"] == PROTOCOL
            assert reference["phase"] == "timing" and reference["harness_hashes"] == harness_hashes
    if args.arm != "baseline":
        assert witness and witness["mode"] == "baseline"
    if args.phase == "timing" and args.arm != "baseline":
        assert previous and previous["mode"] == ARMS[ARMS.index(args.arm) - 1]
    if args.phase == "diagnostic":
        assert timing and timing["mode"] == args.arm
        gate = load(args.timing_comparison)
        assert gate["status"] == "complete" and gate["passed"] is True
        assert gate["protocol"] == PROTOCOL
        assert gate["result_sha256"][args.arm] == file_digest(args.timing_result)

    prompt_data = load(args.prompts)
    immutable = tuple(tuple(p) for p in prompt_data["prompts"])
    assert prompt_data["seed"] == 42 and prompt_data["input_len"] == INPUT_LEN
    assert len(immutable) == 40 and all(len(p) == INPUT_LEN for p in immutable)
    assert all(type(t) is int and 0 <= t < 10000 for p in immutable for t in p)
    assert digest(immutable) == prompt_data["prompts_sha256"] == v1["prompts_sha256"]
    sampling = v1["sampling"]
    assert sampling == {"temperature": 0.0, "seed": 42, "ignore_eos": True,
                        "max_tokens": OUTPUT_LEN, "detokenize": False, "stop": None}
    settings = json.loads(json.dumps(v1["settings"]))
    assert "profiler_config" not in settings and settings["enforce_eager"] is False
    assert settings["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert settings["compilation_config"]["cudagraph_capture_sizes"] == list(BATCHES)
    assert settings["worker_extension_cls"] == "graph_perf_cache_seed.GraphPerfCacheSeed"
    if args.arm.startswith("fp16"):
        settings["attention_config"]["minimax_m3_msa_decode_fp16_dot"] = True
    diagnostic = args.phase == "diagnostic"
    settings.update(disable_log_stats=not diagnostic, cudagraph_metrics=diagnostic)
    assert canonical_settings(settings) == canonical_settings(v1["settings"])
    assert not args.output.exists() and not args.cache_dir.exists(), "Use fresh output/cache directories"
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache_dir.mkdir(parents=True, exist_ok=False)
    for key, value in {
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "1", "VLLM_BATCH_INVARIANT": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": str(args.cache_dir),
        "M3_GRAPH_OUTPUT": str(args.output), "M3_GRAPH_BASELINE_RESULT": str(args.baseline_v1_result),
    }.items():
        os.environ[key] = value
    result = {
        "status": "starting", "protocol": PROTOCOL, "mode": args.arm, "phase": args.phase,
        "source_commit": pin["commit"], "model_revision": REVISION,
        "baseline_v1_result_sha256": file_digest(args.baseline_v1_result),
        "source_pins_sha256": file_digest(args.source_pins), "harness_hashes": harness_hashes,
        "protocol_sha256": harness_hashes["PROTOCOL.md"],
        "settings": settings, "sampling": sampling, "prompts_sha256": digest(immutable),
        "batches": list(BATCHES), "input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
        "warmup_count": 3, "repeats": 7 if not diagnostic else 1,
        "profiling_enabled": False, "dispatch_recorder_installed": False,
        "logging_handler_installed": diagnostic, "trials": [], "warmups": [], "references": {},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "compilation_cache_environment": {key: os.environ.get(key) for key in
                                          ("VLLM_CACHE_ROOT", "TRITON_CACHE_DIR")},
        "interval": "enqueue + wake_up(scheduling) + wait_for_completion; pause and validation excluded",
        "interpretation": "Separate admission-controlled experiment, not comparable with v1 timings. "
                          "Native graph metrics in diagnostic processes establish runtime mode, not a hardware trace.",
    }
    if previous:
        assert previous["finished_at_utc"] <= result["started_at_utc"]
        result["previous_result_sha256"] = file_digest(args.previous_result)
    if witness:
        result["witness_result_sha256"] = file_digest(args.witness_result)
    if diagnostic:
        result["timing_result_sha256"] = file_digest(args.timing_result)
        result["timing_comparison_sha256"] = file_digest(args.timing_comparison)
    result_path = args.output / "result.json"
    write_json(result_path, result)
    llm = None
    capture = None
    logger_states = []
    try:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        import torch
        import vllm

        result.update(torch=torch.__version__, cuda=torch.version.cuda,
                      packages={name: importlib.metadata.version(name) for name in v1["packages"]})
        for key in ("torch", "cuda", "packages"):
            assert result[key] == v1[key], ("Runtime differs", key)
        package = Path(vllm.__file__).parent
        result["source_hashes"] = {name: file_digest(package / name) for name in SOURCE_FILES}
        result["admission_api_hashes"] = {name: file_digest(package / name) for name in API_FILES}
        assert result["source_hashes"] == pin["source_hashes"]
        if witness:
            assert result["admission_api_hashes"] == witness["admission_api_hashes"]
        llm = LLM(**settings)
        workers = result["workers"] = llm.collective_rpc(inspect_worker, timeout=60)
        validate_tuner_startup(workers, seeded=True)
        assert [(w["rank"], w["gpu"], w["num_sms"]) for w in workers] == [
            (w["rank"], w["gpu"], w["num_sms"]) for w in v1["workers"]
        ]
        for worker in workers:
            assert worker["source_sha256"] == pin["source_hashes"][SOURCE_FILES[1]]
            assert worker["resolved_graph_mode"] == "FULL_DECODE_ONLY"
            assert all(d["mode"] == "FULL" and d["padded_tokens"] == d["batch"]
                       for d in worker["dispatch"])
            assert all(any(e["mode"] == "FULL" and e["num_tokens"] == batch
                           for e in worker["captured_entries"]) for batch in BATCHES)
        result["cache_after_startup"] = cache_manifest(args.output)
        verify_cache_matches_baseline(result["cache_after_startup"], v1)
        if diagnostic:
            # vLLM logging configuration can clear handlers; attach only after initialization.
            capture = NamedGraphCapture()
            for name in ("vllm.v1.metrics.loggers", "vllm.compilation.cuda_graph"):
                logger = logging.getLogger(name)
                logger_states.append((logger, logger.level))
                logger.setLevel(logging.DEBUG)
                logger.addHandler(capture)
        seen_ids = set()
        for batch in BATCHES:
            expected_prompts = immutable[:batch]
            reference_tokens = None
            expected_tokens = None
            anchor, anchor_path = None, None
            if diagnostic:
                anchor, anchor_path = timing, args.timing_result
            elif args.arm in ("default-a", "default-b"):
                anchor, anchor_path = witness, args.witness_result
            elif args.arm == "fp16-b":
                anchor, anchor_path = previous, args.previous_result
            if anchor:
                expected_tokens = literal_tokens(anchor_path.parent / f"reference-b{batch}.json",
                                                 anchor["references"][str(batch)])
            for stage, count in (("warmup", 3), ("diagnostic" if diagnostic else "measured", result["repeats"])):
                for repeat in range(count):
                    llm.sleep(level=0, mode="wait")
                    prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in expected_prompts]
                    params = SamplingParams(**sampling)
                    assert params.n == 1 and params.stop == []
                    params_hash = digest({key: getattr(params, key) for key in sampling})
                    assert tuple(tuple(p["prompt_token_ids"]) for p in prompts) == expected_prompts
                    if capture:
                        assert all(capture in logger.handlers for logger, _ in logger_states)
                        llm.llm_engine.do_log_stats()
                    offset = len(capture.rows) if capture else 0
                    measured = stage == "measured"
                    started = time.perf_counter() if measured else None
                    queued_ids = llm.enqueue(prompts, sampling_params=params, use_tqdm=False)
                    awake = llm.wake_up(tags=["scheduling"])
                    outputs = llm.wait_for_completion(use_tqdm=False)
                    seconds = time.perf_counter() - started if measured else None
                    if capture:
                        llm.llm_engine.do_log_stats()
                        assert all(capture in logger.handlers for logger, _ in logger_states)
                    assert awake is True and len(queued_ids) == len(outputs) == batch
                    assert tuple(tuple(p) for p in prompt_data["prompts"]) == immutable
                    assert tuple(tuple(p["prompt_token_ids"]) for p in prompts) == expected_prompts
                    assert tuple(tuple(o.prompt_token_ids) for o in outputs) == expected_prompts
                    assert digest({key: getattr(params, key) for key in sampling}) == params_hash
                    external_ids = [str(i).partition("-")[0] for i in queued_ids]
                    assert all(i.isdigit() for i in external_ids)
                    assert [o.request_id for o in outputs] == external_ids
                    assert list(map(int, external_ids)) == sorted(map(int, external_ids))
                    assert len(set(external_ids)) == batch and not seen_ids.intersection(external_ids)
                    seen_ids.update(external_ids)
                    assert all(o.finished and len(o.outputs) == 1 for o in outputs)
                    tokens = tuple(tuple(o.outputs[0].token_ids) for o in outputs)
                    assert all(len(ids) == OUTPUT_LEN for ids in tokens)
                    row = {"batch": batch, "repeat": repeat, "stage": stage, "seed": 42,
                           "output_tokens": batch * OUTPUT_LEN,
                           "output_token_ids_sha256": digest(tokens),
                           "first_token_ids_sha256": digest([ids[0] for ids in tokens]),
                           "input_before_sha256": digest(expected_prompts),
                           "input_after_sha256": digest([p["prompt_token_ids"] for p in prompts]),
                           "returned_prompts_sha256": digest([o.prompt_token_ids for o in outputs]),
                           "normalized_sampling_sha256": params_hash,
                           "inputs_unchanged": True, "sampling_unchanged": True,
                           "request_ids": external_ids, "queued_request_ids": queued_ids}
                    if measured:
                        row["seconds"] = seconds
                    if capture:
                        row["native_graph_metric_rows"] = capture.rows[offset:]
                        row["unpadded_full_graph_steps"] = sum(r["count"] for r in capture.rows[offset:]
                            if r["mode"] == "FULL" and r["unpadded"] == r["padded"] == batch)
                    stem = f"tokens-b{batch}-r{repeat}" if measured else f"tokens-{stage}-b{batch}-r{repeat}"
                    write_json(args.output / f"{stem}.json", tokens)
                    result["warmups" if stage == "warmup" else "trials"].append(row)
                    write_json(result_path, result)
                    # Warmups are retained, but excluded from the frozen repeatability gate.
                    if stage != "warmup":
                        if reference_tokens is None:
                            reference_tokens = tokens
                            result["references"][str(batch)] = digest(tokens)
                            write_json(args.output / f"reference-b{batch}.json", tokens)
                        assert tokens == reference_tokens, ("Within-arm output variation", args.arm, batch, repeat)
                        if expected_tokens is not None:
                            assert tokens == expected_tokens, ("Cross-arm output mismatch", args.arm, batch, repeat)
                        if capture:
                            assert row["unpadded_full_graph_steps"] >= 128, ("Native FULL mode not observed", row)
                    write_json(result_path, result)
                    print(json.dumps({"mode": args.arm, **row}), flush=True)

        result["tuner_after_generation"] = llm.collective_rpc(read_tuner_statistics, timeout=60)
        validate_tuner_after_generation(workers, result["tuner_after_generation"])
        result["cache_after_generation"] = cache_manifest(args.output)
        assert result["cache_after_generation"] == result["cache_after_startup"]
        verify_cache_matches_baseline(result["cache_after_generation"], v1)
        result["status"] = "complete"
    except BaseException:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        for logger, level in logger_states:
            logger.removeHandler(capture)
            logger.setLevel(level)
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
    parser.add_argument("--phase", choices=("timing", "diagnostic"), default="timing")
    for name in ("baseline-v1-result", "prompts", "source-pins", "cache-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("witness-result", "previous-result", "timing-result", "timing-comparison"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}; shutting down")

    signal.signal(signal.SIGTERM, terminate)
    run(args)


if __name__ == "__main__":
    main()
