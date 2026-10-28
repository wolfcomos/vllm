"""Independently audit raw v2 arms and emit an explicit public-field whitelist.

CPU-only; outside the frozen GPU protocol. Does not issue a performance verdict.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

ARMS = ("baseline", "default-a", "fp16-a", "fp16-b", "default-b")
ACTIVITY = ("tuned_op_successful_configs", "failed_profiling_count", "failed_tactics")
SOURCE_FILES = ("config/attention.py", "models/minimax_m3/common/ops/sparse_attn.py",
                "models/minimax_m3/common/sparse_attention.py", "models/minimax_m3/nvidia/model.py",
                "models/minimax_m3/nvidia/sparse_attention_msa.py")
API_FILES = ("entrypoints/common/offline.py", "entrypoints/llm.py", "sampling_params.py",
             "v1/engine/llm_engine.py", "v1/engine/core_client.py", "v1/engine/core.py",
             "v1/core/sched/scheduler.py", "logger.py", "v1/metrics/loggers.py",
             "compilation/cuda_graph.py")
ROW_FIELDS = ("batch", "repeat", "stage", "seed", "output_tokens",
              "output_token_ids_sha256", "first_token_ids_sha256", "input_before_sha256",
              "input_after_sha256", "returned_prompts_sha256", "normalized_sampling_sha256",
              "inputs_unchanged", "sampling_unchanged")


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def token_file(path, batch, expected):
    tokens = tuple(tuple(row) for row in load(path))
    assert len(tokens) == batch and all(len(row) == 256 for row in tokens)
    assert all(type(t) is int and t >= 0 for row in tokens for t in row)
    assert digest(tokens) == expected
    return tokens


def audit_arm(notes, mode, source_repo=None):
    root = notes / "graph-perf-admission-controlled"
    directory = root / "results" / mode
    result = load(directory / "result.json")
    v1_path = notes / "graph-perf/results/baseline/result.json"
    v1 = load(v1_path)
    pins_path = notes / "staging/source-pins.json"
    pins = load(pins_path)
    pin = pins["baseline" if mode == "baseline" else "candidate"]
    assert (result["status"], result["protocol"], result["phase"], result["mode"]) == (
        "complete", "admission-controlled-v2", "timing", mode)
    assert result["shutdown_completed"] is True
    assert result["source_commit"] == pin["commit"]
    assert result["source_hashes"] == pin["source_hashes"]
    assert set(result["source_hashes"]) == set(SOURCE_FILES)
    assert set(result["admission_api_hashes"]) == set(API_FILES)
    if source_repo is not None:
        for name, expected in {**result["source_hashes"], **result["admission_api_hashes"]}.items():
            raw = subprocess.check_output(["git", "-C", str(source_repo), "show",
                                           result["source_commit"] + ":vllm/" + name])
            assert hashlib.sha256(raw).hexdigest() == expected
    assert result["baseline_v1_result_sha256"] == sha(v1_path)
    assert result["source_pins_sha256"] == sha(pins_path)
    for key in ("torch", "cuda", "packages", "model_revision", "sampling", "prompts_sha256"):
        assert result[key] == v1[key], key
    settings = json.loads(json.dumps(result["settings"]))
    enabled = settings["attention_config"].pop("minimax_m3_msa_decode_fp16_dot", False)
    assert enabled == mode.startswith("fp16") and settings == v1["settings"]
    assert all(result[key] is False for key in (
        "profiling_enabled", "dispatch_recorder_installed", "logging_handler_installed"))
    assert result["warmup_count"] == 3 and result["repeats"] == 7
    assert result["batches"] == [8, 40] and result["input_len"] == 2048 and result["output_len"] == 256
    for name, expected in result["harness_hashes"].items():
        base = root if name in ("run_admission_controlled.py", "PROTOCOL.md") else notes / "graph-perf"
        assert sha(base / name) == expected
    assert result["protocol_sha256"] == result["harness_hashes"]["PROTOCOL.md"]
    if mode != "baseline":
        witness_path = root / "results/baseline/result.json"
        witness = load(witness_path)
        previous_path = root / "results" / ARMS[ARMS.index(mode) - 1] / "result.json"
        assert result["witness_result_sha256"] == sha(witness_path)
        assert result["previous_result_sha256"] == sha(previous_path)
        assert load(previous_path)["finished_at_utc"] <= result["started_at_utc"]
        for key in ("admission_api_hashes", "harness_hashes", "compilation_cache_environment", "interval"):
            assert result[key] == witness[key]

    prompts = load(notes / "graph-perf/prompts.json")["prompts"]
    assert len(prompts) == 40 and all(len(p) == 2048 for p in prompts)
    assert digest(prompts) == result["prompts_sha256"]
    assert [(r["batch"], r["repeat"], r["stage"]) for r in result["trials"]] == [
        (b, i, "measured") for b in (8, 40) for i in range(7)]
    assert [(r["batch"], r["repeat"], r["stage"]) for r in result["warmups"]] == [
        (b, i, "warmup") for b in (8, 40) for i in range(3)]
    seen = set()
    cases = []
    public_rows = {"trials": [], "warmups": []}
    for batch in (8, 40):
        reference = token_file(directory / f"reference-b{batch}.json", batch, result["references"][str(batch)])
        anchor_mode = "baseline" if mode in ("default-a", "default-b") else "fp16-a" if mode == "fp16-b" else None
        if anchor_mode:
            anchor = root / "results" / anchor_mode
            anchor_hash = load(anchor / "result.json")["references"][str(batch)]
            assert reference == token_file(anchor / f"reference-b{batch}.json", batch, anchor_hash)
        measured = [r for r in result["trials"] if r["batch"] == batch]
        warmups = [r for r in result["warmups"] if r["batch"] == batch]
        warmup_hashes = []
        for row in warmups + measured:
            assert row["inputs_unchanged"] is True and row["sampling_unchanged"] is True
            assert row["seed"] == 42 and row["output_tokens"] == batch * 256
            assert row["input_before_sha256"] == row["input_after_sha256"] == row["returned_prompts_sha256"] == digest(prompts[:batch])
            assert row["normalized_sampling_sha256"] == digest({**result["sampling"], "stop": []})
            ids = row["request_ids"]
            assert ids == [x.partition("-")[0] for x in row["queued_request_ids"]]
            assert len(ids) == batch and len(set(ids)) == batch and not seen.intersection(ids)
            assert all(i.isdigit() for i in ids) and list(map(int, ids)) == sorted(map(int, ids))
            seen.update(ids)
            stem = "tokens" if row["stage"] == "measured" else "tokens-warmup"
            tokens = token_file(directory / f"{stem}-b{batch}-r{row['repeat']}.json", batch, row["output_token_ids_sha256"])
            assert digest([t[0] for t in tokens]) == row["first_token_ids_sha256"]
            out = {key: row[key] for key in ROW_FIELDS}
            if row["stage"] == "measured":
                assert tokens == reference
                seconds = row["seconds"]
                assert type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0
                out["seconds"] = seconds
                public_rows["trials"].append(out)
            else:
                assert "seconds" not in row
                warmup_hashes.append(digest(tokens))
                public_rows["warmups"].append(out)
        times = [r["seconds"] for r in measured]
        median = statistics.median(times)
        mad = statistics.median(abs(t - median) for t in times)
        cases.append({"batch": batch, "ordered_seconds": times, "median_seconds": median,
                      "min_seconds": min(times), "max_seconds": max(times),
                      "median_absolute_deviation_seconds": mad, "relative_mad": mad / median,
                      "measured_exact_reference_matches": 7, "measured_unique_output_hashes": 1,
                      "warmup_unique_output_hashes": len(set(warmup_hashes)),
                      "warmups_matching_measured_reference": sum(h == digest(reference) for h in warmup_hashes),
                      "cross_arm_reference": anchor_mode,
                      "cross_arm_exact_match": True if anchor_mode else None})

    assert [(w["rank"], w["gpu"], w["num_sms"]) for w in result["workers"]] == [
        (w["rank"], w["gpu"], w["num_sms"]) for w in v1["workers"]]
    after = {r["rank"]: r["statistics"] for r in result["tuner_after_generation"]}
    assert set(after) == {0, 1, 2, 3}
    assert result["cache_after_startup"] == result["cache_after_generation"]
    cache_hashes = {row["path"]: row["sha256"] for row in result["cache_after_generation"]}
    ranks = []
    for worker in result["workers"]:
        rank = worker["rank"]
        before = worker["flashinfer_statistics"]
        assert before == after[rank] and before["cache_misses"] == 0
        assert all(sum(before[field].values()) == 0 for field in ACTIVITY)
        assert worker["source_sha256"] == pin["source_hashes"][SOURCE_FILES[1]]
        assert worker["resolved_graph_mode"] == "FULL_DECODE_ONLY"
        assert worker["dispatch"] == [{"batch": b, "mode": "FULL", "padded_tokens": b} for b in (8, 40)]
        assert all(any(e["mode"] == "FULL" and e["num_tokens"] == b for e in worker["captured_entries"]) for b in (8, 40))
        sidecar = load(directory / f"cache-rank{rank}.json")
        path = sidecar["actual_file"]
        local = notes / Path(path).relative_to("/work")
        assert sha(local) == cache_hashes[path]
        configs = load(local)
        count = sum(not key.startswith("_") for key in configs)
        assert count == 160
        seed = next(r for r in v1["cache_after_generation"] if Path(r["path"]).name == Path(path).name)
        assert seed["sha256"] == cache_hashes[path]
        assert local.read_bytes() == (notes / Path(seed["path"]).relative_to("/work")).read_bytes()
        ranks.append({"rank_alias": f"rank_{rank}", "sms": worker["num_sms"],
                      "source_sha256": worker["source_sha256"], "resolved_graph_mode": "FULL_DECODE_ONLY",
                      "dispatch": worker["dispatch"],
                      "capture_entries": [{"mode": e["mode"], "num_tokens": e["num_tokens"]}
                                          for e in worker["captured_entries"]],
                      "tactic_cache_sha256": cache_hashes[path], "tactic_configurations": count,
                      "tactic_bytes_equal_v1_seed": True, "all_recorded_tuner_fields_unchanged": True,
                      "statistics_identity_unchanged": True, "cache_misses": 0,
                      "considered_configs": sum(before["tuned_op_total_configs"].values()),
                      "successful_profiles": 0, "failed_profiles": 0, "failed_tactics": 0})
    log = (directory / "run.log").read_text()
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log)
    loads = [(int(count), path) for count, path in re.findall(r"\[Autotuner\]: Loaded (\d+) configs from (\S+)", clean)]
    assert len(loads) == 4 and set(loads) == {(160, path) for path in cache_hashes}
    hits = Counter(re.findall(r"Config cache hit for (.+?) \(runner=([^,]+), source=config file\)", clean))
    assert len(hits) == 2 and all(value == 4 for value in hits.values())
    assert not re.findall(r"\[AutoTuner\]: Tuning [^\r\n]*:\s*\d+%", clean)
    check = load(directory / "cache-log-check.json")
    assert check["status"] == "passed" and check["log_sha256"] == sha(directory / "run.log")

    selected_settings = {key: result["settings"][key] for key in (
        "tensor_parallel_size", "dtype", "kv_cache_dtype", "block_size", "seed", "enforce_eager",
        "max_model_len", "max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes",
        "enable_chunked_prefill", "enable_prefix_caching", "disable_log_stats", "cudagraph_metrics",
        "attention_config", "compilation_config")}
    return {"status": "independently_audited", "mode": mode, "phase": "timing",
            "private_result_sha256": sha(directory / "result.json"),
            "private_log_sha256": sha(directory / "run.log"),
            "cache_log_check_sha256": sha(directory / "cache-log-check.json"),
            "started_at_utc": result["started_at_utc"], "finished_at_utc": result["finished_at_utc"],
            "source_commit": result["source_commit"], "source_hashes": result["source_hashes"],
            "admission_api_hashes": result["admission_api_hashes"],
            "source_git_bytes_verified": source_repo is not None,
            "harness_hashes": result["harness_hashes"], "protocol_sha256": result["protocol_sha256"],
            "model": "nvidia/MiniMax-M3-NVFP4", "model_revision": result["model_revision"],
            "prompts_sha256": result["prompts_sha256"], "input_len": 2048, "output_len": 256,
            "warmup_count": 3, "repeats": 7, "settings": selected_settings,
            "sampling": result["sampling"], "runtime_identical_to_v1": True,
            "ordered_gpu_identities_identical_to_v1": True,
            "profiling_enabled": False, "dispatch_recorder_installed": False,
            "logging_handler_installed": False, "shutdown_completed": True,
            "cache_log_audit": {"rank_loads": 4, "configurations_per_rank": 160,
                                "per_operation_runner_hits": sorted(hits.values()),
                                "tuning_progress_operations": 0},
            "workers": ranks, "cases": cases, **public_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert tuple(args.arms) == ARMS[:len(args.arms)], "Project an ordered completed prefix"
    arms = [audit_arm(args.notes, name, args.source_repo) for name in args.arms]
    report = {"schema_version": 1, "protocol": "admission-controlled-v2",
              "projector_sha256": sha(Path(__file__)), "arms": arms,
              "coverage": {"completed_timing_arms": args.arms, "all_five_present": len(arms) == 5},
              "performance_verdict": "not_evaluated_by_projection",
              "scope": "Explicit selected public fields audited against raw arrays, logs, actual tactic bytes and provenance. "
                       "No raw tokens, prompts, machine identities, private paths, request IDs or statistics object IDs. "
                       "No timing/native replay equivalence claim; the frozen five-arm comparator and separate diagnostic own final gates."}
    text = json.dumps(report, indent=2, allow_nan=False)
    assert not re.search(r"/Users/|/home/|/raid/|/work/|GPU-[0-9a-f]|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", text)
    with args.output.open("x") as stream:
        stream.write(text + "\n")
    print(json.dumps({"status": "passed", "arms": args.arms, "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
