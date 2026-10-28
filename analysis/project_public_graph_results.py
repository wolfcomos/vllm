"""Project completed local graph evidence through an explicit public whitelist.

This does not launch workloads, modify inputs, or certify performance/readiness.
The frozen comparator, drift review and separate replay trace audit remain required.
"""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

MODES = ("baseline", "default", "fp16", "default-repeat")
SOURCE_FILES = (
    "config/attention.py", "models/minimax_m3/common/ops/sparse_attn.py",
    "models/minimax_m3/common/sparse_attention.py",
    "models/minimax_m3/nvidia/model.py",
    "models/minimax_m3/nvidia/sparse_attention_msa.py",
)
ACTIVITY = ("tuned_op_successful_configs", "failed_profiling_count", "failed_tactics")


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_value(value, length=64):
    assert isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value)
    return value


def number(value, integer=False):
    assert type(value) in ((int,) if integer else (int, float))
    assert math.isfinite(value) and value >= 0
    return value


def boolean(value):
    assert type(value) is bool
    return value


def choice(value, allowed):
    assert value in allowed
    return value


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    return parsed.isoformat()


def counts(stats):
    # Operation names and statistics object IDs are intentionally omitted.
    return {"cache_misses": number(stats["cache_misses"], True), **{
        field: sum(number(value, True) for value in stats[field].values())
        for field in ("tuned_op_total_configs", *ACTIVITY)
    }}


def graph_rows(rows, fields):
    return [{key: choice(row[key], ("FULL", "PIECEWISE", "NONE")) if key == "mode"
             else number(row[key], True) for key in fields} for row in rows]


def project(result_path):
    result = load(result_path)
    assert result["status"] == "complete", "Only completed arms may be projected"
    mode = choice(result["mode"], MODES)
    phase = choice(result["phase"], ("timing", "diagnostic"))
    check_path = result_path.parent / "cache-log-check.json"
    check = load(check_path)
    assert check["status"] == "passed" and check["mode"] == mode and check["phase"] == phase
    # These sidecars establish rank association without exposing file names.
    rank_records = [load(path) for path in result_path.parent.glob("cache-rank*.json")]
    assert sorted(row["rank"] for row in rank_records) == [0, 1, 2, 3]
    paths_by_rank = {row["rank"]: row["actual_file"] for row in rank_records}
    startup = {row["path"]: hash_value(row["sha256"]) for row in result["cache_after_startup"]}
    final = {row["path"]: hash_value(row["sha256"]) for row in result["cache_after_generation"]}
    assert set(startup) == set(final) == set(paths_by_rank.values())
    assert set(check["expected_cache_files"]) == set(final)
    after = {row["rank"]: row["statistics"] for row in result["tuner_after_generation"]}
    assert sorted(after) == [0, 1, 2, 3]
    assert sorted(worker["rank"] for worker in result["workers"]) == [0, 1, 2, 3]
    workers = []
    for worker in sorted(result["workers"], key=lambda row: row["rank"]):
        rank = worker["rank"]
        before_stats = worker["flashinfer_statistics"]
        after_stats = after[rank]
        path = paths_by_rank[rank]
        workers.append({
            "rank_alias": f"rank_{rank}", "num_sms": number(worker["num_sms"], True),
            "source_sha256": hash_value(worker["source_sha256"]),
            "resolved_graph_mode": choice(worker["resolved_graph_mode"], ("FULL_DECODE_ONLY",)),
            "dispatch": graph_rows(worker["dispatch"], ("batch", "mode", "padded_tokens")),
            "captured_entries": graph_rows(worker["captured_entries"], ("mode", "num_tokens")),
            "tactic_cache": {
                "startup_sha256": startup[path], "after_generation_sha256": final[path],
                "hash_unchanged": startup[path] == final[path],
                "config_count": number(check["expected_cache_files"][path], True),
            },
            "tuner": {
                "startup": counts(before_stats), "after_generation": counts(after_stats),
                "statistics_identity_stable": before_stats["statistics_object_id"] == after_stats["statistics_object_id"],
                "activity_counters_unchanged": all(before_stats[key] == after_stats[key] for key in ACTIVITY),
            },
        })
    settings = result["settings"]
    compilation = settings["compilation_config"]
    attention = settings["attention_config"]
    projected_settings = {
        key: number(settings[key], True) for key in (
            "tensor_parallel_size", "block_size", "seed", "max_model_len", "max_num_seqs",
            "max_num_batched_tokens", "kv_cache_memory_bytes",
        )
    }
    projected_settings.update({
        key: boolean(settings[key]) for key in (
            "enforce_eager", "enable_chunked_prefill", "enable_prefix_caching", "cudagraph_metrics",
        )
    })
    projected_settings.update({
        "dtype": choice(settings["dtype"], ("bfloat16",)),
        "kv_cache_dtype": choice(settings["kv_cache_dtype"], ("fp8",)),
        "decode_backend": choice(attention["minimax_m3_msa_decode_backend"], ("triton",)),
        "indexer_kv_dtype": choice(attention["indexer_kv_dtype"], ("fp8",)),
        # Absence here is the documented vLLM default; counters never use defaults.
        "fp16_dot_opt_in": boolean(attention.get("minimax_m3_msa_decode_fp16_dot", False)),
        "compilation_mode": number(compilation["mode"], True),
        "cudagraph_mode": choice(compilation["cudagraph_mode"], ("FULL_DECODE_ONLY",)),
        "capture_sizes": [number(value, True) for value in compilation["cudagraph_capture_sizes"]],
        "capture_warmups": number(compilation["cudagraph_num_of_warmups"], True),
    })
    trials = [{
        **{key: number(row[key], key in ("batch", "repeat", "seed", "output_tokens")) for key in (
            "batch", "repeat", "seed", "seconds", "output_tokens", "output_tokens_per_second",
        )}, "output_token_ids_sha256": hash_value(row["output_token_ids_sha256"]),
    } for row in result["trials"]]
    projected = {
        "status": "complete", "mode": mode, "phase": phase,
        "private_result_sha256": sha(result_path), "cache_log_check_sha256": sha(check_path),
        "started_at_utc": timestamp(result["started_at_utc"]),
        "finished_at_utc": timestamp(result["finished_at_utc"]),
        "source_commit": hash_value(result["source_commit"], 40),
        "model_revision": hash_value(result["model_revision"], 40),
        "prompts_sha256": hash_value(result["prompts_sha256"]),
        "source_hashes": {name: hash_value(result["source_hashes"][name]) for name in SOURCE_FILES},
        "harness_hashes": {name: hash_value(result["harness_hashes"][name]) for name in (
            "run_graph_perf.py", "graph_perf_cache_seed.py",
        )},
        "settings": projected_settings,
        **{key: number(result[key], True) for key in ("input_len", "output_len", "warmups", "repeats")},
        "batches": [number(value, True) for value in result["batches"]],
        "startup_seconds": number(result["startup_seconds"]),
        "profiling_enabled": boolean(result["profiling_enabled"]),
        "dispatch_recorder_installed": boolean(result["dispatch_recorder_installed"]),
        "trials": trials, "workers": workers,
        "cache_log_evidence": {
            "status": "passed", "seeded": boolean(check["seeded"]),
            "load_message_count": len(check["load_messages"]),
            "save_message_count": len(check["save_messages"]),
            "per_operation_runner_hit_counts": [number(row["count"], True) for row in check["per_operation_runner_hits"]],
            "tuning_progress_operation_count": len(check["tuning_progress_operations"]),
        },
        "diagnostics": [{
            "batch": number(row["batch"], True),
            "unpadded_full_graph_steps": number(row["unpadded_full_graph_steps"], True),
            "native_graph_metric_rows": graph_rows(row["native_graph_metric_rows"], (
                "unpadded", "padded", "paddings", "mode", "count",
            )),
        } for row in result["diagnostics"]],
    }
    if phase == "timing":
        projected["measured_seconds"] = number(result["measured_seconds"])
    if result["profiling_enabled"]:
        profiler = settings["profiler_config"]
        projected["profiler"] = {
            "type": choice(profiler["profiler"], ("cuda",)),
            "delay_iterations": number(profiler["delay_iterations"], True),
            "max_iterations": number(profiler["max_iterations"], True),
        }
    return projected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    arms = [project(path) for path in args.results]
    keys = [(arm["phase"], arm["mode"]) for arm in arms]
    assert len(set(keys)) == len(keys), "Duplicate phase/mode evidence"
    report = {
        "schema_version": 1, "projector_sha256": sha(Path(__file__)), "arms": arms,
        "coverage": {
            "all_four_timing_arms_present": all(("timing", mode) in keys for mode in MODES),
            "diagnostic_modes_present": [mode for mode in MODES if ("diagnostic", mode) in keys],
        },
        "performance_verdict": "not_evaluated",
        "publication_readiness": "requires_independent_audit",
        "required_final_checks": [
            "All four ordered timing arms and the canonical comparator must pass.",
            "Baseline/default and default/default-repeat token parity must pass; review default-repeat drift.",
            "Audit separate graph diagnostics; captured entries alone do not prove replay.",
            "Nsight cudaGraphLaunch and correlated kernel evidence, if collected, require a separate trace audit.",
        ],
        "scope": "Partial projection is allowed for completed arms. No runtime strings, private paths, cache file names, object IDs or raw tokens are included. Missing required evidence fails projection; no missing counter is converted to zero. Native metrics describe runtime mode selection; this projector does not verify a CUDA trace.",
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "projected", "arm_count": len(arms), **report["coverage"]}))


if __name__ == "__main__":
    main()
