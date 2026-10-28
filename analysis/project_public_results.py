"""Build a whitelisted public projection of complete private GSM8K evidence.

This analysis script never launches models or modifies the executed harnesses.
It publishes hashes and aggregate/shape evidence, never raw generation content.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

from independent_rescore import audit


MODES = {"baseline": "baseline", "bf16": "default", "fp16": "fp16"}
SOURCE_FILES = (
    "config/attention.py", "models/minimax_m3/common/ops/sparse_attn.py",
    "models/minimax_m3/common/sparse_attention.py", "models/minimax_m3/nvidia/model.py",
    "models/minimax_m3/nvidia/sparse_attention_msa.py",
)
SETTING_KEYS = (
    "language_model_only", "tensor_parallel_size", "distributed_executor_backend",
    "dtype", "kv_cache_dtype", "block_size", "seed", "enforce_eager", "max_model_len",
    "max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes",
    "enable_chunked_prefill", "enable_prefix_caching", "generation_config",
)
ACTIVITY = ("tuned_op_successful_configs", "failed_profiling_count", "failed_tactics")
DATA_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/ci-datasets/gsm8k/"
CRITERION = {
    "baseline_sanity_floor": 0.80, "max_accuracy_drop": 0.005,
    "max_invalid_rate": 0.01, "max_truncation_rate": 0.01,
    "default_exact_output_tokens": True,
}


def read(path):
    return json.loads(path.read_text())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_rank(path):
    match = re.match(r"autotune_configs_dp0_rank([0-9]+)", Path(path).name)
    assert match
    return int(match.group(1))


def relative_cache(path):
    return Path(path).relative_to("/work/cache/flashinfer")


def activity(stats):
    return {name: sum(stats[name].values()) for name in ACTIVITY}


def project_rank(notes, result, baseline, mode, rank):
    worker = next(w for w in result["workers"] if w["rank"] == rank)
    base_worker = next(w for w in baseline["workers"] if w["rank"] == rank)
    assert worker["gpu"] == base_worker["gpu"]
    assert worker["source_sha256"] == result["source_hashes"][SOURCE_FILES[1]]
    before = next(w for w in result["autotuner_statistics_after_startup"] if w["rank"] == rank)
    after = next(w for w in result["autotuner_statistics_after_generation"] if w["rank"] == rank)
    assert before["statistics_identity"] == after["statistics_identity"]
    assert all(before[name] == after[name] for name in ACTIVITY)
    if mode != "baseline":
        assert not any(activity(before).values())

    base_files = baseline["autotune_cache_after_generation"]
    matches = [(name, sha) for name, sha in base_files.items() if cache_rank(name) == rank]
    assert len(matches) == 1
    cache_path, expected = matches[0]
    load_count = None
    if mode != "baseline":
        seed = next(s for s in result["validation_cache_seed"] if s["rank"] == rank)
        # The resolver returns the base name; FlashInfer appends DP/rank suffixes.
        seeded = [f for f in seed["files"] if cache_rank(f["destination"]) == rank
                  and Path(f["destination"]).parent == Path(seed["resolver_returned"]).parent]
        assert len(seeded) == 1
        item = seeded[0]
        assert item["sha256"] == item["after_startup_sha256"] == item["after_generation_sha256"] == expected
        assert cache_rank(item["destination"]) == rank
        cache_path = relative_cache(item["destination"])
        log_audit = read(notes / "gsm8k" / mode / "cache-log-verification.json")
        assert log_audit["passed"] and len(log_audit["loaded_files"]) == 4
        load_count = log_audit["loaded_files"][item["destination"]]
        hits = [op for op, _ in log_audit["cache_hits"]]
        assert hits.count("mxfp8_gemm") == hits.count("flashinfer::trtllm_fp4_block_scale_moe") == 4
    raw_path = notes / "cache/flashinfer" / cache_path
    assert file_hash(raw_path) == expected
    operation_count = sum(not key.startswith("_") for key in read(raw_path))
    assert load_count is None or load_count == operation_count

    initial = next(w for w in result["smoke_dispatch"] if w["rank"] == rank)
    final = next(w for w in result["dispatch"] if w["rank"] == rank)
    full = next(w for w in result["full_dispatch"] if w["rank"] == rank)
    assert full["launches"] == final["launches"] - initial["launches"] == sum(full["variants"].values())
    variants = []
    for raw, count in full["variants"].items():
        m = json.loads(raw)
        assert count == final["variants"][raw] - initial["variants"].get(raw, 0)
        assert m["USE_FP16_DOT"] == (mode == "fp16") and m["USE_FP8"]
        assert m["q_dtype"] == "torch.bfloat16"
        denom = max(1, m["total_q"] * m["num_kv_heads"])
        old_target = max(1, min(m["topk"], 256 // denom))
        original = 1 << (old_target.bit_length() - 1)
        target = max(1, min(m["topk"], 2 * worker["num_sms"] // denom))
        blocks = (m["topk"] + target - 1) // target
        tuned = (m["topk"] + blocks - 1) // blocks
        assert m["NUM_TOPK_CHUNKS"] == (tuned if mode == "fp16" else original)
        variants.append({
            "launches": count, "grid": m["grid"], "q_shape": m["q_shape"],
            "q_dtype": "bfloat16", "total_q": m["total_q"],
            "num_kv_heads": m["num_kv_heads"], "topk": m["topk"],
            "use_fp8": True, "use_fp16_dot": mode == "fp16",
            "kv_scale_mode": m["KV_SCALE_MODE"], "num_topk_chunks": m["NUM_TOPK_CHUNKS"],
        })
    return {
        "rank_alias": f"rank_{rank}", "num_sms": worker["num_sms"],
        "source_sha256": worker["source_sha256"],
        "same_device_as_fresh_baseline": True,
        "dispatch": {"full_evaluation_launches": full["launches"],
                     "pilot_subtracted": True, "policy_verified_for_every_variant": True,
                     "variants": variants},
        "cache": {"sha256": expected, "operation_configurations": operation_count,
                  "matches_fresh_baseline": True, "startup_to_generation_unchanged": True,
                  "loaded_configurations": load_count},
        "tuner": {"startup_activity": activity(before), "after_generation_activity": activity(after),
                  "generation_activity_unchanged": True, "statistics_identity_stable": True,
                  "seeded_startup_activity_zero": None if mode == "baseline" else True},
    }


def project(notes):
    private, rows, scored = {}, {}, {}
    pins = read(notes / "attention-tests/source-provenance.json")
    manifest = read(notes / "gsm8k/dataset-manifest.json")
    for mode in MODES:
        arm = notes / "gsm8k" / mode
        private[mode] = read(arm / "result.json")
        rows[mode] = [json.loads(line) for line in (arm / "predictions.jsonl").read_text().splitlines()]
        scored[mode] = audit(arm, notes)
        result = private[mode]
        assert result["criterion"] == CRITERION
        assert result["source_commit"] == pins["baseline" if mode == "baseline" else "candidate"]["commit"]
        assert result["source_hashes"] == pins["baseline" if mode == "baseline" else "candidate"]["source_hashes"]
        assert sorted(w["rank"] for w in result["workers"]) == list(range(4))
        assert result["autotune_cache_after_startup"] == result["autotune_cache_after_generation"]
    base = private["baseline"]
    protocol = {key: base["settings"][key] for key in SETTING_KEYS}
    assert protocol["enforce_eager"]
    report = {
        "schema_version": 1,
        "scope": "Pretrained GSM8K quality validation in eager mode; no performance claim.",
        "model": {"repository": "nvidia/MiniMax-M3-NVFP4", "revision": base["settings"]["revision"],
                  "tokenizer_revision": base["settings"]["tokenizer_revision"]},
        "dataset": {"train_examples": 7473, "test_examples": 1319, "fewshot_train_indices": list(range(5)),
                    "files": {name: {"url": DATA_URL + name, "sha256": sha} for name, sha in manifest["files"].items()},
                    "prompts_sha256": manifest["prompts_sha256"], "labels_sha256": manifest["labels_sha256"]},
        "protocol": protocol, "sampling": {k: base["sampling"][k] for k in ("temperature", "seed", "max_tokens", "stop")},
        "attention": {"minimax_m3_msa_decode_backend": "triton", "indexer_kv_dtype": "fp8"},
        "thinking_mode": "disabled", "criterion": CRITERION, "arms": {}, "paired": {},
    }
    for mode, public_name in MODES.items():
        result, score = private[mode], scored[mode]
        assert {key: result["settings"][key] for key in SETTING_KEYS} == protocol
        assert result["settings"]["revision"] == report["model"]["revision"]
        assert result["sampling"] == base["sampling"] and result["chat_template_kwargs"] == {"thinking_mode": "disabled"}
        assert result["dataset"]["files"] == report["dataset"]["files"]
        assert all(result["settings"]["attention_config"][key] == value for key, value in report["attention"].items())
        assert result["settings"]["attention_config"].get("minimax_m3_msa_decode_fp16_dot", False) == (mode == "fp16")
        public_score = {key: score[key] for key in ("questions", "correct", "accuracy", "invalid", "truncated", "output_tokens", "predictions_sha256", "output_token_sequences_sha256")}
        report["arms"][public_name] = {
            "source_commit": result["source_commit"],
            "requested_fp16_dot": mode == "fp16",
            "source_hashes": {name: result["source_hashes"][name] for name in SOURCE_FILES},
            "metrics": public_score,
            "ranks": [project_rank(notes, result, base, mode, rank) for rank in range(4)],
        }
        if mode == "baseline":
            continue
        lost, gained, same_tokens, changed_answers = [], [], 0, 0
        for old, new in zip(rows["baseline"], rows[mode], strict=True):
            assert (old["id"], old["prompt_token_ids"], old["label"]) == (new["id"], new["prompt_token_ids"], new["label"])
            same_tokens += old["token_ids"] == new["token_ids"]
            changed_answers += old["prediction"] != new["prediction"]
            if old["correct"] and not new["correct"]:
                lost.append(old["id"])
            if new["correct"] and not old["correct"]:
                gained.append(old["id"])
        drop = (scored["baseline"]["correct"] - score["correct"]) / 1319
        gate = drop <= 0.005 and score["invalid"] / 1319 <= 0.01 and score["truncated"] / 1319 <= 0.01
        if mode == "bf16":
            gate &= same_tokens == 1319
        report["paired"][public_name] = {
            "lost_correct": len(lost), "gained_correct": len(gained),
            "lost_example_ids": lost, "gained_example_ids": gained,
            "same_output_token_sequences": same_tokens,
            "changed_answer_predictions": changed_answers,
            "accuracy_delta_percentage_points": -100 * drop,
            "local_quality_gate_passed": gate,
        }
    base_score = scored["baseline"]
    report["baseline_sanity_gate_passed"] = base_score["accuracy"] >= 0.80 and base_score["invalid"] / 1319 <= 0.01 and base_score["truncated"] / 1319 <= 0.01
    report["passed"] = report["baseline_sanity_gate_passed"] and all(v["local_quality_gate_passed"] for v in report["paired"].values())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = project(args.notes)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "arms": list(report["arms"])}))
