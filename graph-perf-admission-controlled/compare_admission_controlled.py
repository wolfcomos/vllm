"""Independent, fail-closed comparison for the predeclared admission-controlled v2."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

PROTOCOL = "admission-controlled-v2"
ARMS = ("baseline", "default-a", "fp16-a", "fp16-b", "default-b")
COMMITS = {"baseline": "9731f30a030c545e7f29a8a4ed96b8562be12c88",
           "candidate": "faf31b04c73400c2af770c280fb8d739a5997fcb"}
ACTIVITY = ("tuned_op_successful_configs", "failed_profiling_count", "failed_tactics")


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def settings_without_opt_in(settings):
    settings = json.loads(json.dumps(settings))
    settings["attention_config"].pop("minimax_m3_msa_decode_fp16_dot", None)
    return settings


def token_file(path, batch, expected_hash):
    tokens = load(path)
    assert len(tokens) == batch and all(len(row) == 256 for row in tokens)
    assert all(type(token) is int and token >= 0 for row in tokens for token in row)
    assert digest(tokens) == expected_hash, ("Token file hash mismatch", path.name)
    return tokens


def summarize(seconds):
    assert len(seconds) == 7 and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in seconds)
    median = statistics.median(seconds)
    mad = statistics.median(abs(value - median) for value in seconds)
    return {"ordered_seconds": seconds, "median_seconds": median,
            "min_seconds": min(seconds), "max_seconds": max(seconds),
            "median_absolute_deviation_seconds": mad, "relative_mad": mad / median,
            "sample_cv": statistics.stdev(seconds) / statistics.mean(seconds)}


def performance_gates(summaries):
    gates = []
    for arm in ARMS:
        for batch in (8, 40):
            value = summaries[arm][str(batch)]["relative_mad"]
            gates.append({"gate": "relative_mad", "arm": arm, "batch": batch,
                          "value": value, "limit": 0.02, "passed": value <= 0.02 + 1e-12})
    for left, right, name in (("default-a", "default-b", "default_anchor_drift"),
                              ("baseline", "default-a", "default_vs_baseline")):
        for batch in (8, 40):
            ratio = summaries[right][str(batch)]["median_seconds"] / summaries[left][str(batch)]["median_seconds"]
            gates.append({"gate": name, "batch": batch, "left": left, "right": right,
                          "median_latency_ratio": ratio, "absolute_relative_difference": abs(ratio - 1),
                          "limit": 0.02, "passed": abs(ratio - 1) <= 0.02 + 1e-12})
    for fp16 in ("fp16-a", "fp16-b"):
        for default in ("default-a", "default-b"):
            ratio = summaries[fp16]["40"]["median_seconds"] / summaries[default]["40"]["median_seconds"]
            gates.append({"gate": "B40_at_least_3_percent_lower_latency", "fp16": fp16,
                          "default": default, "batch": 40, "median_latency_ratio": ratio,
                          "limit": 0.97, "passed": ratio <= 0.97 + 1e-12})
    return gates


def compare(paths, prompts_path, baseline_v1_path, source_pins_path):
    assert len(paths) == 5
    arms = {name: load(path) for name, path in zip(ARMS, paths, strict=True)}
    witness = arms["baseline"]
    v1 = load(baseline_v1_path)
    pins = load(source_pins_path)
    prompts = load(prompts_path)
    assert (v1["status"], v1["mode"], v1["phase"]) == ("complete", "baseline", "timing")
    assert v1["source_commit"] == COMMITS["baseline"]
    assert v1["model_revision"] == "901464083161bf8612a29ff7ad29914cd4ab4a85"
    assert all(pins[key]["commit"] == value for key, value in COMMITS.items())
    assert pins["baseline"]["source_hashes"] == v1["source_hashes"]
    assert len(prompts["prompts"]) == 40 and all(len(row) == 2048 for row in prompts["prompts"])
    assert prompts["seed"] == 42 and prompts["input_len"] == 2048
    assert all(type(token) is int and 0 <= token < 10000 for row in prompts["prompts"] for token in row)
    assert digest(prompts["prompts"]) == prompts["prompts_sha256"] == v1["prompts_sha256"]
    protocol_hash = sha(Path(__file__).parent / "PROTOCOL.md")
    driver_hash = sha(Path(__file__).parent / "run_admission_controlled.py")
    expected_cache = {Path(row["path"]).name: row["sha256"] for row in v1["cache_after_generation"]}
    result_hashes = {name: sha(path) for name, path in zip(ARMS, paths, strict=True)}
    normalized_sampling = {**v1["sampling"], "stop": []}
    summaries, references, metadata = {}, {}, {}
    for index, (name, arm) in enumerate(arms.items()):
        directory = paths[index].parent
        assert (arm["status"], arm["protocol"], arm["mode"], arm["phase"]) == ("complete", PROTOCOL, name, "timing")
        assert arm["shutdown_completed"] is True
        assert arm["warmup_count"] == 3 and arm["repeats"] == 7
        assert arm["batches"] == [8, 40] and arm["input_len"] == 2048 and arm["output_len"] == 256
        assert arm["protocol_sha256"] == protocol_hash == arm["harness_hashes"]["PROTOCOL.md"]
        assert arm["harness_hashes"]["run_admission_controlled.py"] == driver_hash
        assert all(arm["harness_hashes"][key] == value for key, value in v1["harness_hashes"].items())
        assert arm["baseline_v1_result_sha256"] == sha(baseline_v1_path)
        assert arm["source_pins_sha256"] == sha(source_pins_path)
        for key in ("torch", "cuda", "packages", "model_revision", "sampling", "prompts_sha256"):
            assert arm[key] == v1[key], ("Different runtime/input control", name, key)
        for key in ("harness_hashes", "admission_api_hashes", "compilation_cache_environment", "interval"):
            assert arm[key] == witness[key], ("Different harness/API/control", name, key)
        assert settings_without_opt_in(arm["settings"]) == v1["settings"]
        assert arm["settings"]["attention_config"].get("minimax_m3_msa_decode_fp16_dot", False) == name.startswith("fp16")
        assert not arm["profiling_enabled"] and not arm["dispatch_recorder_installed"] and not arm["logging_handler_installed"]
        assert arm["settings"]["disable_log_stats"] is True and arm["settings"]["cudagraph_metrics"] is False
        pin = pins["baseline" if name == "baseline" else "candidate"]
        assert arm["source_commit"] == pin["commit"] and arm["source_hashes"] == pin["source_hashes"]
        if index:
            assert arm["witness_result_sha256"] == result_hashes["baseline"]
            assert arm["previous_result_sha256"] == result_hashes[ARMS[index - 1]]
            assert datetime.fromisoformat(arms[ARMS[index - 1]]["finished_at_utc"]) <= datetime.fromisoformat(arm["started_at_utc"])
        assert sorted(row["rank"] for row in arm["workers"]) == [0, 1, 2, 3]
        assert [(row["rank"], row["gpu"], row["num_sms"]) for row in arm["workers"]] == [
            (row["rank"], row["gpu"], row["num_sms"]) for row in v1["workers"]]
        final_stats = {row["rank"]: row["statistics"] for row in arm["tuner_after_generation"]}
        assert sorted(final_stats) == [0, 1, 2, 3]
        for worker in arm["workers"]:
            assert worker["source_sha256"] == pin["source_hashes"]["models/minimax_m3/common/ops/sparse_attn.py"]
            assert worker["resolved_graph_mode"] == "FULL_DECODE_ONLY"
            assert worker["dispatch"] == [{"batch": batch, "mode": "FULL", "padded_tokens": batch} for batch in (8, 40)]
            assert all(any(entry["mode"] == "FULL" and entry["num_tokens"] == batch for entry in worker["captured_entries"]) for batch in (8, 40))
            before, after = worker["flashinfer_statistics"], final_stats[worker["rank"]]
            assert before["statistics_object_id"] == after["statistics_object_id"]
            for field in ACTIVITY:
                assert all(value == 0 for value in before[field].values()) and before[field] == after[field]
        assert arm["cache_after_startup"] == arm["cache_after_generation"]
        assert {Path(row["path"]).name: row["sha256"] for row in arm["cache_after_generation"]} == expected_cache
        log_check = load(directory / "cache-log-check.json")
        assert log_check["status"] == "passed" and log_check["seeded"] is True and log_check["mode"] == name
        assert log_check["phase"] == "timing" and log_check["log_sha256"] == sha(directory / "run.log")
        assert not log_check["tuning_progress_operations"]
        assert len(log_check["load_messages"]) == 4
        assert set(log_check["expected_cache_files"].values()) == {160}
        assert len(log_check["expected_cache_files"]) == 4
        assert log_check["per_operation_runner_hits"] and all(row["count"] == 4 for row in log_check["per_operation_runner_hits"])
        if index:
            assert log_check["per_operation_runner_hits"] == load(paths[0].parent / "cache-log-check.json")["per_operation_runner_hits"]
        assert [(row["batch"], row["repeat"], row["stage"]) for row in arm["trials"]] == [(batch, i, "measured") for batch in (8, 40) for i in range(7)]
        assert [(row["batch"], row["repeat"], row["stage"]) for row in arm["warmups"]] == [(batch, i, "warmup") for batch in (8, 40) for i in range(3)]
        summaries[name], references[name], seen_ids = {}, {}, set()
        for batch in (8, 40):
            reference = token_file(directory / f"reference-b{batch}.json", batch, arm["references"][str(batch)])
            references[name][batch] = reference
            input_hash = digest(prompts["prompts"][:batch])
            measured = [row for row in arm["trials"] if row["batch"] == batch]
            for row in [row for row in arm["warmups"] if row["batch"] == batch] + measured:
                assert row["seed"] == 42 and row["output_tokens"] == batch * 256
                assert row["inputs_unchanged"] is True and row["sampling_unchanged"] is True
                assert row["input_before_sha256"] == row["input_after_sha256"] == row["returned_prompts_sha256"] == input_hash
                assert row["normalized_sampling_sha256"] == digest(normalized_sampling)
                ids = row["request_ids"]
                assert len(ids) == batch and all(value.isdigit() for value in ids)
                assert ids == [value.partition("-")[0] for value in row["queued_request_ids"]]
                assert len(set(ids)) == batch and not seen_ids.intersection(ids)
                assert list(map(int, ids)) == sorted(map(int, ids))
                seen_ids.update(ids)
                stem = "tokens" if row["stage"] == "measured" else "tokens-warmup"
                tokens = token_file(directory / f"{stem}-b{batch}-r{row['repeat']}.json", batch, row["output_token_ids_sha256"])
                assert digest([token[0] for token in tokens]) == row["first_token_ids_sha256"]
                if row["stage"] == "measured":
                    assert tokens == reference, ("Within-arm output variation", name, batch, row["repeat"])
            summaries[name][str(batch)] = summarize([row["seconds"] for row in measured])
        metadata[name] = {key: arm[key] for key in ("source_commit", "started_at_utc", "finished_at_utc")}
    for batch in (8, 40):
        assert references["baseline"][batch] == references["default-a"][batch] == references["default-b"][batch], ("Default token parity failed", batch)
        assert references["fp16-a"][batch] == references["fp16-b"][batch], ("FP16 repeat token parity failed", batch)
    gates = performance_gates(summaries)
    passed = all(gate["passed"] for gate in gates)
    return {"protocol": PROTOCOL, "protocol_sha256": protocol_hash,
            "comparator_sha256": sha(Path(__file__)), "result_sha256": result_hashes,
            "status": "complete" if passed else "failed", "passed": passed,
            "integrity_passed": True, "performance_gates": gates,
            "arms": metadata, "workloads": summaries,
            "interpretation": "Five ordered engines with admission control; all seven trials retained. B40 claim requires every declared gate. B8 is reported even if slower. Native runtime graph diagnostics remain a separate requirement; this comparator does not establish hardware replay."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs=5, type=Path)
    for name in ("prompts", "baseline-v1-result", "source-pins", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(args.results, args.prompts, args.baseline_v1_result, args.source_pins)
    except (AssertionError, KeyError, ValueError, TypeError, OSError) as error:
        report = {"protocol": PROTOCOL, "status": "failed", "passed": False,
                  "integrity_passed": False, "failure": str(error),
                  "comparator_sha256": sha(Path(__file__))}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("protocol", "status", "passed", "integrity_passed")}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
