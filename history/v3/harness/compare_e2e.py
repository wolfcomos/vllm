"""Offline integrity checks and balanced-arm E2E/TPOT comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from run_e2e import ARMS, COMMITS, PROTOCOL, normalized_settings


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summary(values):
    median = statistics.median(values)
    return {"values": values, "median": median, "min": min(values), "max": max(values),
            "relative_mad": statistics.median(abs(value - median) for value in values) / median}


def compare(paths):
    results = {arm: json.loads(paths[arm].read_text()) for arm in ARMS}
    baseline = results["baseline"]
    checks = []

    def check(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})

    tokens = {}
    stats = {}
    for arm in ARMS:
        result = results[arm]
        assert result["status"] == "complete" and result["shutdown_completed"] is True
        assert result["protocol"] == PROTOCOL and result["mode"] == arm
        assert result["source_commit"] == COMMITS["baseline" if arm == "baseline" else "candidate"]
        assert (result["warmup_count"], result["repeats"], result["batches"], result["input_len"], result["output_len"]) == (5, 7, [8, 40], 2048, 256)
        assert normalized_settings(result["settings"]) == normalized_settings(baseline["settings"])
        assert result["settings"]["disable_log_stats"] is False
        assert result["settings"]["attention_config"].get("minimax_m3_msa_decode_fp16_dot", False) == arm.startswith("fp16")
        for key in ("model_revision", "source_pins_sha256", "tactic_witness_sha256", "harness_hashes", "sampling", "prompts_sha256", "torch", "cuda", "packages", "api_hashes"):
            assert result[key] == baseline[key], ("Manifest differs", arm, key)
        if arm != "baseline":
            assert result["source_hashes"] == results["default-a"]["source_hashes"]
            assert result["baseline_result_sha256"] == hashlib.sha256(paths["baseline"].read_bytes()).hexdigest()
        assert result["cache_after_startup"] == result["cache_after_generation"]
        cache_hashes = lambda record: {Path(item["path"]).name: item["sha256"] for item in record["cache_after_generation"]}
        assert cache_hashes(result) == cache_hashes(baseline)
        assert len(result["trials"]) == 14 and len(result["warmups"]) == 10
        tokens[arm], stats[arm] = {}, {}
        for batch in (8, 40):
            trials = [row for row in result["trials"] if row["batch"] == batch]
            warmups = [row for row in result["warmups"] if row["batch"] == batch]
            assert [row["repeat"] for row in trials] == list(range(7))
            assert [row["repeat"] for row in warmups] == list(range(5))
            reference = json.loads((paths[arm].parent / f"reference-b{batch}.json").read_text())
            assert digest(reference) == result["references"][str(batch)]
            tokens[arm][batch] = reference
            for row in warmups + trials:
                actual = json.loads((paths[arm].parent / row["tokens_file"]).read_text())
                assert digest(actual) == row["output_token_ids_sha256"]
                assert len(actual) == batch and all(len(seq) == 256 for seq in actual)
                assert row["input_before_sha256"] == row["input_after_sha256"] == row["returned_prompts_sha256"]
                assert row["seconds"] > 0 and len(row["request_metrics"]) == batch
                for metric in row["request_metrics"]:
                    assert metric["num_generation_tokens"] == 256 and metric["num_preemptions"] == 0
                    decode = metric["last_token_ts_monotonic"] - metric["first_token_ts_monotonic"]
                    assert decode > 0 and metric["mean_tpot_seconds"] == decode / 255
                assert row["mean_tpot_seconds"] == statistics.mean(m["mean_tpot_seconds"] for m in row["request_metrics"])
                assert row["mean_ttft_seconds"] == statistics.mean(m["ttft_seconds"] for m in row["request_metrics"])
                if row["stage"] == "measured":
                    assert actual == reference, ("Within-arm output mismatch", arm, batch)
            stats[arm][str(batch)] = {metric: summary([row[metric] for row in trials])
                                      for metric in ("seconds", "mean_tpot_seconds", "mean_ttft_seconds")}
            for metric in ("seconds", "mean_tpot_seconds"):
                value = stats[arm][str(batch)][metric]["relative_mad"]
                check(f"{arm}/B{batch}/{metric}/MAD", value <= .02, value=value, maximum=.02)
    for arm in ("default-a", "default-b"):
        assert tokens[arm] == tokens["baseline"], ("Default/baseline outputs differ", arm)
    assert tokens["fp16-a"] == tokens["fp16-b"], "FP16 repeat outputs differ"
    for earlier, later in zip(ARMS, ARMS[1:]):
        assert results[earlier]["finished_at_utc"] <= results[later]["started_at_utc"], "Arm order/overlap differs"

    ratios = []
    for batch in (8, 40):
        for metric in ("seconds", "mean_tpot_seconds", "mean_ttft_seconds"):
            median = lambda arm: stats[arm][str(batch)][metric]["median"]
            for later, earlier in (("default-a", "baseline"), ("default-b", "default-a"),
                                   ("fp16-b", "fp16-a")):
                ratio = median(later) / median(earlier)
                ratios.append({"batch": batch, "metric": metric, "numerator": later,
                               "denominator": earlier, "ratio": ratio, "percent_change": (ratio - 1) * 100})
                if later.startswith("default") and metric != "mean_ttft_seconds":
                    check(f"B{batch}/{metric}/{later}:{earlier}/control", abs(ratio - 1) <= .02,
                          ratio=ratio, maximum_absolute_change=.02)
            for fp16 in ("fp16-a", "fp16-b"):
                for default in ("default-a", "default-b"):
                    ratio = median(fp16) / median(default)
                    ratios.append({"batch": batch, "metric": metric, "numerator": fp16,
                                   "denominator": default, "ratio": ratio, "percent_change": (ratio - 1) * 100})
                    if batch == 40 and metric == "mean_tpot_seconds":
                        check(f"B40/TPOT/{fp16}:{default}/benefit", ratio <= .97,
                              ratio=ratio, maximum=.97)
                    if batch == 40 and metric == "seconds":
                        check(f"B40/E2E/{fp16}:{default}/no-regression", ratio <= 1.02,
                              ratio=ratio, maximum=1.02)
    return {
        "status": "complete", "protocol": PROTOCOL, "integrity_passed": True,
        "performance_passed": all(item["passed"] for item in checks),
        "checks": checks, "failed_checks": [item for item in checks if not item["passed"]],
        "result_sha256": {arm: hashlib.sha256(paths[arm].read_bytes()).hexdigest() for arm in ARMS},
        "statistics": stats, "ratios": ratios,
        "interpretation": "Offline full-model fixed-work latency. Mean TPOT includes scheduler/overlap time. "
                          "Local gates are not upstream policy. No serving QPS or p99 ITL inference. "
                          "GPU telemetry and exact physical GPU identity require separate orchestration audit.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare({arm: args.results_root / arm / "result.json" for arm in ARMS})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "integrity_passed", "performance_passed", "failed_checks")}, indent=2))


if __name__ == "__main__":
    main()
