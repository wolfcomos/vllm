"""Recompute v4 evidence, retaining every arm and the historical 3% criterion."""

import argparse
import json
from pathlib import Path
import statistics

from e2e_support import (
    ARMS, COMMITS, OUTPUT_LEN, PROTOCOL, cache_identity, digest, file_digest, load,
    normalized_settings, summary, tail_metrics, write_json,
)

METRICS = ("seconds", "mean_tpot_seconds", "mean_ttft_seconds", "common_tail_seconds", "common_tail_tpot_seconds")


def compare(paths):
    results = {arm: load(paths[arm]) for arm in ARMS}
    baseline = results["baseline"]
    summaries, tokens = {}, {}
    for arm in ARMS:
        result = results[arm]
        assert result["status"] == "complete" and result["shutdown_completed"] is True
        assert result["protocol"] == PROTOCOL and result["mode"] == arm
        source = "baseline" if arm in ("baseline", "cutlass") else "candidate"
        assert result["source_commit"] == COMMITS[source]
        assert (result["warmup_count"], result["repeats"], result["batches"], result["input_len"], result["output_len"]) == (5, 7, [8, 40], 2048, OUTPUT_LEN)
        assert normalized_settings(result["settings"]) == normalized_settings(baseline["settings"])
        assert result["settings"]["disable_log_stats"] is False
        attention = result["settings"]["attention_config"]
        assert attention.get("minimax_m3_msa_decode_fp16_dot", False) == arm.startswith("fp16")
        if arm == "cutlass":
            assert "minimax_m3_msa_decode_backend" not in attention
        else:
            assert attention["minimax_m3_msa_decode_backend"] == "triton"
        for key in ("model_revision", "manifest_sha256", "harness_hashes", "sampling", "prompts_sha256", "runtime", "api_hashes"):
            assert result[key] == baseline[key], ("Manifest/runtime differs", arm, key)
        expected_sources = baseline["source_hashes"] if source == "baseline" else results["default-a"]["source_hashes"]
        assert result["source_hashes"] == expected_sources
        if arm != "baseline":
            assert result["baseline_result_sha256"] == file_digest(paths["baseline"])
        assert result["cache_after_startup"] == result["cache_after_generation"]
        assert cache_identity(result["cache_after_generation"]) == cache_identity(baseline["cache_after_generation"])
        assert len(result["trials"]) == 14 and len(result["warmups"]) == 10
        assert sorted(worker["rank"] for worker in result["workers"]) == [0, 1, 2, 3]
        for worker in result["workers"]:
            assert worker["implementations"]
            assert all(impl["fp16_dot"] == arm.startswith("fp16") for impl in worker["implementations"])
            assert all(impl["use_cutlass_decode"] == (arm == "cutlass") for impl in worker["implementations"])
            assert worker["native_backend_policies"]
            for policy in worker["native_backend_policies"]:
                assert policy["min_batch_override"] is None
                shapes = {row["batch"]: row for row in policy["shapes"]}
                assert sorted(shapes) == list(range(1, 41))
                assert policy["geometry"]["decode_backend"] == ("cutlass" if arm == "cutlass" else "triton")
                if arm == "cutlass":
                    assert not shapes[8]["should_prepare_at_unpadded_batch"]
                    assert not shapes[8]["should_prepare_at_graph_shape"]
                    assert shapes[40]["should_prepare_at_unpadded_batch"]
                    assert shapes[40]["should_prepare_at_graph_shape"]
                else:
                    assert not any(row["should_prepare_at_unpadded_batch"] or row["should_prepare_at_graph_shape"]
                                   for row in policy["shapes"])
        summaries[arm], tokens[arm] = {}, {}
        for batch in (8, 40):
            trials = [row for row in result["trials"] if row["batch"] == batch]
            warmups = [row for row in result["warmups"] if row["batch"] == batch]
            assert [row["repeat"] for row in trials] == list(range(7))
            assert [row["repeat"] for row in warmups] == list(range(5))
            reference = load(paths[arm].parent / f"reference-b{batch}.json")
            assert digest(reference) == result["references"][str(batch)]
            tokens[arm][batch] = reference
            for row in warmups + trials:
                actual = load(paths[arm].parent / row["tokens_file"])
                assert digest(actual) == row["output_token_ids_sha256"]
                assert len(actual) == batch and all(len(sequence) == OUTPUT_LEN for sequence in actual)
                assert row["input_before_sha256"] == row["input_after_sha256"] == row["returned_prompts_sha256"]
                assert row["seconds"] > 0 and len(row["request_metrics"]) == batch
                for metric in row["request_metrics"]:
                    assert metric["num_generation_tokens"] == OUTPUT_LEN and metric["num_preemptions"] == 0
                    delta = metric["last_token_ts_monotonic"] - metric["first_token_ts_monotonic"]
                    assert delta > 0 and metric["mean_tpot_seconds"] == delta / (OUTPUT_LEN - 1)
                assert row["mean_tpot_seconds"] == statistics.mean(metric["mean_tpot_seconds"] for metric in row["request_metrics"])
                assert row["mean_ttft_seconds"] == statistics.mean(metric["ttft_seconds"] for metric in row["request_metrics"])
                recomputed = tail_metrics(row["request_metrics"])
                assert all(row[key] == value for key, value in recomputed.items()), "Common-tail summary differs"
                if row["stage"] == "measured":
                    assert actual == reference, ("Within-arm outputs vary", arm, batch)
            summaries[arm][str(batch)] = {metric: summary([row[metric] for row in trials]) for metric in METRICS}
    assert tokens["default-a"] == tokens["default-b"] == tokens["baseline"]
    assert tokens["fp16-a"] == tokens["fp16-b"]
    # CUTLASS has its own numerical path; no cross-backend exact-output requirement.
    for earlier, later in zip(ARMS, ARMS[1:]):
        assert results[earlier]["finished_at_utc"] <= results[later]["started_at_utc"]

    ratios, decisions = [], {}
    for batch in (8, 40):
        median = lambda arm, metric: summaries[arm][str(batch)][metric]["median"]
        control_drift = max(abs(median("default-a", "common_tail_seconds") / median("baseline", "common_tail_seconds") - 1),
                            abs(median("default-b", "common_tail_seconds") / median("default-a", "common_tail_seconds") - 1))
        primary, old_three_percent, noise = [], [], []
        for arm in ARMS:
            for metric in ("seconds", "common_tail_seconds"):
                value = summaries[arm][str(batch)][metric]["relative_mad"]
                noise.append({"name": f"{arm}/{metric}/MAD", "value": value, "passed": value <= .02})
        for numerator, denominator in (("default-a", "baseline"), ("default-b", "default-a"),
                                        ("fp16-b", "fp16-a"), ("cutlass", "baseline"),
                                        ("cutlass", "default-a"), ("cutlass", "default-b"),
                                        ("fp16-a", "cutlass"), ("fp16-b", "cutlass")):
            for metric in METRICS:
                ratio = median(numerator, metric) / median(denominator, metric)
                ratios.append({"batch": batch, "metric": metric, "numerator": numerator,
                               "denominator": denominator, "ratio": ratio, "percent_change": (ratio - 1) * 100})
        for fp16 in ("fp16-a", "fp16-b"):
            for default in ("default-a", "default-b"):
                tail_ratio = median(fp16, "common_tail_seconds") / median(default, "common_tail_seconds")
                wall_ratio = median(fp16, "seconds") / median(default, "seconds")
                benefit = 1 - tail_ratio
                primary.append({"fp16": fp16, "default": default, "tail_benefit": benefit,
                                "required_benefit_greater_than": 2 * control_drift,
                                "e2e_ratio": wall_ratio,
                                "passed": benefit > 0 and benefit > 2 * control_drift and wall_ratio <= 1.02})
                # Report the prior unadjusted mean-TPOT >=3% rule separately.
                tpot_ratio = median(fp16, "mean_tpot_seconds") / median(default, "mean_tpot_seconds")
                old_three_percent.append({"fp16": fp16, "default": default,
                                          "mean_tpot_ratio": tpot_ratio,
                                          "passed": tpot_ratio <= .97 and wall_ratio <= 1.02})
                for metric in METRICS:
                    ratio = median(fp16, metric) / median(default, metric)
                    ratios.append({"batch": batch, "metric": metric, "numerator": fp16,
                                   "denominator": default, "ratio": ratio, "percent_change": (ratio - 1) * 100})
        decisions[str(batch)] = {
            "maximum_control_tail_drift": control_drift,
            "primary_comparisons": primary,
            "review_evidence_gate_passed": all(row["passed"] for row in primary),
            "repeatability": noise, "repeatability_passed": all(row["passed"] for row in noise),
            "previous_3pct_mean_tpot_comparisons": old_three_percent,
            "previous_3pct_mean_tpot_rule_passed": all(row["passed"] for row in old_three_percent),
        }
    return {
        "status": "complete", "protocol": PROTOCOL, "integrity_passed": True,
        "result_sha256": {arm: file_digest(paths[arm]) for arm in ARMS},
        "statistics": summaries, "ratios": ratios, "decisions": decisions,
        "cutlass_output_hashes": results["cutlass"]["references"],
        "arm_labels": {arm: ("current-main default backend with native Triton fallback" if arm == "cutlass"
                             else "explicit Triton backend" + (" with FP16 opt-in" if arm.startswith("fp16") else ""))
                       for arm in ARMS},
        "current_default_backend_policies": [{"rank": worker["rank"], "policies": worker["native_backend_policies"]}
                                            for worker in results["cutlass"]["workers"]],
        "cutlass_matches_triton_baseline": {str(batch): tokens["cutlass"][batch] == tokens["baseline"][batch] for batch in (8, 40)},
        "scope": "Prospective draining-batch common tail, not fixed-batch per-step latency or serving p99 ITL. "
                 "The current-default arm uses native fallback: B8 does not select CUTLASS metadata; "
                 "draining B40 dispatch also depends on graph padding. Policy snapshots are not timed dispatch counts. "
                 "This is one late engine, so its ratios are reference evidence with order confounding. "
                 "No local gate is an upstream rule. Preserve the separate previous-3-percent results.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare({arm: args.results_root / arm / "result.json" for arm in ARMS})
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "integrity_passed": result["integrity_passed"],
                      "decisions": result["decisions"]}, indent=2))


if __name__ == "__main__":
    main()
