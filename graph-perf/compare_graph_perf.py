"""Validate all graph timing arms and report ordered, workload-matched ratios."""

import argparse
import json
from pathlib import Path
import statistics

from run_graph_perf import (
    BATCHES, canonical_settings, digest, OUTPUT_LEN,
    validate_tuner_startup, validate_tuner_after_generation,
)


def compare(paths):
    names = ("baseline", "default", "fp16", "default-repeat")
    arms = {name: json.loads(path.read_text()) for name, path in zip(names, paths, strict=True)}
    baseline = arms["baseline"]
    for name, arm in arms.items():
        assert arm["status"] == "complete" and arm["phase"] == "timing" and arm["mode"] == name
        assert arm["repeats"] == baseline["repeats"] and arm["repeats"] in (3, 5)
        assert arm["warmups"] == 2
        assert not arm["profiling_enabled"] and not arm["dispatch_recorder_installed"]
        assert arm["prompts_sha256"] == baseline["prompts_sha256"]
        assert arm["sampling"] == baseline["sampling"]
        for key in ("torch", "cuda", "packages", "harness_hashes", "model_revision", "input_len", "output_len", "batches"):
            assert arm[key] == baseline[key], f"Different {key}: {name}"
        assert canonical_settings(arm["settings"]) == canonical_settings(baseline["settings"])
        assert arm["cache_after_startup"] == arm["cache_after_generation"]
        assert {Path(row["path"]).name: row["sha256"] for row in arm["cache_after_generation"]} == {
            Path(row["path"]).name: row["sha256"] for row in baseline["cache_after_generation"]}
        assert [(w["gpu"], w["num_sms"]) for w in arm["workers"]] == [
            (w["gpu"], w["num_sms"]) for w in baseline["workers"]]
        validate_tuner_startup(arm["workers"], seeded=name != "baseline")
        validate_tuner_after_generation(arm["workers"], arm["tuner_after_generation"])
        keyed = {(row["batch"], row["repeat"]): row for row in arm["trials"]}
        assert len(keyed) == len(arm["trials"]) == len(BATCHES) * arm["repeats"]
        assert set(keyed) == {(batch, repeat) for batch in BATCHES for repeat in range(arm["repeats"])}
        for (batch, repeat), row in keyed.items():
            assert row["seed"] == 42 and row["seconds"] > 0
            assert row["output_tokens"] == batch * OUTPUT_LEN
            raw = json.loads((paths[names.index(name)].parent / f"tokens-b{batch}-r{repeat}.json").read_text())
            assert len(raw) == batch and all(len(ids) == OUTPUT_LEN for ids in raw)
            assert digest(raw) == row["output_token_ids_sha256"]
        arm["keyed_trials"] = keyed
    assert arms["default"]["source_commit"] == arms["fp16"]["source_commit"] == arms["default-repeat"]["source_commit"]
    assert arms["default"]["source_hashes"] == arms["fp16"]["source_hashes"] == arms["default-repeat"]["source_hashes"]
    for earlier, later in zip(names, names[1:]):
        assert arms[earlier]["finished_at_utc"] <= arms[later]["started_at_utc"], "Unexpected arm execution order"
    comparisons = []
    for left, right in (("baseline", "default"), ("default", "fp16"), ("default", "default-repeat")):
        for batch in BATCHES:
            rows = []
            for repeat in range(baseline["repeats"]):
                a = arms[left]["keyed_trials"][(batch, repeat)]
                b = arms[right]["keyed_trials"][(batch, repeat)]
                equal = a["output_token_ids_sha256"] == b["output_token_ids_sha256"]
                if right != "fp16":
                    assert equal, ("Default output differs", left, right, batch, repeat)
                rows.append({"repeat": repeat, "left_seconds": a["seconds"], "right_seconds": b["seconds"],
                             "latency_ratio_right_over_left": b["seconds"] / a["seconds"],
                             "output_token_ids_equal": equal})
            ratios = [row["latency_ratio_right_over_left"] for row in rows]
            comparisons.append({"left": left, "right": right, "batch": batch, "ordered_repetitions": rows,
                                "median_latency_ratio_right_over_left": statistics.median(ratios),
                                "min_latency_ratio": min(ratios), "max_latency_ratio": max(ratios)})
    summaries = {}
    for name, arm in arms.items():
        summaries[name] = {"started_at_utc": arm["started_at_utc"], "finished_at_utc": arm["finished_at_utc"],
                           "source_commit": arm["source_commit"], "startup_seconds": arm["startup_seconds"],
                           "cases": []}
        for batch in BATCHES:
            seconds = [row["seconds"] for row in arm["trials"] if row["batch"] == batch]
            summaries[name]["cases"].append({"batch": batch, "ordered_seconds": seconds,
                                             "median_seconds": statistics.median(seconds),
                                             "mean_seconds": statistics.mean(seconds),
                                             "coefficient_of_variation": statistics.stdev(seconds) / statistics.mean(seconds)})
    return {"status": "complete", "repeats": baseline["repeats"], "arms": summaries,
            "comparisons": comparisons,
            "interpretation": "Ordered repeated workloads, not randomized or independent process-paired trials. Default-repeat is the time/startup-drift anchor. Ratios include prefill, decode and engine overhead with detokenization disabled."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs=4, type=Path, metavar="RESULT_JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.results)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
