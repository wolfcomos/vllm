"""Independently check FlashInfer file-load and per-operation hit evidence."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LOAD = re.compile(r"\[Autotuner\]: Loaded (\d+) configs from (\S+)")
SAVE = re.compile(r"\[Autotuner\]: Saved (\d+) configs to (\S+)")
HIT = re.compile(r"Config cache hit for (.+?) \(runner=([^,]+), source=config file\)")
TUNING = re.compile(r"\[AutoTuner\]: Tuning ([^:\r\n]+(?:::[^:\r\n]+)*):\s*\d+%")


def check(result_path, log_path, seeded, reference=None):
    result = json.loads(result_path.read_text())
    assert result["status"] == "complete"
    text = ANSI.sub("", log_path.read_text(errors="replace"))
    expected = {}
    for item in result["cache_after_generation"]:
        path = Path(item["path"])
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        configs = json.loads(raw)
        expected[str(path)] = sum(not key.startswith("_") for key in configs)
    assert len(expected) == 4
    assert {Path(path).name for path in expected} == {
        f"autotune_configs_dp0_rank{rank}.json" for rank in range(4)
    }
    loads = [{"path": path, "configs": int(count)} for count, path in LOAD.findall(text)]
    saves = [{"path": path, "configs": int(count)} for count, path in SAVE.findall(text)]
    relevant = [row for row in (loads if seeded else saves) if row["path"] in expected]
    assert Counter(row["path"] for row in relevant) == Counter({path: 1 for path in expected}), (
        "Expected one successful load/save per rank cache file", seeded, relevant, expected
    )
    assert all(row["configs"] == expected[row["path"]] for row in relevant)
    hits = Counter(HIT.findall(text))
    tuning = TUNING.findall(text)
    if seeded:
        assert not tuning, ("Seeded arm logged tuning progress", tuning)
        required_ops = set()
        for worker in result["workers"]:
            required_ops.update(worker["flashinfer_statistics"]["tuned_op_total_configs"])
        assert required_ops, "No operation set available for independent cache-hit checking"
        assert required_ops <= {op for op, _ in hits}, ("Missing per-operation config hits", required_ops, hits)
        # FlashInfer logs once per (operation, runner) in each of four processes.
        assert hits and all(count == 4 for count in hits.values()), ("Expected four per-operation/runner hits", hits)
    report = {
        "status": "passed", "mode": result["mode"], "phase": result["phase"], "seeded": seeded,
        "result": str(result_path), "log": str(log_path),
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "expected_cache_files": expected, "load_messages": loads, "save_messages": saves,
        "per_operation_runner_hits": [{"operation": op, "runner": runner, "count": count}
                                      for (op, runner), count in sorted(hits.items())],
        "tuning_progress_operations": sorted(set(tuning)),
        "scope": "File paths identify all four rank loads; hit messages are aggregate per operation/runner and do not carry rank IDs.",
    }
    if reference is not None:
        other = json.loads(reference.read_text())
        assert other["status"] == "passed" and other["seeded"]
        assert seeded
        assert report["per_operation_runner_hits"] == other["per_operation_runner_hits"], (
            "Per-operation cache-hit evidence differs from seeded reference"
        )
        report["reference"] = str(reference)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeded", action="store_true")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    report = check(args.result, args.log, args.seeded, args.reference)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "mode": report["mode"], "seeded": report["seeded"]}))


if __name__ == "__main__":
    main()
