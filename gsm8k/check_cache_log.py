"""Confirm all candidate ranks actually loaded the expected FlashInfer files."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from gsm8k_cache_seed import PROTOCOL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = json.loads((args.output / "result.json").read_text())
    assert result["status"] == "complete"
    assert result["protocol"] == PROTOCOL
    log = (args.output / "run.log").read_text()
    expected = {item["path"]: item["sha256"] for item in result["autotune_cache_after_generation"]}
    loaded = {path: int(count) for count, path in re.findall(
        r"\[Autotuner\]: Loaded (\d+) configs from (\S+)", log)}
    if result["mode"] != "baseline":
        assert set(expected) == set(loaded), (expected, loaded)
    else:
        assert not loaded, ("Fresh baseline unexpectedly loaded saved tactics", loaded)
    for path, digest in expected.items():
        raw = Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        configs = json.loads(raw)
        count = sum(not key.startswith("_") for key in configs)
        if result["mode"] != "baseline":
            assert loaded[path] == count, (path, loaded[path], count)
    hits = re.findall(r"Config cache hit for (.+?) \(runner=(.+?), source=config file\)", log)
    # Current main skips the dummy-run sweep after unanimous cache load. Cache
    # hit log counts need not match the older path's one warmup call per rank.
    report = {"passed": True, "loaded_files": loaded, "cache_hits": hits,
              "expected_hashes": expected,
              "profiling_counters_after_startup": result["autotuner_statistics_after_startup"],
              "profiling_counters_after_generation": result["autotuner_statistics_after_generation"]}
    (args.output / "cache-log-verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"cache_log_passed": True, "mode": result["mode"], "loaded_ranks": len(loaded)}))


if __name__ == "__main__":
    main()
