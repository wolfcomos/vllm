"""Confirm all candidate ranks actually loaded the expected FlashInfer files."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = json.loads((args.output / "result.json").read_text())
    assert result["status"] == "complete"
    if result["mode"] == "baseline":
        return
    log = (args.output / "run.log").read_text()
    expected = {item["destination"]: item["sha256"]
                for rank in result["validation_cache_seed"] for item in rank["files"]}
    loaded = {path: int(count) for count, path in re.findall(
        r"\[Autotuner\]: Loaded (\d+) configs from (\S+)", log)}
    assert set(expected) == set(loaded), (expected, loaded)
    for path, digest in expected.items():
        raw = Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        configs = json.loads(raw)
        count = sum(not key.startswith("_") for key in configs)
        assert loaded[path] == count, (path, loaded[path], count)
    hits = re.findall(r"Config cache hit for (.+?) \(runner=(.+?), source=config file\)", log)
    assert sum(op == "mxfp8_gemm" for op, _ in hits) == 4, hits
    assert sum(op == "flashinfer::trtllm_fp4_block_scale_moe" for op, _ in hits) == 4, hits
    report = {"passed": True, "loaded_files": loaded, "cache_hits": hits,
              "expected_hashes": expected,
              "profiling_counters_after_startup": result["autotuner_statistics_after_startup"],
              "profiling_counters_after_generation": result["autotuner_statistics_after_generation"]}
    (args.output / "cache-log-verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"cache_log_passed": True, "mode": result["mode"], "loaded_ranks": len(loaded)}))


if __name__ == "__main__":
    main()
