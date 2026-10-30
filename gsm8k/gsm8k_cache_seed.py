"""Validation-only fresh GSM8K tuning and exact per-rank baseline reuse."""

import hashlib
import json
import os
from pathlib import Path

PROTOCOL = "gb200-current-main-gsm8k-v4"


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def cache_manifest(output):
    reports = [json.loads(path.read_text()) for path in sorted(Path(output).glob("cache-rank*.json"))]
    assert sorted(row["rank"] for row in reports) == [0, 1, 2, 3]
    records = []
    for report in reports:
        path = Path(report["actual_file"])
        contents = json.loads(path.read_text())
        assert isinstance(contents, dict) and "_metadata" in contents
        count = sum(not key.startswith("_") for key in contents)
        assert count > 0, "Expected baseline to produce actual tactics"
        records.append({"rank": report["rank"], "path": str(path),
                        "sha256": file_digest(path), "configuration_count": count,
                        "metadata": contents["_metadata"]})
    return records


def cache_identity(records):
    return {row["rank"]: {key: row[key] for key in ("sha256", "configuration_count", "metadata")}
            for row in records}


class FlashInferCacheSeed:
    """Observe baseline resolution; seed each candidate's own rank before load."""

    def init_worker_extension(self):
        import torch
        from vllm.model_executor.warmup import kernel_warmup

        original = kernel_warmup.resolve_rank_flashinfer_autotune_file
        original_mode = kernel_warmup.use_flashinfer_autotune_v2
        output = Path(os.environ["GSM8K_RESULT_DIR"])
        baseline_path = os.environ.get("GSM8K_BASELINE_RESULT")
        baseline = json.loads(Path(baseline_path).read_text()) if baseline_path else None
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if baseline:
            assert baseline["status"] == "complete" and baseline["protocol"] == PROTOCOL
            assert baseline["mode"] == "baseline"
            assert len(baseline["autotune_cache_after_generation"]) == 4
            peers = [worker for worker in baseline["workers"] if worker["rank"] == self.rank]
            assert len(peers) == 1 and str(props) == peers[0]["gpu"]
            assert str(getattr(props, "uuid", "unavailable")) == peers[0]["gpu_uuid"]

        def checked_mode(runner):
            actual = original_mode(runner)
            assert actual is False, "TP4 no longer selects legacy v1; stop without changing policy"
            return actual

        def resolve_and_seed(runner):
            destination = original(runner)
            cfg = runner.vllm_config
            assert cfg.model_config.dtype == torch.bfloat16
            assert cfg.cache_config.cache_dtype == "fp8"
            assert cfg.parallel_config.tensor_parallel_size == 4
            assert cfg.scheduler_config.max_num_batched_tokens == 4096
            assert cfg.scheduler_config.max_num_seqs == 40
            assert cfg.model_config.enforce_eager
            assert cfg.attention_config.minimax_m3_msa_decode_backend == "triton"
            assert destination.name == f"autotune_configs_dp0_rank{self.rank}.json"
            seeded = None
            if baseline:
                assert cfg.model_config.model == baseline["settings"]["model"]
                matches = [row for row in baseline["autotune_cache_after_generation"] if row["rank"] == self.rank]
                assert len(matches) == 1
                seeded = matches[0]
                source = Path(seeded["path"])
                assert source.name == destination.name and source != destination
                assert file_digest(source) == seeded["sha256"]
                raw = source.read_bytes()
                if destination.exists():
                    assert destination.read_bytes() == raw, "Preexisting candidate tactics differ"
                else:
                    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
                    temporary.write_bytes(raw)
                    temporary.replace(destination)
            else:
                assert not destination.exists(), "Baseline must autotune into a fresh GSM8K cache"
            report = {"rank": self.rank, "actual_file": str(destination), "seeded_from": seeded,
                      "cache_mode": "legacy-v1-selected-by-current-main-TP4",
                      "gpu": str(props), "gpu_uuid": str(getattr(props, "uuid", "unavailable"))}
            write_json(output / f"cache-rank{self.rank}.json", report)
            print("GSM8K_CACHE_SEED " + json.dumps(report), flush=True)
            return destination

        kernel_warmup.use_flashinfer_autotune_v2 = checked_mode
        kernel_warmup.resolve_rank_flashinfer_autotune_file = resolve_and_seed
