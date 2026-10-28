"""External validation hook: reuse fresh graph-baseline FlashInfer tactics."""

import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GraphPerfCacheSeed:
    def init_worker_extension(self):
        from vllm.model_executor.warmup import kernel_warmup

        original = kernel_warmup.resolve_flashinfer_autotune_file
        output = Path(os.environ["M3_GRAPH_OUTPUT"])
        baseline_path = os.environ.get("M3_GRAPH_BASELINE_RESULT")
        baseline = json.loads(Path(baseline_path).read_text()) if baseline_path else None
        if baseline is not None:
            assert baseline["status"] == "complete"
            assert baseline["mode"] == "baseline"
            assert baseline["phase"] in {"timing", "diagnostic"}
            assert len(baseline["cache_after_generation"]) == 4

        def resolve_and_seed(runner):
            destination = original(runner)
            files = []
            if baseline is not None:
                for item in baseline["cache_after_generation"]:
                    source = Path(item["path"])
                    assert source.name.startswith("autotune_configs_dp0_rank")
                    assert source.suffix == ".json"
                    assert sha256(source) == item["sha256"]
                    target = destination.parent / source.name
                    raw = source.read_bytes()
                    if target.exists():
                        assert target.read_bytes() == raw, f"Tactics differ: {target}"
                    else:
                        temporary = target.with_suffix(f".{os.getpid()}.tmp")
                        temporary.write_bytes(raw)
                        temporary.replace(target)
                    files.append({"source": str(source), "destination": str(target),
                                  "sha256": item["sha256"]})
            # kernel_warmup adds this suffix after the original resolver returns.
            actual_file = destination.with_name(
                f"{destination.stem}_dp0_rank{self.rank}{destination.suffix}"
            )
            report = {"rank": self.rank, "resolver_returned": str(destination),
                      "actual_file": str(actual_file), "seeded_files": files,
                      "live_config": repr(runner.vllm_config)}
            (output / f"cache-rank{self.rank}.json").write_text(json.dumps(report, indent=2))
            return destination

        kernel_warmup.resolve_flashinfer_autotune_file = resolve_and_seed
