"""Freeze source hashes and fixed prospective workload before allocation runs."""

import argparse
from pathlib import Path

from e2e_support import API_FILES, COMMITS, PROTOCOL, SOURCE_FILES, file_digest, load, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "model", "prompts", "output"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--model-revision", default="901464083161bf8612a29ff7ad29914cd4ab4a85")
    args = parser.parse_args()
    pins = {}
    apis = {}
    for key, root in (("baseline", args.baseline), ("candidate", args.candidate)):
        pins[key] = {"commit": COMMITS[key], "root": str(root),
                     "source_hashes": {name: file_digest(root / "vllm" / name) for name in SOURCE_FILES}}
        apis[key] = {name: file_digest(root / "vllm" / name) for name in API_FILES}
    assert apis["baseline"] == apis["candidate"], "Runner API source differs between arms"
    settings = {
        "model": str(args.model), "language_model_only": True,
        "revision": args.model_revision, "tokenizer_revision": args.model_revision,
        "tensor_parallel_size": 4, "distributed_executor_backend": "mp", "dtype": "bfloat16",
        "kv_cache_dtype": "fp8", "block_size": 128, "seed": 42, "enforce_eager": False,
        "max_model_len": 8192, "max_num_seqs": 40, "max_num_batched_tokens": 4096,
        "kv_cache_memory_bytes": 8 * 2**30, "enable_chunked_prefill": True,
        "enable_prefix_caching": False, "generation_config": "vllm",
        "attention_config": {"indexer_kv_dtype": "fp8"},
        "disable_log_stats": False, "cudagraph_metrics": False,
        "kernel_config": {"enable_flashinfer_autotune": True},
        "compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                               "cudagraph_capture_sizes": [8, 40], "cudagraph_num_of_warmups": 1},
        "worker_extension_cls": "e2e_support.CurrentCacheSeed",
    }
    result = {"protocol": PROTOCOL, "source_pins": pins, "api_hashes": apis["baseline"],
              "model_revision": args.model_revision, "prompts_path": str(args.prompts),
              "prompts_sha256": load(args.prompts)["prompts_sha256"], "settings": settings}
    assert not args.output.exists(), "Manifest must be newly frozen"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(file_digest(args.output))


if __name__ == "__main__":
    main()
