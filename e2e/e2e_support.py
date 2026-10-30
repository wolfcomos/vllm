"""External v4 validation utilities; no production source modifications."""

import hashlib
import json
import os
from pathlib import Path
import statistics

PROTOCOL = "gb200-current-main-e2e-v4"
ARMS = ("baseline", "default-a", "fp16-a", "fp16-b", "default-b", "cutlass")
COMMITS = {"baseline": "a20280f30e3169d33a7e21d1d117e9719bace00b",
           "candidate": "25acf1953bf42857aec52d5f90cb7eac174d3189"}
BATCHES = (8, 40)
INPUT_LEN = 2048
OUTPUT_LEN = 2048
PACKAGES = ("vllm", "torch", "triton", "flashinfer-python", "nvidia-cutlass-dsl", "transformers", "tokenizers")
SOURCE_FILES = ("config/attention.py", "models/minimax_m3/common/ops/sparse_attn.py",
                "models/minimax_m3/common/sparse_attention.py", "models/minimax_m3/nvidia/model.py",
                "models/minimax_m3/nvidia/sparse_attention_msa.py")
API_FILES = ("entrypoints/common/offline.py", "entrypoints/llm.py", "sampling_params.py",
             "v1/engine/llm_engine.py", "v1/engine/core_client.py", "v1/engine/core.py",
             "v1/core/sched/scheduler.py", "v1/engine/output_processor.py", "v1/metrics/stats.py",
             "outputs.py", "compilation/cuda_graph.py", "model_executor/warmup/kernel_warmup.py",
             "model_executor/warmup/flashinfer_autotune_cache.py",
             "models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py")


def load(path):
    return json.loads(Path(path).read_text())


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def normalized_settings(settings):
    result = json.loads(json.dumps(settings))
    for name in ("minimax_m3_msa_decode_fp16_dot", "minimax_m3_msa_decode_backend"):
        result["attention_config"].pop(name, None)
    return result


def metric_record(output):
    stats = output.metrics
    assert stats is not None, "Native stats must be enabled"
    assert stats.num_generation_tokens == len(output.outputs[0].token_ids) == OUTPUT_LEN
    assert stats.num_preemptions == 0, "Preemption changes workload"
    assert 0 < stats.queued_ts <= stats.scheduled_ts <= stats.first_token_ts < stats.last_token_ts
    assert stats.first_token_latency > 0
    decode = stats.last_token_ts - stats.first_token_ts
    return {"request_id": output.request_id, "num_generation_tokens": stats.num_generation_tokens,
            "num_preemptions": stats.num_preemptions, "arrival_time_wall": stats.arrival_time,
            "queued_ts_monotonic": stats.queued_ts, "scheduled_ts_monotonic": stats.scheduled_ts,
            "first_token_ts_monotonic": stats.first_token_ts, "last_token_ts_monotonic": stats.last_token_ts,
            "ttft_seconds": stats.first_token_latency,
            "queue_seconds": stats.scheduled_ts - stats.queued_ts,
            "prefill_seconds": stats.first_token_ts - stats.scheduled_ts,
            "decode_seconds": decode, "mean_tpot_seconds": decode / (OUTPUT_LEN - 1)}


def tail_metrics(metrics):
    """Common tail starts after every request emits its first output token."""
    assert metrics
    first = max(row["first_token_ts_monotonic"] for row in metrics)
    last = max(row["last_token_ts_monotonic"] for row in metrics)
    assert last > first
    # Match IDs, not a coincidence between distinct first/last maxima.
    last_first = [row for row in metrics if row["first_token_ts_monotonic"] == first]
    witnesses = [row for row in last_first if row["last_token_ts_monotonic"] == last]
    assert witnesses, "Last-first request is not last-finished; tail/2047 is invalid"
    assert all(row["num_generation_tokens"] == OUTPUT_LEN for row in metrics)
    return {"common_tail_seconds": last - first,
            "common_tail_tpot_seconds": (last - first) / (OUTPUT_LEN - 1),
            "last_first_request_ids": [row["request_id"] for row in last_first],
            "tail_witness_request_ids": [row["request_id"] for row in witnesses],
            "tail_token_intervals": OUTPUT_LEN - 1,
            "tail_start_core_monotonic": first, "tail_end_core_monotonic": last,
            "tail_scope": "Draining-batch interval; never a fixed-batch per-step kernel metric"}


def read_tuner(worker):
    from flashinfer.autotuner import AutoTuner

    stats = AutoTuner.get().stats
    return {"rank": worker.rank, "statistics": {
        "statistics_object_id": id(stats), "cache_misses": int(stats.cache_misses),
        "tuned_op_total_configs": dict(stats.tuned_op_total_configs),
        "tuned_op_successful_configs": dict(stats.tuned_op_successful_configs),
        "failed_profiling_count": {op: len(keys) for op, keys in stats.failed_profiling_count.items()},
        "failed_tactics": {op: len(tactics) for op, tactics in stats.failed_tactics.items()},
    }}


def inspect_worker(worker):
    import torch
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseImpl
    from vllm.models.minimax_m3.nvidia.msa_cutlass_sparse_decode import should_prepare_decode_metadata
    from vllm.models.minimax_m3.nvidia.sparse_attention_msa import MiniMaxM3SparseMSAMetadataBuilder, SPARSE_BLOCK_SIZE

    runner = worker.model_runner
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    implementations = []
    for name, module in runner.model.named_modules():
        impl = getattr(module, "impl", None)
        if isinstance(impl, MiniMaxM3SparseImpl):
            implementations.append({"module": name, "class": type(impl).__name__,
                                    "fp16_dot": getattr(impl, "fp16_dot", False),
                                    "use_cutlass_decode": getattr(impl, "use_cutlass_decode", False),
                                    "kv_cache_dtype": impl.kv_cache_dtype,
                                    "num_q_heads": impl.num_heads, "num_kv_heads": impl.num_kv_heads,
                                    "head_size": impl.head_size, "page_size": impl.block_size,
                                    "topk_blocks": impl.topk_blocks,
                                    "k_scale_numel": None if getattr(module, "_k_scale", None) is None else module._k_scale.numel(),
                                    "v_scale_numel": None if getattr(module, "_v_scale", None) is None else module._v_scale.numel()})
    entries = [{"mode": str(wrapper.runtime_mode), "num_tokens": descriptor.num_tokens}
               for wrapper in CUDAGraphWrapper._all_instances
               for descriptor, entry in wrapper.concrete_cudagraph_entries.items()
               if entry.cudagraph is not None]
    dispatch = []
    for batch in BATCHES:
        mode, descriptor = runner.cudagraph_dispatcher.dispatch(batch, uniform_decode=True)
        dispatch.append({"batch": batch, "mode": str(mode), "padded_tokens": descriptor.num_tokens})
    policies = []
    assert runner.uniform_decode_query_len == 1
    assert os.getenv("VLLM_MINIMAX_M3_MSA_CUTLASS_MIN_BATCH") is None, "Native threshold must not be overridden"
    for group_index, groups in enumerate(runner.attn_groups):
        for attention_index, group in enumerate(groups):
            for builder_index, builder in enumerate(group.metadata_builders):
                if not isinstance(builder, MiniMaxM3SparseMSAMetadataBuilder):
                    continue
                geometry = {"decode_backend": builder.decode_backend,
                            "num_q_heads": builder.num_q_heads, "num_kv_heads": builder.num_kv_heads,
                            "kv_cache_dtype": builder.kv_cache_dtype, "page_size": SPARSE_BLOCK_SIZE,
                            "topk_blocks": builder.topk_blocks}
                assert builder.msa_cutlass_min_batch_size is None
                shapes = []
                for batch in range(1, 41):
                    mode, descriptor = runner.cudagraph_dispatcher.dispatch(batch, uniform_decode=True)
                    padded = descriptor.num_tokens
                    shapes.append({"batch": batch, "graph_mode": str(mode), "graph_padded_tokens": padded,
                                   "should_prepare_at_unpadded_batch": should_prepare_decode_metadata(
                                       batch, 1, **geometry, min_batch_size=None),
                                   "should_prepare_at_graph_shape": should_prepare_decode_metadata(
                                       padded, 1, **geometry, min_batch_size=None)})
                policies.append({"group_index": group_index, "attention_index": attention_index,
                                 "builder_index": builder_index, "class": type(builder).__name__,
                                 "layer_names": list(group.layer_names),
                                 "geometry": geometry, "decode_query_len": 1,
                                 "min_batch_override": None, "shapes": shapes,
                                 "query_preparation": ("Model emits FP8 Q; native Triton fallback dequantizes Q to BF16"
                                                       if builder.decode_backend == "cutlass"
                                                       else "Model emits BF16 Q for explicit Triton"),
                                 "scope": "Untimed helper/dispatcher policy, not observed timed dispatch counts"})
    return {"rank": worker.rank, "gpu": str(props), "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
            "num_sms": props.multi_processor_count, "capability": [props.major, props.minor],
            "resolved_graph_mode": str(runner.cudagraph_dispatcher.cudagraph_mode),
            "captured_entries": entries, "dispatch": dispatch, "implementations": implementations,
            "native_backend_policies": policies,
            "attention_config": str(worker.vllm_config.attention_config),
            "statistics": read_tuner(worker)["statistics"]}


class CurrentCacheSeed:
    """Fresh baseline autotunes; each later engine loads baseline's rank caches."""

    def init_worker_extension(self):
        from vllm.model_executor.warmup import kernel_warmup

        original = kernel_warmup.resolve_rank_flashinfer_autotune_file
        mode = kernel_warmup.use_flashinfer_autotune_v2
        output = Path(os.environ["M3_V4_OUTPUT"])
        baseline_path = os.environ.get("M3_V4_BASELINE_RESULT")
        baseline = load(baseline_path) if baseline_path else None
        if baseline:
            assert baseline["protocol"] == PROTOCOL and baseline["status"] == "complete"
            assert baseline["mode"] == "baseline" and len(baseline["cache_after_generation"]) == 4

        def checked_mode(runner):
            actual = mode(runner)
            assert actual is False, "TP4 cache policy changed; stop rather than patch policy"
            return actual

        def resolve_and_seed(runner):
            destination = original(runner)
            seeded = None
            if baseline:
                matches = [item for item in baseline["cache_after_generation"]
                           if item["rank"] == self.rank]
                assert len(matches) == 1
                seeded = matches[0]
                source = Path(seeded["path"])
                assert file_digest(source) == seeded["sha256"]
                assert source.name == destination.name
                raw = source.read_bytes()
                if destination.exists():
                    assert destination.read_bytes() == raw, "Preexisting tactic file differs"
                else:
                    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
                    temporary.write_bytes(raw)
                    temporary.replace(destination)
            write_json(output / f"cache-rank{self.rank}.json", {
                "rank": self.rank, "actual_file": str(destination), "seeded_from": seeded,
                "cache_mode": "legacy-v1-selected-by-current-main-TP4"})
            return destination

        kernel_warmup.use_flashinfer_autotune_v2 = checked_mode
        kernel_warmup.resolve_rank_flashinfer_autotune_file = resolve_and_seed


def cache_manifest(output):
    reports = [load(path) for path in sorted(Path(output).glob("cache-rank*.json"))]
    assert sorted(row["rank"] for row in reports) == [0, 1, 2, 3]
    records = []
    for row in reports:
        path = Path(row["actual_file"])
        configs = load(path)
        assert isinstance(configs, dict) and "_metadata" in configs
        count = len(configs) - 1
        assert count > 0, "Expected fresh baseline to produce real tactics"
        records.append({"rank": row["rank"], "path": str(path), "sha256": file_digest(path),
                        "configuration_count": count, "metadata": configs["_metadata"]})
    return records


def cache_identity(records):
    return {row["rank"]: {key: row[key] for key in ("sha256", "configuration_count", "metadata")}
            for row in records}


def stats_unchanged(before, after):
    initial = {row["rank"]: row["statistics"] for row in before}
    final = {row["rank"]: row["statistics"] for row in after}
    assert initial.keys() == final.keys() == {0, 1, 2, 3}
    assert initial == final, "FlashInfer state changed during warmups/generation"


def summary(values):
    median = statistics.median(values)
    return {"values": values, "median": median, "min": min(values), "max": max(values),
            "relative_mad": statistics.median(abs(value - median) for value in values) / median}
