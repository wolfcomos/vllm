# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import faulthandler

import pytest
import ray
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.platforms import current_platform

from ..utils import init_test_distributed_environment, multi_process_parallel


@ray.remote(num_gpus=1, max_calls=1)
def _indexer_sharding(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.models.minimax_m3.nvidia import indexer_msa as msa

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        config = create_vllm_config(
            model_name="Qwen/Qwen3-0.6B",
            tensor_parallel_size=tp_size,
            block_size=128,
            max_model_len=4096,
            max_num_seqs=4,
            max_num_batched_tokens=256,
            hf_config_override={
                "num_hidden_layers": 5,
                "sparse_attention_config": {
                    "sparse_num_index_heads": 4,
                    "sparse_index_dim": 128,
                    "sparse_local_block": 1,
                    "sparse_attention_freq": [1] * 5,
                },
            },
        )
        config.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=3
        )
        config.attention_config.indexer_cache_sharding = "on"
        topk = torch.empty((256, 4 // tp_size, 16), dtype=torch.int32, device=device)
        with set_current_vllm_config(config):
            init_test_distributed_environment(
                tp_size, pp_size, rank, distributed_init_port
            )
            _check_initialization_fallback(config, rank, topk)
            assert (
                msa.resolve_index_k_shard(
                    config, 4, "fp8", 16, topk_indices_buffer=topk
                )
                == tp_size
            )
            comm = msa.get_comm()
            assert comm is not None
            _check_indexer_forwards(config, device, rank, tp_size, comm, topk)
        get_tp_group().barrier()


def _check_initialization_fallback(config, rank, topk):
    import torch.distributed._symmetric_memory as symm_mem
    from vllm.third_party.fmha_sm100.icp import _build

    from vllm.models.minimax_m3.nvidia import indexer_msa as msa

    for failure in ("allocation", "compile"):
        with pytest.MonkeyPatch.context() as m:
            msa._ICP_STATES.pop(id(config), None)
            injected = False

            def fail(*args, **kwargs):
                nonlocal injected
                injected = True
                raise RuntimeError("injected indexer initialization failure")

            if rank == 1:
                if failure == "allocation":
                    m.setattr(symm_mem, "empty", fail)
                else:
                    m.setattr(_build, "_fused", fail)
            # A peer stuck in a GPU polling kernel cannot raise a Python timeout.
            faulthandler.dump_traceback_later(90, exit=True)
            try:
                shard = msa.resolve_index_k_shard(
                    config, 4, "fp8", 16, topk_indices_buffer=topk
                )
                assert injected == (rank == 1)
                assert shard == 1
                assert msa.get_comm() is None
                assert msa._ICP_STATES[id(config)][1] is None
            finally:
                faulthandler.cancel_dump_traceback_later()


def _check_indexer_forwards(config, device, rank, tp_size, comm, topk):
    from vllm.models.minimax_m3.nvidia import indexer_msa as msa

    torch.manual_seed(123)
    heads, rows = 4 // tp_size, 128 // tp_size
    layers = []
    for sharded in (False, True):
        layers.append(
            [
                msa.MiniMaxM3IndexerMSAImpl(
                    num_kv_heads=heads,
                    scale=128**-0.5,
                    topk_blocks=16,
                    sparse_block_size=128,
                    num_index_heads=heads,
                    index_head_dim=128,
                    init_blocks=0,
                    local_blocks=1,
                    prefix=f"indexer_{sharded}_{layer}",
                    cache_config=config.cache_config,
                    indexer_kv_dtype="fp8",
                    index_k_shard=tp_size if sharded else 1,
                    topk_indices_buffer=topk if sharded else torch.empty_like(topk),
                )
                for layer in range(5)
            ]
        )
    builders = [
        msa.MiniMaxM3IndexerMSAMetadataBuilder(
            impls[0].index_cache.get_kv_cache_spec(config),
            [impl.index_cache.prefix for impl in impls],
            config,
            device,
        )
        for impls in layers
    ]
    actual = [torch.empty_like(topk) for _ in layers[1]]
    table = torch.zeros((4, 32), dtype=torch.int32, device=device)

    def run(sharded, metadata, queries):
        impls = layers[sharded]
        md = {impl.index_cache.prefix: metadata[sharded] for impl in impls}
        with set_forward_context(md, config):
            for layer, (impl, query) in enumerate(zip(impls, queries)):
                local_q = query[:, rank * heads : (rank + 1) * heads].contiguous()
                impl(query if sharded else local_q)
                if sharded:
                    actual[layer].copy_(impl.topk_indices_buffer)

    def check(metadata, queries, num_tokens):
        run(False, metadata, queries)
        for reference, result in zip(layers[0], actual):
            torch.testing.assert_close(
                result[:num_tokens].sort(dim=-1).values,
                reference.topk_indices_buffer[:num_tokens].sort(dim=-1).values,
                rtol=0,
                atol=0,
            )
        assert comm.errors.item() == 0
        assert comm.k2_errors.item() == 0

    batches = [
        BatchSpec(seq_lens=[31, 32, 33, 65], query_lens=[1] * 4),
        BatchSpec(seq_lens=[127, 128, 129, 2305], query_lens=[1] * 4),
        BatchSpec(seq_lens=[128, 2306], query_lens=[4, 4]),
        BatchSpec(seq_lens=[128, 2307], query_lens=[8, 8]),
        BatchSpec(seq_lens=[2305, 2561, 2624, 2720], query_lens=[1, 1, 64, 96]),
    ]
    for batch in batches:
        common = create_common_attn_metadata(
            batch, 128, device, arange_block_indices=True
        )
        old_table = common.block_table_tensor
        num_pages = old_table.numel()
        table.zero_()
        table[: old_table.shape[0], : old_table.shape[1]].copy_(
            torch.randperm(num_pages, device=device)[old_table.long()].int()
        )
        common.block_table_tensor = table[: old_table.shape[0]]
        if batch.query_lens[-1] > 8:
            common.block_table_tensor[-1, :8] = common.block_table_tensor[-2, :8]
        common.positions = torch.cat(
            [
                torch.arange(s - q, s, device=device)
                for s, q in zip(batch.seq_lens, batch.query_lens)
            ]
        )
        num_tokens = batch.compute_num_tokens()
        bits = torch.arange(7, device=device)
        page_codes = (
            (torch.arange(num_pages, device=device)[:, None] >> bits) & 1
        ) * 64
        head_codes = torch.tensor([0, 21, 42, 85], device=device)[:, None]
        queries, caches = [], []
        for layer, (reference, sharded) in enumerate(zip(*layers)):
            # Integer page codes give each head distinct winners without cutoff ties.
            cache = torch.randint(-4, 5, (num_pages, 128, 128), device=device)
            cache[:, :, :7] = page_codes[:, None]
            cache = cache.to(torch.float8_e4m3fn)
            reference.index_cache.bind_kv_cache(cache.unsqueeze(1))
            shard = cache[:, rank * rows : (rank + 1) * rows].contiguous()
            sharded.index_cache.bind_kv_cache(
                shard.view(num_pages, 1, 128, 128 // tp_size)
            )
            caches.append(cache)
            query = (
                torch.randint(-2, 3, (num_tokens, 4, 128), device=device).float() / 64
            )
            query[:, :, :7] = (1 << bits) * (
                1 - 2 * (((head_codes ^ (layer * 7)) >> bits) & 1)
            )
            queries.append(query.to(torch.float8_e4m3fn))
        # A short prefill can replay a decode graph; scheduler phase must not
        # switch the CP transport independently of the attention split.
        common.is_prefilling = torch.ones(len(batch.seq_lens), dtype=torch.bool)
        metadata = [builder.build(0, common) for builder in builders]
        run(False, metadata, queries)
        for _ in range(3):
            run(True, metadata, queries)
        check(metadata, queries, num_tokens)
        if metadata[1].num_prefills == 0:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(True, metadata, queries)
            common.is_prefilling.zero_()
            builders[1].build(0, common)
            for _ in range(3):
                for query, cache, impl in zip(queries, caches, layers[1]):
                    query.copy_((-query.float()).to(query.dtype))
                    cache.view(torch.uint8).copy_(cache.view(torch.uint8).flip(0))
                    impl.index_cache.kv_cache.copy_(
                        cache[:, rank * rows : (rank + 1) * rows]
                    )
                graph.replay()
            check(metadata, queries, num_tokens)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="MiniMax M3 MSA indexer CP requires capability family 100",
)
def test_msa_indexer_cp_matches_replicated(monkeypatch, tp_size):
    """Decode/prefill CP with independent pages and changing Q/K across slot reuse."""
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} GPUs")
    if not all(
        torch.cuda.can_device_access_peer(src, dst)
        for src in range(tp_size)
        for dst in range(tp_size)
        if src != dst
    ):
        pytest.skip("Requires peer access between every pair of GPUs")
    multi_process_parallel(monkeypatch, tp_size, 1, _indexer_sharding)
