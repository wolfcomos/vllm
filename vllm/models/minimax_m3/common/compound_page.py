# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One page containing head-sharded NVFP4 main KV and token-sharded FP8 index K."""

from dataclasses import dataclass

import torch


class CompoundPageError(ValueError):
    pass


@dataclass(frozen=True)
class CompoundPageLayout:
    physical_page_tokens: int
    tp_size: int

    @classmethod
    def build(
        cls,
        *,
        tp_size: int,
        physical_page_tokens: int = 128,
        ranking_block_tokens: int = 128,
        total_main_kv_heads: int = 4,
        main_head_dim: int = 128,
        index_head_dim: int = 128,
    ) -> "CompoundPageLayout":
        if (
            tp_size not in (2, 4)
            or physical_page_tokens not in (128, 256)
            or ranking_block_tokens != 128
            or total_main_kv_heads != 4
            or main_head_dim != 128
            or index_head_dim != 128
        ):
            raise CompoundPageError(
                "Compound pages require TP2/TP4, P128/P256, KV4 and D128/B128"
            )
        return cls(physical_page_tokens, tp_size)

    @property
    def main_num_kv_heads(self) -> int:
        return 4 // self.tp_size

    @property
    def main_head_dim(self) -> int:
        return 128

    @property
    def index_head_dim(self) -> int:
        return 128

    @property
    def index_rows_per_rank(self) -> int:
        return self.physical_page_tokens // self.tp_size

    @property
    def main_span_bytes(self) -> int:
        return 2 * self.main_num_kv_heads * self.physical_page_tokens * 72

    @property
    def index_span_bytes(self) -> int:
        return self.index_rows_per_rank * self.index_head_dim

    @property
    def compound_span_bytes(self) -> int:
        return self.main_span_bytes + self.index_span_bytes


@dataclass(frozen=True)
class BoundCompoundCache:
    main: torch.Tensor
    index: torch.Tensor
    tp_rank: int


def bind_compound_cache(
    raw: torch.Tensor, layout: CompoundPageLayout, tp_rank: int
) -> BoundCompoundCache:
    """Bind aliases, preserving a layer's offset in its shared backing allocation."""
    page_bytes = layout.compound_span_bytes
    if (
        raw.dtype != torch.uint8
        or raw.ndim != 4
        or tuple(raw.shape[1:]) != (1, 1, page_bytes)
        or not raw.is_contiguous()
        or raw.shape[0] < 1
        or raw.stride(0) != page_bytes
        or raw.data_ptr() % 16
        or not 0 <= tp_rank < layout.tp_size
    ):
        raise CompoundPageError("Expected aligned uint8 [pages,1,1,page_bytes] storage")
    page_tokens = layout.physical_page_tokens
    main = torch.as_strided(
        raw,
        size=(raw.shape[0], 2 * layout.main_num_kv_heads, page_tokens, 72),
        stride=(page_bytes, page_tokens * 72, 72, 1),
        storage_offset=raw.storage_offset(),
    )
    index = torch.as_strided(
        raw.view(torch.float8_e4m3fn),
        size=(raw.shape[0], layout.index_rows_per_rank, 128),
        stride=(page_bytes, 128, 1),
        storage_offset=raw.storage_offset() + layout.main_span_bytes,
    )
    return BoundCompoundCache(main, index, tp_rank)
