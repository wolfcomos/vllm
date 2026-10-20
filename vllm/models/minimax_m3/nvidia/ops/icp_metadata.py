# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare ICP live metadata from scheduler buffers, independently of KV writes."""

from typing import Any

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _request_for_row(query_start_loc, row, num_reqs):
    lo = tl.full(row.shape, 0, tl.int32)
    hi = tl.full(row.shape, num_reqs, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(query_start_loc + mid + 1, mid < num_reqs, other=0)
        right = (lo < hi) & (end <= row)
        hi = tl.where((lo < hi) & ~right, mid, hi)
        lo = tl.where(right, mid + 1, lo)
    return lo


@triton.jit(do_not_specialize=["num_reqs", "rows", "rank"])
def _live_metadata(
    qsl,
    seq_lens,
    positions,
    active,
    local_nvalid,
    local_forced,
    global_nvalid,
    forced,
    n_ordinary,
    candidates,
    num_reqs,
    rows,
    rank,
    FRAGMENT: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = _request_for_row(qsl, row, num_reqs)
    begin = tl.load(qsl + req, req < num_reqs, other=0)
    end = tl.load(qsl + req + 1, req < num_reqs, other=0)
    seq = tl.load(seq_lens + req, req < num_reqs, other=0)
    query = end - begin
    pos = seq.to(tl.int64) - query + row - begin
    live = (
        (row < rows)
        & (req < num_reqs)
        & (query > 0)
        & (seq >= query)
        & (row >= begin)
        & (row < end)
        & (pos >= 0)
    )
    visible = tl.minimum(pos + 1, seq).to(tl.int32)
    last = (pos // 128).to(tl.int32)
    local = visible // PAGE_SIZE + (visible % PAGE_SIZE > rank * FRAGMENT)
    forced_column = last // (PAGE_SIZE // 128)
    forced_owned = (last % (PAGE_SIZE // 128) == rank * FRAGMENT // 128) & (
        forced_column < local
    )
    tl.store(positions + row, tl.where(live, pos, 0), row < rows)
    tl.store(active + row, live, row < rows)
    tl.store(local_nvalid + row, tl.where(live, local, 0), row < rows)
    tl.store(
        local_forced + row,
        tl.where(live & forced_owned, forced_column, -1),
        row < rows,
    )
    tl.store(global_nvalid + row, tl.where(live, last + 1, 0), row < rows)
    tl.store(forced + row, tl.where(live, last, -1), row < rows)
    tl.store(n_ordinary + row, tl.where(live, tl.minimum(last, 15), 0), row < rows)
    candidate = tl.arange(0, 64)
    tl.store(
        candidates + (row[:, None] * 64 + candidate[None, :]) * 2 + 1,
        -1,
        (row[:, None] < rows) & ~live[:, None],
    )


@triton.jit(do_not_specialize=["num_reqs", "rows", "width"])
def _device_plan(
    qsl,
    seq_lens,
    segments,
    work,
    header,
    qo_offsets,
    num_reqs,
    rows,
    width,
    SEGMENT_STRIDE: tl.constexpr,
    WORK_STRIDE: tl.constexpr,
    QO_STRIDE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    MAX_WORK: tl.constexpr,
    BLOCK_SEGMENTS: tl.constexpr,
    BLOCK_CTAS: tl.constexpr,
    BLOCK_WORK: tl.constexpr,
):
    # Device-plan ABI4: the native OnlyScoreIcp direct-greedy schedule.
    slot = tl.program_id(0)
    t0 = slot * width
    t1 = tl.minimum(t0 + width, rows)
    live_end = tl.load(qsl + num_reqs)
    end = tl.minimum(t1, live_end)
    first = _request_for_row(qsl, tl.full((1,), t0, tl.int32), num_reqs)
    last = _request_for_row(qsl, tl.full((1,), end - 1, tl.int32), num_reqs)
    first = tl.sum(tl.where(t0 < end, first, 0), 0)
    n = tl.sum(tl.where(t0 < end, last - first + 1, 0), 0)
    tl.device_assert(n < SEGMENT_STRIDE, "ICP plan segment capacity exceeded")

    b = tl.arange(0, BLOCK_SEGMENTS)
    req = first + b
    begin = tl.load(qsl + req, b < n, other=t0)
    stop = tl.load(qsl + req + 1, b < n, other=t0)
    seq = tl.load(seq_lens + req, b < n, other=0)
    lo, hi = tl.maximum(begin, t0), tl.minimum(stop, t1)
    qlen = tl.where(b < n, hi - lo, 0)
    causal = seq - (stop - lo)
    seg = segments + slot * 5 * SEGMENT_STRIDE
    tl.store(seg, 0)
    tl.store(seg + SEGMENT_STRIDE, 0)
    tl.store(seg + b + 1, hi - t0, b < n)
    tl.store(seg + SEGMENT_STRIDE + b + 1, tl.cumsum(seq), b < n)
    tl.store(seg + 2 * SEGMENT_STRIDE + b, qlen, b < n)
    tl.store(seg + 3 * SEGMENT_STRIDE + b, seq, b < n)
    tl.store(seg + 4 * SEGMENT_STRIDE + b, causal, b < n)
    tl.store(qo_offsets + slot * QO_STRIDE + b, causal, b < n)
    tl.debug_barrier()

    cta = tl.arange(0, BLOCK_CTAS)
    cost = tl.full((BLOCK_CTAS,), 165, tl.int32)
    count = tl.full((BLOCK_CTAS,), 0, tl.int32)
    ranges = work + slot * WORK_STRIDE
    info = ranges + NUM_CTAS
    scratch_info = info + MAX_WORK
    scratch_slot = scratch_info + MAX_WORK
    items = 0
    qt = tl.max(tl.cdiv(qlen, 128), 0) - 1
    while qt >= 0:
        for segment in range(n):
            query_len = tl.load(seg + 2 * SEGMENT_STRIDE + segment)
            kv_len = tl.load(seg + 3 * SEGMENT_STRIDE + segment)
            offset = tl.load(seg + 4 * SEGMENT_STRIDE + segment)
            effective = tl.minimum((qt + 1) * 128 + offset, kv_len)
            if (qt < tl.cdiv(query_len, 128)) & (effective > 0):
                tile_cost = 43 * tl.cdiv(effective, 256) + 110
                available = cta < NUM_CTAS
                for head in tl.static_range(4):
                    key = tl.where(
                        available,
                        cost.to(tl.int64) * BLOCK_CTAS + cta,
                        0x7FFFFFFFFFFFFFFF,
                    )
                    selected = tl.min(key, 0) % BLOCK_CTAS
                    take = cta == selected
                    position = tl.sum(tl.where(take, count, 0), 0)
                    tl.device_assert(
                        items < MAX_WORK, "ICP plan work capacity exceeded"
                    )
                    packed = (qt.to(tl.uint64) << 32) | (head << 16) | segment
                    where = (selected.to(tl.uint64) << 32) | position
                    tl.store(scratch_info + items, packed)
                    tl.store(scratch_slot + items, where)
                    count += take.to(tl.int32)
                    cost += tl.where(take, tile_cost, 0)
                    available &= ~take
                    items += 1
        qt -= 1

    exclusive = tl.cumsum(count) - count
    packed_range = ((exclusive + count).to(tl.uint64) << 32) | exclusive
    tl.store(ranges + cta, packed_range, cta < NUM_CTAS)
    tl.debug_barrier()
    item = tl.arange(0, BLOCK_WORK)
    where = tl.load(scratch_slot + item, item < items, other=0).to(tl.uint64)
    bucket = (where >> 32).to(tl.int32)
    start = tl.load(ranges + bucket, item < items, other=0).to(tl.uint64)
    destination = ((start & 0xFFFFFFFF) + (where & 0xFFFFFFFF)).to(tl.int32)
    packed = tl.load(scratch_info + item, item < items, other=0)
    tl.store(info + destination, packed, item < items)
    tl.store(header + slot * 4, n)
    tl.store(header + slot * 4 + 1, first)
    tl.store(header + slot * 4 + 2, items)
    tl.store(header + slot * 4 + 3, t0)


def prepare_icp_metadata(
    step: Any, rank: int, tp: int, *, page_size: int = 128
) -> None:
    """Refresh stable live buffers once per scheduler step, before graph replay."""
    args = step.writer_args
    rows = step.token_capacity
    if rows == 0:
        return
    _live_metadata[(triton.cdiv(rows, 32),)](
        args["icp_query_start_loc"],
        args["icp_seq_lens"],
        args["icp_positions"],
        args["icp_active"],
        args["icp_local_nvalid"],
        args["icp_local_forced"],
        args["icp_global_nvalid"],
        args["icp_forced"],
        args["icp_n_ordinary"],
        args["icp_candidates"].view(torch.int32),
        args["icp_num_reqs"],
        rows,
        rank,
        FRAGMENT=page_size // tp,
        PAGE_SIZE=page_size,
        BLOCK=32,
    )
    if "icp_plan_segments" not in args:
        return
    segments = args["icp_plan_segments"]
    work = args["icp_plan_work"]
    qo_offsets = args["icp_qo_offsets"]
    num_ctas = args["icp_plan_num_ctas"]
    max_work = (work.shape[1] - num_ctas) // 3
    _device_plan[(qo_offsets.shape[0],)](
        args["icp_query_start_loc"],
        args["icp_seq_lens"],
        segments,
        work,
        args["icp_plan_header"],
        qo_offsets,
        args["icp_num_reqs"],
        rows,
        args["icp_chunk_width"],
        SEGMENT_STRIDE=segments.shape[2],
        WORK_STRIDE=work.shape[1],
        QO_STRIDE=qo_offsets.shape[1],
        NUM_CTAS=num_ctas,
        MAX_WORK=max_work,
        BLOCK_SEGMENTS=triton.next_power_of_2(segments.shape[2] - 1),
        BLOCK_CTAS=triton.next_power_of_2(num_ctas),
        BLOCK_WORK=triton.next_power_of_2(max_work),
        num_warps=8,
    )
