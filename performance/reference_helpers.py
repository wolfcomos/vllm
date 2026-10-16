"""Independent all-active-query references and changed-Q CUDA graph checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import triton

from bench_helpers import errors, graph_for, load, make_case, timed_graph


def references(kwargs, active):
    q, cache = kwargs["q"], kwargs["kv_cache"]
    dql, hkv = kwargs["decode_query_len"], kwargs["num_kv_heads"]
    group = q.shape[1] // hkv
    exact = torch.empty_like(q[:active], dtype=torch.float32)
    rounded = torch.empty_like(exact)
    for request in range(active // dql):
        length = int(kwargs["seq_lens"][request])
        positions = torch.arange(length, device="cuda")
        pages = kwargs["block_table"][request, positions // 128].long()
        qs = slice(request * dql, (request + 1) * dql)
        query_pos = length - dql + torch.arange(dql, device="cuda")
        for h in range(hkv):
            hs = slice(h * group, (h + 1) * group)
            kv = cache[pages, h, positions % 128].float()
            ks, vs = kwargs["k_scale"], kwargs["v_scale"]
            if ks is not None:
                ks = ks if ks.ndim == 0 else ks[h, pages * 128 + positions % 128, None]
                vs = vs if vs.ndim == 0 else vs[h, pages * 128 + positions % 128, None]
                kv[:, :128] *= ks
                kv[:, 128:] *= vs
            qh = q[qs, hs].float().transpose(0, 1)
            selected = kwargs["topk_idx"][h, qs]
            mask = ((positions[None, :, None] // 128) == selected[:, None, :]).any(-1)
            mask &= positions[None, :] <= query_pos[:, None]
            for dst, values in ((exact, kv), (rounded, kv.to(q.dtype).float())):
                scores = qh @ values[:, :128].T * kwargs["sm_scale"]
                probs = scores.masked_fill(~mask[None, :, :], -float("inf")).softmax(-1)
                dst[qs, hs] = (probs @ values[:, 128:]).transpose(0, 1)
    return exact, rounded


def verify_graph(fn, output, q, active):
    fn()
    expected = output.clone()
    graph = graph_for(fn, 1)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output[:active], expected[:active])
    q.mul_(0.5)
    fn()
    expected_changed = output.clone()
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output[:active], expected_changed[:active])
    assert not torch.equal(expected_changed[:active], expected[:active])
    q.mul_(2.0)
    fn()

