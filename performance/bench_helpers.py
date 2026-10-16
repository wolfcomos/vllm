"""Synthetic M3 sparse decode ablation; full decode plus merge CUDA graph timing."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import torch
import triton


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_case(batch, dql, cache_dtype="fp8", scale_mode="none", heads=16,
              kv_heads=1, layout="HND", context=4096, padded=0, invalid=False):
    torch.manual_seed(20261016 + batch * 31 + dql)
    requests = batch + padded
    lengths = [context] * batch + [0] * padded
    pages_per = triton.cdiv(context, 128)
    pages = batch * pages_per
    table = torch.zeros(requests, pages_per, dtype=torch.int32, device="cuda")
    permutation = torch.randperm(pages, device="cuda").to(torch.int32)
    table[:batch] = permutation.reshape(batch, pages_per)
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    q = torch.randn(requests * dql, heads, 128, dtype=torch.bfloat16, device="cuda")
    raw = torch.randn(pages, kv_heads, 128, 256, dtype=torch.bfloat16, device="cuda")
    dtype = torch.float8_e4m3fn if cache_dtype == "fp8" else torch.bfloat16
    k_scale = v_scale = None
    if cache_dtype == "fp8" and scale_mode != "none":
        if scale_mode == "scalar":
            k_scale = torch.tensor(0.3, device="cuda")
            v_scale = torch.tensor(0.7, device="cuda")
        else:
            k_scale = torch.full((kv_heads, pages * 128), 0.3, device="cuda")
            v_scale = torch.full((kv_heads, pages * 128), 0.7, device="cuda")
        raw[..., :128] = (raw[..., :128].float() / 0.3).to(raw.dtype)
        raw[..., 128:] = (raw[..., 128:].float() / 0.7).to(raw.dtype)
    kv = raw.to(dtype)
    if layout == "NHD":
        kv = kv.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
    topk = torch.full((kv_heads, requests * dql, 16), -1,
                      dtype=torch.int32, device="cuda")
    for r in range(batch):
        for t in range(dql):
            nb = triton.cdiv(max(context - dql + t + 1, 0), 128)
            n = min(16, nb)
            if n:
                selected = torch.linspace(0, nb - 1, n, device="cuda").round().int()
                topk[:, r * dql + t, :n] = selected
    if invalid:
        topk[..., 1::3] = 1_000_000
        topk[..., 2::3] = -1
    kwargs = dict(q=q, kv_cache=kv, topk_idx=topk, block_table=table,
                  seq_lens=seq_lens, num_kv_heads=kv_heads, sm_scale=128**-0.5,
                  decode_query_len=dql, k_scale=k_scale, v_scale=v_scale)
    return kwargs, batch * dql


def reference_first_query(kwargs):
    q, kv = kwargs["q"], kwargs["kv_cache"]
    hkv = kwargs["num_kv_heads"]
    group = q.shape[1] // hkv
    kv_len = int(kwargs["seq_lens"][0]) - kwargs["decode_query_len"] + 1
    out = torch.empty_like(q[0], dtype=torch.float32)
    for h in range(hkv):
        blocks = kwargs["topk_idx"][h, 0]
        blocks = blocks[(blocks >= 0) & (blocks < triton.cdiv(kv_len, 128))].long()
        pos = (blocks[:, None] * 128 + torch.arange(128, device="cuda")).flatten()
        pos = pos[pos < kv_len]
        pages = kwargs["block_table"][0, pos // 128].long()
        values = kv[pages, h, pos % 128].float()
        if kwargs["k_scale"] is not None:
            ks, vs = kwargs["k_scale"], kwargs["v_scale"]
            ks = ks if ks.ndim == 0 else ks[h, pages * 128 + pos % 128, None]
            vs = vs if vs.ndim == 0 else vs[h, pages * 128 + pos % 128, None]
            values[:, :128] *= ks
            values[:, 128:] *= vs
        values = values.to(q.dtype).float()
        qh = q[0, h * group:(h + 1) * group].float()
        scores = qh @ values[:, :128].T * kwargs["sm_scale"]
        out[h * group:(h + 1) * group] = scores.softmax(-1) @ values[:, 128:]
    return out


def errors(actual, expected):
    error = (actual.float() - expected.float()).abs()
    return dict(max_abs=error.max().item(), mean_abs=error.mean().item(),
                nrmse=(error.square().mean().sqrt() /
                       expected.float().square().mean().sqrt()).item())


def graph_for(fn, unroll):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            fn()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def timed_graph(graph, unroll, replays):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (unroll * replays)


