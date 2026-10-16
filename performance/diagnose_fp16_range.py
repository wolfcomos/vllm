"""Finite FP8 inputs whose scaled values exceed FP16; diagnostic only."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from bench_helpers import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).parent
    os.environ['VLLM_BATCH_INVARIANT'] = '0'
    torch.backends.cuda.matmul.allow_tf32 = False
    modules = {name: load('range_' + name, root / 'sources' / (name + '.py'))
               for name in ('baseline', 'candidate')}
    variants = dict(baseline=(modules['baseline'], {}),
                    candidate_bf16=(modules['candidate'], dict(fp16_dot=False)),
                    candidate_fp16=(modules['candidate'], dict(fp16_dot=True)))
    q = torch.full((1, 16, 128), .001, dtype=torch.bfloat16, device='cuda')
    topk = torch.zeros((1, 1, 1), dtype=torch.int32, device='cuda')
    table = torch.zeros((1, 1), dtype=torch.int32, device='cuda')
    lens = torch.tensor([128], dtype=torch.int32, device='cuda')
    rows = []
    for dtype, large_scale in ((torch.float8_e4m3fn, 200.), (torch.float8_e5m2, 2.)):
        maximum = torch.finfo(dtype).max
        for mode in ('bounded', 'k_overflow', 'v_overflow', 'both_overflow'):
            k_large = mode in ('bounded', 'k_overflow', 'both_overflow')
            v_large = mode in ('bounded', 'v_overflow', 'both_overflow')
            raw = torch.empty((1, 1, 128, 256), device='cuda')
            raw[..., :128] = maximum if k_large else .25
            raw[..., 128:] = maximum if v_large else .25
            cache = raw.to(dtype)
            ks = torch.tensor(large_scale if mode in ('k_overflow', 'both_overflow') else 1., device='cuda')
            vs = torch.tensor(large_scale if mode in ('v_overflow', 'both_overflow') else 1., device='cuda')
            assert torch.isfinite(q).all() and torch.isfinite(cache.float()).all()
            row = dict(dtype=str(dtype), mode=mode, max_finite_cache=maximum,
                       k_scale=ks.item(), v_scale=vs.item(), outputs={})
            outputs = {}
            for name, (module, extra) in variants.items():
                output = torch.empty_like(q)
                module.minimax_m3_sparse_attn_decode(q, cache, topk, table, lens,
                                                    1, 128 ** -.5, output, 1,
                                                    k_scale=ks, v_scale=vs, **extra)
                torch.cuda.synchronize()
                outputs[name] = output
                row['outputs'][name] = dict(nan=int(torch.isnan(output).sum()),
                                            inf=int(torch.isinf(output).sum()),
                                            finite=int(torch.isfinite(output).sum()),
                                            total=output.numel())
                if torch.isfinite(output).all():
                    row['outputs'][name]['min'] = float(output.min())
                    row['outputs'][name]['max'] = float(output.max())
            assert torch.isfinite(outputs['baseline']).all()
            assert torch.equal(outputs['baseline'], outputs['candidate_bf16'])
            row['bf16_matches_baseline_bitwise'] = True
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = dict(scope='Diagnostic extreme finite FP8 cache values; not model-distribution evidence',
                  source_sha256={n: hashlib.sha256((root / 'sources' / (n + '.py')).read_bytes()).hexdigest() for n in modules},
                  device=str(torch.cuda.get_device_properties(0)), rows=rows)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    with torch.inference_mode():
        main()
