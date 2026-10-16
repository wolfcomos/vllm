"""Synthetic full decode+merge comparison of upstream baseline and explicit dtype flag."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import triton
import vllm.envs as envs

from bench_helpers import errors, graph_for, load, make_case, timed_graph
from reference_helpers import references, verify_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['main', 'edge'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    os.environ['VLLM_BATCH_INVARIANT'] = '0'
    torch.backends.cuda.matmul.allow_tf32 = False
    modules = {name: load('upstream_' + name, root / 'sources' / (name + '.py'))
               for name in ('baseline', 'candidate')}
    variants = {
        'baseline': (modules['baseline'], {}),
        'candidate_bf16': (modules['candidate'], {'fp16_dot': False}),
        'candidate_fp16': (modules['candidate'], {'fp16_dot': True}),
    }
    if args.suite == 'main':
        cases = [dict(batch=b, dql=4, scale_mode='scalar') for b in (1, 4, 8, 9, 10, 12, 16, 32, 64)]
        cases += [dict(batch=b, dql=1, scale_mode='scalar') for b in (1, 17, 19, 20, 36, 40, 48, 64)]
        cases += [dict(batch=b, dql=8, scale_mode='scalar') for b in (1, 4, 9, 16)]
        cases += [dict(batch=10, dql=4, scale_mode=s) for s in ('none', 'per_token_head')]
        cases += [dict(batch=10, dql=4, cache_dtype='bf16')]
    else:
        cases = [dict(batch=3, dql=d, heads=32, kv_heads=2, layout=l,
                      context=c, padded=2, invalid=i, scale_mode=s)
                 for d in (1, 4) for l in ('HND', 'NHD')
                 for c, i in ((130, False), (4096, False), (4096, True))
                 for s in ('none', 'scalar', 'per_token_head')]
        cases += [dict(batch=3, dql=4, heads=32, kv_heads=2, layout=l,
                       context=4096, padded=2, cache_dtype='bf16') for l in ('NHD', 'HND')]
    result = dict(
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                     triton=triton.__version__, device=str(torch.cuda.get_device_properties(0)),
                     vllm_envs=envs.__file__, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
        source_sha256={n: hashlib.sha256((root / 'sources' / (n + '.py')).read_bytes()).hexdigest() for n in modules},
        harness_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), root / 'bench_helpers.py', root / 'reference_helpers.py']},
        suite=args.suite, rounds=7, unroll=32, replays=4,
        fp32_reference_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        scope='synthetic; full decode+merge; warm CUDA graph; paired rotated rounds; no scalar-scale hoisting',
        scalar_scales=dict(k=.3, v=.7), rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for number, case in enumerate(cases):
        kwargs, active = make_case(**case)
        exact_ref, rounded_ref = references(kwargs, active)
        row = dict(case=case, errors={}, timings_us={}, graph_eager_bitwise={})
        funcs, outputs = {}, {}
        for name, (module, extra) in variants.items():
            output = torch.empty_like(kwargs['q'])
            outputs[name] = output
            funcs[name] = lambda m=module, o=output, e=extra: m.minimax_m3_sparse_attn_decode(**kwargs, output=o, **e)
            funcs[name]()
            torch.cuda.synchronize()
            assert torch.isfinite(output[:active]).all(), (case, name)
            row['errors'][name] = dict(exact_dequant_fp32=errors(output[:active], exact_ref),
                                        rounded_bf16_dequant_fp32=errors(output[:active], rounded_ref),
                                        vs_baseline=errors(output[:active], outputs['baseline'][:active]))
            for ref in (exact_ref, rounded_ref):
                error = errors(output[:active], ref)
                assert error['nrmse'] < .01 and error['max_abs'] < .017, (case, name, error)
        if case.get('scale_mode') == 'per_token_head' or case.get('cache_dtype') == 'bf16':
            assert torch.equal(outputs['candidate_bf16'][:active], outputs['candidate_fp16'][:active])
            row['dtype_fallback_bitwise'] = True
        for name in variants:
            verify_graph(funcs[name], outputs[name], kwargs['q'], active)
            row['graph_eager_bitwise'][name] = True
        if args.suite == 'edge':
            os.environ['VLLM_BATCH_INVARIANT'] = '1'
            funcs['baseline']()
            for name in ('candidate_bf16', 'candidate_fp16'):
                funcs[name]()
                torch.cuda.synchronize()
                assert torch.equal(outputs[name][:active], outputs['baseline'][:active])
            os.environ['VLLM_BATCH_INVARIANT'] = '0'
            row['batch_invariant_bitwise_baseline_both_flags'] = True
        else:
            graphs = {name: graph_for(fn, 32) for name, fn in funcs.items()}
            names = list(graphs)
            samples = {n: [] for n in names}
            for repeat in range(7):
                order = names[repeat % len(names):] + names[:repeat % len(names)]
                for name in order:
                    samples[name].append(timed_graph(graphs[name], 32, 4))
            row['timings_us'] = {n: dict(median=statistics.median(v), samples=v) for n,v in samples.items()}
            row['speedups'] = {n: statistics.median(a/b for a,b in zip(samples['baseline'], samples[n]))
                               for n in ('candidate_bf16', 'candidate_fp16')}
            del graphs
        result['rows'].append(row)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(progress=f'{number+1}/{len(cases)}', case=case,
                              speedups=row.get('speedups'),
                              timings={n:v['median'] for n,v in row['timings_us'].items()})), flush=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
