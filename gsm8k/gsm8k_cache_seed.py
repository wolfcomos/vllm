"""Validation-only reuse of baseline FlashInfer tactics for paired GSM8K runs."""
import hashlib
import json
import os
from pathlib import Path


class FlashInferCacheSeed:
    def init_worker_extension(self):
        import torch
        from vllm.model_executor.warmup import kernel_warmup
        from vllm.model_executor.warmup.flashinfer_autotune_cache import (
            flashinfer_autotune_cache_hash,
        )

        baseline = json.loads(Path('/work/gsm8k/baseline/result.json').read_text())
        assert baseline['status'] == 'complete'
        original = kernel_warmup.resolve_flashinfer_autotune_file
        rank = self.rank
        gpu = str(torch.cuda.get_device_properties(torch.cuda.current_device()))
        assert gpu == baseline['workers'][rank]['gpu']

        def resolve_and_seed(runner):
            destination = original(runner)
            cfg = runner.vllm_config
            assert cfg.model_config.model == baseline['settings']['model']
            assert cfg.model_config.dtype == torch.bfloat16
            assert cfg.cache_config.cache_dtype == 'fp8'
            assert cfg.parallel_config.tensor_parallel_size == 4
            assert cfg.scheduler_config.max_num_batched_tokens == 4096
            assert cfg.scheduler_config.max_num_seqs == 40
            sources = baseline['autotune_cache_after_generation']
            assert len(sources) == 4
            records = []
            for relative, expected_sha in sources.items():
                source = Path('/work/cache/flashinfer') / relative
                assert source.name.startswith('autotune_configs_dp0_rank')
                assert source.suffix == '.json'
                raw = source.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == expected_sha
                target = destination.parent / source.name
                if target.exists():
                    assert target.read_bytes() == raw
                else:
                    temporary = target.with_suffix(f'.{os.getpid()}.tmp')
                    temporary.write_bytes(raw)
                    temporary.replace(target)
                records.append({'source': str(source), 'destination': str(target),
                                'sha256': expected_sha})
            report = {'rank': rank, 'gpu': gpu, 'cache_hash': flashinfer_autotune_cache_hash(runner),
                      'resolver_returned': str(destination), 'files': records,
                      'live_config': repr(cfg)}
            out = Path(os.environ['GSM8K_RESULT_DIR']) / f'cache-seed-rank{rank}.json'
            out.write_text(json.dumps(report, indent=2))
            print('GSM8K_CACHE_SEED ' + json.dumps({k: v for k, v in report.items() if k != 'live_config'}), flush=True)
            return destination

        kernel_warmup.resolve_flashinfer_autotune_file = resolve_and_seed
