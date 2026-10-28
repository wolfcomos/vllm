"""Owned-allocation orchestration for the separately frozen v2 protocol."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback

ARMS = ('baseline', 'default-a', 'fp16-a', 'fp16-b', 'default-b')
CLEANUP_RESERVE = datetime(2026, 10, 28, 23, 49, 28, tzinfo=timezone.utc)
PYTHON = Path('/work/.venv/bin/python')
ROOT = Path('/work/graph-perf-admission-controlled')
FROZEN = Path('/work/graph-perf')
V1_BASELINE = FROZEN / 'results/baseline/result.json'
PINS = Path('/work/source-pins.json')
CACHE_ROOT = Path('/work/cache/admission-controlled-v2')
REQUIRED_STATISTICS = {'statistics_object_id', 'cache_misses', 'tuned_op_total_configs',
                       'tuned_op_successful_configs', 'failed_profiling_count', 'failed_tactics'}


def now():
    return datetime.now(timezone.utc)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def command(arm, phase, output, cache, results, comparison):
    argv = [str(PYTHON), '-u', str(ROOT / 'run_admission_controlled.py'),
            '--arm', arm, '--phase', phase,
            '--baseline-v1-result', str(V1_BASELINE),
            '--prompts', str(FROZEN / 'prompts.json'),
            '--source-pins', str(PINS), '--cache-dir', str(cache), '--output', str(output)]
    if arm != 'baseline':
        argv += ['--witness-result', str(results / 'baseline/result.json')]
    if phase == 'timing' and arm != 'baseline':
        previous = ARMS[ARMS.index(arm) - 1]
        argv += ['--previous-result', str(results / previous / 'result.json')]
    if phase == 'diagnostic':
        argv += ['--timing-result', str(results / arm / 'result.json'),
                 '--timing-comparison', str(comparison)]
    return argv


def snapshot(path):
    query = ['nvidia-smi', '--query-gpu=index,uuid,name,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used', '--format=csv']
    result = subprocess.run(query, text=True, capture_output=True, timeout=30, check=True)
    path.write_text(result.stdout)


def require_idle():
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                           '--format=csv,noheader'], text=True, capture_output=True,
                          timeout=30, check=True)
    assert not apps.stdout.strip(), ('GPU compute processes remain', apps.stdout)
    processes = subprocess.run(['ps', '-eo', 'comm='], text=True, capture_output=True,
                               timeout=30, check=True).stdout.splitlines()
    assert not any(Path(p.strip()).name in {'nsys', 'ncu'} for p in processes), 'Profiler process active'


def terminate_owned_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def run_process(argv, env, log, timeout_seconds):
    with log.open('w') as stream:
        process = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout_seconds), False
        except subprocess.TimeoutExpired:
            terminate_owned_group(process)
            return 124, True
        except BaseException:
            terminate_owned_group(process)
            raise


def verify_full_statistics(result_path, output):
    result = json.loads(result_path.read_text())
    before = {w['rank']: w['flashinfer_statistics'] for w in result['workers']}
    after = {w['rank']: w['statistics'] for w in result['tuner_after_generation']}
    assert set(before) == set(after) == {0, 1, 2, 3}
    rows = []
    for rank in range(4):
        assert REQUIRED_STATISTICS <= set(before[rank]), ('Missing startup fields', rank)
        assert REQUIRED_STATISTICS <= set(after[rank]), ('Missing final fields', rank)
        rows.append({'rank': rank, 'before': before[rank], 'after': after[rank],
                     'identical': before[rank] == after[rank],
                     'changed_fields': {key: {'before': before[rank].get(key),
                                             'after': after[rank].get(key)}
                         for key in before[rank].keys() | after[rank].keys()
                         if key not in before[rank] or key not in after[rank]
                         or before[rank][key] != after[rank][key]}})
    report = {'passed': all(row['identical'] for row in rows), 'ranks': rows,
              'scope': 'Every recorded tuner statistic is compared; missing values are not treated as zero.'}
    save(output, report)
    return report['passed']


def run(args):
    release = datetime.fromisoformat(args.release_utc.replace('Z', '+00:00'))
    assert release.tzinfo is not None
    deadline = min(release + timedelta(minutes=60), CLEANUP_RESERVE)
    assert release <= now() < deadline
    assert os.environ.get('M3_ADMISSION_START_AUTHORIZED') == '1'
    quality = json.loads(Path('/work/gsm8k/comparison.json').read_text())
    assert quality['passed'] is True
    assert quality['comparisons']['bf16']['same_output_token_sequences'] == 1319
    assert args.comparison_script.is_absolute() and args.comparison_script.is_file()
    results, logs = ROOT / 'results', ROOT / 'logs'
    assert not results.exists() and not logs.exists() and not CACHE_ROOT.exists()
    results.mkdir(); logs.mkdir(); CACHE_ROOT.mkdir(parents=True)
    comparison = results / 'comparison.json'
    assert not comparison.exists()
    report_path = ROOT / 'orchestration.json'
    assert not report_path.exists()
    report = {'status': 'starting', 'release_utc': release.isoformat(),
              'deadline_utc': deadline.isoformat(), 'cleanup_reserve_utc': CLEANUP_RESERVE.isoformat(),
              'order': list(ARMS), 'with_diagnostic': args.with_diagnostic,
              'per_engine_timeout_cap_seconds': 900, 'term_grace_seconds': 90,
              'minimum_remaining_before_engine_seconds': 720,
              'argv': sys.argv, 'orchestration_sha256': sha(__file__),
              'input_hashes': {str(p): sha(p) for p in [ROOT / 'run_admission_controlled.py',
                  ROOT / 'PROTOCOL.md', args.comparison_script, FROZEN / 'check_cache_logs.py',
                  FROZEN / 'run_graph_perf.py', FROZEN / 'graph_perf_cache_seed.py',
                  FROZEN / 'prompts.json', V1_BASELINE, PINS]}, 'runs': []}
    save(report_path, report)
    common_env = os.environ.copy()
    common_env.pop('RAY_ADDRESS', None)
    common_env.update(CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='4',
                      VLLM_CACHE_ROOT='/work/cache/vllm', TRITON_CACHE_DIR='/work/cache/triton')

    def arm(arm_name, phase):
        remaining = (deadline - now()).total_seconds()
        assert remaining >= 720, f'Fewer than 12 minutes remain before deadline: {remaining:.0f}s'
        require_idle()
        tag = arm_name if phase == 'timing' else f'diagnostic-{arm_name}'
        output = results / arm_name if phase == 'timing' else ROOT / tag
        cache = CACHE_ROOT / tag
        assert not output.exists() and not cache.exists()
        log = logs / f'{tag}.log'
        env = common_env.copy()
        source = '/work/baseline' if arm_name == 'baseline' else '/work/source'
        env['PYTHONPATH'] = f'{source}:{ROOT}:{FROZEN}'
        argv = command(arm_name, phase, output, cache, results, comparison)
        row = {'arm': arm_name, 'phase': phase, 'started_utc': now().isoformat(),
               'argv': argv, 'environment': {key: env[key] for key in ['CUDA_VISIBLE_DEVICES',
                   'OMP_NUM_THREADS', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'PYTHONPATH']},
               'output': str(output), 'cache': str(cache), 'log': str(log)}
        report['runs'].append(row); save(report_path, report)
        save(logs / f'{tag}.invocation.json', row)
        snapshot(logs / f'{tag}.gpu-before.csv')
        limit = min(900, int((deadline - now()).total_seconds()) - 240)
        assert limit > 0
        code, timed_out = run_process(argv, env, log, limit)
        row.update(exit_code=code, timed_out=timed_out, finished_utc=now().isoformat(),
                   log_sha256=sha(log))
        (logs / f'{tag}.exit.txt').write_text(str(code) + '\n')
        snapshot(logs / f'{tag}.gpu-after.csv')
        save(report_path, report)
        if output.is_dir():
            shutil.copyfile(log, output / 'run.log')
            assert sha(log) == sha(output / 'run.log')
        assert code == 0, ('Driver failed; no automatic retry', arm_name, phase, code)
        require_idle()
        checker = [str(PYTHON), str(FROZEN / 'check_cache_logs.py'),
                   '--result', str(output / 'result.json'), '--log', str(log), '--seeded',
                   '--output', str(output / 'cache-log-check.json')]
        if arm_name != 'baseline' or phase == 'diagnostic':
            checker += ['--reference', str(results / 'baseline/cache-log-check.json')]
        check = subprocess.run(checker, text=True, capture_output=True, timeout=60)
        (logs / f'{tag}.cache-check.log').write_text(check.stdout + check.stderr)
        row['cache_check_argv'] = checker
        row['cache_check_exit_code'] = check.returncode
        save(report_path, report)
        assert check.returncode == 0, ('Cache evidence check failed', tag)
        row['full_tuner_statistics_unchanged'] = verify_full_statistics(
            output / 'result.json', output / 'full-tuner-statistics-check.json')
        save(report_path, report)
        print(json.dumps({'finished_arm': tag, 'status': 'passed'}), flush=True)

    try:
        for name in ARMS:
            arm(name, 'timing')
        comparison_argv = [str(PYTHON), str(args.comparison_script),
                           *[str(results / name / 'result.json') for name in ARMS],
                           '--prompts', str(FROZEN / 'prompts.json'),
                           '--baseline-v1-result', str(V1_BASELINE),
                           '--source-pins', str(PINS), '--output', str(comparison)]
        report['comparison_argv'] = comparison_argv
        result = subprocess.run(comparison_argv, text=True, capture_output=True, timeout=60)
        (ROOT / 'comparison.log').write_text(result.stdout + result.stderr)
        report['comparison_exit_code'] = result.returncode
        save(report_path, report)
        assert result.returncode == 0, 'Frozen v2 comparison failed; diagnostic held'
        gate = json.loads(comparison.read_text())
        assert gate['status'] == 'complete' and gate['passed'] is True
        if args.with_diagnostic:
            arm('fp16-a', 'diagnostic')
        report['status'] = 'complete'
    except BaseException:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['finished_utc'] = now().isoformat()
        save(report_path, report)


def main():
    def handle_term(signum, frame):
        raise KeyboardInterrupt('Owned orchestration received SIGTERM')
    signal.signal(signal.SIGTERM, handle_term)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-utc', required=True)
    parser.add_argument('--comparison-script', type=Path, required=True)
    parser.add_argument('--with-diagnostic', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
