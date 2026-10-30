import datetime, hashlib, json, os, pathlib, signal, subprocess, sys, time, traceback
ROOT=pathlib.Path('/work/e2e')
PYTHON='/work/.venv/bin/python'
ARMS=['baseline','default-a','fp16-a','fp16-b','default-b']
START=time.monotonic()
report={'status':'running','order':ARMS,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':[]}
(ROOT/'results').mkdir(exist_ok=False)
(ROOT/'logs').mkdir(exist_ok=False)
def save():
    tmp=ROOT/'orchestration.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(ROOT/'orchestration.json')
def idle():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=20)
    assert not p.stdout.strip(),p.stdout

def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=60)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=20)

def snapshot(path):
    p=subprocess.run(['nvidia-smi','-q'],capture_output=True,text=True,timeout=20,check=True)
    path.write_text(p.stdout)

save()
telemetry=None
try:
    idle()
    fields='timestamp,index,uuid,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used'
    telemetry_file=(ROOT/'gpu-telemetry.csv').open('w')
    telemetry=subprocess.Popen(['nvidia-smi','--query-gpu='+fields,'--format=csv','--loop-ms=1000'],stdout=telemetry_file,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(2)
    assert telemetry.poll() is None,'GPU telemetry failed'
    for arm in ARMS:
        assert time.monotonic()-START<75*60,'Time budget exceeded before next engine'
        idle()
        source='/work/baseline' if arm=='baseline' else '/work/source'
        env=os.environ.copy();env.pop('RAY_ADDRESS',None)
        env.update(PYTHONPATH=source+':/work/e2e:/work/graph-perf',CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='4',VLLM_CACHE_ROOT='/work/cache/vllm-e2e',TRITON_CACHE_DIR='/work/cache/triton-e2e')
        argv=[PYTHON,'-u',str(ROOT/'run_e2e.py'),'--arm',arm,'--helper-dir','/work/graph-perf','--tactic-witness','/work/graph-perf/results/baseline/result.json','--source-pins','/work/source-pins.json','--prompts','/work/graph-perf/prompts.json','--model','/work/models/MiniMax-M3-NVFP4','--cache-dir','/work/cache/e2e-flashinfer/'+arm,'--output',str(ROOT/'results'/arm)]
        if arm!='baseline':argv+=['--baseline-result',str(ROOT/'results/baseline/result.json')]
        if arm=='fp16-b':argv+=['--reference-result',str(ROOT/'results/fp16-a/result.json')]
        row={'arm':arm,'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'environment':{k:env[k] for k in ['PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','VLLM_CACHE_ROOT','TRITON_CACHE_DIR']}}
        report['runs'].append(row);save();snapshot(ROOT/'logs'/f'{arm}.gpu-before.txt')
        with (ROOT/'logs'/f'{arm}.log').open('w') as stream:
            p=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=1000)
            except BaseException:stop(p);raise
        row.update(exit_code=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
        snapshot(ROOT/'logs'/f'{arm}.gpu-after.txt')
        print(json.dumps(row),flush=True)
        assert code==0,(arm,code)
        idle()
    argv=[PYTHON,str(ROOT/'compare_e2e.py'),'--results-root',str(ROOT/'results'),'--output',str(ROOT/'comparison.json')]
    p=subprocess.run(argv,capture_output=True,text=True,timeout=90)
    (ROOT/'comparison.log').write_text(p.stdout+p.stderr)
    report.update(status='complete' if p.returncode==0 else 'comparison_failed',comparison_exit_code=p.returncode)
except BaseException:
    report.update(status='failed',traceback=traceback.format_exc());raise
finally:
    if telemetry is not None:stop(telemetry)
    report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
