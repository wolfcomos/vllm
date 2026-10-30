import datetime,json,os,pathlib,subprocess,time
root=pathlib.Path('/work/tests-ready');root.mkdir(exist_ok=False)
files=['tests/kernels/attention/test_minimax_m3.py','tests/kernels/test_minimax_m3_sparse_attn_fp8_scale.py','tests/kernels/attention/test_minimax_m3_fp8_triton_indexer.py','tests/kernels/attention/test_minimax_m3_msa_cutlass_sparse_decode.py']
report={'files':files,'runs':[]}
for arm,source in [('baseline','/work/baseline'),('candidate','/work/source')]:
 env=dict(os.environ,PYTHONPATH=source,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',TRITON_CACHE_DIR='/work/cache/triton-tests',VLLM_CACHE_ROOT='/work/cache/vllm-tests')
 cmd=['/work/.venv/bin/python','-m','pytest','-q','-ra',*files,'--junitxml='+str(root/(arm+'.xml'))]
 row={'arm':arm,'argv':cmd,'environment':{k:env[k] for k in ['PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS']},'start':datetime.datetime.now(datetime.timezone.utc).isoformat()};report['runs'].append(row)
 (root/'status.json').write_text(json.dumps(report,indent=2))
 with (root/(arm+'.log')).open('w') as log:
  p=subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
 row.update(exit_code=p.returncode,end=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (root/'status.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
 if p.returncode>1:break
