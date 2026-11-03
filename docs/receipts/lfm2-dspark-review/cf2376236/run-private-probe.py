import argparse, fcntl, json, os, re, sys
from pathlib import Path
from types import SimpleNamespace

p=argparse.ArgumentParser()
p.add_argument('--driver',required=True)
p.add_argument('--binary',required=True)
p.add_argument('--target',required=True)
p.add_argument('--drafter',required=True)
p.add_argument('--output',required=True)
p.add_argument('--lock-inode',required=True,type=int)
a=p.parse_args()
sys.path.insert(0,a.driver)
import bench_lfm2_dspark as d

root=Path(a.output)
root.mkdir(parents=True,exist_ok=True)
settings={'label':'PRIVATE_CAUSAL_DIAGNOSTIC_NOT_QUALIFICATION','base_source':'5bed4bd92c7c257f963046b945d83fc931dcbf29','binary_sha256':d.sha256_file(Path(a.binary)),'context':4096,'parallel':16,'tokens':512,'cases':[{'prompt':'code','temperature':0.0,'streams':1},{'prompt':'code','temperature':1.0,'streams':8}]}
(root/'binding.json').write_text(json.dumps(settings,indent=2)+'\n')
rows=[]
lock=open(d.LOCK,'a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert os.fstat(lock.fileno()).st_ino==a.lock_inode
assert d.pressure_level() in (None,1)
assert not d.competing_processes(Path(a.binary).name)
try:
 for case in settings['cases']:
  prompt=next(item for item in d.bc.PROMPTS if item['name']==case['prompt'])
  width=case['streams'];temp=case['temperature'];name=f"{case['prompt']}-T{temp:g}-w{width}"
  refs={}
  for arm in ['plain','bf16','bits8']:
   out=root/(name+'-'+arm)
   out.mkdir()
   args=SimpleNamespace(binary=a.binary,target=a.target,drafter=a.drafter,widths=str(width),temperatures=str(temp),tokens=512,reps=1,seed=1234,port=18194,context=4096,startup_timeout=300)
   ctx=d.Context(args,out);ctx.logs.mkdir()
   with d.Server(ctx,name+'-'+arm,arm) as srv:
    child=srv.proc
    specs=[(prompt,1234+i) for i in range(width)]
    warm=d.bc.together(ctx.base,d.SERVED_NAME,specs,16,temp)
    ctx.raw(name,'warmup',warm)
    srv.check()
    runs=d.bc.together(ctx.base,d.SERVED_NAME,specs,512,temp)
    ctx.raw(name,'measured',runs)
    srv.check()
    for run in runs:
     errors=d.problems(run,512,False)
     assert not errors,errors
     if arm=='plain':refs[run['seed']]=run
     else:
      mismatch=d.compare(refs[run['seed']],run,'same_head_arm',run['seed'])
      assert not mismatch,mismatch
    aggregate=d.bc.aggregates(runs)
   assert child.returncode==0,child.returncode
   text=ctx.logs.joinpath(name+'-'+arm+'.server.log').read_text()
   targets={};enrollments=[]
   for line in text.splitlines():
    match=re.search(r'DSPARK_PRIVATE_TARGET phase=(\S+) calls=(\d+) rows=(\d+) captured_calls=(\d+) encode_cpu_ms=([0-9.]+) gpu_ms=([0-9.]+) timed_gpu_calls=(\d+)',line)
    if match:targets[match[1]]={'calls':int(match[2]),'rows':int(match[3]),'captured_calls':int(match[4]),'encode_cpu_ms':float(match[5]),'gpu_ms':float(match[6]),'timed_gpu_calls':int(match[7])}
    match=re.search(r'DSPARK_PRIVATE_ENROLL bits=(\d+) max_new=(\d+) streams=(\d+) heads=(\d+) tracked=(true|false) observed=(true|false) remaining=(\d+) rate0=([0-9.]+)',line)
    if match:enrollments.append({'bits':int(match[1]),'max_new':int(match[2]),'streams':int(match[3]),'heads':int(match[4]),'tracked':match[5]=='true','observed':match[6]=='true','remaining':int(match[7]),'rate0':float(match[8])})
   assert set(targets)=={'prefill','decode'},'missing target timing receipts'
   assert all(value['calls']==value['timed_gpu_calls'] for value in targets.values()),'target GPU duration unavailable'
   measured_enrollment=[value for value in enrollments if value['max_new']==512]
   assert len(measured_enrollment)==(0 if arm=='plain' else width),'missing actual request enrollment decisions'
   counters={}
   for line in text.splitlines():
    m=re.search(r'DSPARK_PRIVATE_COUNTER kind=(\S+) calls=(\d+)(?: wall_ms=([0-9.]+))?',line)
    if m:counters[m[1]]={'calls':int(m[2]),'wall_ms':float(m[3]) if m[3] else None}
   m=re.search(r'DSPARK_PRIVATE_HOLDS passes=(\d+) blocks=(\d+) retained=(\d+) staged_rows=(\d+) encode_gpu_ms=([0-9.]+)',text)
   holds=dict(zip(['passes','blocks','retained','staged_rows','encode_gpu_ms'],[int(m[i]) for i in range(1,5)]+[float(m[5])])) if m else None
   assert 'step' in counters, 'missing private attribution output'
   assert (arm=='plain')==(holds is None)
   row={'case':case,'arm':arm,'aggregate_tps':aggregate['aggregate_tps'],'activity':d.total_activity(runs),'private_counters':counters,'private_holds':holds,'private_target':targets,'enrollments':enrollments,'returncode':child.returncode,'log_sha256':d.sha256_file(ctx.log_path if hasattr(ctx,'log_path') else srv.log_path)}
   rows.append(row)
   (root/'observed.json').write_text(json.dumps({'binding':settings,'records':rows},indent=2)+'\n')
   print(json.dumps(row),flush=True)
finally:
 fcntl.flock(lock,fcntl.LOCK_UN)
 lock.close()
print(json.dumps({'complete':True,'label':settings['label'],'records':len(rows)}),flush=True)
