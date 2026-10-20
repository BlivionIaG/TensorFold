#!/usr/bin/env python3
"""Run only after a source-bound resource pass and explicit exclusive GPU allocation."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,re,shlex,signal,subprocess,threading,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--head',required=True);p.add_argument('--tree',default='sfopt');p.add_argument('--resources',type=Path,required=True);p.add_argument('--allocation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
H=Path.home();S=Path(__file__).resolve().parent;T=H/'nh'/a.tree;D=a.output.resolve();D.mkdir(parents=True,exist_ok=a.resume)
OLD='8649e9858b38e9ef3dcd8d579879c6512a363050';REVIEW='1596156e21769324b675ef5e0caae0452f361660';HARNESS='725bb8efa17532c15b3bc20df9f21491953f53377f8aab1ce909b892bfef2606'
PASS='PASS kernel-contract-limits: every registered launch fits gfx1030, gfx1100, gfx1151 and gfx1201, every record current, every control red'
R=json.loads((D/'result.json').read_text()) if a.resume else {'source_commit':a.head,'baseline_source_commit':OLD,'state':'starting','stages':[]};mu=threading.Lock()
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as b:
  for x in iter(lambda:b.read(4<<20),b''):h.update(x)
 return h.hexdigest()
def save(): (D/'result.json').write_text(json.dumps(R,indent=2)+'\n')
def git(*args):return subprocess.check_output(['git','-C',str(T),*args],text=True)
def identity():
 assert git('rev-parse','HEAD').strip()==a.head
 assert not git('status','--porcelain','--untracked-files=no').strip()
 assert all(x.split('/')[0].startswith('out-') for x in git('ls-files','--others','--exclude-standard').splitlines())
def run(name,cmd,env=None,timeout=5400):
 identity();log=D/(name+'.log');start=time.monotonic()
 with log.open('w') as out:
  proc=subprocess.Popen([str(x) for x in cmd],cwd=T,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124
 row={'name':name,'command':[str(x) for x in cmd],'source_commit':a.head,'exit_status':code,'seconds':round(time.monotonic()-start,3),'log':log.name,'log_sha256':sha(log)}
 with mu:R['stages'].append(row);save()
 identity();assert code==0,(name,code);return row,log

def validate_arch(arch,device):
 env=os.environ.copy();env.pop('ROCR_VISIBLE_DEVICES',None);env['HIP_VISIBLE_DEVICES']=str(device);env['LD_LIBRARY_PATH']=str(H/'.local/rocm-rdna/lib')
 for side in ('old','new'):
  binary=(H/'hcx/streamfix/finite_o3/builds'/('old_'+arch)) if side=='old' else D/'builds'/('new_'+arch)
  meta=json.loads(binary.with_suffix('.json').read_text());assert meta['source_commit']==(OLD if side=='old' else a.head) and meta['harness_sha256']==HARNESS and meta['binary_sha256']==sha(binary)
  compiler_log=binary.with_suffix('.compiler.log');assert sha(compiler_log)==meta['compiler_verbose_log_sha256']
  device_lines=[shlex.split(x) for x in compiler_log.read_text().splitlines() if re.search(r'(?<!\S)-triple amdgcn-amd-amdhsa(?:\s|$)',x) and '-cc1' in x]
  assert len(device_lines)==1 and all(flag in device_lines[0] for flag in ('-O3','-ffp-contract=off','-std=c++20',arch))
  dump=D/'dumps'/(side+'_'+arch);dump.mkdir(parents=True)
  row,_=run('finite_'+side+'_'+arch,[binary,dump],env,1800);
  with mu:row.update(source_commit=meta['source_commit'],validation_candidate_source_commit=a.head,binary_sha256=meta['binary_sha256'],harness_sha256=HARNESS,devices=str(device));save()
 comparison=D/('comparison_'+arch+'.json')
 run('compare_'+arch,['python3',S/'compare.py',D/'dumps'/('old_'+arch),D/'dumps'/('new_'+arch),H/'hcx/streamfix/finite_o3/builds'/('old_'+arch+'.json'),D/'builds'/('new_'+arch+'.json'),'--output',comparison])
 cr=json.loads(comparison.read_text());count,nan=(1080,60) if arch=='gfx1100' else (540,24)
 assert cr['pass'] and not cr['failed'] and len(cr['fixtures'])==count and cr['finite_input_nan_output_cases']==nan
 binary=T/('out-validate-'+arch)/'bin/tf-hip-kernel-contract';digest=sha(binary)
 row,log=run('contracts_'+arch,[binary],env,1800);text=log.read_text()
 pattern=rf'^SUMMARY {arch} 49 kernels: 49 pass gates 1-4, 0 fail, 0 known findings; 5 controls: 5 red as planted, 0 missed; 0 limit problems; gates \d+ ms, limits \d+ ms$'
 summaries=re.findall(pattern,text,re.M);assert len(summaries)==1
 failures=re.findall(r'^FAIL.*$',text,re.M)
 plants=['FAIL nonfinite control: fmax row max:','FAIL repeat control: atomic-add row sum:','FAIL repeat control: row sum writing its input:','FAIL shapes control: row sum losing a tail input:','FAIL sums control: dot summed in half:']
 assert len(failures)==5 and all(sum(v.startswith(w) for v in failures)==1 for w in plants)
 assert digest==sha(binary)
 with mu:row.update(binary_sha256=digest,devices=str(device),summary=summaries[0],controls='all original five plants red; no other FAIL');save()
 return {'arch':arch,'devices':str(device),'finite_fixtures':count,'finite_nan_controls':nan,'byte_mismatches':0,'comparison_sha256':sha(comparison),'summary':summaries[0],'contract_binary_sha256':digest}
try:
 assert re.fullmatch('[0-9a-f]{40}',a.head);identity();assert sha(S/'finite.hip')==HARNESS
 resources=json.loads(a.resources.read_text());allocation=json.loads(a.allocation.read_text())
 assert resources['validated'] and resources['source_commit']==a.head
 resource_log=Path(resources['four_chip_limits_log']);assert sha(resource_log)==resources['four_chip_limits_sha256'] and resource_log.read_text().rstrip().endswith(PASS)
 assert allocation['parent_allocated'] and allocation['exclusive_gpu_reservation'] and allocation['source_commit']==a.head
 devices=allocation['devices'];assert set(devices)=={'gfx1100','gfx1030'} and devices['gfx1100']!=devices['gfx1030']
 assert all(str(v) in {'0','1','2','3','4','5'} for v in devices.values()),'GPU indices outside authorized0to5'
 assert str(devices['gfx1100']) in {'0','1'} and str(devices['gfx1030']) in {'2','3','4','5'},'GPU architecture allocation mismatch'
 # Preserve the entire review fixture, core predicates and planted GPU controls.
 assert not git('diff','--name-only',REVIEW,a.head,'--','zig/tests/hip','zig/kernels/hip/contract_controls.hip').strip()
 changed=git('diff','--name-only',REVIEW,a.head,'--','zig/src/core/contract').splitlines();assert all(Path(x).name.startswith('hip_records') for x in changed),changed
 if a.resume:
  assert R['source_commit']==a.head and R['baseline_source_commit']==OLD
  assert R['resources_binding_sha256']==sha(a.resources) and R['allocation_sha256']==sha(a.allocation) and R['harness_sha256']==HARNESS
 R.update(resources_binding_sha256=sha(a.resources),allocation_sha256=sha(a.allocation),harness_sha256=HARNESS);save()
 if a.resume:
  previous=D/'result.json';prior=json.loads(previous.read_text());assert prior['source_commit']==a.head and prior['state']=='failed'
  expected={'build_finite_gfx1100','build_finite_gfx1030','build_contract_gfx1100','build_contract_gfx1030','limits'}
  assert len(prior['stages'])==5 and {x['name'] for x in prior['stages']}==expected
  for x in prior['stages']:assert x['exit_status']==0 and x['source_commit']==a.head and sha(D/x['log'])==x['log_sha256']
  assert sha(D/'four_chip_limits.log')==prior['four_chip_limits_sha256'] and (D/'four_chip_limits.log').read_text().rstrip().endswith(PASS)
  R=prior;R['resumed_after_infrastructure_parser_failure']=True;R.pop('error',None);R.pop('pass',None);R['state']='resuming_gpu_checks';save()
 else:
  with (H/'w1012/build.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   for arch in ('gfx1100','gfx1030'):
    run('build_finite_'+arch,['python3',S/'build.py',T,arch,D/'builds'/('new_'+arch),'--hipcc','/opt/rocm/core-7.14/bin/hipcc'])
  for arch in ('gfx1100','gfx1030'):
   name='inline_validate_build_'+arch+'.log';run('build_contract_'+arch,[H/'hcx/zb.sh',a.tree,name,'tf-hip-kernel-contract','-Dhip-arch='+arch,'-Doptimize=ReleaseFast','--prefix',T/('out-validate-'+arch)])
  # Re-audit all four chips using actual hydrated records at the exact candidate head.
  name='inline_validate_limits.log';run('limits',[H/'hcx/zb.sh',a.tree,name,'hip-kernel-contract-limits','-Dhip-arch=gfx1100','-Doptimize=ReleaseFast'])
  limitlog=H/'hcx/logs'/name;assert limitlog.read_text().rstrip().endswith(PASS);R['four_chip_limits_sha256']=sha(limitlog);(D/'four_chip_limits.log').write_bytes(limitlog.read_bytes());save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  jobs=[pool.submit(validate_arch,a,devices[a]) for a in ('gfx1100','gfx1030')];R['architectures']=[j.result() for j in jobs]
 identity();R['state']='complete';R['pass']=True;save();(D/'INLINE_CORRECTNESS_ALL_PASS').write_text(a.head+'\n');print('INLINE_CORRECTNESS_ALL_PASS',a.head,flush=True)
except BaseException as exc:
 R.update({'state':'failed','pass':False,'error':str(exc)});save();raise
