#!/usr/bin/env python3
"""Bounded exact-head proofs. Staged offline; run only after parent release of recovery A/B."""
import argparse,concurrent.futures,hashlib,json,os,re,signal,subprocess,threading,time
from pathlib import Path
p=argparse.ArgumentParser(__doc__);p.add_argument('--allocation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);a=p.parse_args()
HEADS=[('foundation','9f7d265d802cf394b4f60a142aa9d5214cdbfdbf',23),('relu2','f8cd5a4d72805fabe7350746ce9ba91274b54d29',43)]
CANDIDATE='729833615595df66c9f9a438ce9a85061843f010'
LIMIT='PASS kernel-contract-limits: every registered launch fits gfx1030, gfx1100, gfx1151 and gfx1201, every record current, every control red'
PLANTS=['FAIL nonfinite control: fmax row max:','FAIL repeat control: atomic-add row sum:','FAIL repeat control: row sum writing its input:','FAIL shapes control: row sum losing a tail input:','FAIL sums control: dot summed in half:']
H=Path.home();D=a.output.resolve();R={'schema_version':1,'candidate_context':CANDIDATE,'state':'starting','heads':[]};TREE=None;SHA=None;created=False;mu=threading.RLock()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save():
 with mu:(D/'result.json').write_text(json.dumps(R,indent=2)+'\n')
def guard():
 g=json.loads(a.allocation.read_text());assert g['parent_allocated'] and g['exclusive_gpu_reservation'] and g['recovery_ab_finished'] and g['purpose']=='ownhead_contracts'
 assert g['candidate_context']==CANDIDATE and g['source_commits']==[h for _,h,_ in HEADS]
 assert g['devices']=={'gfx1100':'0','gfx1030':'2'}
 if TREE:
  assert subprocess.check_output(['git','-C',str(TREE),'rev-parse','HEAD'],text=True).strip()==SHA
  assert not subprocess.check_output(['git','-C',str(TREE),'status','--porcelain','--untracked-files=no'],text=True).strip()
 return g

def run(name,cmd,row,env=None,timeout=5400):
 guard();log=D/(name+'.log');start=time.monotonic()
 with log.open('w') as f:
  proc=subprocess.Popen([str(x) for x in cmd],cwd=TREE,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124
 rec={'name':name,'source_commit':SHA,'command':[str(x) for x in cmd],'exit_status':code,'elapsed_seconds':round(time.monotonic()-start,3),'log':log.name,'log_sha256':sha(log)}
 with mu:row['stages'].append(rec);save()
 guard();assert code==0,(name,code)
 return rec,log
try:
 # Guard precedes all output/worktree/build mutations.
 g=guard();D.mkdir(parents=True,exist_ok=False);created=True;R['allocation_sha256']=sha(a.allocation);save()
 repo=H/'nh/sfopt';assert sha(a.bundle)=='81586133deadfb29f5130b09035eb303f9c672caaba6d4bd166487763bc2cc41'
 refs={'refs/tags/validation-inline729-foundation':HEADS[0][1],'refs/tags/validation-inline729-relu2':HEADS[1][1]}
 listed=subprocess.check_output(['git','bundle','list-heads',str(a.bundle)],text=True)
 assert {line.split()[1]:line.split()[0] for line in listed.splitlines()}==refs
 # Import only the two private proof refs. Existing branches and checkouts are unchanged.
 for name,cmd in [('verify',['git','-C',str(repo),'bundle','verify',str(a.bundle)]),('import',['git','-C',str(repo),'fetch','--no-tags',str(a.bundle),*[src+':refs/validation/inline729/'+src.rsplit('/',1)[1] for src in refs]])]:
  guard();log=D/('bundle_'+name+'.log')
  with log.open('w') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
  with mu:R.setdefault('bundle_operations',[]).append({'name':name,'command':cmd,'exit_status':code,'log':log.name,'log_sha256':sha(log),'bundle_sha256':sha(a.bundle)});save()
  assert code==0,(name,code)
 for label,head,count in HEADS:
  SHA=head;TREE=None;guard();repo=H/'nh/sfopt';tree=H/'nh'/('inline729-proof-'+label)
  subprocess.run(['git','-C',str(repo),'cat-file','-e',head+'^{commit}'],check=True)
  assert not tree.exists(),'fresh isolated proof tree required'
  subprocess.run(['git','-C',str(repo),'worktree','add','--detach',str(tree),head],check=True)
  TREE=tree;guard();row={'name':label,'source_commit':head,'expected_kernels':count,'stages':[],'architectures':[]};R['heads'].append(row);save()
  # zb.sh owns the shared build lock. Every run uses a fresh local cache; no cached test-count claim.
  cache=D/('cache-'+label)
  for arch in ('gfx1100','gfx1030'):
   prefix=tree/('out-proof-'+arch);rawname='inline729_'+label+'_'+arch+'_'+str(os.getpid())+'_build.log'
   rec,_=run(label+'_build_'+arch,[H/'hcx/zb.sh',tree.name,rawname,'tf-hip-kernel-contract','-Dhip-arch='+arch,'-Doptimize=ReleaseFast','--cache-dir',cache,'--prefix',prefix],row)
   raw=H/'hcx/logs'/rawname;dst=D/(label+'_build_'+arch+'.raw.log');dst.write_bytes(raw.read_bytes());rec.update(raw_log=dst.name,raw_log_sha256=sha(dst));save()
  rawname='inline729_'+label+'_'+str(os.getpid())+'_limits.log'
  rec,_=run(label+'_fourchip_limits',[H/'hcx/zb.sh',tree.name,rawname,'hip-kernel-contract-limits','-Dhip-arch=gfx1100','-Doptimize=ReleaseFast','--cache-dir',cache],row)
  raw=H/'hcx/logs'/rawname;dst=D/(label+'_fourchip_limits.raw.log');dst.write_bytes(raw.read_bytes());assert dst.read_text().rstrip().endswith(LIMIT)
  rec.update(raw_log=dst.name,raw_log_sha256=sha(dst),exact_summary=LIMIT);save()
  def contract_arch(arch):
   binary=tree/('out-proof-'+arch)/'bin/tf-hip-kernel-contract';digest=sha(binary);env=os.environ.copy();env.pop('ROCR_VISIBLE_DEVICES',None);env['HIP_VISIBLE_DEVICES']=g['devices'][arch];env['LD_LIBRARY_PATH']=str(H/'.local/rocm-rdna/lib')
   rec,log=run(label+'_contracts_'+arch,[binary],row,env,1800);text=log.read_text();summaries=re.findall(rf'^SUMMARY {arch} {count} kernels: {count} pass gates 1-4, 0 fail, 0 known findings; 5 controls: 5 red as planted, 0 missed; 0 limit problems; gates \d+ ms, limits \d+ ms$',text,re.M)
   failures=re.findall(r'^FAIL.*$',text,re.M);assert len(summaries)==1 and len(failures)==5 and all(sum(x.startswith(p) for x in failures)==1 for p in PLANTS)
   assert sha(binary)==digest
   with mu:
    rec.update(binary_sha256=digest,devices=g['devices'][arch],summary=summaries[0]);row['architectures'].append({'arch':arch,'summary':summaries[0],'binary_sha256':digest});save()
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   jobs=[pool.submit(contract_arch,arch) for arch in ('gfx1100','gfx1030')]
   for job in jobs:job.result()
  row['validated']=True;save()
 R['state']='complete';R['validated']=True;save();(D/'OWNHEAD_CONTRACTS_ALL_PASS').write_text('\n'.join(h for _,h,_ in HEADS)+'\n');print('OWNHEAD_CONTRACTS_ALL_PASS',flush=True)
except BaseException as e:
 if created:R.update(state='failed',validated=False,error=str(e));save()
 raise
