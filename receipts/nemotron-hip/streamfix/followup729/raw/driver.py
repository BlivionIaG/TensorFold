#!/usr/bin/env python3
"""Exact-head shared-kernel/model checks; GPU stages wait for explicit resource-release markers."""
import hashlib,json,os,re,shlex,signal,subprocess,time
from pathlib import Path

H=Path.home(); D=H/'hcx/streamfix/optimization/followup729'; D.mkdir(parents=True,exist_ok=True)
TREE=H/'nh/sfopt'; HEAD='729833615595df66c9f9a438ce9a85061843f010'
R={'schema_version':1,'source_commit':HEAD,'state':'starting','stages':[]}
if (D/'summary.json').exists():
 previous=json.loads((D/'summary.json').read_text())
 if previous['source_commit']==HEAD:R=previous;R.pop('error',None);R.pop('result',None)
def save():
 (D/'summary.json').write_text(json.dumps(R,indent=2)+'\n')
def require(ok,msg):
 if not ok: raise RuntimeError(msg)
def identity():
 got=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip()
 require(got==HEAD,'source head changed')
 dirty=subprocess.check_output(['git','status','--porcelain'],cwd=TREE,text=True).splitlines()
 require(all(x.startswith('?? out-') and x.endswith('/') for x in dirty),'source tree became dirty: '+repr(dirty))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def env(devs=None,rdna=False):
 e=os.environ.copy();e.pop('ROCR_VISIBLE_DEVICES',None);e.pop('LD_LIBRARY_PATH',None)
 if devs is not None:e['HIP_VISIBLE_DEVICES']=devs
 if rdna:e['LD_LIBRARY_PATH']=str(H/'.local/rocm-rdna/lib')
 return e
def run(name,cmd,e=None,timeout=5500):
 identity(); log=D/(name+'.log');start=time.monotonic();R['state']=name;save()
 with log.open('w') as out:
  proc=subprocess.Popen(cmd,cwd=TREE,env=e,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  try:status=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGKILL);status=proc.wait();status=124
 row={'name':name,'command':cmd,'source_commit':HEAD,'exit_status':status,'elapsed_seconds':round(time.monotonic()-start,3),'log':log.name,'log_sha256':sha(log)}
 R['stages'].append(row);save();identity();require(status==0,name+' nonzero exit '+str(status));return row

def build_stage(name,args):
 rel='inline729_final_'+name+'.log';row=run(name,[str(H/'hcx/zb.sh'),'sfopt',rel,*args])
 raw=H/'hcx/logs'/rel;data=raw.read_text(errors='replace');(D/(name+'.raw.log')).write_bytes(raw.read_bytes())
 sums=re.findall(r'^Build Summary: .*$',data,re.M);require(len(sums)==1,name+' completed summary missing')
 m=re.search(r'Build Summary: (\d+)/(\d+) steps succeeded',sums[0]);require(m and m[1]==m[2] and '(1 failed)' not in sums[0],name+' unsuccessful build')
 row.update({'exact_summary':sums[0],'raw_log':name+'.raw.log','raw_log_sha256':sha(raw)});save();return row,data

def wait_ready():
 marker=D/'model_tools_built.json'; allocation=D/'allocation.json'
 require(marker.is_file() and allocation.is_file(),'parent allocation and source-bound model build required')
 m=json.loads(marker.read_text());g=json.loads(allocation.read_text())
 require(g['source_commit']==HEAD and g['parent_allocated'] and g['exclusive_gpu_reservation'] and g['purpose']=='followup_models' and g['recovery_ab_finished'],'models cannot overlap recovery A/B')
 require(g['devices']=={'gfx1100':'0,1','gfx1030':'2,3,4,5'},'unexpected allocation')
 require(m['source_commit']==HEAD,'wrong build source')
 for arch in ('gfx1100','gfx1030'):require(m['architectures'][arch]['exit_status']==0,'model build failed')
 R['allocation_sha256']=sha(allocation);R['build_marker_sha256']=sha(marker);save();return m

def binary(marker,arch,name):
 b=marker['architectures'][arch]['binaries'][name];p=Path(b['path']);require(p.is_file() and sha(p)==b['sha256'],'binary changed or lacks measured build binding: '+name)
 return p,b['sha256']

def kernel_tests(arch,dev):
 rel='inline729_final_kernels_'+arch+'.log';raw=H/'hcx/logs'/rel
 completed=next((x for x in reversed(R['stages']) if x['name']=='kernels_'+arch and x.get('exit_status')==0),None)
 if completed and raw.is_file() and 'Build Summary:' in raw.read_text(errors='replace'):row=completed
 else:row=run('kernels_'+arch,[str(H/'hcx/zb.sh'),'sfopt',rel,'hip-kernel-test','-Dhip-arch='+arch,'-Doptimize=ReleaseFast','--summary','all','--verbose'],env(dev))
 data=raw.read_text(errors='replace');(D/('kernels_'+arch+'.raw.log')).write_bytes(raw.read_bytes())
 sums=re.findall(r'^Build Summary: .*$',data,re.M);require(len(sums)==1,'kernel completed summary missing')
 m=re.search(r'Build Summary: (\d+)/(\d+) steps succeeded; (\d+)/(\d+) tests passed',sums[0]);require(m and m[1]==m[2] and m[3]==m[4] and int(m[3])>=103,'kernel tests failed, skipped or cached without executed counts')
 binaries=set()
 for line in data.splitlines():
  try:words=shlex.split(line)
  except ValueError:continue
  for word in words:
   if (word.startswith('/') or word.startswith(('.zig-cache/','./.zig-cache/'))) and re.search(r'/o/[0-9a-f]+/(?:test|root)$',word):
    p=Path(word) if Path(word).is_absolute() else TREE/word
    if p.is_file() and os.access(p,os.X_OK):binaries.add(p)
 require(binaries,'verbose kernel log lacks actual test binary identity')
 row.update({'arch':arch,'devices':dev,'exact_summary':sums[0],'raw_log_sha256':sha(raw),'test_binaries':[{'sha256':sha(p),'path':str(p)} for p in sorted(binaries)]});save()

def qwen(marker,arch,dev):
 p,digest=binary(marker,arch,'tf-qwen35-test')
 for model,ids in [('Qwen3.5-9B-MLX-4bit','ids512'),('Qwen3.6-35B-A3B-MLX-4bit-MTP','ids35_512'),('Qwen3.8-27B-MLX-4bit','ids512')]:
  output=D/('qwen_'+arch+'_'+model+'.npy');ref=H/'nh/logs'/('ql110_'+arch+'_'+model+'_m110.npy')
  require(not output.exists(),'candidate output already exists');require(ref.is_file() and ref.stat().st_size>0,'baseline logits absent')
  reference_sha=sha(ref);row=run('qwen_'+arch+'_'+model,[str(p),'logits',str(H/'models'/model),str(H/'tf-truth'/(ids+'.npy')),str(output),'--decode'],env(dev,True),1800)
  require(output.is_file() and output.stat().st_size==ref.stat().st_size and sha(output)==reference_sha,'Qwen byte mismatch: '+model)
  require(sha(p)==digest,'Qwen binary changed during check')
  row.update({'arch':arch,'devices':dev,'model':model,'binary_sha256':digest,'output_bytes':output.stat().st_size,'output_sha256':sha(output),'reference_file':ref.name,'reference_sha256':reference_sha,'equality':'byte-identical'});save();output.unlink()

def flash(marker,arch,devs,world,model,want,port):
 p,digest=binary(marker,arch,'tf-flashnext-hip-test')
 for mode in ('on','off'):
  identity();name='flash_'+arch+'_'+mode;R['state']=name;save();procs=[];start=time.monotonic()
  try:
   for rank in [*range(1,world),0]:
    log=D/(name+'.rank'+str(rank)+'.log');handle=log.open('w');cmd=[str(p),'generate',str(H/'models'/model),'--tp',str(world),'--rank',str(rank),'--port',str(port),'--hash','--repeat','3']
    if mode=='off':cmd+=['--policy','graphs=off']
    proc=subprocess.Popen(cmd,cwd=TREE,env=env(devs),stdout=handle,stderr=subprocess.STDOUT,start_new_session=True);procs.append((rank,proc,handle,log,cmd))
   while any(proc.poll() is None for _,proc,_,_,_ in procs):
    require(time.monotonic()-start<1800,'Flash Next timed out')
    require(all(proc.poll() in (None,0) for _,proc,_,_,_ in procs),'Flash Next rank failed')
    time.sleep(1)
   require(all(proc.returncode==0 for _,proc,_,_,_ in procs),'Flash Next rank nonzero')
  finally:
   for _,proc,handle,_,_ in procs:
    if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    handle.close()
  root=next(log for rank,_,_,log,_ in procs if rank==0);text=root.read_text(errors='replace');hashes=re.findall(r'^logits hash ([0-9a-f]{16})$',text,re.M)
  require(hashes==[want]*3,'Flash Next requires exactly three expected hashes')
  require(not re.search(r'^FAIL\b',text,re.M),'Flash Next reports failure')
  if mode=='on':require(re.search(r'graphs: [1-9]\d* captured',text),'graphs-on did not capture')
  else:require(not re.search(r'graphs: [1-9]\d* captured',text),'graphs-off captured unexpectedly')
  require(sha(p)==digest,'Flash Next binary changed');identity()
  R['stages'].append({'name':name,'source_commit':HEAD,'binary_sha256':digest,'arch':arch,'devices':devs,'world':world,'mode':mode,'hashes':hashes,'elapsed_seconds':round(time.monotonic()-start,3),'ranks':[{'rank':rank,'exit_status':proc.returncode,'command':cmd,'log':log.name,'log_sha256':sha(log)} for rank,proc,_,log,cmd in procs]});save();port+=1

try:
 identity();save();marker=wait_ready()
 if not any(x['name']=='linux_full' and x['exit_status']==0 for x in R['stages']):build_stage('linux_full',['test','test-kernel-contract','test-golden','fuzz-cortheon','--cache-dir','/tmp/inline729-final-host-cache','--summary','all'])
 if not any(x['name']=='linux_plain' and x['exit_status']==0 for x in R['stages']):build_stage('linux_plain',['--cache-dir','/tmp/inline729-final-host-cache','--summary','all'])
 for arch,dev in [('gfx1100','0'),('gfx1030','2')]:kernel_tests(arch,dev);qwen(marker,arch,dev)
 flash(marker,'gfx1100','0,1',2,'Qwen3.8-Flash-Next-MLX-3bit-MTP','6fbb0a3f895f7fa1',29940)
 flash(marker,'gfx1030','2,3,4,5',4,'Qwen3.8-Flash-Next-MLX-4bit-MTP','349d29968cc98589',29944)
 identity();R['state']='complete';R['result']='PASS';save();(D/'MODEL_CHECKS_ALL_PASS').write_text(HEAD+'\n');print('MODEL_CHECKS_ALL_PASS',flush=True)
except BaseException as exc:
 R['state']='failed';R['result']='FAIL';R['error']=str(exc);save();print('MODEL_CHECKS_FAILED '+str(exc),flush=True);raise
