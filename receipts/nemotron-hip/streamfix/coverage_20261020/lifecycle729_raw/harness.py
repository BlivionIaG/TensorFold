"""Run existing source-bound model window and mixed-stream tests on one owned rig at a time."""
from pathlib import Path
import os,json,hashlib,subprocess,time,signal,fcntl,ctypes,socket,sys
H=Path.home();S=Path(__file__).resolve().parent;HEAD='729833615595df66c9f9a438ce9a85061843f010'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
admission={k:os.environ.get(k) for k in ['TENSORFOLD_MEMORY_RESERVE_GIB','TENSORFOLD_HIP_MEMORY_LIMIT_GB']};assert all(v is None for v in admission.values()),'Expected unchanged default admission environment'
lock=(H/'opt/gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
repo=H/'nh/sfopt';assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==HEAD
for args in [['diff','--quiet'],['diff','--cached','--quiet']]:subprocess.run(['git','-C',str(repo),*args],check=True)
mf=H/'hcx/streamfix/optimization/followup729/model_tools_built.json';m=json.loads(mf.read_text());assert m['source_commit']==HEAD
model=H/'models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit';meta=model/'.cache/huggingface/download/config.json.metadata';assert meta.read_text().splitlines()[0]=='d9d758fb83953437f7263256b0d96157e2a348b8'
ids=H/'nem-work/ids/chat_full.ids.bin';assert ids.stat().st_size%4==0 and 8<ids.stat().st_size//4<4096
h=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');mapping=[]
for i,pci in [(0,'0000:c7:00.0'),(1,'0000:83:00.0'),(2,'0000:47:00.0')]:
 b=ctypes.create_string_buffer(64);assert h.hipDeviceGetPCIBusId(b,64,i)==0 and b.value.decode()==pci;mapping.append({'HIP':i,'pci':pci})
D=S/'runs'/time.strftime('%Y%m%dT%H%M%S');D.mkdir(parents=True,exist_ok=False)
r={'source_commit':HEAD,'checkpoint_revision':meta.read_text().splitlines()[0],'manifest_sha256':sha(mf),'harness_sha256':sha(Path(__file__)),'input':{'path':str(ids),'sha256':sha(ids),'tokens':ids.stat().st_size//4},'gpu_mapping':mapping,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'memory_admission_environment':admission,'existing_lock':{'path':str(H/'opt/gpu.lock'),'held':True,'method':'fcntl.flock EX NB, descriptor retained entire run'},'runs':[],'complete':False,'pass':False}
save(D/'summary.json',r);(D/'model_tools_built.raw.json').write_bytes(mf.read_bytes());(D/'harness.py').write_bytes(Path(__file__).read_bytes())
STOP=False
class Interrupted(Exception):pass
def stop(*_):
 global STOP
 STOP=True
for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
try:
 for rig,arch,devices,world in [('W7800tp2','gfx1100','0,1',2),('V620tp1','gfx1030','2',1)]:
  entry=m['architectures'][arch];assert entry['exit_status']==0;b=entry['binaries']['tf-nemotron-hip-test'];assert sha(Path(b['path']))==b['sha256']
  for mode in ['window','rounds']:
   # Only an explicitly masked, identified overlapping GPU model job prevents our allocation.
   conflicts=[]
   for pp in Path('/proc').iterdir():
    if not pp.name.isdigit() or pp.name==str(os.getpid()):continue
    try:
     av=(pp/'cmdline').read_bytes().split(b'\0');exe=Path(av[0].decode()).name
     if not (exe.startswith(('tensorfold-native','tf-nemotron-hip-test','tf-qwen35-test','tf-flashnext-hip-test')) or b'vllm serve' in b' '.join(av)):continue
     masks=[e.partition(b'=')[2].decode() for e in (pp/'environ').read_bytes().split(b'\0') if e.startswith((b'HIP_VISIBLE_DEVICES=',b'ROCR_VISIBLE_DEVICES=',b'CUDA_VISIBLE_DEVICES='))]
     if len(masks)==1 and all(v.isdigit() for v in masks[0].split(',')) and set(map(int,masks[0].split(',')))&{0,1,2}:conflicts.append(pp.name)
    except (OSError,ValueError):pass
   assert not conflicts,conflicts
   port=30600+len(r['runs']);
   with socket.socket() as sock:sock.bind(('127.0.0.1',port))
   env=os.environ.copy()
   for key in ['HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES']:env.pop(key,None)
   env.update(HIP_VISIBLE_DEVICES=devices,LD_LIBRARY_PATH=str(H/'.local/rocm-rdna/lib'))
   if mode=='window':env['TF_NEMOTRON_PAGE_SHIFT']='1'
   else:env.pop('TF_NEMOTRON_PAGE_SHIFT',None)
   row={'rig':rig,'mode':mode,'binary_sha256':b['sha256'],'devices':devices,'page_shift':mode=='window','ranks':[],'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};procs=[];logs=[]
   r['runs'].append(row);save(D/'summary.json',r)
   try:
    for rank in reversed(range(world)):
     argv=[b['path'],mode,str(model),str(ids),str(D/(rig+'_'+mode+'.unused')),'--steps','4']+(['--streams','4'] if mode=='rounds' else [])+(['--tp',str(world),'--rank',str(rank),'--master','127.0.0.1','--port',str(port)] if world>1 else [])
     log=D/(rig+'_'+mode+'.r'+str(rank)+'.log');fd=log.open('wb');logs.append(fd);p=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=fd,stderr=subprocess.STDOUT,start_new_session=True);procs.append((rank,p,log));row['ranks'].append({'rank':rank,'pid':p.pid,'argv':argv,'log':log.name});save(D/'summary.json',r)
    identity=D/(rig+'_'+mode+'.identity.json');capture=H/'hcx/streamfix/optimization/followup729/capture_identity.py'
    for _ in range(60):
     c=subprocess.run([sys.executable,str(capture),str(identity),*[str(p.pid) for _,p,_ in procs]],capture_output=True)
     if c.returncode==0:break
     if all(p.poll() is not None for _,p,_ in procs):break
     time.sleep(.5)
    assert c.returncode==0 and identity.is_file(),'No successful actual runtime identity capture'
    identity_data=json.loads(identity.read_text());assert len(identity_data['processes'])==world and all(z['executable']['sha256']==b['sha256'] for z in identity_data['processes'])
    assert identity_data['checkpoint_metadata']['config.json.metadata'].splitlines()[0]==r['checkpoint_revision']
    row['runtime_identity']={'path':identity.name,'sha256':sha(identity),'actual_executable_hashes_verified':True}
    deadline=time.monotonic()+900
    while any(p.poll() is None for _,p,_ in procs):
     if STOP:raise Interrupted('User or signal interrupted')
     assert time.monotonic()<deadline,'test timeout'
     time.sleep(1)
    for rank,p,log in procs:
     x=next(z for z in row['ranks'] if z['rank']==rank);x.update(exit_status=p.wait(),log_sha256=sha(log));assert x['exit_status']==0
     text=log.read_text(errors='replace')
     if mode=='window':assert 'the same bits; first differing row null' in text and 'the snapshotted window equals the plain one; rows after a rollback equal the single steps' in text
     else:assert '4 streams, 10 rows' in text and 'the same bits (0 rows differ)' in text
    row['pass']=True
   finally:
    for rank,p,log in procs:
     if p.poll() is None:
      os.killpg(p.pid,signal.SIGTERM)
      try:p.wait(timeout=10)
      except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
     x=next(z for z in row['ranks'] if z['rank']==rank);x['exit_status']=p.returncode
    for fd in logs:fd.close()
    for _,_,log in procs:
     next(z for z in row['ranks'] if z['log']==log.name)['log_sha256']=sha(log)
    row['cleanup_all_owned_ranks_reaped']=all(p.poll() is not None for _,p,_ in procs)
    remaining_groups=[]
    for _,p,_ in procs:
     try:os.killpg(p.pid,0);remaining_groups.append(p.pid)
     except ProcessLookupError:pass
    row['remaining_owned_process_groups']=remaining_groups;assert not remaining_groups,'Owned descendants remain'
    row['ended_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save(D/'summary.json',r)
 r['complete']=True;r['pass']=all(x['pass'] for x in r['runs']);save(D/'summary.json',r);print('COMPLETE '+str(D),flush=True)
except BaseException as e:
 r['error']=str(e);save(D/'summary.json',r);raise
finally:
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
