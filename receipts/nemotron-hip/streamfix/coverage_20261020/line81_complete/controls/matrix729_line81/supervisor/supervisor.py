"""LOCAL prepared supervisor, dry by default; actual reviewed authorization and completed activeVon and independent collection release required."""
from pathlib import Path
import argparse,json,hashlib,os,sys,time,subprocess,signal,fcntl,ctypes,re
S=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=S/'config.json');p.add_argument('--execute',action='store_true');a=p.parse_args();C=json.loads(a.config.read_text());H=Path.home();expand=lambda s:Path(s).expanduser();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();HEAD='729833615595df66c9f9a438ce9a85061843f010';KEYS=['TENSORFOLD_MEMORY_RESERVE_GIB','TENSORFOLD_HIP_MEMORY_LIMIT_GB']
def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
if not a.execute:print(json.dumps({'execute':False,'blocks':C['blocks'],'active_von_state':C['active_von_state'],'required_boot':C['boot_id'],'no_launch':True},indent=2));raise SystemExit
assert C['root_authorized'] is True and C['source_commit']==HEAD
assert re.fullmatch(r'[0-9a-f]{64}',C['active_von_state_sha256'] or '') and re.fullmatch(r'[0-9a-f]{64}',C['collection_release_sha256'] or '')
assert [b['name']for b in C['blocks']]==['v_on_close','v_off_close','w_on_close','w_off_ono']
assert {k:os.environ.get(k)for k in KEYS}=={k:None for k in KEYS}
assert all(os.environ.get(k)is None for k in ['HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES']),'Enumeration requires unchanged unmasked ordinal mapping'
D=expand(C['output_root'])/time.strftime('%Y%m%dT%H%M%S');D.mkdir(parents=True,exist_ok=False);R={'started_utc':now(),'complete':False,'phase':'waiting_for_active_von_and_collection_release','blocks':[],'scope':'OwnedGPU0-2 serial groups, sharedhostjobs6-9 permitted; no physicalhostexclusiveclaim'};write(D/'config.raw.json',C)
def save():write(D/'summary.json',R)
def reading(p):
 try:return json.loads(p.read_text())
 except(FileNotFoundError,json.JSONDecodeError):return None # Live writer may not have completed its checkpoint write.
def boot():assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==C['boot_id'],'Boot changed; stop without retry'
def alive(pid):
 try:os.kill(int(pid),0);return True
 except ProcessLookupError:return False

def groupgone(pid):
 try:os.killpg(int(pid),0);return False
 except ProcessLookupError:return True
def lockfree():
 for q in [H/'opt/gpu.lock',H/'w1012/build.lock']:
  assert q.is_file()
  with q.open('r')as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
 # Probe only: reviewed child runner acquires and retains both canonical locks itself.
def overlap():
 observed=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit()or int(p.name)==os.getpid():continue
  try:
   args=[x.decode(errors='replace')for x in(p/'cmdline').read_bytes().split(b'\0')if x]
   if not args:continue
   exe=Path(args[0]).name;line=' '.join(args)
   if not(exe.startswith(('tensorfold-native','tf-nemotron-hip-test','tf-qwen35-test','tf-flashnext-hip-test'))or any(t in line for t in ['vllm.entrypoints','vllm serve','sglang.launch_server','llama-server'])):continue
   env={}
   for e in(p/'environ').read_bytes().split(b'\0'):
    k,sep,v=e.partition(b'=')
    if sep and k.decode(errors='replace')in ['HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES']:env[k.decode()]=v.decode(errors='replace')
   direct=len(env)==1 and re.fullmatch(r'\d+(?:,\d+)*',next(iter(env.values())))
   conflict=bool(direct and set(map(int,next(iter(env.values())).split(',')))&{0,1,2});observed.append({'pid':int(p.name),'executable':exe,'masks':env,'positive_reserved_overlap':conflict});assert not conflict,'Known model overlaps assignedGPU0-2'
  except(FileNotFoundError,ProcessLookupError):continue
  except PermissionError:observed.append({'pid':int(p.name),'unknown_permission_denied':True})
 return observed

def mapping():
 lib=expand(C['hip_library']);assert sha(lib)==C['hip_library_sha256'];h=ctypes.CDLL(str(lib));h.hipDeviceGetPCIBusId.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int];h.hipDeviceGetPCIBusId.restype=ctypes.c_int;rows=[]
 for i,pci in [(0,'0000:c7:00.0'),(1,'0000:83:00.0'),(2,'0000:47:00.0')]:
  b=ctypes.create_string_buffer(64);assert h.hipDeviceGetPCIBusId(b,64,i)==0 and b.value.decode()==pci;rows.append({'ordinal':i,'pci':pci})
 return rows

STOP=False
def stop(*_):
 global STOP
 STOP=True
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
def cleanup_state(state):
 ranks=[]
 for groups in state['conditions'].values():
  for g in groups:
   rig=g['rig'];world=2 if rig=='W7800tp2'else 1;arch='gfx1100'if world==2 else'gfx1030'
   for hold in g['holds']:
    assert hold['teardown'] is True and len(hold['ranks'])==world and {r['rank']for r in hold['ranks']}==set(range(world))
    assert hold['source_commit']==C['source_bindings'][hold['tree']]['source_commit'] and hold['binary_sha256']==C['source_bindings'][hold['tree']]['binary_sha256'][arch]
    for r in hold['ranks']:
     assert r['process_group_gone'] and r['alive_before_intentional_teardown'] and r['exit_reason']=='intentional native-server teardown' and not r.get('forced_kill_after_term_timeout',False)
     assert r['actual_exit_status']==0 or(rig=='W7800tp2'and r['rank']==1 and r['actual_exit_status']==-15),'Unexpected rank exit'
     assert not alive(r['pid'])and groupgone(r['pid'])
    ranks.extend(hold['ranks'])
 return ranks
try:
 boot();save();active=expand(C['active_von_state']);release_path=expand(C['collection_release'])
 while True:
  boot();assert not STOP,'Supervisor stop requested before launch';j=reading(active)
  if j and j.get('complete'):
   assert sha(active)==C['active_von_state_sha256'] and j['selected_block']=={'rig':'V620tp1','mode':'on'} and j['pair_block_complete'] and j['actual_holds']==2
   assert j['measured_sources']=={k:v['source_commit']for k,v in C['source_bindings'].items()} and j['boot_id_start']==j['boot_id_end']==C['boot_id']
   assert j['global_output_equality']['validated_cell_files']==8 and j['global_output_equality']['validated_requests_including_warmups']==144
   cleanup_state(j)
   launch=reading(expand(C['active_von_launch_result']));started=reading(expand(C['active_von_launched']))
   if not launch or not started:time.sleep(5);continue
   assert launch['complete']and launch['coordinator_exit_status']==launch['observer_exit_status']==0 and Path(launch['run_directory'])==active.parent
   assert all(not alive(started[k])for k in ['coordinator_pid','observer_pid','launcher_pid'])
   release=reading(release_path)
   if release is None:time.sleep(10);continue
   assert sha(release_path)==C['collection_release_sha256'] and release['source_commit']==HEAD and release['boot_id']==C['boot_id']
   assert Path(release['run_directory'])==active.parent and release['state_sha256']==sha(active) and release['all_downloads_verified']is True and release['owned_groups_gone']is True and release['independent_audit_pass']is True
   assert re.fullmatch(r'[0-9a-f]{64}',release['independent_audit_sha256']) and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',release['released_at_utc'])
   R['active_von_gate']={'state_sha256':sha(active),'release_sha256':sha(release_path),'release':release,'observed_utc':now()};save();break
  time.sleep(10)
 for B in C['blocks']:
  assert not STOP,'Supervisor stop requested before nextblock';boot();lockfree();obs=overlap();pci=mapping();P=expand(B['directory']);assert not(P/'parent_allocation.json').exists()and not(P/'launch_result.json').exists()
  if B['holds']==1:
   opening=B['opening_pair'];sf=expand(opening['state_path']);assert sha(sf)==opening['state_sha256'];prior=json.loads(sf.read_text());assert prior['complete']and prior['pair_block_complete']and prior['selected_block']==B['selected_block']and prior['boot_id_start']==prior['boot_id_end']==C['boot_id'];cleanup_state(prior)
  manifest=P/'payload_sha256.json';assert sha(manifest)==B['payload_sha256']
  for n,x in json.loads(manifest.read_text()).items():assert sha(P/n)==x
  alloc=json.loads((P/'parent_allocation.example.json').read_text());assert not alloc['parent_allocated'];alloc.update(parent_allocated=True,prior_gpu_stage_cleanup_verified=True,active_von_cleanup_verified=True,harness_manifest_sha256=B['payload_sha256'],root_allocated_at_utc=now());alloc['opening_pair']=B.get('opening_pair');assert alloc['selected_block']==B['selected_block'];assert alloc['memory_admission_environment']=={k:None for k in KEYS};write(P/'parent_allocation.json',alloc)
  R['phase']='running_reviewed_block';rec={'selected_block':B['selected_block'],'pci_observation':pci,'model_process_observation':obs,'allocation_sha256':sha(P/'parent_allocation.json'),'started_utc':now()};R['blocks'].append(rec);save()
  with(D/(P.name+'.launcher.log')).open('wb')as log:child=subprocess.Popen([sys.executable,str(P/'launch_off_dec83.py')],stdout=log,stderr=subprocess.STDOUT);rec['launcher_pid']=child.pid;save();rc=child.wait()
  rec['launcher_exit_status']=rc;save();assert rc==0
  launched=json.loads((P/'launched.json').read_text());result=json.loads((P/'launch_result.json').read_text());assert result['complete']and result['coordinator_exit_status']==result['observer_exit_status']==0
  for name in ['coordinator_pid','observer_pid','launcher_pid']:assert not alive(launched[name]),'Owned process stillpresent'
  run=Path(result['run_directory']);state=json.loads((run/'state.json').read_text());assert state['complete']and state['line81_block_complete']and state['actual_holds']==B['holds'] and not state['requested_matrix_complete'];assert state['global_output_equality']['validated_cell_files']==4*B['holds'] and state['global_output_equality']['validated_requests_including_warmups']==72*B['holds'];assert state['boot_id_start']==state['boot_id_end']==C['boot_id']
  cleanup_state(state);boot();lockfree();rec.update(complete=True,finished_utc=now(),state_sha256=sha(run/'state.json'),launch_result_sha256=sha(P/'launch_result.json'));save();write(P/'ALLOCATION_RELEASED.json',{'source_commit':HEAD,'allocation_sha256':sha(P/'parent_allocation.json'),'owned_groups_gone':True,'released_at_utc':now()})
 R.update(complete=True,phase='complete',finished_utc=now());save()
except BaseException as e:R.update(failed=True,phase='failed_stopped',error=type(e).__name__+': '+str(e),failed_utc=now());save();raise
