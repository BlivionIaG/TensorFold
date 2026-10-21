"""Two-shape C1/C8 descriptive old/new pair: one selected rig/mode, 2 server holds; dry-run unless --execute."""
import argparse,contextlib,hashlib,io,json,os,re,signal,socket,statistics,subprocess,time,urllib.request,threading,fcntl
from pathlib import Path

S=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(__doc__);ap.add_argument('--config',type=Path,default=S/'config.json');ap.add_argument('--execute',action='store_true');ap.add_argument('--modes',nargs='+',choices=['off','on'],default=['off','on']);a=ap.parse_args();C=json.loads(a.config.read_text());block=C['selected_block'];assert block['rig'] in ['W7800tp2','V620tp1'] and block['mode'] in ['off','on'];a.modes=[block['mode']];expected_holds=2
def path(s):return Path(s).expanduser()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
if not a.execute:
 print(json.dumps({'scope':'paired_benchmark_measurement; three measured plus one warmup percell; no statistical confidence claim','expected_holds':expected_holds,'order':['old','new'],'modes':a.modes,'rigs':C['rigs'],'candidate_identity_pending':not C['candidate']['source_commit'],'owned_groups_serial_shared_host':True,'own_model_groups_serial':True,'physical_host_isolated':False,'user_shared_host_override':True,'rig_order':[block['rig']],'shared_build_lock_held_during_execution':True,'required_gates':['exact candidate native build','exact INLINE_CORRECTNESS_ALL_PASS + result','explicit parent GPU allocation']},indent=2));raise SystemExit(0)
head=C['candidate']['source_commit'];assert re.fullmatch(r'[0-9a-f]{40}',head or '')
assert path(C['correctness_marker']).read_text().strip()==head
assert sha(path(C['correctness_result']))==C['correctness_result_sha256']
proof=json.loads(path(C['correctness_result']).read_text());assert proof['pass'] is True and proof['state']=='complete' and proof['source_commit']==head
assert proof['baseline_source_commit']==C['baseline']['source_commit']
assert proof['stages'] and all(x['exit_status']==0 for x in proof['stages'])
limits=path(C['correctness_result']).parent/'four_chip_limits.log';assert sha(limits)==proof['four_chip_limits_sha256']
assert limits.read_text().rstrip().endswith('PASS kernel-contract-limits: every registered launch fits gfx1030, gfx1100, gfx1151 and gfx1201, every record current, every control red')
archives=proof.get('archived_infrastructure_attempts',[])
if archives:
 assert {x['result'] for x in archives}=={'attempt1/result.json','attempt2/result.json'}
 for item in archives:
  historical=path(C['correctness_result']).parent/item['result'];assert sha(historical)==item['sha256']
  record=json.loads(historical.read_text());assert record['source_commit']==head and record['state']=='failed'
  failures=[x for x in record['stages'] if x['exit_status']!=0]
  if item['result'].startswith('attempt1/'):assert not failures
  else:
   assert {x['name'] for x in failures}=={'compare_gfx1100','compare_gfx1030'}
   for failed in failures:
    raw=path(C['correctness_result']).parent/failed['log'];assert failed['exit_status']==1 and sha(raw)==failed['log_sha256'] and 'AssertionError: unexpected final candidate' in raw.read_text()
rows={x['arch']:x for x in proof['architectures']};assert set(rows)=={'gfx1100','gfx1030'}
for arch,count,nan in [('gfx1100',1080,60),('gfx1030',540,24)]:
 r=rows[arch];assert r['finite_fixtures']==count and r['finite_nan_controls']==nan and r['byte_mismatches']==0
 assert re.fullmatch(r'SUMMARY '+arch+r' 49 kernels: 49 pass gates 1-4, 0 fail, 0 known findings; 5 controls: 5 red as planted, 0 missed; 0 limit problems; gates \d+ ms, limits \d+ ms',r['summary'])
allocation=json.loads(path(C['parent_allocation']).read_text());assert allocation['parent_allocated'] and allocation['exclusive_gpu_reservation'] and allocation['source_commit']==head and allocation['purpose']=='matrix_published729_continuation' and allocation['phase']==block['mode'];assert allocation['selected_block']==block and allocation['prior_gpu_stage_cleanup_verified'] is True
pilot=json.loads(path(C['pilot_qualification']).read_text());assert pilot['parent_approved_full_matrix'] is True and pilot['candidate_source_commit']==head and pilot['client_sha256']==sha(S/'matrix_client.py')
assert sha(path(pilot['pilot_summary_path']))==pilot['pilot_summary_sha256']
frozen=json.loads((S/'payload_sha256.json').read_text());assert sha(S/'payload_sha256.json')==allocation['harness_manifest_sha256']
for name,digest in frozen.items():assert sha(S/name)==digest,'Frozen serial helper changed: '+name
assert allocation['single_model_scheduling'] is True and allocation['single_model_scheduling_scope']=='OWN_GROUPS_ONLY' and allocation['user_shared_host_override'] is True
assert allocation['exclusive_owned_gpu_reservation'] is True
assert {k:v['devices'] for k,v in C['rigs'].items()}=={'W7800tp2':'0,1','V620tp1':'2'}
assert all(x in {'0','1','2'} for x in ','.join(v['devices'] for v in C['rigs'].values()).split(','))
assert allocation['devices']=={k:v['devices'] for k,v in C['rigs'].items()}
assert len(set(','.join(v['devices'] for v in C['rigs'].values()).split(',')))==3
gpu_lock_path=Path.home()/'opt/gpu.lock';assert gpu_lock_path.is_file(),'Existing canonical GPU lock required'
gpu_lock=gpu_lock_path.open('r');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu_lock_stat=os.fstat(gpu_lock.fileno());gpu_lock_proof={'path':str(gpu_lock_path),'mode':'LOCK_EX|LOCK_NB','coordinator_pid':os.getpid(),'inode':gpu_lock_stat.st_ino,'device':gpu_lock_stat.st_dev,'acquired':True,'scope':'Existing canonical advisory lock held by coordinator; owned groups serial, shared disjoint activity allowed; no physical-host lease'}
host_build_lock=(Path.home()/'w1012/build.lock').open('a');fcntl.flock(host_build_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
run_lock=(S/'run.lock').open('a');fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
D=path(C['output_root'])/time.strftime('%Y%m%dT%H%M%S');D.mkdir(parents=True,exist_ok=False)
(D/'correctness_four_chip_limits.log').write_bytes(limits.read_bytes())
for item in archives:
 dest=D/'correctness_archives'/item['result'];dest.parent.mkdir(parents=True,exist_ok=True);origin=path(C['correctness_result']).parent/item['result'];dest.write_bytes(origin.read_bytes())
 for stage in json.loads(origin.read_text())['stages']:
  raw=path(C['correctness_result']).parent/stage['log'];assert sha(raw)==stage['log_sha256'];dest=D/'correctness_archives'/stage['log'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw.read_bytes())
write(D/'config.raw.json',C);(D/'correctness.raw.json').write_bytes(path(C['correctness_result']).read_bytes());(D/'allocation.raw.json').write_bytes(path(C['parent_allocation']).read_bytes())
boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
state={'selected_block':block,'boot_id_start':boot_id,'historical_data_loaded':False,'gpu_lock_held_proof':gpu_lock_proof,'scope':'paired_benchmark_measurement','expected_holds':expected_holds,'cold_prefill_run':False,'measured_sources':{'old':C['baseline']['source_commit'],'new':head},'owned_groups_serial_shared_host':True,'own_model_groups_serial':True,'physical_host_isolated':False,'user_shared_host_override':True,'modes':a.modes,'conditions':{},'complete':False,'publication_qualified':False}
write(D/'state.json',state)
model=path(C['model']);assert (model/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0]==C['checkpoint_revision']
client=path(C['client'])
for name,digest in C['frozen_client_sha256'].items():assert sha(S/name)==digest,'Frozen client changed: '+name
identity={'harness':{n:sha(S/n) for n in ['run.py','capture_identity.py','observe_memory_env.py']},'clients':{n:sha(S/n) for n in C['frozen_client_sha256']},'python':{'path':str(path(C['python'])),'sha256':sha(path(C['python']))},'checkpoint_revision':C['checkpoint_revision']}
write(D/'identity.raw.json',identity);(D/'payload_sha256.json').write_bytes((S/'payload_sha256.json').read_bytes())
inputs=D/'inputs';inputs.mkdir()
for name in identity['harness']:(inputs/name).write_bytes((S/name).read_bytes())
for name in identity['clients']:
 dest=inputs/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((S/name).read_bytes())
for variant in ['baseline','candidate']:(inputs/(variant+'_native_builds.raw.json')).write_bytes(path(C[variant]['build_manifest']).read_bytes())
def bound(tree,arch):
 c=C['baseline' if tree=='old' else 'candidate'];repo=path(c['tree'])
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==c['source_commit']
 for args in [['diff','--quiet'],['diff','--cached','--quiet']]:subprocess.run(['git','-C',str(repo),*args],check=True)
 assert sha(path(c['build_manifest']))==C['native_build_manifest_sha256'][tree]
 m=json.loads(path(c['build_manifest']).read_text());assert m['source_commit']==c['source_commit'];r=m['architectures'][arch];assert r['exit_status']==0
 binary=r['binaries']['tensorfold-native'];p=path(binary['path']);assert sha(p)==binary['sha256']==C['frozen_native_sha256'][tree][arch];return p,binary['sha256'],c['source_commit']
def unused(port):
 with socket.socket() as s:s.bind(('127.0.0.1',port))
STOP=threading.Event()
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:STOP.set())
def host_idle(allowed_pids=(),allowed_pgids=(),scan_only=False):
 blocked=[];shared=[];scanned=0
 for process in Path('/proc').iterdir():
  if not process.name.isdigit() or int(process.name)==os.getpid() or int(process.name) in allowed_pids:continue
  try:
   if allowed_pgids and os.getpgid(int(process.name)) in allowed_pgids:continue
   args=[x.decode(errors='replace') for x in (process/'cmdline').read_bytes().split(b'\0') if x]
   if not args:continue
   scanned+=1;exe=Path(args[0]).name;line=' '.join(args)
   native=exe.startswith(('tensorfold-native','tf-nemotron-hip-test','tf-qwen35-test','tf-flashnext-hip-test'))
   server=any(x in line for x in ['vllm.entrypoints','vllm serve','sglang.launch_server','mlx_lm.server','llama-server'])
   orchestration=any(Path(x).name in {'pipeline.py','nem_compare.py','ab.sh'} for x in args) or (any(Path(x).name=='driver.py' for x in args) and 'followup729' in line)
   other_ab=any(Path(x).name=='run.py' and ('decode_recovery/' in x or 'decode_serial729/' in x or 'successor_validation/' in x or 'off_dec83/' in x) for x in args)
   build=exe in {'hipcc','ninja','cmake','gcc','g++','cc','c++','rustc','cargo'} or exe.startswith('clang') or (exe in {'zig','maker'} and 'build' in args)
   comm=(process/'comm').read_text().strip()
   http_client=len(args)>=5 and args[1:5]==['-m','vllm.entrypoints.cli.main','bench','serve']
   model_job=(native or server or orchestration or other_ab or 'vllm' in exe.lower() or comm.startswith('VLLM::')) and not http_client
   if model_job:
    keys={'HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'};selected={k:None for k in keys}
    for item in (process/'environ').read_bytes().split(b'\0'):
     key,sep,value=item.partition(b'=');name=key.decode(errors='replace')
     if sep and name in keys:selected[name]=value.decode(errors='replace')
    present=[(k,v) for k,v in selected.items() if v is not None]
    # Only positive overlap in one direct numeric HIP/ROCR/CUDA mask blocks; unknown mappings remain observations.
    direct=len(present)==1 and present[0][0] in keys and bool(re.fullmatch(r'\d+(?:,\d+)*',present[0][1]))
    devices=set(map(int,present[0][1].split(','))) if direct else set()
    conflict=direct and bool(devices & {0,1,2})
    row={'pid':int(process.name),'executable':exe,'device_environment':selected,'mapping_status':'direct_mask_overlap_our_gpu0_to2' if conflict else 'direct_mask_no_overlap' if direct else 'unknown_or_ambiguous_observed_not_blocking','mapping_scope':'Numeric mask observation only; unknown is not evidence of collision or physical-host isolation'}
    (blocked if conflict else shared).append(row)
   elif build or http_client:shared.append({'pid':int(process.name),'executable':exe,'scope':'CPU HTTP benchmark client allowed' if http_client else 'Unrelated CPU build allowed by user; no capacity or GPU-runtime claim'})
  except (FileNotFoundError,ProcessLookupError):continue
  except PermissionError:shared.append({'pid':int(process.name),'mapping_status':'permission_denied_unknown_observed_not_blocking'})
 assert not blocked,'Positively identified direct-mask overlap with reserved GPUs0-2: '+json.dumps(blocked)
 mem={line.split(':',1)[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
 sample={'processes_inspected':scanned,'shared_host':True,'known_assigned_gpu_conflicts':blocked,'allowed_other_activity_observed':shared[:32],'allowed_shared_jobs':shared[:32],'allowed_shared_jobs_total':len(shared),'shared_job_list_truncated':len(shared)>32,'host_loadavg_1_5_15':list(os.getloadavg()),'host_ram_kib':mem,'at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 if scan_only:return sample
 cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'not reported')
 return {'single_logical_model_instance_per_host':False,'one_owned_logical_group_scheduled':True,'shared_host':True,'shared_host_observation':sample,'host_idleness_scope':'User allows shared CPU builds and other jobs; only positive direct-mask overlap with GPUs0-2 blocks, unknown mappings observed without blocking; no physical-host isolation or capacity guarantee','tp2_rank_processes_are_one_logical_model':True,'allowed_other_activity_observed':shared[:32],'conflicting_gpu_jobs':blocked,'processes_inspected':scanned,'shared_build_lock_held':True,'existing_gpu_lock_found_in_reviewed_scripts':True,'existing_gpu_lock_qualification':'Existing canonical lock acquired and held by current coordinator under explicit root instruction','gpu_lock_held_proof':gpu_lock_proof,'native_admission_guard':'unchanged frozen native binary and limits; no admission override flags','host_loadavg_1_5_15':list(os.getloadavg()),'host_ram_kib':mem,'host_ram_class':'physical host RAM from /proc/meminfo; DIMM/ECC class not inferred','cpu_model':cpu,'logical_cpus':os.cpu_count(),'context_tokens':65536,'parallel_streams':8,'at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
def memory_snapshot(procs,phase):
 rows=[]
 for rank,p in procs:
  row={'rank':rank,'pid':p.pid}
  try:
   status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path(f'/proc/{p.pid}/status').read_text().splitlines() if ':' in line}
   row['status_kib']={k:int(status[k].split()[0]) for k in ['VmRSS','VmHWM','VmPin','VmLck'] if k in status}
  except OSError as exc:row['status_unavailable']=str(exc)
  try:
   rollup={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path(f'/proc/{p.pid}/smaps_rollup').read_text().splitlines() if ':' in line}
   row['smaps_rollup_kib']={k:int(rollup[k].split()[0]) for k in ['Rss','Pss','Private_Clean','Private_Dirty','Shared_Clean','Shared_Dirty','Locked'] if k in rollup}
  except OSError as exc:row['smaps_unavailable']=str(exc)
  rows.append(row)
 return {'phase':phase,'at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'processes':rows,'scope':'Per-rank Linux VmHWM is kernel peak resident memory through this point; cumulative hold including startup/warmups. PSS is a snapshot, not exact peak. VmPin is Linux accounting, not complete ROCm pinned allocation. Raw /stats memory counters remain in client receipts; no pureGPU/pinned inference.'}

def hold(rig,settings,mode,index,tree):
 assert not STOP.is_set(),"Recovery run cancelled"
 preflight=host_idle()
 rd=D/rig;rd.mkdir(exist_ok=True);base=f'{mode}_{index}_{tree}';f=rd/(base+'.json');port=settings['port_start']+index+(0 if mode=='off' else 4);world=settings['world'];unused(port)
 if world>1:unused(port+1000)
 binary,bhash,commit=bound(tree,settings['arch']);env=os.environ.copy();env.pop('ROCR_VISIBLE_DEVICES',None);env['HIP_VISIBLE_DEVICES']=settings['devices'];env['LD_LIBRARY_PATH']=str(path(C['runtime_library_dir']))
 extra=['--dashboard','--context','65536','--parallel','8','--prompt-cache-gib','0','--checkpoint-slots','0','--snapshot-dir','none','--max-snapshots','0']+(['--no-drafts'] if mode=='off' else [])
 rec={'rig':rig,'mode':mode,'hold_index':index,'tree':tree,'source_commit':commit,'binary_sha256':bhash,'port':port,'ranks':[],'host_preflight':preflight,'teardown':False};procs=[];handles=[];bench=None
 rec['continuous_collision_guard']={'interval_seconds':2,'checks':0,'scope':'Positive direct-mask GPU0-2 overlap scan; CPU/HTTP/shared and unknown mappings observed without blocking, own groups excluded; no physical-host isolation claim','shared_host_samples':[],'sample_limit':128};last_scan=0
 def collision_guard(force=False):
  nonlocal last_scan
  now=time.monotonic()
  if not force and now-last_scan<2:return
  own=[p.pid for _,p in procs]+([bench.pid] if bench is not None else [])
  try:
   scan=host_idle(allowed_pids=own,allowed_pgids=own,scan_only=True)
  except BaseException as exc:
   rec['contaminated']=True;rec['collision_error']=str(exc);rec['collision_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
   write(D/'CONTAMINATED.json',{'rig':rig,'hold':base,'error':str(exc),'at_utc':rec['collision_utc'],'action':'Abort own client/rank groups only; unrelated jobs untouched'})
   raise
  last_scan=now
  if len(rec['continuous_collision_guard']['shared_host_samples'])<128:rec['continuous_collision_guard']['shared_host_samples'].append(scan)
  rec['continuous_collision_guard']['checks']+=1;rec['continuous_collision_guard']['last_scan_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 try:
  for rank in reversed(range(world)):
   if rank==0 and world>1:time.sleep(1)
   argv=[str(binary),'serve',str(model)]+(['--tp',str(world),'--rank',str(rank),'--master','127.0.0.1','--master-port',str(port+1000)] if world>1 else [])+(['--name','m','--port',str(port)] if rank==0 else [])+extra
   log=rd/(base+f'.r{rank}.log');h=log.open('wb');handles.append(h);p=subprocess.Popen(argv,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);procs.append((rank,p));rec['ranks'].append({'rank':rank,'pid':p.pid,'argv':argv,'log':log.name})
  for attempt in range(300):
   collision_guard()
   assert not STOP.is_set(),"Recovery run cancelled"
   assert all(p.poll() is None for _,p in procs),'Unexpected rank exit before health'
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as response:
     if response.status==200:
      health=json.load(response);assert health['max_batch_size']==8,'Health must confirm8admittedslots';rec['health_observed']=health;break
   except (OSError,TimeoutError):pass
   time.sleep(2)
  else:raise RuntimeError('Health startup timeout')
  collision_guard(force=True)
  sidecar=rd/(base+'.identity.json');subprocess.run(['python3',str(S/'capture_identity.py'),str(sidecar),*[str(p.pid) for _,p in procs]],check=True,timeout=60)
  mapped=json.loads(sidecar.read_text());assert all(x['executable']['sha256']==bhash for x in mapped['processes'])
  rec['runtime_identity']={'file':sidecar.name,'raw_sha256':sha(sidecar)}
  with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=3) as response:models=json.load(response)
  assert models['data'] and all(x.get('context_length')==65536 for x in models['data']);rec['models_observed']=models
  startup=(rd/(base+'.r0.log')).read_text(errors='replace')
  assert re.search(r'\b8 streams at once\b',startup),'Fixed8 capacity not observed; do not silently queue8onfewer slots'
  rec['actual_capacity_observed']=8;rec['cache_budget_requested_gib']=0
  rec['memory_snapshots']=[memory_snapshot(procs,'server_ready')];rec['benchmarks']=[]
  for shape in C['fixtures']:
   fixture=path(shape['path']);assert sha(fixture)==shape['sha256'];reference=None
   for concurrency in [1,8]:
    output=rd/(base+'.'+shape['name']+'.c'+str(concurrency)+'.json')
    argv=[str(path(C['python'])),str(client),f'http://127.0.0.1:{port}','--model','m','--fixture',str(fixture),'--prompt-tokens',str(shape['prompt_tokens']),'--tokens',str(shape['tokens']),'--concurrency',str(concurrency),'--mode',mode,'--reps','3','--warmups','1','--output',str(output),'--execute']
    if concurrency==8:argv+=['--reference-output',str(reference)]
    with (rd/(base+'.'+shape['name']+'.c'+str(concurrency)+'.client.log')).open('wb') as log:
     bench=subprocess.Popen(argv,stdout=log,stderr=log,start_new_session=True);deadline=time.monotonic()+1800
     while bench.poll() is None:
      collision_guard();assert not STOP.is_set(),'Pilot cancelled';assert all(p.poll() is None for _,p in procs),'Unexpected native rank death'
      if time.monotonic()>deadline:raise TimeoutError('Pilot client timeout')
      time.sleep(.5)
     status=bench.returncode
    collision_guard(force=True)
    entry={'shape':shape['name'],'concurrency':concurrency,'file':output.name,'client_exit_status':status,'argv':argv}
    if output.is_file():entry['raw_sha256']=sha(output)
    rec['benchmarks'].append(entry);rec['memory_snapshots'].append(memory_snapshot(procs,'after_'+shape['name']+'_c'+str(concurrency)));write(rd/(base+'.run.json'),rec)
    assert status==0
    result=json.loads(output.read_text());assert result['complete'] and not result.get('error') and result['concurrency']==concurrency and result['mode']==mode
    assert result['fixture_sha256']==shape['sha256'] and len(result['batches'])==4
    if concurrency==1:reference=output
  assert all(p.poll() is None for _,p in procs),'Unexpected native exit before teardown'

 except BaseException:
  STOP.set();raise
 finally:
  if bench is not None and bench.poll() is None:os.killpg(bench.pid,signal.SIGKILL);bench.wait()
  rec['teardown']=True
  for rank,p in procs:
   row=next(x for x in rec['ranks'] if x['rank']==rank);row['alive_before_intentional_teardown']=p.poll() is None
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  for rank,p in procs:
   row=next(x for x in rec['ranks'] if x['rank']==rank)
   try:status=p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);status=p.wait();row['forced_kill_after_term_timeout']=True
   row['actual_exit_status']=status;row['exit_reason']='intentional native-server teardown' if row['alive_before_intentional_teardown'] else 'unexpected earlier exit'
  for h in handles:h.close()
  time.sleep(3)
  for rank,p in procs:
   try:os.killpg(p.pid,0);gone=False
   except ProcessLookupError:gone=True
   next(x for x in rec['ranks'] if x['rank']==rank)['process_group_gone']=gone
  write(rd/(base+'.run.json'),rec)
  assert all(r['process_group_gone'] for r in rec['ranks']), 'Native process group survived cleanup'
 return rec

assert {(x['name'],x['prompt_tokens'],x['tokens']) for x in C['fixtures']}=={('code_1024_512',1024,512),('prose_16384_1024',16384,1024)} and len(C['fixtures'])==2
for mode in a.modes:
 groups=[]
 for rig in [block['rig']]:
  rows=[]
  for index,side in enumerate(['old','new'],1):
   assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot_id
   rows.append(hold(rig,C['rigs'][rig],mode,index,side))
   assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot_id
   state['completed_holds_checkpoint']=rows;write(D/'state.json',state)
  groups.append({'rig':rig,'order':['old','new'],'holds':rows})
  state['conditions'][mode]=groups;write(D/'state.json',state);print('MATRIX_PAIR_COMPLETE '+mode+' '+rig,flush=True)
# Validate every retained warmup and measured request across source/mode/order/concurrency.
fingerprints={};validated_files=[];validated_requests=0
for mode in a.modes:
 for group in state['conditions'][mode]:
  rig=group['rig']
  for rec in group['holds']:
   assert len(rec['benchmarks'])==4
   assert {(b['shape'],b['concurrency']) for b in rec['benchmarks']}=={(x['name'],c) for x in C['fixtures'] for c in [1,8]}
   for b in rec['benchmarks']:
    file=D/rig/b['file'];assert sha(file)==b['raw_sha256'] and b['client_exit_status']==0
    result=json.loads(file.read_text());shape=next(x for x in C['fixtures'] if x['name']==b['shape'])
    assert result['complete'] and result['fixture_sha256']==shape['sha256'] and result['prompt_tokens']==shape['prompt_tokens'] and result['requested_tokens']==shape['tokens']
    assert [batch['phase'] for batch in result['batches']]==['warmup','measured','measured','measured']
    key=(rig,b['shape']);fingerprints.setdefault(key,set())
    for batch in result['batches']:
     assert len(batch['runs'])==b['concurrency']
     for r in batch['runs']:
      fingerprints[key].add((r['raw_runtime']['token_sha256'],r['output_sha256']));validated_requests+=1
      assert len(fingerprints[key])==1,'Cross-source/mode/hold/concurrency output mismatch: '+str(key)
    validated_files.append(str(file.relative_to(D)))
assert len(validated_files)==8 and validated_requests==144
state['global_output_equality']={'validated_cell_files':8,'validated_requests_including_warmups':144,'fingerprints':{':'.join(k):list(next(iter(v))) for k,v in fingerprints.items()}}
state['complete']=True;state['actual_holds']=2;state['pair_block_complete']=True;state['requested_matrix_complete']=False;state['boot_id_end']=Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert state['boot_id_end']==boot_id;state['statistical_confidence_claimed']=False;state['full_contributing_qualified']=False;write(D/'state.json',state)
print('MATRIX_PAIR_BLOCK_COMPLETE '+str(D),flush=True)
