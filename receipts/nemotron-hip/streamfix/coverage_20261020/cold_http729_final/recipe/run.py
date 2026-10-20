"""External729 receipt coordinator; reuse public requests/Evidence, no source edit or build; dry by default."""
import argparse,contextlib,fcntl,hashlib,json,math,os,re,signal,socket,subprocess,sys,time,urllib.request,urllib.error
from pathlib import Path
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S/'clients'))
from bench_evidence import Evidence
p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,default=S/'config.json');p.add_argument('--modes',nargs='+',choices=['cold','resume','cancel'],default=['cold','resume','cancel']);p.add_argument('--rigs',nargs='+',choices=['W7800tp2','V620tp1'],default=['W7800tp2','V620tp1']);p.add_argument('--execute',action='store_true');a=p.parse_args();assert len(a.rigs)==len(set(a.rigs));C=json.loads(a.config.read_text());HEAD='729833615595df66c9f9a438ce9a85061843f010';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();path=lambda s:Path(s).expanduser()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
if not a.execute:print(json.dumps({'execute':False,'source':HEAD,'modes':a.modes,'selected_rigs':a.rigs,'rigs':{rig:C['rigs'][rig]for rig in a.rigs},'cold_sizes':C['cold_sizes'],'fresh_servers':True,'no_model_launched':True},indent=2));raise SystemExit
assert len(a.modes)==len(set(a.modes));allocation=json.loads(path(C['allocation']).read_text());assert allocation['parent_allocated'] and allocation['source_commit']==HEAD and allocation['previous_owned_gpu_jobs_absent_verified'];assert allocation['devices']=={rig:{'W7800tp2':'0,1','V620tp1':'2'}[rig]for rig in a.rigs}
manifest=S/'payload_sha256.json';assert sha(manifest)==allocation['payload_sha256']
for n,h in json.loads(manifest.read_text()).items():assert sha(S/n)==h,n
repo=path(C['tree']);model=path(C['model']);assert (model/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0]==C['checkpoint_revision'];build=path(C['native_manifest']);assert sha(build)==C['native_manifest_sha256'];M=json.loads(build.read_text());assert M['source_commit']==HEAD
for cmd in [['rev-parse','HEAD'],['status','--porcelain','--untracked-files=no']]:
 out=subprocess.check_output(['git','-C',str(repo),*cmd],text=True).strip();assert out==(HEAD if cmd[0]=='rev-parse' else '')
admission_keys=['TENSORFOLD_MEMORY_RESERVE_GIB','TENSORFOLD_HIP_MEMORY_LIMIT_GB']
admission={k:os.environ.get(k)for k in admission_keys};assert admission==allocation['memory_admission_environment'],'Admission environment differs from allocation'
assert sha(path(C['capture_identity']))==C['capture_identity_sha256']
locks=[]
for q in [Path.home()/'opt/gpu.lock',Path.home()/'w1012/build.lock']:
 assert q.is_file();f=q.open('r');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
D=path(C['output_root'])/time.strftime('%Y%m%dT%H%M%S');D.mkdir(parents=True,exist_ok=False);R={'source_commit':HEAD,'checkpoint_revision':C['checkpoint_revision'],'modes':a.modes,'selected_rigs':a.rigs,'cases':[],'complete':False,'all_selected_pass':False,'lock_proofs':[{'path':str(f.name),'inode':os.fstat(f.fileno()).st_ino,'held':True} for f in locks],'scope':'Owned rigs serial; shared host, no physical exclusivity or unreported pinned-memory claim'}
R['memory_admission_environment']=admission;R['capture_identity_sha256']=C['capture_identity_sha256'];R['host_context']={'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'ram_kib':{line.split(':',1)[0]:int(line.split()[1])for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith(('MemTotal:','MemAvailable:'))},'loadavg':list(os.getloadavg()),'context_tokens':262144,'parallel':1,'capability_scope':'Actual native health and request/refusal records; no inferred support or pinned-memory total'}
write(D/'config.raw.json',C);write(D/'allocation.raw.json',allocation);write(D/'native_builds.raw.json',M)
def save():write(D/'summary.json',R)
def guard():
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==HEAD
 assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
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
 return {'shared_observations':shared,'processes_inspected':scanned,'scope':'Known model process/direct numeric mask scan only; unknown mappings are observations, not physical-host exclusivity'}
def positive(x):return type(x)in(int,float) and math.isfinite(x) and x>0
def stats(base):
 with urllib.request.urlopen(base+'/stats',timeout=3) as r:return json.load(r)
def proc_memory(procs,base,phase):
 rows=[]
 for rank,p in procs:
  row={'rank':rank,'pid':p.pid}
  for name,keys in [('status',{'VmRSS','VmHWM','VmPin','VmLck'}),('smaps_rollup',{'Rss','Pss','Locked','Private_Clean','Private_Dirty'})]:
   try:row[name+'_kib']={k:int(v.split()[0]) for line in Path(f'/proc/{p.pid}/{name}').read_text().splitlines() if ':' in line for k,v in [line.split(':',1)] if k in keys}
   except OSError as e:row[name+'_unavailable']=type(e).__name__
  rows.append(row)
 try:runtime=stats(base)
 except Exception as e:runtime={'unavailable':type(e).__name__}
 return {'phase':phase,'at':time.time(),'ranks':rows,'server_stats':runtime,'scope':'VmHWM resident peak; PSS snapshot not exact peak; VmPin/Locked not complete ROCm pinned allocation. HIP allocator fields only as reported.'}
# Same standard-library SSE flow and Evidence hashing as existing bench_concurrent; preserve raw events/errors.
def request(base,body,dest,cancel=False):
 req=urllib.request.Request(base+('/v1/chat/completions' if 'messages'in body else '/v1/completions'),json.dumps(body).encode(),{'Content-Type':'application/json'});x={'request':body,'request_sha256':hashlib.sha256(req.data).hexdigest(),'sent':time.perf_counter(),'events':[],'complete':False,'intentional_disconnect':False};ev=Evidence();usage={};runtime={};texts=[];pieces=[]
 try:
  with urllib.request.urlopen(req,timeout=1800) as response:
   for raw in response:
    line=raw.decode().strip()
    if line=='data: [DONE]':x['complete']=True;break
    if not line.startswith('data:'):continue
    chunk=json.loads(line[5:]);now=time.perf_counter();x['events'].append({'received':now,'chunk':chunk})
    if 'error'in chunk:raise RuntimeError('server SSE error: '+json.dumps(chunk['error']))
    usage=chunk.get('usage')or usage;runtime=chunk.get('tensorfold')or runtime
    for choice in chunk.get('choices',[]):
     ev.add(choice);delta=choice.get('delta')or{};text=choice.get('text')or delta.get('content')or'';reason=delta.get('reasoning_content')or'';texts.append(text)
     if text or reason:pieces.append(now)
    if cancel and pieces:x['intentional_disconnect']=True;break
  x.update(completed=time.perf_counter(),raw_usage=usage,raw_runtime=runtime,evidence=ev.finish(usage,runtime,x['complete']),content=''.join(texts),pieces=pieces)
  if pieces:x['client_ttft_s']=pieces[0]-x['sent']
 except urllib.error.HTTPError as e:x.update(error='HTTPError',http_status=e.code,error_body=e.read().decode(errors='replace'));raise
 except BaseException as e:x['error']=type(e).__name__+': '+str(e);raise
 finally:write(dest,x)
 return x

def validate(x,n=None,tokens=16,cached=None):
 e=x['evidence'];rt=x['raw_runtime'];assert x['complete'] and not x.get('error');assert e['tokens']==tokens
 if n is not None:assert e['prompt_tokens']==n
 assert e['cached_tokens'] is not None
 if cached=='positive':assert e['cached_tokens']>0
 elif cached is not None:assert e['cached_tokens']==cached
 assert re.fullmatch('[0-9a-f]{64}',e['token_sha256']or'') and e['output_sha256'];assert positive(rt.get('prefill_seconds')) and positive(rt.get('seconds'));assert positive(x.get('client_ttft_s'))
 finish=[c.get('finish_reason')for q in x['events']for c in q['chunk'].get('choices',[])if c.get('finish_reason')is not None];assert finish==['length'];return e['token_sha256'],e['output_sha256']
@contextlib.contextmanager
def server(rig,mode,cache_gib,index):
 guard();R.setdefault('overlap_observations',[]).append(host_idle());save();spec=C['rigs'][rig];arch=spec['arch'];b=M['architectures'][arch];assert b['exit_status']==0;binary=b['binaries']['tensorfold-native'];assert sha(binary['path'])==C['native_sha256'][arch]==binary['sha256'];port=spec['port_start']+index
 for n in [port]+([port+1000]if spec['world']>1 else[]):
  with socket.socket()as sock:sock.bind(('127.0.0.1',n))
 procs=[];handles=[];rec={'rig':rig,'mode':mode,'cache_gib':cache_gib,'source_commit':HEAD,'binary_sha256':binary['sha256'],'ranks':[],'memory':[]};R['servers'].append(rec);save();base=f'http://127.0.0.1:{port}'
 try:
  env=os.environ.copy();assert {k:env.get(k)for k in admission_keys}==admission
  for k in ['ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES']:env.pop(k,None)
  env['HIP_VISIBLE_DEVICES']=spec['devices'];env['LD_LIBRARY_PATH']=str(path(C['runtime_library_dir']));rec['memory_admission_environment']={k:env.get(k)for k in ['TENSORFOLD_MEMORY_RESERVE_GIB','TENSORFOLD_HIP_MEMORY_LIMIT_GB']}
  for rank in reversed(range(spec['world'])):
   cmd=[binary['path'],'serve',str(model)]+(['--tp',str(spec['world']),'--rank',str(rank),'--master','127.0.0.1','--master-port',str(port+1000)]if spec['world']>1 else[])+(['--name','m','--port',str(port)]if rank==0 else[])+['--dashboard','--context','262144','--parallel','1','--prompt-cache-gib',str(cache_gib)]
   log=D/f'{rig}_{mode}_{index}.r{rank}.log';fd=log.open('wb');handles.append(fd);p=subprocess.Popen(cmd,env=env,stdout=fd,stderr=subprocess.STDOUT,start_new_session=True);procs.append((rank,p));rec['ranks'].append({'rank':rank,'pid':p.pid,'argv':cmd,'log':log.name});save()
  for _ in range(300):
   assert all(p.poll()is None for _,p in procs),'native exited during admission/startup'
   try:
    with urllib.request.urlopen(base+'/health',timeout=1)as q:health=json.load(q)
    assert health['max_batch_size']==1;rec['health']=health;break
   except(OSError,TimeoutError):time.sleep(2)
  else:raise TimeoutError('native health timeout')
  assert sha(path(C['capture_identity']))==C['capture_identity_sha256'];identity=D/f'{rig}_{mode}_{index}.identity.json';subprocess.run([sys.executable,str(path(C['capture_identity'])),str(identity),*[str(p.pid)for _,p in procs]],check=True,timeout=60);ident=json.loads(identity.read_text());assert ident['processes'] and all(q['executable']['sha256']==binary['sha256']for q in ident['processes']);rec['identity']={'file':identity.name,'sha256':sha(identity)};rec['memory'].append(proc_memory(procs,base,'after_health'));save();yield base,rec,procs
  assert all(p.poll() is None for _,p in procs),'unexpected native exit before intentional teardown'
 finally:
  for rank,p in procs:
   next(q for q in rec['ranks']if q['rank']==rank)['alive_before_intentional_teardown']=p.poll()is None
   if p.poll()is None:os.killpg(p.pid,signal.SIGTERM)
  for rank,p in procs:
   try:code=p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
   row=next(q for q in rec['ranks']if q['rank']==rank);row['actual_exit_status']=code
   try:os.killpg(p.pid,0);row['process_group_gone']=False
   except ProcessLookupError:row['process_group_gone']=True
  for fd in handles:fd.close()
  for row in rec['ranks']:row['log_sha256']=sha(D/row['log'])
  rec['cleanup_owned_groups_gone']=all(q['process_group_gone']for q in rec['ranks']);save();assert rec['cleanup_owned_groups_gone'];guard()
def signals(*_):raise InterruptedError('signal interrupted owned receipt')
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,signals)
R['servers']=[];save()
try:
 for rig in a.rigs:
  for mi,mode in enumerate(a.modes):
   case={'rig':rig,'mode':mode,'pass':False};R['cases'].append(case);save()
   try:
    if mode=='cold':
     case['sizes']=[]
     for ni,n in enumerate(C['cold_sizes']):
      size={'prompt_tokens':n,'pass':False};case['sizes'].append(size);save()
      try:
       with server(rig,mode+'_'+str(n),0,100+ni)as(base,rec,procs):
        f=S/'fixtures'/f'cold_{n}.json';body=json.loads(f.read_text());assert len(body['prompt'])==n and body['max_tokens']==16;rows=[]
        for repeat in range(3):
         x=request(base,body,D/f'{rig}_cold_{n}_{repeat}.json');rows.append(validate(x,n,16,0))
        plain=json.loads((S/'fixtures'/f'plain_{n}.json').read_text());assert plain=={**body,'draft':False};q=request(base,plain,D/f'{rig}_plain_{n}.json');want=validate(q,n,16,0);assert all(v==want for v in rows);size['three_cold_equal_plain']=True;rec['memory'].append(proc_memory(procs,base,'after_context_'+str(n)));save()
       size['pass']=True
      except InterruptedError:raise
      except Exception as e:size['error']=type(e).__name__+': '+str(e);size['scope']='Actual failed/refused context; later contexts still attempted'
      save()
     assert all(q['pass']for q in case['sizes']),'One or more cold contexts failed/refused; all contexts retained'
    else:
     doc=(S/'fixtures/document.txt').read_text();m1=[{'role':'user','content':f'Here is a document:\n\n{doc}\n\nIn two sentences, what is it about?'}];body={'model':'m','messages':m1,'max_tokens':16,'temperature':0,'seed':1234,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
     with server(rig,mode+'_cacheon',1,mi*3)as(base,rec,procs):
      first=request(base,body,D/f'{rig}_{mode}_turn1.json');validate(first,tokens=16)
      frozen={**body,'messages':m1+[{'role':'assistant','content':first['content']},{'role':'user','content':'List three numbers the document reports, with what each measures.'}]};write(D/f'{rig}_{mode}_frozen_turn2.request.json',frozen)
      if mode=='cancel':
       cancelled=request(base,{**frozen,'max_tokens':1024},D/f'{rig}_cancel_prefix.json',cancel=True);assert cancelled['intentional_disconnect']and not cancelled['complete'];samples=[]
       for _ in range(60):
        q=stats(base);samples.append(q)
        if q['running']==q['waiting']==0:break
        time.sleep(.5)
       else:raise TimeoutError('owned HTTP request did not drain')
       case['drain_samples']=samples
      resumed=request(base,frozen,D/f'{rig}_{mode}_reused.json');want=validate(resumed,tokens=16,cached='positive' if mode=='resume' else None);case['cache_reuse_observed']=resumed['evidence']['cached_tokens']>0;case['cancel_scope']='First-output disconnect/drain and later frozen-request equality; not forced partial-MTP/reset API' if mode=='cancel' else None;rec['memory'].append(proc_memory(procs,base,'after_reuse'));save()
     with server(rig,mode+'_cacheoff',0,mi*3+1)as(base,rec,procs):
      fresh=request(base,frozen,D/f'{rig}_{mode}_fresh.json');got=validate(fresh,tokens=16,cached=0);assert resumed['request_sha256']==fresh['request_sha256'];assert resumed['evidence']['prompt_tokens']==fresh['evidence']['prompt_tokens'];assert got==want;case['identical_frozen_turn2_hashes']=True;rec['memory'].append(proc_memory(procs,base,'after_fresh'))
    case['pass']=True
   except InterruptedError:raise
   except Exception as e:case['error']=type(e).__name__+': '+str(e);case['scope']='Actual failed/refused case retained; no unsupported capability inferred';save()
 R['complete']=True;R['all_selected_pass']=all(q['pass']for q in R['cases']);save();print(D);raise SystemExit(0 if R['all_selected_pass']else 1)
finally:
 for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()
