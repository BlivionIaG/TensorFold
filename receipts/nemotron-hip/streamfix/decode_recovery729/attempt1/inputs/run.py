"""Decode-only recovery A/B: OFF barrier then ON, 16 holds; dry-run unless --execute."""
import argparse,concurrent.futures,contextlib,hashlib,io,json,os,re,signal,socket,statistics,subprocess,time,urllib.request,threading,fcntl
from pathlib import Path
import ab_validate as validator
S=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(__doc__);ap.add_argument('--config',type=Path,default=S/'config.json');ap.add_argument('--execute',action='store_true');a=ap.parse_args();C=json.loads(a.config.read_text())
def path(s):return Path(s).expanduser()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
if not a.execute:
 print(json.dumps({'scope':'decode-only; no cold-prefill claim','expected_holds':16,'order':['old','new','new','old'],'modes':['off','on'],'rigs':C['rigs'],'candidate_identity_pending':not C['candidate']['source_commit'],'required_gates':['exact candidate native build','exact INLINE_CORRECTNESS_ALL_PASS + result','explicit parent GPU allocation']},indent=2));raise SystemExit(0)
head=C['candidate']['source_commit'];assert re.fullmatch(r'[0-9a-f]{40}',head or '')
assert path(C['correctness_marker']).read_text().strip()==head
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
allocation=json.loads(path(C['parent_allocation']).read_text());assert allocation['parent_allocated'] and allocation['exclusive_gpu_reservation'] and allocation['source_commit']==head and allocation['purpose']=='decode_recovery_ab'
assert allocation['devices']=={k:v['devices'] for k,v in C['rigs'].items()}
assert len(set(','.join(v['devices'] for v in C['rigs'].values()).split(',')))==3
run_lock=(S/'run.lock').open('a');fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
D=path(C['output_root'])/time.strftime('%Y%m%dT%H%M%S');D.mkdir(parents=True,exist_ok=False)
(D/'correctness_four_chip_limits.log').write_bytes(limits.read_bytes())
for item in archives:
 dest=D/'correctness_archives'/item['result'];dest.parent.mkdir(parents=True,exist_ok=True);origin=path(C['correctness_result']).parent/item['result'];dest.write_bytes(origin.read_bytes())
 for stage in json.loads(origin.read_text())['stages']:
  raw=path(C['correctness_result']).parent/stage['log'];assert sha(raw)==stage['log_sha256'];dest=D/'correctness_archives'/stage['log'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw.read_bytes())
write(D/'config.raw.json',C);(D/'correctness.raw.json').write_bytes(path(C['correctness_result']).read_bytes());(D/'allocation.raw.json').write_bytes(path(C['parent_allocation']).read_bytes())
state={'scope':'decode-only','expected_holds':16,'cold_prefill_run':False,'measured_sources':{'old':C['baseline']['source_commit'],'new':head},'conditions':{},'complete':False,'publication_qualified':False}
write(D/'state.json',state)
model=path(C['model']);assert (model/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0]==C['checkpoint_revision']
client=path(C['client']);identity={'harness':{n:sha(S/n) for n in ['run.py','ab_validate.py','capture_identity.py','summarize.py']},'clients':{n:sha(client.with_name(n)) for n in ['bench_openai.py','bench_evidence.py']},'python':{'path':str(path(C['python'])),'sha256':sha(path(C['python']))},'checkpoint_revision':C['checkpoint_revision']}
write(D/'identity.raw.json',identity)
inputs=D/'inputs';inputs.mkdir()
for name in identity['harness']:(inputs/name).write_bytes((S/name).read_bytes())
for name in identity['clients']:(inputs/name).write_bytes(client.with_name(name).read_bytes())
for variant in ['baseline','candidate']:(inputs/(variant+'_native_builds.raw.json')).write_bytes(path(C[variant]['build_manifest']).read_bytes())
def bound(tree,arch):
 c=C['baseline' if tree=='old' else 'candidate'];repo=path(c['tree'])
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==c['source_commit']
 for args in [['diff','--quiet'],['diff','--cached','--quiet']]:subprocess.run(['git','-C',str(repo),*args],check=True)
 m=json.loads(path(c['build_manifest']).read_text());assert m['source_commit']==c['source_commit'];r=m['architectures'][arch];assert r['exit_status']==0
 binary=r['binaries']['tensorfold-native'];p=path(binary['path']);assert sha(p)==binary['sha256'];return p,binary['sha256'],c['source_commit']
def unused(port):
 with socket.socket() as s:s.bind(('127.0.0.1',port))
STOP=threading.Event()
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:STOP.set())
def hold(rig,settings,mode,index,tree):
 assert not STOP.is_set(),"Recovery run cancelled"
 rd=D/rig;rd.mkdir(exist_ok=True);base=f'{mode}_{index}_{tree}';f=rd/(base+'.json');port=settings['port_start']+index+(0 if mode=='off' else 4);world=settings['world'];unused(port)
 if world>1:unused(port+1000)
 binary,bhash,commit=bound(tree,settings['arch']);env=os.environ.copy();env.pop('ROCR_VISIBLE_DEVICES',None);env['HIP_VISIBLE_DEVICES']=settings['devices'];env['LD_LIBRARY_PATH']=str(path(C['runtime_library_dir']))
 extra=['--context','65536','--parallel','4']+(['--no-drafts'] if mode=='off' else [])
 rec={'rig':rig,'mode':mode,'hold_index':index,'tree':tree,'source_commit':commit,'binary_sha256':bhash,'port':port,'ranks':[],'teardown':False};procs=[];handles=[];bench=None
 try:
  for rank in reversed(range(world)):
   if rank==0 and world>1:time.sleep(1)
   argv=[str(binary),'serve',str(model)]+(['--tp',str(world),'--rank',str(rank),'--master','127.0.0.1','--master-port',str(port+1000)] if world>1 else [])+(['--name','m','--port',str(port)] if rank==0 else [])+extra
   log=rd/(base+f'.r{rank}.log');h=log.open('wb');handles.append(h);p=subprocess.Popen(argv,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);procs.append((rank,p));rec['ranks'].append({'rank':rank,'pid':p.pid,'argv':argv,'log':log.name})
  for attempt in range(300):
   assert not STOP.is_set(),"Recovery run cancelled"
   assert all(p.poll() is None for _,p in procs),'Unexpected rank exit before health'
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as response:
     if response.status==200:break
   except (OSError,TimeoutError):pass
   time.sleep(2)
  else:raise RuntimeError('Health startup timeout')
  sidecar=rd/(base+'.identity.json');subprocess.run(['python3',str(S/'capture_identity.py'),str(sidecar),*[str(p.pid) for _,p in procs]],check=True,timeout=60)
  mapped=json.loads(sidecar.read_text());assert all(x['executable']['sha256']==bhash for x in mapped['processes'])
  rec['runtime_identity']={'file':sidecar.name,'raw_sha256':sha(sidecar)}
  argv=[str(path(C['python'])),str(client),f'http://127.0.0.1:{port}','m','--tokens','256','--reps','3','--temperatures','0','--output',str(f)]
  with (rd/(base+'.client.log')).open('wb') as log:
   bench=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);deadline=time.monotonic()+900
   while bench.poll() is None:
    assert not STOP.is_set(),"Recovery run cancelled"
    assert all(p.poll() is None for _,p in procs),'Unexpected native rank death during benchmark'
    if time.monotonic()>deadline:raise TimeoutError('Benchmark timeout')
    time.sleep(.5)
   rec['client_exit_status']=bench.returncode;assert bench.returncode==0
  assert all(p.poll() is None for _,p in procs),'Unexpected rank exit before intentional teardown'
  with contextlib.redirect_stdout(io.StringIO()):validator.read(f,'decode',0)
  rec['benchmark']={'file':f.name,'raw_sha256':sha(f)}
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

def quartet(rig,settings,mode):
 rows=[hold(rig,settings,mode,i,tree) for i,tree in enumerate(['old','new','new','old'],1)]
 paths=[D/rig/r['benchmark']['file'] for r in rows]
 with contextlib.redirect_stdout(io.StringIO()):hashes=[validator.read(p,'decode',0) for p in paths]
 assert all(h==hashes[0] for h in hashes),'Complete reply/token hash mismatch in quartet'
 docs=[json.loads(p.read_text()) for p in paths];prompts={}
 for prompt in ['fibonacci-raw','gpu-chat-no-think']:
  values=[next(x for x in d if x['prompt']==prompt and x['temperature']==0)['decode_tps_median'] for d in docs];old=statistics.mean([values[0],values[3]]);new=statistics.mean([values[1],values[2]])
  prompts[prompt]={'hold_order':values,'old_mean':old,'new_mean':new,'delta_percent':100*(new/old-1)}
 return {'rig':rig,'mode':mode,'order':['old','new','new','old'],'prompts':prompts,'complete_reply_and_token_sha_equal':True,'holds':rows}
for mode in ['off','on']:
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(quartet,k,v,mode) for k,v in C['rigs'].items()];result=[f.result() for f in futures]
 state['conditions'][mode]=result;write(D/'state.json',state);write(D/(mode+'_checkpoint.json'),{'scope':'decode-only checkpoint','complete_full_decode_ab':mode=='on','cold_prefill_run':False,'results':result})
 with (D/(mode+'_validated_summary.json')).open('w') as report:subprocess.run(['python3',str(S/'summarize.py'),str(D)],stdout=report,check=True)
 print(mode.upper()+'_QUARTETS_PASS '+str(D),flush=True)
state['complete']=True;state['actual_holds']=16;write(D/'state.json',state)
with (D/'validated_summary.json').open('w') as report:subprocess.run(['python3',str(S/'summarize.py'),str(D)],stdout=report,check=True)
print('DECODE_ONLY_16_HOLDS_DONE '+str(D),flush=True)
