import hashlib,json,math,re,statistics
from pathlib import Path
import sys
H=Path.home();D=Path(sys.argv[1]).expanduser();S=H/'hcx/streamfix/optimization/decode_serial729'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state=json.loads((D/'state.json').read_text());cfg=json.loads((D/'config.raw.json').read_text());identity=json.loads((D/'identity.raw.json').read_text());assert state['measured_sources']=={'old':'8649e9858b38e9ef3dcd8d579879c6512a363050','new':'729833615595df66c9f9a438ce9a85061843f010'};assert state['scope']=='decode-only' and state['cold_prefill_run'] is False
for kind in ['harness','clients']:
 for name,digest in identity[kind].items():assert sha(D/'inputs'/name)==digest
assert sha(Path(identity['python']['path']))==identity['python']['sha256']
assert state['complete'] and state['actual_holds']==16 and state['single_model_host_serial'] and state['modes']==['off','on']
allocation=json.loads((D/'allocation.raw.json').read_text());frozen=json.loads((D/'payload_sha256.json').read_text());assert sha(D/'payload_sha256.json')==allocation['harness_manifest_sha256']
assert sha(D/'config.raw.json')==frozen['config.json'];assert allocation['single_model_scheduling'] and allocation['exclusive_owned_gpu_reservation']
for n in ['run.py','summarize.py','ab_validate.py','capture_identity.py']:assert sha(D/'inputs'/n)==frozen[n]
allbindings={};libraries={};checkpoint=None;results=[];holds=[]
for mode,groups in state['conditions'].items():
 assert [g['rig'] for g in groups]==['W7800tp2','V620tp1']
 assert mode in ['off','on'] and len(groups)==2 and {g['rig'] for g in groups}=={'W7800tp2','V620tp1'}
 for group in groups:
  rig=group['rig'];rd=D/rig;assert group['order']==['old','new','new','old'] and len(group['holds'])==4;docs=[];allhashes={}
  for i,hold in enumerate(group['holds'],1):
   pre=hold['host_preflight'];assert pre['known_other_model_or_build_processes']==[] and pre['shared_build_lock_held'] and pre['single_logical_model_instance_per_host']
   side=group['order'][i-1];build=json.loads((D/'inputs'/('baseline_native_builds.raw.json' if side=='old' else 'candidate_native_builds.raw.json')).read_text());arch=cfg['rigs'][rig]['arch'];assert build['source_commit']==state['measured_sources'][side] and build['architectures'][arch]['exit_status']==0
   assert hold['binary_sha256']==build['architectures'][arch]['binaries']['tensorfold-native']['sha256']
   assert json.loads((rd/f'{mode}_{i}_{side}.run.json').read_text())==hold;assert hold['hold_index']==i and hold['source_commit']==state['measured_sources'][side] and hold['mode']==mode
   assert hold['client_exit_status']==0 and hold['teardown'];world=cfg['rigs'][rig]['world'];assert len(hold['ranks'])==world
   for rank in hold['ranks']:
    argv=rank['argv'];assert ('--no-drafts' in argv)==(mode=='off');assert argv[argv.index('--context')+1]=='65536' and argv[argv.index('--parallel')+1]=='4';assert rank['alive_before_intentional_teardown'] and rank['process_group_gone'] and rank['exit_reason']=='intentional native-server teardown' and type(rank['actual_exit_status']) is int
    assert sha(Path(argv[0]))==hold['binary_sha256'];assert (rd/rank['log']).exists()
   for key in ['benchmark','runtime_identity']:assert sha(rd/hold[key]['file'])==hold[key]['raw_sha256']
   mapped=json.loads((rd/hold['runtime_identity']['file']).read_text());assert {p['pid'] for p in mapped['processes']}=={p['pid'] for p in hold['ranks']};libsets=[]
   for proc in mapped['processes']:
    assert proc['executable']['sha256']==hold['binary_sha256'];libset=tuple(sorted((Path(p).name,v['sha256']) for p,v in proc['libraries'].items()));assert any('libamdhip64' in k for k,v in libset)
    if world==2:assert any('librccl' in k for k,v in libset)
    libsets.append(libset)
   assert len(set(libsets))==1
   if rig not in libraries:libraries[rig]=libsets[0]
   assert libraries[rig]==libsets[0]
   metadata=mapped['checkpoint_metadata'];assert metadata and metadata['config.json.metadata'].splitlines()[0]==cfg['checkpoint_revision']
   if checkpoint is None:checkpoint=metadata
   assert metadata==checkpoint
   bind=(rig,side)
   if bind not in allbindings:allbindings[bind]=hold['binary_sha256']
   assert allbindings[bind]==hold['binary_sha256']
   doc=json.loads((rd/hold['benchmark']['file']).read_text());assert len(doc)==2 and {r['prompt'] for r in doc}=={'fibonacci-raw','gpu-chat-no-think'}
   for row in doc:
    assert row['temperature']==0 and row['tokens']==256;rates=row['decode_tps_all'];assert len(rates)==3 and all(math.isfinite(x) and x>0 for x in rates);assert round(row['decode_tps_median'],2)==statistics.median(rates)
    evidence=row['evidence'];assert len(evidence)==3 and all(e['complete'] and e['tokens']==256 and e['token_sha'] and e['output_sha256'] for e in evidence);hashes={(e['token_sha'],e['output_sha256']) for e in evidence};assert len(hashes)==1
    key=row['prompt'];allhashes.setdefault(key,set()).update(hashes)
   docs.append(doc);holds.append({'rig':rig,'mode':mode,'index':i,'side':side,'source_commit':hold['source_commit'],'binary_sha256':hold['binary_sha256'],'client_exit':0,'native_actual_teardown_exits':[r['actual_exit_status'] for r in hold['ranks']],'benchmark_sha256':hold['benchmark']['raw_sha256'],'runtime_identity_sha256':hold['runtime_identity']['raw_sha256']})
  assert all(len(v)==1 for v in allhashes.values())
  for prompt in ['fibonacci-raw','gpu-chat-no-think']:
   values=[next(r for r in d if r['prompt']==prompt)['decode_tps_median'] for d in docs];old=statistics.mean([values[0],values[3]]);new=statistics.mean([values[1],values[2]]);delta=100*(new/old-1);assert group['prompts'][prompt]=={'hold_order':values,'old_mean':old,'new_mean':new,'delta_percent':delta};results.append({'mode':mode,'rig':rig,'prompt':prompt,'hold_order':values,'old_tps':old,'new_tps':new,'throughput_delta_percent':delta,'denominator':'old_mean_tps','all12_complete_256_token_replies_equal':True,'token_sha':next(iter(allhashes[prompt]))[0],'output_sha256':next(iter(allhashes[prompt]))[1]})
cross_mode={}
for row in results:cross_mode.setdefault((row['rig'],row['prompt']),set()).add((row['token_sha'],row['output_sha256']))
assert all(len(v)==1 for v in cross_mode.values()), 'Draft/plain full-output hash mismatch on same rig'
assert len(holds)==16
obs=json.loads((D/'observation.json').read_text());env=allocation['memory_admission_environment'];assert set(env)=={'TENSORFOLD_MEMORY_RESERVE_GIB','TENSORFOLD_HIP_MEMORY_LIMIT_GB'}
assert obs['observer_complete'] and obs['errors']==[] and obs['allocation_snapshot']==env and obs['coordinator']['environment']==env and obs['coordinator']['matches_allocation']
allpids={r['pid'] for gs in state['conditions'].values() for g in gs for h in g['holds'] for r in h['ranks']};assert len(allpids)==24
assert len({r['pid'] for r in obs['ranks']})==len(obs['ranks']) and {r['pid'] for r in obs['ranks']}<=allpids
assert all(r['environment']==env and r['matches_allocation'] and r['parent_pid']==obs['coordinator']['pid'] for r in obs['ranks'])
complete=state['complete'] and len(holds)==16 and set(state['conditions'])=={'off','on'}
print(json.dumps({'schema_version':1,'validated_completed_checkpoints':list(state['conditions']),'full16_decode_complete':bool(complete),'cold_prefill_qualified':False,'overall_publication_qualified':False,'hold_count':len(holds),'state_sha256':sha(D/'state.json'),'measured_sources':state['measured_sources'],'settings':{'context':65536,'parallel':4,'completion_tokens':256,'repeats':3},'checkpoint_revision':cfg['checkpoint_revision'],'checkpoint_binding_scope':'Downloaded per-file checkpoint cache metadata, not rehash of all weight bytes','measured_input_sha256':{'harness':identity['harness'],'clients':identity['clients'],'python':identity['python']['sha256']},'all_harness_client_input_sha_verified':True,'draft_plain_complete_hash_equality_verified':set(state['conditions'])=={'off','on'},'results':results,'holds':holds,'runtime_libraries':{k:dict(v) for k,v in libraries.items()},'audit_remote_mutations':False},indent=2))
