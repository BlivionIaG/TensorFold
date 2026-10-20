"""Read-only revalidation of recovery receipts; decode-only, never cold qualification."""
import contextlib,hashlib,io,json,sys,statistics
from pathlib import Path
import ab_validate
root=Path(sys.argv[1]);state=json.loads((root/'state.json').read_text())
assert state['scope']=='decode-only' and state['cold_prefill_run'] is False
assert state['measured_sources']=={'old':'8649e9858b38e9ef3dcd8d579879c6512a363050','new':'729833615595df66c9f9a438ce9a85061843f010'}
assert state['single_model_host_serial'] is True
count=0;library_binding={};model_binding=None
for mode,groups in state['conditions'].items():
 assert mode in ('off','on') and {x['rig'] for x in groups}=={'W7800tp2','V620tp1'}
 for g in groups:
  assert g['order']==['old','new','new','old'] and len(g['holds'])==4
  hashes=[];docs=[]
  for i,h in enumerate(g['holds']):
   assert h['mode']==mode and h['tree']==g['order'][i] and h['source_commit']==state['measured_sources'][h['tree']]
   rd=root/g['rig'];raw=json.loads((rd/f'{mode}_{i+1}_{h["tree"]}.run.json').read_text());assert raw==h
   assert h['host_preflight']['single_logical_model_instance_per_host'] is True and h['host_preflight']['known_other_model_or_build_processes']==[] and h['host_preflight']['shared_build_lock_held'] is True
   assert h['client_exit_status']==0 and h['teardown'] is True
   assert len(h['ranks'])==(2 if g['rig']=='W7800tp2' else 1)
   assert all(r['alive_before_intentional_teardown'] and r['process_group_gone'] and r['exit_reason']=='intentional native-server teardown' and isinstance(r['actual_exit_status'],int) for r in h['ranks'])
   for key in ['benchmark','runtime_identity']:
    p=rd/h[key]['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==h[key]['raw_sha256']
   with contextlib.redirect_stdout(io.StringIO()):hashes.append(ab_validate.read(rd/h['benchmark']['file'],'decode',0))
   identity=json.loads((rd/h['runtime_identity']['file']).read_text());assert len(identity['processes'])==len(h['ranks'])
   libs={tuple(sorted((Path(k).name,v['sha256']) for k,v in p['libraries'].items())) for p in identity['processes']};assert len(libs)==1
   binding=next(iter(libs))
   if g['rig'] not in library_binding:library_binding[g['rig']]=binding
   if model_binding is None:model_binding=identity['checkpoint_metadata']
   assert binding==library_binding[g['rig']] and identity['checkpoint_metadata']==model_binding
   assert identity['processes'] and all(p['executable']['sha256']==h['binary_sha256'] for p in identity['processes'])
   docs.append(json.loads((rd/h['benchmark']['file']).read_text()));count+=1
  assert all(h==hashes[0] for h in hashes)
  for prompt in ['fibonacci-raw','gpu-chat-no-think']:
   values=[next(x for x in d if x['prompt']==prompt and x['temperature']==0)['decode_tps_median'] for d in docs]
   old=statistics.mean([values[0],values[3]]);new=statistics.mean([values[1],values[2]])
   assert g['prompts'][prompt]=={'hold_order':values,'old_mean':old,'new_mean':new,'delta_percent':100*(new/old-1)}
modes=state.get('modes',['off','on']);assert modes and len(modes)==len(set(modes)) and set(modes)<={'off','on'}
expected=8*len(modes);assert state['expected_holds']==expected
complete=state['complete'] is True and set(state['conditions'])==set(modes) and count==expected
assert not state['complete'] or complete
print(json.dumps({'validated':True,'scope':'decode-only recovery A/B; no cold-prefill run or qualification','measured_sources':state['measured_sources'],'actual_validated_holds':count,'expected_holds':expected,'selected_modes':modes,'complete_selected_decode_modes':complete,'complete_decode_only':complete and set(modes)=={'off','on'},'publication_qualified':False,'conditions':[{ 'mode':m,'rig':g['rig'],'prompts':g['prompts']} for m,groups in state['conditions'].items() for g in groups]},indent=2))
