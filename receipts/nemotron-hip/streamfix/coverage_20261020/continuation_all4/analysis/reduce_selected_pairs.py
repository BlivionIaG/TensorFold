"""Read retained explicitly selected two-hold paired matrix JSON only; recompute metrics, never launch jobs."""
import argparse,json,math,statistics,hashlib,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--partial',action='store_true');p.add_argument('--derived-index',type=Path);p.add_argument('--selection',type=Path,required=True);a=p.parse_args()
R={'schema_version':1,'scope':'Explicit selected old/new pairs, three measured batches; not ABBA, statistical confidence or full CONTRIBUTING qualification','true_token_itl_distribution_available':False,'pure_gpu_prefill_compute_claimed':False,'cells':[],'comparisons':[],'errors':[]};groups={};hashes={};total=measured=0

def close(got,want,label):
 assert type(got) in (int,float) and math.isfinite(got) and math.isclose(got,want,rel_tol=1e-12,abs_tol=1e-12),(label,got,want)
def median(v):return statistics.median(v)
def classify_rank(rank,rig):
 assert rank['process_group_gone'] is True and rank['alive_before_intentional_teardown'] is True
 assert rank['exit_reason']=='intentional native-server teardown'
 assert not rank.get('forced_kill_after_term_timeout',False)
 code=rank['actual_exit_status'];role=rank['rank']
 assert code==0 or (rig=='W7800tp2' and role>0 and code==-15),'Unexpected native exit'
 return {'rank':role,'pid':rank['pid'],'actual_exit_status':code,'alive_before_intentional_teardown':rank['alive_before_intentional_teardown'],'exit_reason':rank['exit_reason'],'process_group_gone':rank['process_group_gone'],'forced_kill_flag':rank.get('forced_kill_after_term_timeout',False),'classification':'clean_exit_0' if code==0 else 'intentional_tp_worker_sigterm'}

def load_selection():
 selection=json.loads(a.selection.read_text());assert selection['schema_version']==1
 for name,binding in selection['fixtures'].items():
  file=(a.root/binding['file']).resolve();assert file.is_relative_to(a.root.resolve());assert hashlib.sha256(file.read_bytes()).hexdigest()==SHAPES[name][2];FIXTURE_IDS[name]=json.loads(file.read_text())
 expected={('W7800tp2','off'),('V620tp1','off'),('W7800tp2','on'),('V620tp1','on')};assert {(b['rig'],b['mode'])for b in selection['blocks']}==expected and len(selection['blocks'])==4
 state={'complete':False,'conditions':{}};verified=[];pending=[]
 for block in selection['blocks']:
  key=(block['rig'],block['mode'])
  if block['run_root'] is None:pending.append({'rig':key[0],'mode':key[1]});continue
  root=(a.root/block['run_root']).resolve();assert root.is_relative_to(a.root.resolve())
  sf=root/'state.json';assert hashlib.sha256(sf.read_bytes()).hexdigest()==block['state_sha256'];runtime=json.loads(sf.read_text())
  assert runtime['measured_sources']==SOURCES
  if block['scope']=='recovered_first_pair':
   assert key==('W7800tp2','off') and not runtime['complete'];holds=[]
   coordinator=runtime['gpu_lock_held_proof']['coordinator_pid']
   for binding in block['hold_records']:
    rf=root/binding['file'];assert hashlib.sha256(rf.read_bytes()).hexdigest()==binding['sha256'];hold=json.loads(rf.read_text());assert hold['host_preflight']['gpu_lock_held_proof']['coordinator_pid']==coordinator;holds.append(hold)
   boot_scope='Same uninterrupted recorded coordinator/lock and original run; explicit perhold bootUUID was not captured by historical runner. No closing quartet or crossing reboot is inferred.'
  else:
   assert block['scope']=='completed_fresh_pair';assert runtime['complete'] and runtime['pair_block_complete'] and not runtime['requested_matrix_complete'];assert runtime['actual_holds']==2 and runtime['selected_block']=={'rig':key[0],'mode':key[1]};assert re.fullmatch('[0-9a-f-]{36}',runtime['boot_id_start']) and runtime['boot_id_start']==runtime['boot_id_end'];assert set(runtime['conditions'])=={key[1]};groups=runtime['conditions'][key[1]];assert len(groups)==1 and groups[0]['rig']==key[0];holds=groups[0]['holds'];boot_scope={'start':runtime['boot_id_start'],'end':runtime['boot_id_end'],'same_within_block':True}
  assert len(holds)==2 and [(h['hold_index'],h['tree'])for h in holds]==[(1,'old'),(2,'new')]
  for hold in holds:
   assert hold['rig']==key[0] and hold['mode']==key[1];assert hold['source_commit']==SOURCES[hold['tree']];assert hold['binary_sha256']==BINARIES[hold['tree']][key[0]];assert hold['teardown'];assert {r['rank']for r in hold['ranks']}==({'W7800tp2':{0,1},'V620tp1':{0}}[key[0]]);hold['native_exit_classifications']=[classify_rank(r,key[0])for r in hold['ranks']];assert hold['actual_capacity_observed']==8;assert len(hold['benchmarks'])==4 and {(b['shape'],b['concurrency'])for b in hold['benchmarks']}=={(s,c)for s in SHAPES for c in [1,8]};hold['run_root']=block['run_root']
  state['conditions'].setdefault(key[1],[]).append({'rig':key[0],'holds':holds});verified.append({'rig':key[0],'mode':key[1],'state_sha256':block['state_sha256'],'boot_scope':boot_scope,'native_exit_classifications':[{'hold_index':h['hold_index'],'tree':h['tree'],'ranks':h['native_exit_classifications']}for h in holds]})
 state['complete']=not pending;R['selected_blocks']=verified;R['pending_blocks']=pending;R['selection_sha256']=hashlib.sha256(a.selection.read_bytes()).hexdigest();R['independent_blocks_not_single_boot_run']=True;assert a.partial or not pending
 return state
SOURCES={'old':'8649e9858b38e9ef3dcd8d579879c6512a363050','new':'729833615595df66c9f9a438ce9a85061843f010'}
BINARIES={'old':{'W7800tp2':'a71d528e226baceab0d1d6fd01de0dcaadb7940ce88469cba6d954ae15440256','V620tp1':'73290b400a23f52e442a59bfbac1b0d6080dc67bfb53bd1a689ed84dad96e225'},'new':{'W7800tp2':'4387690303ac2a7d8050ecd4db766085d55605f2f235bc552fb8255f29ad52dd','V620tp1':'ec661b4569807ba529f567f728bb44edca4e5fe6a8bda564de9bc809efa1ae49'}}
FIXTURE_IDS={}
SHAPES={'code_1024_512':(1024,512,'fdecab4eb1392c27d160b5d0bfbd236274614fb80b1c01e4edc5828e0b15499e'),'prose_16384_1024':(16384,1024,'51e13c227097b4fa1e0a6cc854f2a77208bf2301c1aa478bb8e76b2f47f4defa')}

try:
 assert a.derived_index is None,'Use explicit source-bound selection, not an overriding derived index'
 state=load_selection() if a.selection else json.loads((a.root/'state.json').read_text());R['state_complete']=state.get('complete',False)
 if a.derived_index:
  assert a.partial,'Derived partial index cannot claim full runtime completion'
  index_raw=a.derived_index.read_bytes();index=json.loads(index_raw);assert index['analytical_only'] is True
  for binding in index['runtime_run_record_bindings']:assert hashlib.sha256((a.root/binding['file']).read_bytes()).hexdigest()==binding['sha256']
  R['derived_partial_index']={'path':str(a.derived_index),'sha256':hashlib.sha256(index_raw).hexdigest(),'runtime_state_unmodified':True};state={**state,'conditions':index['conditions']}
 for mode,rigs in state['conditions'].items():
  for group in rigs:
   rig=group['rig']
   for hold in group['holds']:
    for b in hold['benchmarks']:
     file=a.root/hold.get('run_root','')/rig/b['file'];raw=file.read_bytes();assert hashlib.sha256(raw).hexdigest()==b['raw_sha256'];assert b['client_exit_status']==0
     x=json.loads(raw);expected_prompt,expected_output,expected_fixture=SHAPES[b['shape']];assert (x['prompt_tokens'],x['requested_tokens'],x['fixture_sha256'])==(expected_prompt,expected_output,expected_fixture);body=x['exact_request_body'];assert body['prompt']==FIXTURE_IDS[b['shape']];assert body.get('draft',True) is(mode=='on');assert len(body['prompt'])==expected_prompt and all(type(v)is int and 0<=v<2**32 for v in body['prompt']);assert body['max_tokens']==expected_output and body['temperature']==0 and body['seed']==1234 and body['ignore_eos'];assert hashlib.sha256(json.dumps(body).encode()).hexdigest()==x['request_body_sha256'];assert x['complete'] and not x.get('error');assert x['mode']==mode and x['concurrency']==b['concurrency'];assert [z['phase'] for z in x['batches']]==['warmup','measured','measured','measured']
     rows=[]
     for batch in x['batches']:
      runs=batch['runs'];assert len(runs)==b['concurrency'];metrics=[]
      for r in runs:
       assert r['complete'] and not r.get('error') and r['cached_tokens']==0 and r['tokens']==x['requested_tokens'] and r['prompt_tokens']==x['prompt_tokens']
       assert r['actual_request_body']==body and r['actual_request_body_sha256']==x['request_body_sha256'];rt=r['raw_runtime'];assert re.fullmatch('[0-9a-f]{64}',rt['token_sha256']) and rt['token_sha256']==r['token_sha256'];content=[];reason=[]
       for event in r['raw_sse_events']:
        for choice in event['chunk'].get('choices',[]):
         delta=choice.get('delta')or{};content.append(choice.get('text')or delta.get('content')or'');reason.append(delta.get('reasoning_content')or'')
       reconstructed=hashlib.sha256(hashlib.sha256(''.join(content).encode()).digest()+hashlib.sha256(''.join(reason).encode()).digest()).hexdigest();assert reconstructed==r['output_sha256'];assert rt['drafts'] is (mode=='on');finish=[c.get('finish_reason') for e in r['raw_sse_events'] for c in e['chunk'].get('choices',[]) if c.get('finish_reason') is not None];assert finish==['length']
       n=r['tokens'];span=r['last']-r['first'];prefill=rt['prefill_seconds'];decode=rt['seconds'];assert min(span,prefill,decode)>0
       m={'client_ttft_s':r['first']-r['sent'],'server_ttft_s':rt['time_to_first_token'],'end_to_end_s':r['completed']-r['sent'],'server_prefill_elapsed_s':prefill,'server_prefill_elapsed_tps':r['prompt_tokens']/prefill,'server_decode_elapsed_s':decode,'server_decode_tps':(n-1)/decode,'delivery_decode_tps':(n-1)/span,'delivery_mean_tpot_s':span/(n-1),'server_mean_decode_interval_s':decode/(n-1)}
       close(r['ttft_s'],m['client_ttft_s'],'ttft');close(r['decode_tps'],m['delivery_decode_tps'],'deliveryrate');close(rt['tokens_per_second'],m['server_decode_tps'],'serverrate');close(r['delivery_tpot_seconds'],m['delivery_mean_tpot_s'],'tpot')
       gaps=[y[0]-z[0] for z,y in zip(r['pieces'],r['pieces'][1:])];assert gaps==r['chunk_gap_seconds'];m['chunk_gap_count']=len(gaps);m['chunk_gap_seconds']=gaps
       key=(rig,b['shape']);hashes.setdefault(key,set()).add((rt['token_sha256'],r['output_sha256']));assert len(hashes[key])==1,'Cross-source/mode/concurrency equality failure'
       metrics.append(m);total+=1;measured+=batch['phase']=='measured'
      wall=max(r['completed'] for r in runs)-min(r['sent'] for r in runs);delivery=max(r['last'] for r in runs)-min(r['first'] for r in runs);first=max(r['first'] for r in runs)-min(r['sent'] for r in runs);pt=sum(r['prompt_tokens'] for r in runs);ct=sum(r['tokens'] for r in runs)
      ag={'wall_seconds':wall,'send_spread_seconds':max(r['sent'] for r in runs)-min(r['sent'] for r in runs),'completion_end_to_end_tps':ct/wall,'prompt_end_to_end_tps':pt/wall,'total_end_to_end_tps':(pt+ct)/wall,'completion_delivery_window_tps':sum(r['tokens']-1 for r in runs)/delivery,'weighted_server_prefill_service_tps':pt/sum(r['raw_runtime']['prefill_seconds'] for r in runs),'batch_first_content_window_prefill_proxy_tps':pt/first}
      for k,v in ag.items():close(batch['aggregate'][k],v,k)
      rows.append({'phase':batch['phase'],'per_request':metrics,'aggregate':ag})
     cell={'rig':rig,'mode':mode,'shape':b['shape'],'concurrency':b['concurrency'],'tree':hold['tree'],'hold_index':hold['hold_index'],'file':str(file.relative_to(a.root)),'sha256':hashlib.sha256(raw).hexdigest(),'batches':rows};R['cells'].append(cell)
     measured_rows=rows[1:];series={}
     for k in measured_rows[0]['per_request'][0]:
      if k not in ['chunk_gap_count','chunk_gap_seconds']:series['request_batch_median_'+k]=[median([v[k] for v in z['per_request']])for z in measured_rows]
     for k in measured_rows[0]['aggregate']:series['batch_'+k]=[z['aggregate'][k]for z in measured_rows]
     summaries={k:{'batch_values':v,'hold_mean':statistics.mean(v),'hold_median':median(v)}for k,v in series.items()}
     raw_measured=x['batches'][1:];completion=sum(r['tokens']for z in raw_measured for r in z['runs']);active_wall=sum(z['aggregate']['wall_seconds']for z in measured_rows);span=max(r['completed']for z in raw_measured for r in z['runs'])-min(r['sent']for z in raw_measured for r in z['runs'])
     cell['three_measured_batch_summaries']=summaries;cell['completion_denominators']={'tokens':completion,'sum_batch_wall_seconds':active_wall,'first_measured_sent_to_last_completed_seconds':span,'pooled_active_batch_completion_tps':completion/active_wall,'elapsed_span_completion_tps_including_interbatch_gaps':completion/span}
     values={k+'_'+which:detail[which]for k,detail in summaries.items()for which in ['hold_mean','hold_median']};values.update({k:v for k,v in cell['completion_denominators'].items()if k.endswith('tps')or k.endswith('gaps')});groups.setdefault((rig,mode,b['shape'],b['concurrency']),[]).append((hold['hold_index'],hold['tree'],values))
 for key,holds in groups.items():
  if len(holds)!=2:continue
  holds.sort();assert [h[0]for h in holds]==[1,2];assert [h[1] for h in holds]==['old','new'];c={'rig':key[0],'mode':key[1],'shape':key[2],'concurrency':key[3],'metrics':{}}
  for metric in holds[0][2]:
   vals=[h[2][metric] for h in holds];old=vals[0];new=vals[1];c['metrics'][metric]={'hold_order':vals,'old_first_hold_statistic':old,'new_first_hold_statistic':new,'change_percent':100*(new/old-1) if old else None,'baseline_drift_available':False,'order':'old,new descriptive pair'}
  R['comparisons'].append(c)
 R.update(validated_cell_files=len(R['cells']),validated_requests_including_warmups=total,validated_measured_requests=measured,output_equality_groups={':'.join(k):list(next(iter(v))) for k,v in hashes.items()})
 if not a.partial:assert len(R['cells'])==32 and total==576 and measured==432 and len(R['comparisons'])==16;assert {(c['rig'],c['mode'],c['shape'],c['concurrency'])for c in R['comparisons']}=={(r,m,s,c)for r in ['W7800tp2','V620tp1']for m in ['off','on']for s in ['code_1024_512','prose_16384_1024']for c in [1,8]}
 R['validated_available_raw']=True;R['selected_pair_matrix_raw_complete']=len(R['cells'])==32 and total==576 and measured==432 and len(R['comparisons'])==16 and len({(c['rig'],c['mode'],c['shape'],c['concurrency'])for c in R['comparisons']})==16;R['original_full_matrix_complete']=False;R['statistical_confidence_claimed']=False;R['full_contributing_qualified']=False;R['token_hash_scope']='Full server-reported token hashes; independently rebuilt content/reasoning fingerprint; no pertoken output ID arrays available'
except Exception as e:R['errors'].append(type(e).__name__+': '+str(e));R['validated_available_raw']=False
R['metric_scope']='Server prefill elapsed excludes initial queue but includes scheduler elapsed time, not pureGPUcompute. Client TTFT is first nonempty text arrival; delivery TPOT and server mean decode interval are averages, not per-token ITL distributions. Chunk gaps are transport chunk gaps. Rate and latency change percentages retain their own denominators; no acceptance threshold.'
a.output.write_text(json.dumps(R,indent=2)+'\n');print(a.output,'valid',R['validated_available_raw'],'cells',len(R['cells']),'errors',R['errors']);raise SystemExit(0 if R['validated_available_raw'] else 1)
