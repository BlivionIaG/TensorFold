"""Read retained explicitly selected three-hold alternating control JSON only; recompute metrics, never launch jobs."""
import argparse,json,math,statistics,hashlib,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--partial',action='store_true');p.add_argument('--derived-index',type=Path);p.add_argument('--selection',type=Path,required=True);a=p.parse_args()
R={'schema_version':1,'scope':'Explicit O/N/O controls, three measured batches perhold; descriptive drift, no ABBA/confidence or readiness inference','true_token_itl_distribution_available':False,'pure_gpu_prefill_compute_claimed':False,'cells':[],'comparisons':[],'errors':[]};groups={};hashes={};total=measured=0

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
 selection=json.loads(a.selection.read_text());assert selection['schema_version']==1 and selection['protocol']=='O/N/O controls'
 for name,binding in selection['fixtures'].items():
  file=(a.root/binding['file']).resolve();assert file.is_relative_to(a.root.resolve());assert hashlib.sha256(file.read_bytes()).hexdigest()==SHAPES[name][2];FIXTURE_IDS[name]=json.loads(file.read_text())
 expected={('W7800tp2','off'),('V620tp1','off'),('W7800tp2','on'),('V620tp1','on')};assert {(b['rig'],b['mode'])for b in selection['blocks']}==expected and len(selection['blocks'])==4
 state={'complete':False,'conditions':{}};verified=[];pending=[]
 for block in selection['blocks']:
  key=(block['rig'],block['mode']);holds=[];boots=[];bindings=[]
  if any(entry['run_root'] is None for entry in block['entries']):pending.append({'rig':key[0],'mode':key[1]});continue
  for entry in block['entries']:
   root=(a.root/entry['run_root']).resolve();assert root.is_relative_to(a.root.resolve());sf=root/'state.json';assert hashlib.sha256(sf.read_bytes()).hexdigest()==entry['state_sha256'];runtime=json.loads(sf.read_text());assert runtime['measured_sources']==SOURCES and runtime['complete'] and runtime['selected_block']=={'rig':key[0],'mode':key[1]};assert runtime.get('pair_block_complete')or runtime.get('line81_block_complete');assert runtime['boot_id_start']==runtime['boot_id_end'] and re.fullmatch('[0-9a-f-]{36}',runtime['boot_id_start']);boots.append(runtime['boot_id_start']);assert set(runtime['conditions'])=={key[1]};gg=runtime['conditions'][key[1]];assert len(gg)==1 and gg[0]['rig']==key[0] and runtime['actual_holds']==len(gg[0]['holds'])
   chosen=[h for h in gg[0]['holds']if h['hold_index']in entry['hold_indices']];assert {h['hold_index']for h in chosen}==set(entry['hold_indices'])
   for hold in chosen:
    hold['run_root']=entry['run_root'];holds.append(hold)
   bindings.append({'run_root':entry['run_root'],'state_sha256':entry['state_sha256'],'selected_hold_indices':entry['hold_indices'],'boot_id':runtime['boot_id_start']})
  assert len(set(boots))==1,'Crossboot O/N/O forbidden';holds.sort(key=lambda h:h['hold_index']);assert [(h['hold_index'],h['tree'])for h in holds]==[(1,'old'),(2,'new'),(3,'old')]
  for hold in holds:
   assert hold['rig']==key[0] and hold['mode']==key[1];assert hold['source_commit']==SOURCES[hold['tree']]and hold['binary_sha256']==BINARIES[hold['tree']][key[0]];assert hold['teardown'];assert {r['rank']for r in hold['ranks']}==({'W7800tp2':{0,1},'V620tp1':{0}}[key[0]]);hold['native_exit_classifications']=[classify_rank(r,key[0])for r in hold['ranks']];assert hold['actual_capacity_observed']==8;assert len(hold['benchmarks'])==4 and {(b['shape'],b['concurrency'])for b in hold['benchmarks']}=={(s,c)for s in SHAPES for c in [1,8]}
  state['conditions'].setdefault(key[1],[]).append({'rig':key[0],'holds':holds});verified.append({'rig':key[0],'mode':key[1],'bindings':bindings,'same_boot_within_block':True,'gaps_not_waived':True,'scope':'Actual O/N/O source-bound holds, possibly separated by intervening work; no ABBA or confidence claim'})
 state['complete']=not pending;R['selected_blocks']=verified;R['pending_blocks']=pending;R['selection_sha256']=hashlib.sha256(a.selection.read_bytes()).hexdigest();assert a.partial or not pending
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
     cell['actual_request_monotonic_span']={'first_sent':min(r['sent']for z in x['batches']for r in z['runs']),'last_completed':max(r['completed']for z in x['batches']for r in z['runs'])};cell['three_measured_batch_summaries']=summaries;cell['completion_denominators']={'tokens':completion,'sum_batch_wall_seconds':active_wall,'first_measured_sent_to_last_completed_seconds':span,'pooled_active_batch_completion_tps':completion/active_wall,'elapsed_span_completion_tps_including_interbatch_gaps':completion/span}
     values={k+'_'+which:detail[which]for k,detail in summaries.items()for which in ['hold_mean','hold_median']};values.update({k:v for k,v in cell['completion_denominators'].items()if k.endswith('tps')or k.endswith('gaps')});groups.setdefault((rig,mode,b['shape'],b['concurrency']),[]).append((hold['hold_index'],hold['tree'],values))
 for key,holds in groups.items():
  holds.sort();assert len(holds)==3 and [h[0]for h in holds]==[1,2,3] and [h[1]for h in holds]==['old','new','old'];c={'rig':key[0],'mode':key[1],'shape':key[2],'concurrency':key[3],'metrics':{}}
  cells=sorted([x for x in R['cells']if(x['rig'],x['mode'],x['shape'],x['concurrency'])==key],key=lambda x:x['hold_index']);times=[x['actual_request_monotonic_span']for x in cells];assert all(t['first_sent']<t['last_completed']for t in times) and times[0]['last_completed']<=times[1]['first_sent'] and times[1]['last_completed']<=times[2]['first_sent'],'Nonchronological or overlapping O/N/O requests';c['actual_same_boot_monotonic_timing']={'holds':times,'opening_to_candidate_gap_seconds':times[1]['first_sent']-times[0]['last_completed'],'candidate_to_closing_gap_seconds':times[2]['first_sent']-times[1]['last_completed'],'scope':'Actual per-case first/last request times include warmups; intervening cells/work and long control gaps explicitly retained, no latency attribution'}
  for metric in holds[0][2]:
   v=[h[2][metric]for h in holds];book=(v[0]+v[2])/2;c['metrics'][metric]={'hold_order':['old_first','candidate','old_last'],'values':v,'old_first_to_old_last_change_percent':100*(v[2]/v[0]-1)if v[0]else None,'candidate_vs_old_first_percent':100*(v[1]/v[0]-1)if v[0]else None,'candidate_vs_old_last_percent':100*(v[1]/v[2]-1)if v[2]else None,'arithmetic_bookend_mean':book,'candidate_vs_bookend_mean_percent':100*(v[1]/book-1)if book else None,'scope':'Unrounded descriptive statistics; bookend mean is not a drift model/confidence interval or waiver'}
  R['comparisons'].append(c)
 R.update(validated_cell_files=len(R['cells']),validated_requests_including_warmups=total,validated_measured_requests=measured,output_equality_groups={':'.join(k):list(next(iter(v))) for k,v in hashes.items()})
 if not a.partial:assert len(R['cells'])==48 and total==864 and measured==648 and len(R['comparisons'])==16;assert {(c['rig'],c['mode'],c['shape'],c['concurrency'])for c in R['comparisons']}=={(r,m,s,c)for r in ['W7800tp2','V620tp1']for m in ['off','on']for s in ['code_1024_512','prose_16384_1024']for c in [1,8]}
 R['validated_available_raw']=True;R['alternating_control_campaign_raw_complete']=len(R['cells'])==48 and total==864 and measured==648 and len(R['comparisons'])==16 and len({(c['rig'],c['mode'],c['shape'],c['concurrency'])for c in R['comparisons']})==16;R['original_full_matrix_complete']=False;R['automatic_drift_acceptance_threshold']=None;R['readiness_inferred']=False;R['statistical_confidence_claimed']=False;R['full_contributing_qualified']=False;R['token_hash_scope']='Full server-reported token hashes; independently rebuilt content/reasoning fingerprint; no pertoken output ID arrays available'
except Exception as e:R['errors'].append(type(e).__name__+': '+str(e));R['validated_available_raw']=False
R['metric_scope']='Server prefill elapsed excludes initial queue but includes scheduler elapsed time, not pureGPUcompute. Client TTFT is first nonempty text arrival; delivery TPOT and server mean decode interval are averages, not per-token ITL distributions. Chunk gaps are transport chunk gaps. Rate and latency change percentages retain their own denominators; no acceptance threshold.'
a.output.write_text(json.dumps(R,indent=2)+'\n');print(a.output,'valid',R['validated_available_raw'],'cells',len(R['cells']),'errors',R['errors']);raise SystemExit(0 if R['validated_available_raw'] else 1)
