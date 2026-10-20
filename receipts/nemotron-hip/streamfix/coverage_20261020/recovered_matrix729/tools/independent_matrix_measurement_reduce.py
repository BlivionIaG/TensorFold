"""Read retained three-repetition matrix JSON only; recompute metrics, never launch jobs."""
import argparse,json,math,statistics,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--partial',action='store_true');p.add_argument('--derived-index',type=Path);a=p.parse_args()
R={'schema_version':1,'scope':'Independent raw reduction of Three-measured-batch requested matrix; not statistical confidence or full CONTRIBUTING qualification','true_token_itl_distribution_available':False,'pure_gpu_prefill_compute_claimed':False,'cells':[],'comparisons':[],'errors':[]};groups={};hashes={};total=measured=0

def close(got,want,label):
 assert type(got) in (int,float) and math.isfinite(got) and math.isclose(got,want,rel_tol=1e-12,abs_tol=1e-12),(label,got,want)
def median(v):return statistics.median(v)
try:
 state=json.loads((a.root/'state.json').read_text());R['state_complete']=state.get('complete',False)
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
     file=a.root/rig/b['file'];raw=file.read_bytes();assert hashlib.sha256(raw).hexdigest()==b['raw_sha256'];assert b['client_exit_status']==0
     x=json.loads(raw);assert x['complete'] and not x.get('error');assert x['mode']==mode and x['concurrency']==b['concurrency'];assert [z['phase'] for z in x['batches']]==['warmup','measured','measured','measured']
     rows=[]
     for batch in x['batches']:
      runs=batch['runs'];assert len(runs)==b['concurrency'];metrics=[]
      for r in runs:
       assert r['complete'] and not r.get('error') and r['cached_tokens']==0 and r['tokens']==x['requested_tokens'] and r['prompt_tokens']==x['prompt_tokens']
       rt=r['raw_runtime'];assert rt['drafts'] is (mode=='on');finish=[c.get('finish_reason') for e in r['raw_sse_events'] for c in e['chunk'].get('choices',[]) if c.get('finish_reason') is not None];assert finish==['length']
       n=r['tokens'];span=r['last']-r['first'];prefill=rt['prefill_seconds'];decode=rt['seconds'];assert min(span,prefill,decode)>0
       m={'client_ttft_s':r['first']-r['sent'],'server_ttft_s':rt['time_to_first_token'],'end_to_end_s':r['completed']-r['sent'],'server_prefill_elapsed_s':prefill,'server_prefill_elapsed_tps':r['prompt_tokens']/prefill,'server_decode_elapsed_s':decode,'server_decode_tps':(n-1)/decode,'delivery_decode_tps':(n-1)/span,'delivery_mean_tpot_s':span/(n-1),'server_mean_decode_interval_s':decode/(n-1)}
       close(r['ttft_s'],m['client_ttft_s'],'ttft');close(r['decode_tps'],m['delivery_decode_tps'],'deliveryrate');close(rt['tokens_per_second'],m['server_decode_tps'],'serverrate');close(r['delivery_tpot_seconds'],m['delivery_mean_tpot_s'],'tpot')
       gaps=[y[0]-z[0] for z,y in zip(r['pieces'],r['pieces'][1:])];assert gaps==r['chunk_gap_seconds'];m['chunk_gap_count']=len(gaps)
       key=(rig,b['shape']);hashes.setdefault(key,set()).add((rt['token_sha256'],r['output_sha256']));assert len(hashes[key])==1,'Cross-source/mode/concurrency equality failure'
       metrics.append(m);total+=1;measured+=batch['phase']=='measured'
      wall=max(r['completed'] for r in runs)-min(r['sent'] for r in runs);delivery=max(r['last'] for r in runs)-min(r['first'] for r in runs);first=max(r['first'] for r in runs)-min(r['sent'] for r in runs);pt=sum(r['prompt_tokens'] for r in runs);ct=sum(r['tokens'] for r in runs)
      ag={'wall_seconds':wall,'send_spread_seconds':max(r['sent'] for r in runs)-min(r['sent'] for r in runs),'completion_end_to_end_tps':ct/wall,'prompt_end_to_end_tps':pt/wall,'total_end_to_end_tps':(pt+ct)/wall,'completion_delivery_window_tps':sum(r['tokens']-1 for r in runs)/delivery,'weighted_server_prefill_service_tps':pt/sum(r['raw_runtime']['prefill_seconds'] for r in runs),'batch_first_content_window_prefill_proxy_tps':pt/first}
      for k,v in ag.items():close(batch['aggregate'][k],v,k)
      rows.append({'phase':batch['phase'],'per_request':metrics,'aggregate':ag})
     cell={'rig':rig,'mode':mode,'shape':b['shape'],'concurrency':b['concurrency'],'tree':hold['tree'],'hold_index':hold['hold_index'],'file':str(file.relative_to(a.root)),'sha256':hashlib.sha256(raw).hexdigest(),'batches':rows};R['cells'].append(cell)
     measured_rows=rows[1:];series={}
     for k in measured_rows[0]['per_request'][0]:
      if k!='chunk_gap_count':series['request_batch_median_'+k]=[median([v[k] for v in z['per_request']])for z in measured_rows]
     for k in measured_rows[0]['aggregate']:series['batch_'+k]=[z['aggregate'][k]for z in measured_rows]
     summaries={k:{'batch_values':v,'hold_mean':statistics.mean(v),'hold_median':median(v)}for k,v in series.items()}
     raw_measured=x['batches'][1:];completion=sum(r['tokens']for z in raw_measured for r in z['runs']);active_wall=sum(z['aggregate']['wall_seconds']for z in measured_rows);span=max(r['completed']for z in raw_measured for r in z['runs'])-min(r['sent']for z in raw_measured for r in z['runs'])
     cell['three_measured_batch_summaries']=summaries;cell['completion_denominators']={'tokens':completion,'sum_batch_wall_seconds':active_wall,'first_measured_sent_to_last_completed_seconds':span,'pooled_active_batch_completion_tps':completion/active_wall,'elapsed_span_completion_tps_including_interbatch_gaps':completion/span}
     values={k+'_'+which:detail[which]for k,detail in summaries.items()for which in ['hold_mean','hold_median']};values.update({k:v for k,v in cell['completion_denominators'].items()if k.endswith('tps')or k.endswith('gaps')});groups.setdefault((rig,mode,b['shape'],b['concurrency']),[]).append((hold['hold_index'],hold['tree'],values))
 for key,holds in groups.items():
  if len(holds)!=4:continue
  holds.sort();assert [h[1] for h in holds]==['old','new','new','old'];c={'rig':key[0],'mode':key[1],'shape':key[2],'concurrency':key[3],'metrics':{}}
  for metric in holds[0][2]:
   vals=[h[2][metric] for h in holds];old=(vals[0]+vals[3])/2;new=(vals[1]+vals[2])/2;c['metrics'][metric]={'hold_order':vals,'old_mean_of_hold_statistics':old,'new_mean_of_hold_statistics':new,'change_percent':100*(new/old-1) if old else None,'old_first_to_last_drift_percent':100*(vals[3]/vals[0]-1) if vals[0] else None}
  R['comparisons'].append(c)
 R.update(validated_cell_files=len(R['cells']),validated_requests_including_warmups=total,validated_measured_requests=measured,output_equality_groups={':'.join(k):list(next(iter(v))) for k,v in hashes.items()})
 if not a.partial:assert state['complete'] and len(R['cells'])==64 and total==1152 and measured==864 and len(R['comparisons'])==16
 R['validated_available_raw']=True;R['requested_matrix_raw_complete']=len(R['cells'])==64 and total==1152 and measured==864
except Exception as e:R['errors'].append(type(e).__name__+': '+str(e));R['validated_available_raw']=False
R['metric_scope']='Server prefill elapsed excludes initial queue but includes scheduler elapsed time, not pureGPUcompute. Client TTFT is first nonempty text arrival; delivery TPOT and server mean decode interval are averages, not per-token ITL distributions. Chunk gaps are transport chunk gaps. Rate and latency change percentages retain their own denominators; no acceptance threshold.'
a.output.write_text(json.dumps(R,indent=2)+'\n');print(a.output,'valid',R['validated_available_raw'],'cells',len(R['cells']),'errors',R['errors']);raise SystemExit(0 if R['validated_available_raw'] else 1)
