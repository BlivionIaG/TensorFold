"""Post-run read-only serial receipt audit. Does not execute models or grant independent review."""
import argparse,contextlib,hashlib,io,json,math,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();S=Path(__file__).resolve().parent;D=a.raw
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
state=json.loads((D/'state.json').read_text());assert state['complete'] and state['actual_holds']==16 and state['single_model_host_serial'] and state['modes']==['off','on']
assert sha(S/'summarize.py')==sha(D/'inputs/summarize.py'),'Audit summarizer differs from measured frozen helper'
report=json.loads(subprocess.check_output([sys.executable,str(S/'summarize.py'),str(D)],text=True));assert report['validated'] and report['complete_decode_only'] and report['actual_validated_holds']==16
cfg=json.loads((D/'config.raw.json').read_text());allocation=json.loads((D/'allocation.raw.json').read_text());frozen=json.loads((D/'payload_sha256.json').read_text());assert sha(D/'payload_sha256.json')==allocation['harness_manifest_sha256']
assert sha(D/'config.raw.json')==frozen['config.json']
assert allocation['source_commit']=='729833615595df66c9f9a438ce9a85061843f010' and allocation['purpose']=='decode_serial_recovery_ab' and allocation['exclusive_gpu_reservation']
assert allocation['single_model_scheduling'] and allocation['exclusive_owned_gpu_reservation'] and 'exclusive_physical_host' not in allocation
assert {k:v['devices'] for k,v in cfg['rigs'].items()}=={'W7800tp2':'0,1','V620tp1':'2'}
for n in ['run.py','summarize.py','ab_validate.py','capture_identity.py']:assert sha(D/'inputs'/n)==frozen[n]
assert 'ThreadPool' not in (D/'inputs/run.py').read_text() and 'concurrent.futures' not in (D/'inputs/run.py').read_text()
for n,digest in cfg['frozen_client_sha256'].items():assert sha(D/'inputs'/n)==digest
holds=[];starts=[];fingerprints={};rank_pids=set()
for mode in ['off','on']:
 groups=state['conditions'][mode];assert [g['rig'] for g in groups]==['W7800tp2','V620tp1']
 for g in groups:
  for h in g['holds']:
   pre=h['host_preflight'];assert pre['single_logical_model_instance_per_host'] and pre['known_other_model_or_build_processes']==[] and pre['shared_build_lock_held']
   assert len(pre['host_loadavg_1_5_15'])==3 and all(math.isfinite(x) and x>=0 for x in pre['host_loadavg_1_5_15'])
   mem=pre['host_ram_kib'];assert 0<mem['MemAvailable']<=mem['MemTotal'];assert pre['context_tokens']==65536 and pre['parallel_streams']==4
   assert h['binary_sha256']==cfg['frozen_native_sha256'][h['tree']][cfg['rigs'][g['rig']]['arch']]
   doc=json.loads((D/g['rig']/h['benchmark']['file']).read_text())
   for prompt in doc:
    key=(g['rig'],prompt['prompt']);fingerprints.setdefault(key,set()).update((x['token_sha'],x['output_sha256']) for x in prompt['evidence'])
   rank_pids.update(x['pid'] for x in h['ranks'])
   starts.append(pre['at_utc']);holds.append({'mode':mode,'rig':g['rig'],'index':h['hold_index'],'tree':h['tree'],'source_commit':h['source_commit'],'binary_sha256':h['binary_sha256'],'preflight':pre,'actual_native_exit_statuses':[x['actual_exit_status'] for x in h['ranks']],'client_exit_status':h['client_exit_status'],'groups_gone':all(x['process_group_gone'] for x in h['ranks'])})
assert starts==sorted(starts) and len(holds)==16 and len(fingerprints)==4 and all(len(x)==1 for x in fingerprints.values())
obs=json.loads((D/'observation.json').read_text());keys={'TENSORFOLD_MEMORY_RESERVE_GIB','TENSORFOLD_HIP_MEMORY_LIMIT_GB'};expected_env=allocation['memory_admission_environment']
assert set(expected_env)==keys and obs['source_context']==report['measured_sources']['new'] and obs['allocation_snapshot']==expected_env
assert obs['observer_complete'] and obs['errors']==[] and obs['coordinator']['environment']==expected_env and obs['coordinator']['matches_allocation']
observed={x['pid'] for x in obs['ranks']};assert observed and len(observed)==len(obs['ranks']) and observed<=rank_pids
assert len(rank_pids)==24 and all(x['environment']==expected_env and set(x['environment'])==keys and x['matches_allocation'] and x['parent_pid']==obs['coordinator']['pid'] for x in obs['ranks'])
memory_scope={'coordinator_environment':obs['coordinator']['environment'],'actual_rank_environment_samples':len(observed),'total_recorded_rank_processes':len(rank_pids),'all_sampled_rank_environments_match_allocation':True,'unobserved_rank_environments_inferred':False,'observation_sha256':sha(D/'observation.json')}
r={'schema_version':1,'validated_by_helper':True,'memory_admission_environment_observed':True,'memory_admission_environment_scope':memory_scope,'independent_reviewer_required':True,'scope':'Fresh serial16 decode-only; not full CONTRIBUTING qualification','measured_sources':report['measured_sources'],'actual_validated_holds':16,'expected_holds':16,'single_model_scheduling':True,'physical_host_lease_claimed':False,'known_process_preflight_limitation':'Named known model/build process scan only; arbitrary Python/renamed GPU executables may escape detection. Scheduling is coordinator assertion plus recorded observations, not a physical-host lease.','existing_gpu_lock_qualified':False,'existing_gpu_lock_scope':'No existing GPU lock identified in requested shell-script scan; CPU build lock is not GPU lock qualification. Frozen native admission limits unchanged.','all_reply_and_token_hashes_equal_across_modes':True,'complete_decode_only':True,'cold_prefill_benchmark_run':False,'full_contributing_qualified':False,'conditions':report['conditions'],'holds':holds,'raw_file_bindings':[{'path':str(x.relative_to(D)),'sha256':sha(x),'bytes':x.stat().st_size} for x in sorted(D.rglob('*')) if x.is_file()]}
a.output.write_text(json.dumps(r,indent=2)+'\n');print(a.output)
