"""Attach an actual independently reviewed summarySHA without mutating that summary. Local only."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('public_stage',type=Path);p.add_argument('audit',type=Path);a=p.parse_args();D=a.public_stage;raw=a.audit.read_bytes();r=json.loads(raw);summary=D/'summary.json'
assert r['schema_version']==1 and r['independently_validated'] is True and r['source_commit']=='729833615595df66c9f9a438ce9a85061843f010'
assert r['summary_sha256']==hashlib.sha256(summary.read_bytes()).hexdigest() and r['actual_validated_holds']==16 and r['single_model_scheduling_observed'] is True and r['existing_gpu_lock_qualified'] is False and r['memory_admission_environment_observed'] is True
assert b'/Users/' not in raw and b'/home/' not in raw,'Independent public audit must not contain personal home paths'
name='independent_audit.json';(D/name).write_bytes(raw);m=json.loads((D/'manifest.json').read_text());m['files']=[x for x in m['files'] if x['path']!=name];m['files'].append({'path':name,'raw_sha256':hashlib.sha256(raw).hexdigest(),'public_sha256':hashlib.sha256(raw).hexdigest(),'raw_bytes':len(raw),'public_bytes':len(raw)});m['file_count']=len(m['files']);(D/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('Actual independent audit attached; summary unchanged')
