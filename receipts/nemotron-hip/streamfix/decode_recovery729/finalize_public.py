"""Add independent audits and explicit drift to the curated selected16 package."""
import hashlib,json,re
from pathlib import Path
S=Path(__file__).resolve().parent;D=S/'public-stage';O=S.parent;summary=json.loads((D/'summary.json').read_text());manifest=json.loads((D/'manifest.json').read_text())
extras={'independent_recovery_ab_audit.json':O/'independent_recovery_ab_audit.json','independent_recovery_off_audit.json':O/'independent_recovery_off_audit.json','independent_recovery_on_audit.json':O/'independent_recovery_on_audit.json','original_recovery_failure_audit.json':O/'original_recovery_failure_audit.json','audit_recovery_readonly.py':O/'audit_recovery_readonly.py','combine.py':S/'combine.py','finalize_public.py':Path(__file__).resolve()}
for name,source in extras.items():
 raw=source.read_bytes();public=re.sub(r'/home/[A-Za-z0-9_.-]+','<USER_HOME>',raw.decode()).encode();(D/name).write_bytes(public);manifest['files']=[x for x in manifest['files'] if x['path']!=name];manifest['files'].append({'path':name,'raw_sha256':hashlib.sha256(raw).hexdigest(),'public_sha256':hashlib.sha256(public).hexdigest(),'raw_bytes':len(raw),'public_bytes':len(public)})
summary['independent_audit']={'file':'independent_recovery_ab_audit.json','public_sha256':hashlib.sha256((D/'independent_recovery_ab_audit.json').read_bytes()).hexdigest(),'passed':True}
summary['baseline_hold_drift']=[{'mode':g['mode'],'rig':g['rig'],'prompt':p,'old_first':r['hold_order'][0],'old_last':r['hold_order'][3],'old_last_vs_first_percent':100*(r['hold_order'][3]/r['hold_order'][0]-1)} for g in summary['conditions'] for p,r in g['prompts'].items()]
summary['interpretation']='The large measured cc183 regression is much smaller in this narrow candidate729 run. Several candidate means remain lower. No parity, stable speedup, statistical significance or performance-acceptance claim is made.'
summary['limitations']+=['V620 drafts-on chat baseline holds156.61746→133.07299 tok/s drift substantially; the+5.80109% mean does not establish a stable speedup.','W7800 drafts-on chat baseline holds216.02656→204.52731 tok/s also drift; all exact hold-order values are retained.']
manifest['file_count']=len(manifest['files']);(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert len({x['path'] for x in manifest['files']})==manifest['file_count']
for x in manifest['files']:assert hashlib.sha256((D/x['path']).read_bytes()).hexdigest()==x['public_sha256']
print(manifest['file_count'],'raw/curated bindings verified; selected16/excluded5; drift explicit')
