import json,hashlib
from pathlib import Path
HEAD='729833615595df66c9f9a438ce9a85061843f010'
def load(root):
 a=root/'allocation.json';g=json.loads(a.read_text());assert g['source_commit']==HEAD and g['parent_allocated'] and g['exclusive_gpu_reservation'] and g['recovery_ab_finished'] and g['purpose']=='followup_models'
 assert g['devices']=={'gfx1100':'0,1','gfx1030':'2,3,4,5'},'unexpected allocation'
 p=root/'MODEL_CHECKS_ALL_PASS';assert p.read_text().strip()==HEAD
 r=root/'summary.json';s=json.loads(r.read_text());assert s['source_commit']==HEAD and s['result']=='PASS' and s['state']=='complete'
 return {'source_commit':HEAD,'model_summary_sha256':hashlib.sha256(r.read_bytes()).hexdigest(),'allocation_sha256':hashlib.sha256(a.read_bytes()).hexdigest()}
