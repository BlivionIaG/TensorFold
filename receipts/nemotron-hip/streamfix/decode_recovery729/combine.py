"""Validate original OFF8 + fresh ON8; preserve and exclude original partial ON5."""
import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('original',type=Path);p.add_argument('retry',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();S=Path(__file__).resolve().parent
old=json.loads(subprocess.check_output([sys.executable,str(S/'summarize.py'),str(a.original)],text=True));new=json.loads(subprocess.check_output([sys.executable,str(S/'retry_on/summarize.py'),str(a.retry)],text=True))
assert old['validated'] and old['actual_validated_holds']==8 and {x['mode'] for x in old['conditions']}=={'off'}
assert new['validated'] and new['actual_validated_holds']==8 and new['complete_selected_decode_modes'] and new['selected_modes']==['on'] and not new['complete_decode_only']
assert old['measured_sources']==new['measured_sources'];accepted=[];excluded=[]
for label,root,mode in [('attempt1',a.original,'off'),('retry_on',a.retry,'on')]:
 state=json.loads((root/'state.json').read_text())
 for rig in ['W7800tp2','V620tp1']:
  for file in sorted((root/rig).glob('*.run.json')):
   r=json.loads(file.read_text());row={'attempt':label,'rig':rig,'mode':r['mode'],'hold_index':r['hold_index'],'tree':r['tree'],'source_commit':r['source_commit'],'run_file':str(Path(label)/file.relative_to(root)),'raw_run_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'benchmark':r.get('benchmark'),'runtime_identity':r.get('runtime_identity')}
   (accepted if r['mode']==mode else excluded).append(row)
assert len(accepted)==16 and len(excluded)==5 and all(x['attempt']=='attempt1' and x['mode']=='on' for x in excluded)
a.destination.mkdir(parents=True,exist_ok=False);manifest=[]
for label,root in [('attempt1',a.original),('retry_on',a.retry)]:
 for file in sorted(root.rglob('*')):
  if not file.is_file():continue
  raw=file.read_bytes();public=re.sub(r'/home/[A-Za-z0-9_.-]+','<USER_HOME>',raw.decode('utf-8')).encode();rel=Path(label)/file.relative_to(root);dest=a.destination/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(public)
  manifest.append({'path':str(rel),'raw_sha256':hashlib.sha256(raw).hexdigest(),'public_sha256':hashlib.sha256(public).hexdigest(),'raw_bytes':len(raw),'public_bytes':len(public)})
summary={'validated':True,'scope':'Decode-only recovery protocol: selected original OFF8 + fresh ON8; no cold-prefill run or qualification.','measured_sources':old['measured_sources'],'expected_selected_holds':16,'actual_selected_holds':16,'complete_decode_only':True,'cold_prefill_run':False,'publication_qualified':False,'conditions':old['conditions']+new['conditions'],'selected_receipts':accepted,'excluded_original_partial_on_receipts':excluded,'excluded_receipt_count':5,'retry_reason':'Original attempt stopped at unused(port) with Errno98 before V620 ON hold2; original ON5 retained and excluded. Both rigs ON rerun as fresh old/new/new/old quartets using lower ports.','limitations':['Two fixed prompts,256 completion tokens,3 repetitions per hold,temperature0.','Each reported median checked against median of three rounded rates; reported full-precision medians used for means.','Direct hold-median means and percentage differences; no significance or recovery threshold claimed.','Native server exits are actual intentional teardown signals; all process groups disappeared; client exit0 required.','Pinned checkpoint revision/cache metadata and actual executable/library hashes retained; no claim hashing every weight byte.'],'privacy_redaction':'Personal home directories only; raw/public file hashes preserve transformation bindings.'}
(a.destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.destination/'manifest.json').write_text(json.dumps({'file_count':len(manifest),'files':manifest},indent=2)+'\n');print(a.destination)
