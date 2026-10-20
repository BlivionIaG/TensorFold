#!/usr/bin/env python3
"""Compare independent old/new hardware dumps exactly, including NaN payloads."""
import argparse,hashlib,json,pathlib,struct
p=argparse.ArgumentParser();p.add_argument('old',type=pathlib.Path);p.add_argument('new',type=pathlib.Path);p.add_argument('old_build',type=pathlib.Path);p.add_argument('new_build',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--expected-candidate',required=True);a=p.parse_args()
builds=[json.loads(x.read_text()) for x in [a.old_build,a.new_build]]
assert builds[0]['arch']==builds[1]['arch'],'different architectures'
assert builds[0]['harness_sha256']==builds[1]['harness_sha256'],'different harness source'
assert builds[0]['source_commit']=='8649e9858b38e9ef3dcd8d579879c6512a363050','unexpected baseline'
assert builds[1]['source_commit']==a.expected_candidate,'unexpected final candidate'
assert all('-O3' in b['flags'] and '-O2' not in b['flags'] for b in builds),'incorrect optimization flags'
files=[{p.name:p for p in root.glob('*.bin')} for root in [a.old,a.new]]
expected=1080 if builds[0]['arch']=='gfx1100' else 540
assert len(files[0])==expected and files[0].keys()==files[1].keys(), 'missing fixture outputs'
result=[];failed=[];nan_cases=0;signedzero_cases=0
for name in sorted(files[0]):
 b,c=[m[name].read_bytes() for m in files]
 assert len(b)>0 and len(b)==len(c),'truncated/size mismatch'
 equal=b==c
 if not equal:failed.append(name)
 if '_m0.' in name or '_m4.' in name:
  values=struct.unpack('<'+'I'*(len(b)//4),b)
  nan_cases+=any((v&0x7f800000)==0x7f800000 and v&0x7fffff for v in values)
  signedzero_cases+=any(v==0x80000000 for v in values)
 result.append({'fixture':name,'bytes':len(b),'old_sha256':hashlib.sha256(b).hexdigest(),'new_sha256':hashlib.sha256(c).hexdigest(),'equal':equal})
assert nan_cases>0,'missing finite-input NaN output control'
report={'schema_version':1,'comparison':'exact bytes; no NaN canonicalization','builds':builds,'fixtures':result,'finite_input_nan_output_cases':nan_cases,'negative_zero_output_cases':signedzero_cases,'failed':failed,'pass':not failed}
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(f'{len(result)} fixtures; {nan_cases} finite-input NaN controls; {len(failed)} byte mismatches')
raise SystemExit(bool(failed))
