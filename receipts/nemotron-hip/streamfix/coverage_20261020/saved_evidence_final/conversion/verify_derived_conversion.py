from pathlib import Path
import hashlib,json,struct,os,datetime
H=Path.home();D=H/'models/Nemotron-derived-fp16-890ab464-20261021';R=H/'hcx/streamfix/coverage_20261020/conversion_actual_890ab464_20261021';sha=lambda b:hashlib.sha256(b).hexdigest();M=json.loads((D/'conversion_manifest.json').read_text());res=json.loads((R/'result.json').read_text());intent=json.loads((R/'intent.json').read_text());assert res['exit_status']==0 and res['original_metadata_unchanged']and res['source_after']==intent['source_before'];assert sha((D/'conversion_manifest.json').read_bytes())==res['conversion_manifest_sha256'];assert M['tool_sha256']=='890ab464a8aa1efc8953c5eac335d8a50a6178fae537013598ae81bd0e7ac31b'and M['plan_sha256']=='43e5475f147fe952565a9f90c539f1cb8d4eb6ee311d883ddb020af59947e3d1';assert not list(D.glob('*.tmp'));index=json.loads((D/'model.safetensors.index.json').read_text());plan={t['name']:t for t in M['plan']['tensors']};seen={};rows=[]
for s in M['shards']:
 p=D/s['file'];st=p.stat();h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(8*1024**2):h.update(b)
 assert h.hexdigest()==s['sha256']and st.st_size==s['bytes']
 with p.open('rb')as f:n=int.from_bytes(f.read(8),'little');header=json.loads(f.read(n))
 last=0
 for name,v in header.items():
  if name=='__metadata__':continue
  t=plan[name];assert name not in seen and t['shape']==v['shape']and t['dtype']==v['dtype'];a,b=v['data_offsets'];assert a==last and b-a==t['bytes'];last=b;seen[name]=s['file']
 assert last+8+n==st.st_size
 rows.append({'file':s['file'],'sha256':h.hexdigest(),'bytes':st.st_size,'header_bytes':n,'tensors':len(header)})
assert len(rows)==62 and len(seen)==6513 and seen==index['weight_map']and index['metadata']['total_size']==65826538624
for x in M['copied_metadata']:assert sha((D/x['file']).read_bytes())==x['sha256']
cfg=json.loads((D/'config.json').read_text());assert cfg['dtype']==cfg['torch_dtype']=='float16'and 'quantization'not in cfg and 'quantization_config'not in cfg
proc=json.loads((R/'process.json').read_text());live=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  try:
   text=(p/'stat').read_text();tail=text[text.rfind(')')+2:].split();pgid=int(tail[2]);state=tail[0]
   if pgid==proc['pgid']and state!='Z':live.append(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not live
report={'validated':True,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'conversion_manifest_sha256':res['conversion_manifest_sha256'],'all_shards_sha_verified':rows,'tensor_count':len(seen),'tensor_bytes':65826538624,'index_complete':True,'source_metadata_unchanged':True,'child_group_live_pids':live,'affine_fp16_rounding':M['affine_fp16_rounding'],'stock_loader_executed':False,'scope':'FulloutputSHA/header/index/config verification; sourceonlyheader/config/stat/read-orderhashes notfulloriginalpayloadhash; no inference equivalence'}
with (R/'full_verification.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'validated':True,'shards':62,'tensors':6513,'report_sha256':sha((R/'full_verification.json').read_bytes())}))
