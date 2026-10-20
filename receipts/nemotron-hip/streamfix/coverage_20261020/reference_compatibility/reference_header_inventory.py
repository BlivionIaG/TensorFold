"""CPU metadata only: seek/read safetensors headers, never tensor payloads."""
import argparse,json,struct,math,shutil,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tensors={};headers=[]
for f in sorted(a.model.glob('*.safetensors')):
 with f.open('rb')as q:
  n=struct.unpack('<Q',q.read(8))[0];assert 0<n<64*1024*1024;raw=q.read(n);h=json.loads(raw)
 headers.append({'file':f.name,'header_bytes':n,'header_sha256':hashlib.sha256(raw).hexdigest(),'file_bytes':f.stat().st_size})
 for k,v in h.items():
  if k!='__metadata__':assert k not in tensors;tensors[k]=v
rows=[]
for k,v in tensors.items():
 if k.endswith(('.scales','.biases')):
  stem=k.rsplit('.',1)[0];assert stem+'.weight'in tensors and tensors[stem+'.weight']['dtype']=='U32';continue
 shape=v['shape'];quant=k.endswith('.weight')and k[:-7]+'.scales'in tensors
 if quant:
  assert v['dtype']=='U32';stem=k[:-7];assert tensors[stem+'.scales']['shape']==tensors[stem+'.biases']['shape'];shape=shape[:-1]+[shape[-1]*8];assert tensors[stem+'.scales']['shape']==shape[:-1]+[shape[-1]//64]
 rows.append({'name':k,'shape':shape,'quantized_source':quant,'bf16_bytes':2*math.prod(shape),'fp32_bytes':4*math.prod(shape)})
free=shutil.disk_usage(a.model);out={'scope':'header-only; no tensor payload read/conversion','shards':headers,'derived_tensors':rows,'dense_bf16_bytes':sum(x['bf16_bytes']for x in rows),'dense_fp32_bytes':sum(x['fp32_bytes']for x in rows),'filesystem':{'total_bytes':free.total,'used_bytes':free.used,'free_bytes':free.free},'limits':'Counts all retained tensor names inclMTP; target omission/preserve policy must be reviewed, disk availability is instantaneous not a reservation'};a.output.write_text(json.dumps(out,indent=2)+'\n')
