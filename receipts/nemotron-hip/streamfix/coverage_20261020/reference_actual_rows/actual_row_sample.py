"""Read only headers and at most 3rows of 7 representative affine tensors; no torch/model/GPU."""
from pathlib import Path
import sys,json,struct,hashlib,platform,math
import numpy as np
M=Path.home()/'models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit';expected=json.loads(sys.stdin.read());sha=lambda b:hashlib.sha256(b).hexdigest();assert sha((M/'config.json').read_bytes())==expected['config_sha256'];revision=(M/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0];assert revision==expected['revision'];headers={};loc={};read_bytes=0
for item in expected['shards']:
 p=M/item['file']
 with p.open('rb')as f:n=struct.unpack('<Q',f.read(8))[0];assert n<64*1024*1024;raw=f.read(n)
 assert sha(raw)==item['header_sha256'];h=json.loads(raw);headers[p.name]=(p,n+8,h);read_bytes+=8+n
 for k,v in h.items():
  if k!='__metadata__':assert k not in loc;loc[k]=(p.name,v)
quant=[k for k,(f,v)in loc.items()if v['dtype']=='U32'and k.endswith('.weight')and k[:-7]+'.scales'in loc];selectors={'embedding':lambda k:'embeddings.weight'in k,'lm_head':lambda k:'lm_head.weight'in k,'mamba':lambda k:'.mixer.in_proj.weight'in k,'shared_mlp':lambda k:'shared' in k and k.endswith('.weight'),'routed_3d':lambda k:len(loc[k][1]['shape'])==3,'mtp':lambda k:loc[k][0]=='mtp-4bit.safetensors'};selected={}
for category,pred in selectors.items():
 names=sorted(k for k in quant if pred(k));assert names,category;selected[category]=names[0]
reads=[]
def slice_tensor(k,row):
 global read_bytes
 file,v=loc[k];p,start,h=headers[file];shape=v['shape'];width=shape[-1];size={'U32':4,'BF16':2}[v['dtype']];offset=start+v['data_offsets'][0]+row*width*size;length=width*size;assert offset+length<=start+v['data_offsets'][1]
 with p.open('rb')as f:f.seek(offset);raw=f.read(length)
 assert len(raw)==length;read_bytes+=length;reads.append({'tensor':k,'file':file,'flattened_row':row,'byte_offset':offset,'bytes':length,'sha256':sha(raw)});return raw
samples=[]
for category,k in selected.items():
 shape=loc[k][1]['shape'];rowcount=math.prod(shape[:-1]);kw=shape[-1];stem=k[:-7]
 for row in sorted({0,rowcount//2,rowcount-1}):
  wr=slice_tensor(k,row);sr=slice_tensor(stem+'.scales',row);br=slice_tensor(stem+'.biases',row);w=np.frombuffer(wr,dtype='<u4');s=(np.frombuffer(sr,dtype='<u2').astype(np.uint32)<<16).view(np.float32);b=(np.frombuffer(br,dtype='<u2').astype(np.uint32)<<16).view(np.float32);assert len(s)==len(b)==kw*8//64
  q=((w[:,None]>>(np.arange(8,dtype=np.uint32)*4))&15).reshape(-1).astype(np.float32);v=np.add(np.multiply(q,np.repeat(s,64),dtype=np.float32),np.repeat(b,64),dtype=np.float32)
  f32=lambda n:struct.unpack('<f',struct.pack('<f',n))[0];scalar=np.array([f32(f32(((struct.unpack_from('<I',wr,(i//8)*4)[0]>>(4*(i%8)))&15)*float(s[i//64]))+float(b[i//64]))for i in range(len(q))],dtype=np.float32);assert np.array_equal(v.view(np.uint32),scalar.view(np.uint32));assert np.isfinite(v).all()
  bits=v.view(np.uint32);bf=((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype('<u2');back=(bf.astype(np.uint32)<<16).view(np.float32)
  with np.errstate(over='ignore'):half=v.astype('<f2');halfback=half.astype(np.float32)
  errbf=np.abs(v.astype(np.float64)-back.astype(np.float64));errhalf=np.abs(v.astype(np.float64)-halfback.astype(np.float64));samples.append({'category':category,'tensor':k,'shape':shape,'flattened_row':row,'values':len(v),'vector_scalar_fp32_bits_equal':True,'reconstructed_fp32_sha256':sha(v.astype('<f4').tobytes()),'derived_bf16_sha256':sha(bf.tobytes()),'derived_fp16_sha256':sha(half.tobytes()),'bf16_max_abs_rounding_error':float(errbf.max()),'fp16_nonfinite_after_rounding':int((~np.isfinite(halfback)).sum()),'fp16_max_abs_rounding_error_finite':float(errhalf[np.isfinite(errhalf)].max())if np.isfinite(errhalf).any()else None})
assert read_bytes<8*1024*1024
print(json.dumps({'validated':True,'scope':'Actual bounded CPU checkpoint row reads; no full conversion, model load, torch import or GPU use','checkpoint_revision':revision,'checkpoint_config_sha256':expected['config_sha256'],'versions':{'python':platform.python_version(),'numpy':np.__version__},'formula':'Little-endian low-nibble-first4bit; FP32code*FP32BF16scale then separateFP32addBF16bias; group64','rounding_scope':'Derived BF16RNE/FP16 candidate storage precision only; not native or standard-server inference equivalence','total_bytes_read_including_headers':read_bytes,'tensor_payload_bytes_read':sum(r['bytes']for r in reads),'header_bindings':expected['shards'],'reads':reads,'samples':samples},indent=2,allow_nan=False))
