"""Capture actual loaded HIP/RCCL files for this hold's process groups, plus checkpoint metadata."""
import hashlib,json,os,sys
from pathlib import Path
out=Path(sys.argv[1]);groups={int(x) for x in sys.argv[2:]};records=[];seen=set()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  if os.getpgid(int(d.name)) not in groups:continue
  libs={}
  for line in (d/'maps').read_text().splitlines():
   fields=line.split();p=Path(fields[-1])
   if fields[-1].startswith('/') and any(x in p.name for x in ['librccl','libamdhip64','libhsa-runtime']):
    libs[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
  if libs:seen.add(os.getpgid(int(d.name)));records.append({'pid':int(d.name),'executable':{'path':str((d/'exe').resolve()),'sha256':sha(d/'exe')},'libraries':libs})
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert records and seen==groups,'Missing loaded HIP/RCCL mappings for one or more rank process groups'
if len(groups)>1:
 assert all(any('librccl' in name for name in rec['libraries']) for rec in records),'TP rank has no loaded RCCL mapping yet'
model=Path.home()/'models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit'
metadata={p.name:p.read_text() for p in (model/'.cache/huggingface/download').glob('*.metadata')}
out.write_text(json.dumps({'processes':records,'checkpoint_metadata':metadata},indent=2)+'\n')
