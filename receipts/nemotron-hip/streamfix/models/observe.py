#!/usr/bin/env python3
"""Observe actual loaded file maps only for descendants of the measured driver."""
import hashlib,json,os,sys,time
from pathlib import Path
root=int(sys.argv[1]);d=Path.home()/'hcx/streamfix/final_models';records={};hashes={}
def digest(p):
 st=p.stat();key=(str(p),st.st_size,st.st_mtime_ns)
 if key not in hashes:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for x in iter(lambda:f.read(4<<20),b''):h.update(x)
  hashes[key]=h.hexdigest()
 return hashes[key]
while Path('/proc/'+str(root)).exists():
 parents={}
 for q in Path('/proc').iterdir():
  if not q.name.isdigit():continue
  try:parents[int(q.name)]=int((q/'stat').read_text().split(') ',1)[1].split()[1])
  except (OSError,ValueError,IndexError):pass
 def descendant(pid):
  seen=set()
  while pid not in seen and pid in parents:
   if pid==root:return True
   seen.add(pid);pid=parents[pid]
  return pid==root
 for pid in parents:
  if not descendant(pid):continue
  q=Path('/proc')/str(pid)
  try:
   exe=Path(os.readlink(q/'exe'))
   if not (exe.name.startswith(('tf-qwen35-test','tf-flashnext-hip-test')) or exe.name=='test'):continue
   cmd=(q/'cmdline').read_bytes().split(b'\0');cmd=[x.decode(errors='replace') for x in cmd if x]
   maps=(q/'maps').read_text();paths=set()
   for line in maps.splitlines():
    fields=line.split(maxsplit=5)
    if len(fields)==6 and fields[5].startswith('/') and not fields[5].endswith(' (deleted)'):
     if '.so' in Path(fields[5]).name or fields[5]==str(exe):paths.add(fields[5])
   key=str(pid)
   (d/('maps_'+key+'.txt')).write_text(maps)
   r=records.setdefault(key,{'pid':pid,'command':cmd,'executable':str(exe),'executable_sha256':digest(exe),'stage':json.loads((d/'summary.json').read_text())['state'],'first_observed_unix':time.time(),'mapped_files':{}})
   r['last_observed_unix']=time.time();r['maps_sha256']=hashlib.sha256(maps.encode()).hexdigest()
   for path in paths:
    if path not in r['mapped_files']:
     p=Path(path)
     if p.is_file():r['mapped_files'][path]={'sha256':digest(p),'size_bytes':p.stat().st_size}
  except (OSError,ValueError,KeyError):pass
 (d/'loaded_maps.json').write_text(json.dumps({'driver_pid':root,'observation':'actual /proc/PID/maps samples; union per process','processes':list(records.values())},indent=2)+'\n')
 time.sleep(.1)
