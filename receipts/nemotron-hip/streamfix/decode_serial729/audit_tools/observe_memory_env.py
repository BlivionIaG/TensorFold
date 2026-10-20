"""Observe only two admission-environment keys from an authorized coordinator/direct native children. No GPU actions."""
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--coordinator-pid',type=int,required=True);p.add_argument('--allocation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-seconds',type=int,default=10800);a=p.parse_args();KEYS={'TENSORFOLD_MEMORY_RESERVE_GIB','TENSORFOLD_HIP_MEMORY_LIMIT_GB'}
allocation=json.loads(a.allocation.read_text());expected=allocation['memory_admission_environment'];assert set(expected)==KEYS and all(v is None or isinstance(v,str) for v in expected.values())
assert allocation['source_commit']=='729833615595df66c9f9a438ce9a85061843f010' and allocation['single_model_scheduling'] and allocation['exclusive_owned_gpu_reservation']
def sample(pid):
 # Discard all other environment keys immediately; never serialize or print them.
 data={}
 for item in (Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0'):
  key,sep,value=item.partition(b'=');name=key.decode(errors='replace')
  if sep and name in KEYS:data[name]=value.decode(errors='replace')
 env={k:data.get(k) for k in sorted(KEYS)}
 return {'pid':pid,'environment':env,'matches_allocation':env==expected,'observed_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
r={'schema_version':1,'source_context':allocation['source_commit'],'allocation_snapshot':expected,'coordinator':sample(a.coordinator_pid),'ranks':[],'observer_complete':False,'rank_sample_scope':'Only actually observed direct tensorfold-native child PIDs; absent samples are not invented. No global GPU-process or physical-host isolation claim.','errors':[]};seen=set();started=time.monotonic()
def save():
 a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(a.output.suffix+'.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(a.output)
save()
while time.monotonic()-started<a.max_seconds:
 coordinator=Path('/proc')/str(a.coordinator_pid)
 try:
  stat=(coordinator/'stat').read_text();status=stat[stat.rfind(')')+2:].split()[0]
  if status=='Z':r['observer_complete']=True;r['stop_reason']='coordinator exited (zombie awaiting reap)';break
  children=(coordinator/'task'/str(a.coordinator_pid)/'children').read_text().split()
 except (FileNotFoundError,ProcessLookupError):r['observer_complete']=True;r['stop_reason']='coordinator exited';break
 for pid_text in children:
  pid=int(pid_text)
  if pid in seen:continue
  try:
   exe=(Path('/proc')/pid_text/'exe').resolve()
   if exe.name!='tensorfold-native':continue
   row=sample(pid);row['parent_pid']=a.coordinator_pid;row['executable_name']=exe.name;r['ranks'].append(row);seen.add(pid);save()
  except (FileNotFoundError,ProcessLookupError):continue
  except PermissionError:r['errors'].append({'pid':pid,'error':'PermissionError reading authorized child metadata'});save()
 time.sleep(.2)
else:r['stop_reason']='observer deadline; samples incomplete'
save();assert r['coordinator']['matches_allocation'] and all(x['matches_allocation'] for x in r['ranks']) and not r['errors'],'Admission environment mismatch or observation error'
print('Memory environment observation saved; actual rank samples',len(r['ranks']),'complete',r['observer_complete'])
