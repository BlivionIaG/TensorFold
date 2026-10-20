from pathlib import Path
import subprocess,json,hashlib,os,time,sys
S=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
keys=['TENSORFOLD_MEMORY_RESERVE_GIB','TENSORFOLD_HIP_MEMORY_LIMIT_GB']
a=json.loads((S/'parent_allocation.json').read_text());assert {k:os.environ.get(k) for k in keys}==a['memory_admission_environment']
assert sha(S/'observe_memory_env.py')==a['memory_observer_sha256']
assert not (S/'launch_result.json').exists()
with (S/'coordinator.out').open('wb') as out,(S/'observer.out').open('wb') as obs:
 p=subprocess.Popen([sys.executable,str(S/'run.py'),'--execute'],stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 q=subprocess.Popen([sys.executable,str(S/'observe_memory_env.py'),'--coordinator-pid',str(p.pid),'--allocation',str(S/'parent_allocation.json'),'--output',str(S/'memory_admission_observation.json')],stdout=obs,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 (S/'launched.json').write_text(json.dumps({'coordinator_pid':p.pid,'observer_pid':q.pid,'launcher_pid':os.getpid(),'allocation_sha256':sha(S/'parent_allocation.json'),'observer_sha256':sha(S/'observe_memory_env.py'),'launcher_sha256':sha(Path(__file__)),'memory_admission_environment':a['memory_admission_environment'],'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2)+'\n')
 rc=p.wait();orc=q.wait(timeout=10)
 r={'coordinator_exit_status':rc,'observer_exit_status':orc,'complete':rc==0 and orc==0,'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 runs=list((S/'runs').glob('*'))
 if len(runs)==1 and (S/'memory_admission_observation.json').is_file():
  (runs[0]/'observation.json').write_bytes((S/'memory_admission_observation.json').read_bytes());r['run_directory']=str(runs[0])
 for n in ['launch_serial729.py','launched.json','coordinator.out','observer.out']:
  if len(runs)==1:(runs[0]/n).write_bytes((S/n).read_bytes())
 (S/'launch_result.json').write_text(json.dumps(r,indent=2)+'\n')
 if len(runs)==1:(runs[0]/'launch_result.json').write_bytes((S/'launch_result.json').read_bytes())
 print(json.dumps(r),flush=True);sys.exit(rc or orc)
