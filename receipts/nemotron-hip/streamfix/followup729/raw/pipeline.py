#!/usr/bin/env python3
"""Fresh source-bound followup orchestration; requires explicit post-ownhead allocation."""
import argparse,hashlib,json,os,re,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser(__doc__);p.add_argument('--allocation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
S=Path(__file__).resolve().parent;H=Path.home();HEAD='729833615595df66c9f9a438ce9a85061843f010';OLD='8649e9858b38e9ef3dcd8d579879c6512a363050';M='1a5a823c58f43aaa3f800aa4a57f9117c757ed4fe30eeaf0812f6f9558cfa83c';D=a.output.resolve();created=False;active=[]
R={'schema_version':1,'source_commit':HEAD,'state':'starting','stages':[]}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(): (D/'pipeline_result.json').write_text(json.dumps(R,indent=2)+'\n')
def guard():
 assert a.allocation.resolve()==S/'allocation.json','child scripts must consume the identical allocation'
 g=json.loads(a.allocation.read_text());assert g['source_commit']==HEAD and g['parent_allocated'] and g['exclusive_gpu_reservation'] and g['recovery_ab_finished'] and g['purpose']=='followup_models'
 assert g['devices']=={'gfx1100':'0,1','gfx1030':'2,3,4,5'}
 assert subprocess.check_output(['git','-C',str(H/'nh/sfopt'),'rev-parse','HEAD'],text=True).strip()==HEAD
 assert not subprocess.check_output(['git','-C',str(H/'nh/sfopt'),'status','--porcelain','--untracked-files=no'],text=True).strip()
 assert sha(S/'recipe_manifest.json')==M
 m=json.loads((S/'recipe_manifest.json').read_text())
 for n in ('driver.py','observe.py','build_model_tools.py','nem_compare.py','model_gate.py','capture_identity.py'):assert sha(S/n)==m[n]

def spawn(name,cmd):
 guard();log=D/(name+'.log');f=log.open('w');proc=subprocess.Popen(cmd,cwd=S,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);active.append((proc,f))
 return {'name':name,'command':cmd,'source_commit':HEAD,'pid':proc.pid,'log':log.name,'start':time.monotonic()},proc,f

def finish(row,proc,f,timeout,require_zero=True):
 try:code=proc.wait(timeout=timeout)
 except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124
 f.close();active.remove((proc,f));row.update(exit_status=code,elapsed_seconds=round(time.monotonic()-row.pop('start'),3),log_sha256=sha(D/row['log']));R['stages'].append(row);save();guard()
 if require_zero:assert code==0,(row['name'],code)
 return row

def run(name,cmd,timeout=14400):
 row,proc,f=spawn(name,cmd);return finish(row,proc,f,timeout)
try:
 guard();assert not any((S/n).exists() for n in ('summary.json','MODEL_CHECKS_ALL_PASS','loaded_maps.json','model_tools_built.json')),'fresh child outputs required'
 assert not (S/'logs').exists(),'fresh Nemotron output directory required'
 D.mkdir(parents=True,exist_ok=False);created=True;R.update(allocation_sha256=sha(a.allocation),recipe_manifest_sha256=M,pipeline_sha256=sha(Path(__file__)));save()
 run('build_tools',['python3',str(S/'build_model_tools.py')])
 builds=json.loads((S/'model_tools_built.json').read_text());assert builds['source_commit']==HEAD and set(builds['architectures'])=={'gfx1100','gfx1030'}
 for v in builds['architectures'].values():
  assert v['exit_status']==0
  for b in v['binaries'].values():assert sha(b['path'])==b['sha256']
 R['build_marker_sha256']=sha(S/'model_tools_built.json');save()
 row,proc,f=spawn('shared_models',['python3',str(S/'driver.py')])
 obs,op,of=spawn('loaded_maps_observer',['python3',str(S/'observe.py'),str(proc.pid)])
 finish(row,proc,f,21600,False);finish(obs,op,of,120,False)
 assert row['exit_status']==0 and obs['exit_status']==0,'model driver or observer failed'
 marker=S/'MODEL_CHECKS_ALL_PASS';assert marker.read_text().strip()==HEAD
 models=json.loads((S/'summary.json').read_text());assert models['source_commit']==HEAD and models['state']=='complete' and models['result']=='PASS'
 assert (S/'loaded_maps.json').is_file() and json.loads((S/'loaded_maps.json').read_text())['processes']
 R['models']={'summary_sha256':sha(S/'summary.json'),'marker_sha256':sha(marker),'loaded_maps_sha256':sha(S/'loaded_maps.json')};save()
 for rig in ('w7800tp2','v620tp1'):
  row=run('nem_'+rig,['python3',str(S/'nem_compare.py'),rig],10800)
  matches=re.findall(r'^NEM_DONE (.+)$',(D/row['log']).read_text(),re.M);assert len(matches)==1
  out=Path(matches[0]);assert out.is_relative_to(S/'logs')
  summary=json.loads((out/'summary.json').read_text());assert summary['rig']==rig and summary['done'] and len(summary['runs'])==8 and len(summary['comparisons'])==4
  assert {x['source_commit'] for x in summary['runs']}=={OLD,HEAD}
  assert all(x['finite'] and all(q['exit_status']==0 for q in x['ranks']) for x in summary['runs']) and all(x['identical'] for x in summary['comparisons'])
  R.setdefault('nemotron',[]).append({'rig':rig,'summary':str(out/'summary.json'),'summary_sha256':sha(out/'summary.json')});save()
 guard();R.update(state='complete',validated=True);save();(D/'FOLLOWUP_ALL_PASS').write_text(HEAD+'\n');print('FOLLOWUP_ALL_PASS '+HEAD,flush=True)
except BaseException as exc:
 for proc,f in active:
  if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  f.close()
 if created:R.update(state='failed',validated=False,error=str(exc));save()
 raise
