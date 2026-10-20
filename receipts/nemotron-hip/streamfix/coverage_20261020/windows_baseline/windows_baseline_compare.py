from pathlib import Path
import subprocess,json,hashlib,time,os,datetime
W=Path('<LOCAL_TEMP_PATH>');D=Path('<LOCAL_TEMP_PATH>');D.mkdir(exist_ok=False)
Z=Path('<LOCAL_HOME>/.local/zig-0.17.0/zig');refs={'main_d101':'d1018b318fb51981c4a18f12e40269a4d3ee9713','pr1082_dc':'dc595eeb198c66d3327d7f0ce4a3eb6c9a807ec5'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();R={'scope':'LocalMac Windows-target compilation only; no PE execution, no GPU, no source edits','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'compiler':{'path':str(Z),'sha256':sha(Z),'version':subprocess.check_output([str(Z),'version'],text=True).strip()},'driver_sha256':sha(Path(__file__)),'runs':[],'complete':False}
def save():(D/'result.json').write_text(json.dumps(R,indent=2)+'\n')
save()
for name,head in refs.items():
 T=D/name;subprocess.run(['git','-C',str(W),'worktree','add','--detach',str(T),head],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);assert subprocess.check_output(['git','-C',str(T),'rev-parse','HEAD'],text=True).strip()==head;assert not subprocess.check_output(['git','-C',str(T),'status','--porcelain','--untracked-files=no'],text=True).strip()
 A=[str(Z),'test','--test-no-exec'];target=['-Odebug','-target','x86_64-windows-gnu']
 A+=target+['--dep','sys','--dep','checkpoint_config','--dep','save_guard','-Mroot=zig/src/core/checkpoint_identity.zig']
 A+=target+['--dep','sys','--dep','checkpoint_config','--dep','save_guard','-Msys=zig/src/platform/sys.zig']
 A+=target+['--dep','checkpoint_config','--dep','save_guard','-Mcheckpoint_config=zig/src/core/checkpoint_config.zig']
 A+=target+['--dep','save_guard','-Msave_guard=zig/src/core/save_guard.zig','-lc','--cache-dir',str(D/(name+'_cache')),'--global-cache-dir',str(D/(name+'_global')),'--build-root',str(T),'--name','checkpoint_identity_windows','-femit-bin='+str(D/(name+'.exe')),'--zig-lib-dir',str(Z.parent/'lib')]
 row={'name':name,'source_commit':head,'tree':subprocess.check_output(['git','-C',str(T),'rev-parse','HEAD^{tree}'],text=True).strip(),'worktree':str(T),'argv':A,'source_files':{p:sha(T/p)for p in ['zig/src/core/checkpoint_identity.zig','zig/src/core/digest_key.zig','zig/src/platform/sys.zig','zig/src/core/checkpoint_config.zig','zig/src/core/save_guard.zig']},'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};R['runs'].append(row);log=D/(name+'.log')
 with log.open('wb')as f:
  p=subprocess.Popen(A,cwd=T,stdout=f,stderr=subprocess.STDOUT);row['pid']=p.pid;save()
  try:row['actual_exit_status']=p.wait(timeout=600)
  except subprocess.TimeoutExpired:p.kill();row['actual_exit_status']=p.wait();row['timeout']=True
 row['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();row['log_sha256']=sha(log);row['log']=str(log);row['clean_after']=not subprocess.check_output(['git','-C',str(T),'status','--porcelain','--untracked-files=no'],text=True).strip();save()
R['complete']=True;R['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print(D)
