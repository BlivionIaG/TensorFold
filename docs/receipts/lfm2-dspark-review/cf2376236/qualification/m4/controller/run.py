import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent
b=json.loads((p/'binding.json').read_text());root=Path(b['source_root'])
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==b['source_revision']
proof=json.loads(Path(b['own_head_proof']).read_text());assert proof['source_revision']==b['source_revision'] and proof['base_revision']==b['base_revision']
assert all(r['returncode']==0 for r in proof['records'])
for name,h in b['driver_sha256'].items():assert sha(Path(b['frozen_driver'])/name)==h
for kind in ('target','drafter'):
    for name,h in b['model_sha256'][kind].items():assert sha(Path(b['remote_'+kind])/name)==h
sys.path.insert(0,b['frozen_driver']);import bench_lfm2_dspark as gate
env,_=gate.clean_env();binary=root/'zig-out/native/bin/tensorfold-native';b['binary_sha256']=sha(binary)
records=[]
def run(label,cmd,deadline):
    log=p/(label+'.log');start=time.monotonic()
    with log.open('xb') as out:r=subprocess.run(cmd,cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=deadline)
    row={'label':label,'command':cmd,'returncode':r.returncode,'seconds':time.monotonic()-start,'log_sha256':sha(log)}
    records.append(row);(p/'observed.json').write_text(json.dumps({'binding':b,'records':records},indent=2)+'\n');print(json.dumps(row),flush=True)
    if r.returncode not in (0,3):print(log.read_text()[-8000:],flush=True);raise SystemExit(r.returncode)
    return r.returncode
with Path(b['shared_lock']).open('r+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert os.fstat(lock.fileno()).st_ino==68270044
    assert gate.pressure_level()==1 and not gate.competing_processes('tensorfold-native')
    assert run('norm-tap-contract',[str(root/'zig-out/bin/tf-kernel-contract'),'--only','metal lfm2 norm tap'],1200)==0
    assert run('help',[str(binary),'--help'],120)==0
cmd=[sys.executable,str(Path(b['frozen_driver'])/'bench_lfm2_dspark.py'),'run','--binary',str(binary),'--target',b['remote_target'],'--drafter',b['remote_drafter'],'--source-revision',b['source_revision'],'--target-revision',b['target_revision'],'--drafter-revision',b['drafter_revision'],'--output-dir',str(p/'1394-full-gate')]
code=run('1394-full-gate',cmd,14400)
assert sha(binary)==b['binary_sha256']
summary=p/'1394-full-gate/summary.json'
if summary.exists():print(summary.read_text(),flush=True)
raise SystemExit(code)
