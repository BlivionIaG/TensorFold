import argparse,fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--binding',required=True);p.add_argument('--output',required=True);p.add_argument('--bits',type=int,choices=(0,8),required=True);a=p.parse_args()
root=Path(a.root);out=Path(a.output);out.mkdir(parents=True)
b=json.loads(Path(a.binding).read_text());sys.path.insert(0,b['frozen_driver']);import bench_lfm2_dspark as gate
for name,digest in b['driver_sha256'].items():assert gate.sha256_file(Path(b['frozen_driver'])/name)==digest
env,removed=gate.clean_env();os.environ.clear();os.environ.update(env)
settings={'TF_QUAL_OUT':str(out),'TF_QUAL_MODEL':b['remote_target'],'TF_QUAL_DRAFTER':b['remote_drafter'],'TF_QUAL_CANDIDATE':str(root/'zig-out/native/bin/tensorfold-native')}
os.environ.update(settings)
spec=importlib.util.spec_from_file_location('original_functional',root/'docs/receipts/lfm2-dspark-review/qualification.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
for name in settings:os.environ.pop(name,None)
Base=q.Server;active=None
class CheckedServer(Base):
 def check(self):
  try:return gate.Server.check(self)
  except gate.Fatal as error:raise SystemExit(str(error))
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  if a.bits==8:self.mode='8'
 def __enter__(self):
  global active
  if gate.pressure_level() not in (None,1):raise SystemExit('OS memory pressure before functional server')
  if gate.competing_processes(Path(self.binary).name):raise SystemExit('competing model process')
  self.log_path=out/(self.label+'.server.log')
  try:
   value=super().__enter__();active=self;self.check();return value
  except BaseException:
   if hasattr(self,'proc'):
    try:super().__exit__(None,None,None)
    finally:active=None
   raise
 def __exit__(self,*args):
  global active
  try:
   if self.proc.poll() is None:self.check()
  finally:
   try:super().__exit__(*args)
   finally:active=None
q.Server=CheckedServer
base_post=q.post
def checked_post(*args,**kwargs):
 result=base_post(*args,**kwargs)
 if active is not None:active.check()
 return result
q.post=checked_post
start=time.monotonic();result={'label':'ORIGINAL_CACHE_CANCEL_SAMPLE_FUNCTIONAL_ONLY','source_revision':b['source_revision'],'bits':a.bits,'activation_dtype':'BF16','native_binary_sha256':gate.sha256_file(root/'zig-out/native/bin/tensorfold-native'),'removed_environment':removed}
with open(b['shared_lock'],'r+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert os.fstat(lock.fileno()).st_ino==b['shared_lock_inode_observed']
 try:
  q.functional();result['complete']=True
 except BaseException as error:
  result.update({'complete':False,'error_type':type(error).__name__,'error':str(error)});raise
 finally:
  result['elapsed_s']=time.monotonic()-start;(out/'observed.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
