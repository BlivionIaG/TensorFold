#!/usr/bin/env python3
"""Run on approved host after root review. Dry by default; CPU converter only."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,datetime,time
H=Path.home(); BASE=H/'hcx/streamfix/coverage_20261020/derived_hf_header_plan_890ab464_attempt1'
SOURCE=H/'models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit'
PYTHON=H/'Apps/vllm/venv-7.14.0_0.31.0-final/bin/python'
SITE=H/'Apps/vllm/venv-7.14.0_0.31.0-final/lib/python3.12/site-packages'
TOOL='890ab464a8aa1efc8953c5eac335d8a50a6178fae537013598ae81bd0e7ac31b';PLAN='43e5475f147fe952565a9f90c539f1cb8d4eb6ee311d883ddb020af59947e3d1'
ENV_KEYS=['TENSORFOLD_MEMORY_RESERVE_GIB','TENSORFOLD_HIP_MEMORY_LIMIT_GB']
sha=lambda b:hashlib.sha256(b).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def source_snapshot(plan):
 out={}
 for row in plan['headers']:
  p=SOURCE/row['file'];st=p.stat()
  with p.open('rb')as f:
   n=int.from_bytes(f.read(8),'little');header=f.read(n)
  assert n==row['header_bytes']and sha(header)==row['header_sha256']
  out[row['file']]={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'header_sha256':sha(header)}
 for name in ['config.json','.cache/huggingface/download/config.json.metadata']:
  p=SOURCE/name;out[name]={'sha256':sha(p.read_bytes()),'stat':list(p.stat())}
 return out
WRAPPER="""import sys,runpy,json,hashlib,platform
from pathlib import Path
sys.path.insert(0,sys.argv[1]);script=Path(sys.argv[2]);runtime=Path(sys.argv[3]);sys.argv=[str(script)]+sys.argv[4:]
try:runpy.run_path(str(script),run_name='__main__')
finally:
 import numpy
 forbidden=sorted(k for k in sys.modules if k=='torch' or k.startswith('torch.') or k=='vllm' or k.startswith('vllm.') or k=='transformers' or k.startswith('transformers.'))
 runtime.write_text(json.dumps({'python':platform.python_version(),'executable':sys.executable,'executable_sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),'numpy':numpy.__version__,'numpy_file':numpy.__file__,'forbidden_modules_loaded':forbidden,'isolated':sys.flags.isolated,'no_site':sys.flags.no_site},indent=2)+'\\n')
 assert not forbidden
"""
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--boot-id',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--receipt-dir',type=Path,required=True);a=ap.parse_args()
 assert sha((BASE/'convert.py').read_bytes())==TOOL and sha((BASE/'plan.json').read_bytes())==PLAN
 plan=json.loads((BASE/'plan.json').read_text());assert plan['total_tensor_bytes']==65826538624
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert boot==a.boot_id
 env={k:os.environ.get(k)for k in ENV_KEYS};assert env==dict.fromkeys(ENV_KEYS)
 output=a.output.resolve();receipt=a.receipt_dir.resolve();assert not output.exists()and not receipt.exists();assert SOURCE.resolve()not in output.parents and SOURCE.resolve()not in receipt.parents and output!=SOURCE.resolve();assert output.parent.is_dir()and receipt.parent.is_dir()
 mem={k:int(v.split()[0])*1024 for k,v in (l.split(':',1)for l in Path('/proc/meminfo').read_text().splitlines())};stat=os.statvfs(output.parent);free=stat.f_bavail*stat.f_frsize;assert free>=plan['total_tensor_bytes']+2*1024**3 and mem['MemAvailable']>=8*1024**3
 before=source_snapshot(plan);cmd=[str(PYTHON),'-I','-S','-c',WRAPPER,str(SITE),str(BASE/'convert.py'),str(receipt/'process_runtime.json'),str(SOURCE),'--plan',str(BASE/'plan.json'),'--approved-plan-sha',PLAN,'--output',str(output),'--execute']
 if not a.execute:print(json.dumps({'dry_run':True,'command':cmd,'boot':boot,'disk_available':free,'MemAvailable':mem['MemAvailable'],'memory_environment':env},indent=2));return
 receipt.mkdir();atomic(receipt/'intent.json',{'started_utc':now(),'boot':boot,'converter_sha256':TOOL,'plan_sha256':PLAN,'command':cmd,'disk_available':free,'MemAvailable':mem['MemAvailable'],'loadavg':Path('/proc/loadavg').read_text(),'memory_environment':env,'source_before':before,'scope':'CPU-onlyconversion; no stockloader/model/GPU; no inferenceequivalence'})
 started=time.monotonic()
 with (receipt/'stdout.log').open('xb')as out,(receipt/'stderr.log').open('xb')as err:
  proc=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True);atomic(receipt/'process.json',{'pid':proc.pid,'pgid':proc.pid});code=proc.wait()
 after=source_snapshot(plan);runtime=json.loads((receipt/'process_runtime.json').read_text())if(receipt/'process_runtime.json').exists()else None
 result={'exit_status':code,'finished_utc':now(),'elapsed_seconds':time.monotonic()-started,'boot_end':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'source_after':after,'original_metadata_unchanged':before==after,'source_scope':'Config/revision+fiveheaders/stat identities, not fulloriginalpayloadrehash','runtime':runtime,'stdout_sha256':sha((receipt/'stdout.log').read_bytes()),'stderr_sha256':sha((receipt/'stderr.log').read_bytes()),'conversion_manifest_sha256':sha((output/'conversion_manifest.json').read_bytes())if(output/'conversion_manifest.json').exists()else None};atomic(receipt/'result.json',result)
 assert code==0 and before==after and result['boot_end']==boot and runtime and not runtime['forbidden_modules_loaded']and result['conversion_manifest_sha256']
 print(json.dumps({'conversion_completed':True,'result':str(receipt/'result.json')}))
if __name__=='__main__':main()
