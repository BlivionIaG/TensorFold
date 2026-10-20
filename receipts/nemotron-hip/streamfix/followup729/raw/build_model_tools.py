"""Build source-bound model tools using the existing host build lock; no GPU runs."""
import hashlib,json,os,subprocess
from pathlib import Path
H=Path.home(); out=H/'hcx/streamfix/optimization/followup729';out.mkdir(parents=True,exist_ok=True)
allocation=json.loads((out/'allocation.json').read_text());assert allocation['source_commit']=='729833615595df66c9f9a438ce9a85061843f010' and allocation['parent_allocated'] and allocation['recovery_ab_finished'] and allocation['purpose']=='followup_models'
heads={'sfopt':'729833615595df66c9f9a438ce9a85061843f010'}
for tree,expected in heads.items():
 repo=H/'nh'/tree
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==expected
 subprocess.run(['git','-C',str(repo),'diff','--exit-code','--quiet'],check=True)
 subprocess.run(['git','-C',str(repo),'diff','--cached','--exit-code','--quiet'],check=True)
 result={'source_commit':expected,'tree':tree,'architectures':{}}
 for arch in ['gfx1100','gfx1030']:
  prefix=repo/('out-'+arch)
  targets=['tf-nemotron-hip-test']+(['tf-qwen35-test','tf-flashnext-hip-test','hip-kernel-build'] if tree=='sfopt' else [])
  args=[str(H/'hcx/zb.sh'),tree,f'sf_model_build_{tree}_{arch}.log',*targets,'-Dhip-arch='+arch,'-Doptimize=ReleaseFast','--prefix',str(prefix)]
  status=subprocess.run(args).returncode
  record={'command':args,'exit_status':status,'binaries':{}}
  for name,relative in [('tf-nemotron-hip-test','bin/tf-nemotron-hip-test')]+([('tf-qwen35-test','bin/tf-qwen35-test'),('tf-flashnext-hip-test','bin/tf-flashnext-hip-test')] if tree=='sfopt' else []):
   p=prefix/relative
   if status==0:
    assert p.exists() and p.stat().st_size>0 and os.access(p,os.X_OK),p
    record['binaries'][name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  result['architectures'][arch]=record
  (out/f'build_{tree}_progress.json').write_text(json.dumps(result,indent=2)+'\n')
  if status:raise SystemExit(status)
 name='model_tools_built.json' if tree=='sfopt' else 'baseline_model_tools_built.json'
 (out/name).write_text(json.dumps(result,indent=2)+'\n')
 print(tree+' BOTH_ARCH_BUILT',flush=True)
print('ALL_MODEL_BUILDS_DONE',flush=True)
