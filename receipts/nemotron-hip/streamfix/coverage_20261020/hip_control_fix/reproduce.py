"""Offline CPU-only synthetic proof replay, fresh work directory required."""
from pathlib import Path
import argparse,subprocess,shutil,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--zig',required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();S=Path(__file__).resolve().parent;W=a.work.resolve();assert W!=S and S not in W.parents;W.mkdir(exist_ok=False);shutil.copytree(S/'proof',W/'proof');D=W/'proof';assert subprocess.check_output([a.zig,'version']).decode().strip()=='0.17.0';rows=[]
def run(label,cmd,expected):
 r=subprocess.run(cmd,capture_output=True);log=W/(label+'.log');log.write_bytes(r.stdout+r.stderr);rows.append({'label':label,'command':cmd,'exit_status':r.returncode,'expected_exit_status':expected,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()});(W/'result.json').write_text(json.dumps(rows,indent=2)+'\n');assert r.returncode==expected,label
for phase in ['before','after']:
 for case in ['normal','partial_green','missing_usage','stale_control']:
  q=D/'probes'/phase/case;f=q/'probe.zig';f.write_text(f.read_text().replace('<PROOF_ROOT>',str(D)));cmd=[a.zig,'test','--dep','core','--dep','hip','--dep','hip_controls','-Mroot='+str(f),'-Mcore='+str(q/'core.zig'),'-Mhip='+str(q/'hip.zig'),'-Mhip_controls='+str(q/'controls.zig'),'--cache-dir',str(W/'cache'),'--global-cache-dir',str(W/'global')];run(phase+'_'+case,cmd,0 if phase=='after' or case=='normal' else 1)
run('before_regression',[a.zig,'test',str(D/'before/regression.zig'),'--cache-dir',str(W/'cache'),'--global-cache-dir',str(W/'global')],1)
run('fixed_records',[a.zig,'test',str(D/'records_emitter.zig'),'--cache-dir',str(W/'cache'),'--global-cache-dir',str(W/'global')],0)
print('CPU proof replay passed; no GPU/model execution')
