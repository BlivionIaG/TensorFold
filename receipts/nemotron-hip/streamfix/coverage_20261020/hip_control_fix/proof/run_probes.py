from pathlib import Path
import subprocess,json,hashlib,time
D=Path(__file__).parent;rows=[]
for phase in ['before','after']:
 for case in ['normal','partial_green','missing_usage','stale_control']:
  p=D/'probes'/phase/case;cmd=['<ZIG>','test','--dep','core','--dep','hip','--dep','hip_controls','-Mroot='+str(p/'probe.zig'),'-Mcore='+str(p/'core.zig'),'-Mhip='+str(p/'hip.zig'),'-Mhip_controls='+str(p/'controls.zig'),'--cache-dir',str(D/'cache_probes'),'--global-cache-dir',str(D/'global')];t=time.time();r=subprocess.run(cmd,capture_output=True);log=D/(phase+'_'+case+'.log');log.write_bytes(r.stdout+r.stderr);expected=0 if phase=='after'or case=='normal'else 1
  row={'phase':phase,'case':case,'command':cmd,'actual_exit_status':r.returncode,'expected_exit_status':expected,'log':log.name,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'elapsed_seconds':time.time()-t};rows.append(row);(D/'probe_results.json').write_text(json.dumps(rows,indent=2)+'\n');print(phase,case,r.returncode,flush=True);assert r.returncode==expected
