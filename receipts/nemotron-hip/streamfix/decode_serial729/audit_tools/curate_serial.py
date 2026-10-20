"""Curate a real complete serial receipt; refuses absent/incomplete inputs. No GPU work."""
import argparse,hashlib,json,re,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();S=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
 audit=Path(tmp)/'audit.json';subprocess.run([sys.executable,str(S/'audit_serial.py'),str(a.raw),'--output',str(audit)],check=True);r=json.loads(audit.read_text())
a.destination.mkdir(parents=True,exist_ok=False);manifest=[]
def add(name,raw):
 public=re.sub(r'/home/[A-Za-z0-9_.-]+','<USER_HOME>',raw.decode()).encode();dest=a.destination/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(public);manifest.append({'path':str(name),'raw_sha256':hashlib.sha256(raw).hexdigest(),'public_sha256':hashlib.sha256(public).hexdigest(),'raw_bytes':len(raw),'public_bytes':len(public)})
for f in sorted(a.raw.rglob('*')):
 if f.is_file():add(Path('raw')/f.relative_to(a.raw),f.read_bytes())
for name in ['audit_serial.py','curate_serial.py','attach_independent_audit.py','observe_memory_env.py','independent_audit.schema.json','amd_guard_source_review.json','AMD_GUARD_SCOPE.md']:add(Path('audit_tools')/name,(S/name).read_bytes())
summary={k:v for k,v in r.items() if k!='raw_file_bindings'};summary['actual_holds']=16;summary['raw_manifest_bindings_file']='manifest.json';summary['independent_review_status']='Separate exact-summary-SHA independent audit is required; helper validation alone is not independent approval'
add(Path('summary.json'),(json.dumps(summary,indent=2)+'\n').encode())
text='# Fresh serial16 decode-only receipt\n\nThis separately measured receipt does not replace the21 concurrent raw holds/16 selected results. OFF then ON; each mode completes W7800TP2 old/new/new/old before V620TP1. Two prompts,256 tokens,three repetitions,temperature0. One logicalTP model group is scheduled at a time; TP2 has two rank processes. Actual loaded binary/library/checkpoint identity, full reply/token hashes, host-load/RAM/context preflight and actual native teardown/client exits are retained.\n\n'
for g in r['conditions']:
 for prompt,v in g['prompts'].items():text+=f"- {g['rig']} drafts {g['mode']}, {prompt}: {v['old_mean']:.5f} → {v['new_mean']:.5f} tok/s ({v['delta_percent']:+.5f}%). Old hold order: {v['hold_order'][0]:.5f}, {v['hold_order'][3]:.5f}.\n"
text+=f"\nActual admission environment: coordinator observed plus {r['memory_admission_environment_scope']['actual_rank_environment_samples']} of {r['memory_admission_environment_scope']['total_recorded_rank_processes']} recorded rank PIDs sampled. Only TENSORFOLD_MEMORY_RESERVE_GIB and TENSORFOLD_HIP_MEMORY_LIMIT_GB were captured; observed values match the allocation snapshot. Unsampled rank environments are not invented or inferred.\n"
text+='\nThese are direct hold-median means, not a significance or acceptance threshold. Known-process scans cannot rule out arbitrary renamed/Python GPU processes; scheduling is a coordinator assertion and observations, not a whole-host lease. Existing GPU-lock qualification remains pending; the held existing lock is CPU-build-only. Native admission/limit behavior is unchanged. No cold-prefill benchmark, full CONTRIBUTING or upstream-integration qualification is claimed. Independent review is supplied only by a separate audit binding this exact summarySHA.\n'
add(Path('README.md'),text.encode());(a.destination/'manifest.json').write_text(json.dumps({'file_count':len(manifest),'files':manifest},indent=2)+'\n');print(a.destination)
