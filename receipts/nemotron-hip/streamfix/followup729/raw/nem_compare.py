"""Run source/binary-bound Nemotron comparisons on one released rig; preserve every actual rank status."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
ap=argparse.ArgumentParser(__doc__);ap.add_argument('rig',choices=['w7800tp2','v620tp1']);a=ap.parse_args()
from model_gate import load as load_gate
H=Path.home();R=H/'hcx/streamfix/optimization/followup729';contracts=load_gate(R)
arch,devs,world=('gfx1100','0,1',2) if a.rig=='w7800tp2' else ('gfx1030','2',1)
O=R/'logs'/('nem_'+a.rig+'_'+time.strftime('%Y%m%dT%H%M%S'));O.mkdir(parents=True,exist_ok=False)
summary={'rig':a.rig,'architectures':arch,'devices':devs,'checkpoint_revision':'d9d758fb83953437f7263256b0d96157e2a348b8','runs':[],'comparisons':[],'done':False}
model=H/'models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit';ids=H/'nem-work/ids';acc=H/'nem-acc'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(): (O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
summary['harness_sha256']=sha(Path(__file__));summary['gate_validator_sha256']=sha(Path(__file__).with_name('model_gate.py'));summary['qualification_contract_gate']=contracts
summary['libraries']={}
for path in (H/'.local/rocm-rdna/lib').glob('librccl.so*'):
 if path.is_file():summary['libraries'][str(path)]={'realpath':str(path.resolve()),'sha256':sha(path)}
summary['model_metadata']={str(p.relative_to(model)):p.read_text() for p in (model/'.cache/huggingface/download').glob('*.metadata')}
config_metadata=model/'.cache/huggingface/download/config.json.metadata'
assert config_metadata.is_file() and config_metadata.read_text().splitlines()[0]==summary['checkpoint_revision'],'checkpoint config revision mismatch'
summary['config_metadata_sha256']=sha(config_metadata)
for tree,expected,marker in [('d46','8649e9858b38e9ef3dcd8d579879c6512a363050','baseline_model_tools_built.json'),('sfopt','729833615595df66c9f9a438ce9a85061843f010','model_tools_built.json')]:
 repo=H/'nh'/tree;assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==expected
 subprocess.run(['git','-C',str(repo),'diff','--quiet'],check=True);subprocess.run(['git','-C',str(repo),'diff','--cached','--quiet'],check=True)
 manifest=json.loads(((H/'hcx/streamfix'/marker) if tree=='d46' else (R/marker)).read_text());assert manifest['source_commit']==expected
 binary=manifest['architectures'][arch]['binaries']['tf-nemotron-hip-test'];assert sha(Path(binary['path']))==binary['sha256']
 for kind,name,inputfile,extra in [('decode','decode',acc/'chat_long.ids.bin',['--steps','224']),*[( 'teacher',x,ids/(x+'.ids.bin'),[]) for x in ['chat_full','code_full','story_full']]]:
  output=O/(tree+'_'+name+'.bin');assert not output.exists();port=27500 if world==2 else 28500
  base=[binary['path'],kind,str(model),str(inputfile),str(output),*extra];env=os.environ.copy();env.pop('ROCR_VISIBLE_DEVICES',None);env['HIP_VISIBLE_DEVICES']=devs;env['LD_LIBRARY_PATH']=str(H/'.local/rocm-rdna/lib')
  rec={'tree':tree,'source_commit':expected,'binary_sha256':binary['sha256'],'kind':kind,'name':name,'input_sha256':sha(inputfile),'ranks':[]};procs=[];handles=[]
  try:
   for rank in reversed(range(world)):
    argv=base+(['--tp',str(world),'--rank',str(rank),'--master','127.0.0.1','--port',str(port)] if world>1 else [])
    log=O/(tree+'_'+name+'.r'+str(rank)+'.log');handle=log.open('wb');handles.append(handle);proc=subprocess.Popen(argv,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True);procs.append((rank,proc));rec['ranks'].append({'rank':rank,'argv':argv,'log':log.name})
   identity=O/(tree+'_'+name+'.identity.json')
   for attempt in range(90):
    capture=subprocess.run(['python3',str(Path(__file__).with_name('capture_identity.py')),str(identity),*[str(proc.pid) for _,proc in procs]],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if capture.returncode==0:break
    time.sleep(.5)
   assert identity.is_file(),'No actual runtime identity captured'
   captured=json.loads(identity.read_text());assert len(captured['processes'])==world,'actual rank identity count mismatch'
   assert all(rec['executable']['sha256']==binary['sha256'] for rec in captured['processes']),'captured executable differs from source-bound binary'
   assert captured['checkpoint_metadata']['config.json.metadata'].splitlines()[0]==summary['checkpoint_revision'],'captured checkpoint revision mismatch'
   rec['runtime_identity']={'file':identity.name,'sha256':sha(identity)}
   for rank,proc in reversed(procs):
    try:status=proc.wait(timeout=1800 if rank==0 else 60)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);status=proc.wait();rec['timeout']=True
    next(x for x in rec['ranks'] if x['rank']==rank)['exit_status']=status
   assert all(x['exit_status']==0 for x in rec['ranks']),rec
   assert output.is_file() and output.stat().st_size>0,output
   rank0text=(O/(tree+'_'+name+'.r0.log')).read_text()
   assert 'finite true' in rank0text and 'finite false' not in rank0text,'Nonfinite or absent finite-status output'
   rec['finite']=True
   rec['output']={'path':str(output),'bytes':output.stat().st_size,'sha256':sha(output)}
  finally:
   for _,proc in procs:
    if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   for handle in handles:handle.close()
   summary['runs'].append(rec);save()
for name in ['decode','chat_full','code_full','story_full']:
 matches=[r for r in summary['runs'] if r['name']==name];assert len(matches)==2
 old,new=matches;assert old['output']['bytes']==new['output']['bytes']>0 and old['output']['sha256']==new['output']['sha256'],name
 subprocess.run(['cmp','--silent',old['output']['path'],new['output']['path']],check=True)
 summary['comparisons'].append({'name':name,'bytes':old['output']['bytes'],'sha256':old['output']['sha256'],'identical':True})
summary['done']=True;save();print('NEM_DONE '+str(O),flush=True)
