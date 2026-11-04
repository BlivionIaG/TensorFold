import fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent
old=json.loads((p/'binding.json').read_text())
keys=('driver_sha256','frozen_driver','shared_lock','shared_lock_inode_observed','model_sha256','remote_target','remote_drafter','target_revision','drafter_revision','source_archive_sha256')
b={k:old[k] for k in keys}
root=Path(old['source_root'])
head=old['source_revision']
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(p.parent/'source.tar.gz')==b['source_archive_sha256']
for name,digest in b['driver_sha256'].items():assert sha(Path(b['frozen_driver'])/name)==digest
sys.path.insert(0,b['frozen_driver'])
import bench_lfm2_dspark as gate
assert Path(b['shared_lock']).stat().st_ino==b['shared_lock_inode_observed']==3719605
records=[]
def run(cmd,path,deadline):
    start=time.monotonic()
    with path.open('xb') as out:
        child=subprocess.Popen(cmd,cwd=root,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:code=child.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=90)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
    row={'command':cmd,'returncode':code,'seconds':time.monotonic()-start,'log_sha256':sha(path)}
    records.append(row);print(json.dumps(row),flush=True)
    return code
with Path(b['shared_lock']).open('r+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.fstat(lock.fileno()).st_ino==3719605
    assert gate.pressure_level()==1
    assert not gate.competing_processes('tf-lfm2-dspark')
    env,_=gate.clean_env()
    for kind in ('target','drafter'):
        for name,digest in b['model_sha256'][kind].items():assert sha(Path(b['remote_'+kind])/name)==digest
    build=[old['remote_zig'],'build','native','test-lfm2','test-lfm2-dspark','test-native','test-qwen27-shared-allocation-host','test-draft-metal-host','test-drafter-skip-metal-host','tf-kernel-contract','tf-lfm2-dspark','-Dcpu=apple_m1','-Dversion=1.2.0','-j1','--summary','all']
    assert run(build,p/'build.log',900)==0
    for operator in ('lfm2',):
        assert run([str(root/'zig-out/bin/tf-kernel-contract'),'--only',operator],p/(operator.replace(' ','-')+'.log'),900)==0
    binary=root/'zig-out/bin/tf-lfm2-dspark'
    cases=[]
    for bits in (0,8):
        for depth in (9,):
            cmd=[str(binary),b['remote_target'],b['remote_drafter'],'64',str(depth),str(bits)]
            path=p/('native-bits'+str(bits)+'-depth'+str(depth)+'.log')
            assert run(cmd,path,900)==0
            text=path.read_text()
            assert 'fixed query block: 9 rows;' in text
            for count in (8,16):assert 'split hold, '+str(count)+' logical lanes:' in text
            assert '60 context quotes' in text
            assert 'zero-row rewind: context end 2143, no encode or GPU work' in text
            assert 'pending CPU retained only for EMA' in text and 'pending CPU cleared, not recharged' in text
            producers=[line for line in text.splitlines() if 'Markov producer ' in line]
            assert len(producers)==5 and all(line.startswith('PASS Markov producer ') for line in producers)
            import re
            reference=re.search(r'(\d+) of (\d+) drafts equal the fp32 reference',text)
            assert reference and int(reference[2])==depth
            if bits==0:assert int(reference[1])==depth,'BF16 reference picks differ'
            margin=re.search(r"reference's largest margin over a different GPU pick: ([\d.]+)",text)
            confidence=re.search(r"confidences' worst difference from the fp32 reference: ([\d.]+)",text)
            cases.append({'bits':bits,'requested_prefix':depth,'reference_equal':int(reference[1]),'reference_count':depth,'different_pick_max_margin':float(margin[1]),'confidence_max_difference':(float(confidence[1]) if confidence else None),'prefix_bytes_and_first_pick_invariant':True,'confidence_prefix_invariant':True,'logical_cohorts_8_and_16':True,'log_sha256':sha(path)})
            assert gate.pressure_level()==1
            print(json.dumps(cases[-1]),flush=True)
    result={'scope':'IntegratedPinnedMainM5OriginalGuards;NoServingQualification','source_revision':head,'base_revision':old['base_revision'],'source_tree_ids':old['source_tree_ids'],'production_binary_sha256':sha(root/'zig-out/native/bin/tensorfold-native'),'binary_sha256':sha(binary),'cases':cases,'records':records,'binding':b}
    (p/'observed.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('binding','records')}),flush=True)
