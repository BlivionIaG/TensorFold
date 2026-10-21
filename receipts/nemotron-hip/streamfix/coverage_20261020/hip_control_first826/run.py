#!/usr/bin/env python3
import concurrent.futures, ctypes, datetime, fcntl, hashlib, json, os, pathlib, re, signal, subprocess, threading, time
H=pathlib.Path.home(); D=H/'hcx/streamfix/coverage_20261020/hip_control_fix_qualification_826_a44_attempt1'
BOOT='ccd3c480-76df-44a2-a67f-ee1db46a46e5'
SPECS=[('first','hcffirst826','826a0635634dee3793a68e473064d17d5c139945','bbb17bdf4db45f1084bb6e35442d4b5dde32039e',5),('top','hcftopa44','a44e7af98183721e58f2fb4d7e96ae874dfecd83','51a4fadc948801c456b03995186f612a17e5a7dc',49)]
LIB='/opt/rocm-7.2.0/core-7.14/lib/libamdhip64.so.7.14.60850-0000000'
PINS={str(H/'hcx/zb.sh'):'9464ba58dcf80191bb926aa8e3a10744a51dd5530ea062ad666951fad62218ab',str(H/'.local/zig-0.17.0/zig'):'7c61539af19fa4082c94848f1c2b57b89e76dc6cfd5bd4c57764c3caf8f82272','/opt/rocm/core-7.14/bin/hipcc':'d92839ed940e685fd15b8c60744d9f30efa65102e49e0babd655045219be371d',LIB:'00ca863da1cce26bd91d14bf268903a0ee3107595b531fa27fc06c88c6402e0d'}
INTENT=False
STOP=threading.Event()
def stopped():
    if STOP.is_set():raise InterruptedError('qualification interrupted')
def on_signal(sig,frame):STOP.set()
for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,on_signal)
R={'source_specs':SPECS,'boot_id':BOOT,'stages':[],'complete':False,'validated':False}; LOCK=threading.RLock(); OWN=set()
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save():
    with LOCK:
        p=D/'result.tmp'; p.write_text(json.dumps(R,indent=2)); p.replace(D/'result.json')
def cmd(a,cwd=None): return subprocess.check_output(a,cwd=cwd,text=True).strip()
def source(s):
    p=H/'nh'/s[1]
    assert cmd(['git','rev-parse','HEAD'],p)==s[2]
    assert cmd(['git','rev-parse','HEAD^{tree}'],p)==s[3]
    assert not cmd(['git','status','--porcelain','--untracked-files=no'],p)
    assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
    return p
def overlap():
    found=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid(): continue
        try:
            c=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if not any(x in c for x in ['native-serve','tf-nemotron','vllm.entrypoints','tf-hip-kernel-contract']): continue
            if 'bench serve' in c: continue
            e=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
            masks={k:e[k] for k in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES') if k in e}
            with LOCK:R.setdefault('process_observations',[]).append({'pid':int(p.name),'command':c,'masks':masks,'at':utc()})
            if len(masks)==1:
                value=next(iter(masks.values()))
                if re.fullmatch(r'\d+(,\d+)*',value) and set(value.split(',')) & {'0','1','2'}:found.append((p.name,c,masks))
        except (FileNotFoundError,ProcessLookupError): pass
    assert not found, found

def run(name,a,log,cwd=None,env=None,timeout=5600,observe=False):
    stopped()
    assert not log.exists(),log
    row={'name':name,'argv':list(map(str,a)),'cwd':str(cwd) if cwd else None,'started_at':utc(),'log':str(log)}
    maps={}; identities={}; start=time.monotonic()
    with log.open('xb') as f:
        p=subprocess.Popen(a,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        with LOCK: OWN.add(p.pid)
        row['pid']=p.pid
        try:
            while p.poll() is None:
                stopped()
                assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
                if observe:
                    try:
                        exe=pathlib.Path(f'/proc/{p.pid}/exe');identities['executable_sha256']=sha(exe);identities['executable_path']=str(exe.resolve())
                        observed=dict(x.split('=',1) for x in pathlib.Path(f'/proc/{p.pid}/environ').read_bytes().decode().split('\0') if '=' in x)
                        identities['environment']={k:observed.get(k) for k in ('HIP_VISIBLE_DEVICES','TENSORFOLD_MEMORY_RESERVE_GIB','TENSORFOLD_HIP_MEMORY_LIMIT_GB','LD_LIBRARY_PATH')}
                        for line in pathlib.Path(f'/proc/{p.pid}/maps').read_text().splitlines():
                            q=line.split()[-1]
                            if q.startswith('/') and ('libamdhip64' in q or 'libhsa-runtime64' in q): maps[q]=sha(q)
                    except FileNotFoundError: pass
                if time.monotonic()-start>timeout: raise TimeoutError(name)
                time.sleep(.05 if observe else 1)
            row['exit_status']=p.returncode
            try:os.killpg(p.pid,0)
            except ProcessLookupError:row['owned_process_group_gone']=True
            else:raise RuntimeError('owned process group remains: '+str(p.pid))
        except BaseException:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                try:os.killpg(p.pid,0)
                except ProcessLookupError:break
                p.poll();time.sleep(.1)
            else:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            p.wait()
            try:os.killpg(p.pid,0);row['owned_process_group_gone']=False
            except ProcessLookupError:row['owned_process_group_gone']=True
            row['exit_status']=p.returncode; raise
        finally:
            with LOCK: OWN.discard(p.pid)
            row.update(finished_at=utc(),elapsed_seconds=time.monotonic()-start,log_sha256=sha(log),loaded_libraries=maps,observed_identity=identities)
            with LOCK:R['stages'].append(row);save()
    assert row['exit_status']==0,row
    return row

def build(s,name,args):
    source(s); rel='hip_control_fix_826_a44_attempt1/'+name+'.log'; raw=H/'hcx/logs'/rel
    row=run(name,[str(H/'hcx/zb.sh'),s[1],rel,*args],D/(name+'.outer.log'))
    assert raw.exists();row['wrapper_log']=str(raw);row['wrapper_log_sha256']=sha(raw);source(s);save()
    return raw

def live(s,arch,device,binary):
    source(s);env=os.environ.copy();env['HIP_VISIBLE_DEVICES']=str(device);env['LD_LIBRARY_PATH']=str(H/'.local/rocm-rdna/lib')
    log=D/f'{s[0]}_{arch}_live.log'
    row=run(f'{s[0]}_{arch}_live',[str(binary)],log,env=env,timeout=1800,observe=True)
    text=log.read_text();pat=rf'^SUMMARY {arch} {s[4]} kernels: {s[4]} pass gates 1-4, 0 fail, 0 known findings; 5 controls: 5 red as planted, 0 missed; 0 limit problems;.*$'
    summaries=re.findall(pat,text,re.M);assert len(summaries)==1
    assert 'MISSED' not in text and '(record stale)' not in text
    assert row['observed_identity'].get('executable_sha256')==sha(binary),row
    assert row['observed_identity']['environment']['HIP_VISIBLE_DEVICES']==str(device)
    assert all(row['observed_identity']['environment'][k] is None for k in ('TENSORFOLD_MEMORY_RESERVE_GIB','TENSORFOLD_HIP_MEMORY_LIMIT_GB'))
    assert any('libamdhip64' in q and v==PINS[LIB] for q,v in row['loaded_libraries'].items()),row['loaded_libraries']
    source(s)
    with LOCK:
        row.update(source_commit=s[2],binary_sha256=sha(binary),device=device,summary=summaries[0]);save()

def main():
    global INTENT
    with (D/'EXECUTION_INTENT').open('x') as f:f.write(str(os.getpid())+'\n')
    INTENT=True
    assert not (D/'result.json').exists()
    assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
    assert all(sha(p)==v for p,v in PINS.items());R['tools']=PINS
    R['resolved_tools']={p:str(pathlib.Path(p).resolve()) for p in PINS}
    (D/'zb.sh.retained').write_bytes((H/'hcx/zb.sh').read_bytes())
    clang=pathlib.Path('/opt/rocm/core-7.14/lib/llvm/bin/clang++').resolve()
    if not clang.is_file():clang=pathlib.Path('/opt/rocm/core-7.14/bin/clang++').resolve()
    assert clang.is_file()
    R['clang']={'path':str(clang),'sha256':sha(clang),'version':cmd([str(clang),'--version'])}
    R['tool_versions']={'zig':cmd([str(H/'.local/zig-0.17.0/zig'),'version']),'hipcc':cmd(['/opt/rocm/core-7.14/bin/hipcc','--version'])}
    assert cmd([str(H/'.local/zig-0.17.0/zig'),'version'])=='0.17.0'
    for k in ('TENSORFOLD_MEMORY_RESERVE_GIB','TENSORFOLD_HIP_MEMORY_LIMIT_GB','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'):assert k not in os.environ,k
    assert cmd(['git','rev-parse','HEAD'],H/'nh/sfopt')=='729833615595df66c9f9a438ce9a85061843f010'
    assert not cmd(['git','status','--porcelain','--untracked-files=no'],H/'nh/sfopt')
    overlap()
    for s in SPECS:source(s)
    with (H/'opt/gpu.lock').open('a') as gl:
        fcntl.flock(gl,fcntl.LOCK_EX|fcntl.LOCK_NB)
        overlap()
        with (H/'w1012/build.lock').open('a') as bl:fcntl.flock(bl,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lib=ctypes.CDLL(LIB);assert lib.hipInit(0)==0
        pci={}
        for i,expected in [(0,'0000:c7:00.0'),(1,'0000:83:00.0'),(2,'0000:47:00.0')]:
            b=ctypes.create_string_buffer(64);assert lib.hipDeviceGetPCIBusId(b,64,i)==0;assert b.value.decode().lower()==expected;pci[i]=expected
        R['pci']=pci;R['started_at']=utc();save()
        (H/'hcx/logs/hip_control_fix_826_a44_attempt1').mkdir(exist_ok=False)
        for s in SPECS:
            stopped();overlap()
            resource_prefix=D/('resources_'+s[0]);assert not resource_prefix.exists()
            log=build(s,s[0]+'_limits',['hip-kernel-contract-limits','hip-kernel-contract-objects','-p',str(resource_prefix),'--summary','all','--verbose'])
            t=log.read_text();assert 'PASS kernel-contract-limits: every registered launch fits gfx1030, gfx1100, gfx1151 and gfx1201, every record current, every control red' in t
            assert not re.search(r'\b(FAIL|MISSED)\b|\(record stale\)',t)
            for chip in ('gfx1030','gfx1100','gfx1151','gfx1201'):
                for control in ('control: past its bounds','control: 72 KiB of LDS'):
                    assert len(re.findall(r'^RED +limits '+chip+' '+re.escape(control)+r':.*\(record current\)',t,re.M))==1
            objects={str(p.relative_to(resource_prefix)):sha(p) for p in resource_prefix.rglob('*.hsaco')}
            assert objects and all(any(k.startswith('hsaco/'+chip+'/') for k in objects) for chip in ('gfx1030','gfx1100','gfx1151','gfx1201'))
            R.setdefault('objects',{})[s[0]]=objects;save()
            bins={}
            for arch in ('gfx1100','gfx1030'):
                prefix=D/f'build_{s[0]}_{arch}';assert not prefix.exists()
                build(s,f'{s[0]}_{arch}_build',['tf-hip-kernel-contract','-Dhip-arch='+arch,'-p',str(prefix),'--summary','all','--verbose'])
                binary=prefix/'bin/tf-hip-kernel-contract';assert binary.is_file();bins[arch]=binary
            overlap()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                fs=[ex.submit(live,s,a,d,bins[a]) for a,d in [('gfx1100',0),('gfx1030',2)]]
                for f in fs:f.result()
            source(s)
        assert not OWN;R.update(complete=True,validated=True,finished_at=utc());save()
        (D/'QUALIFICATION_ALL_PASS').write_text(sha(D/'result.json')+'\n')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if INTENT:R.update(error=repr(e),finished_at=utc());save()
        raise
