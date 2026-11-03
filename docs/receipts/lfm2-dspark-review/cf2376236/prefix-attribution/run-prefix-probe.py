import argparse, fcntl, json, os, re, sys
from pathlib import Path
from types import SimpleNamespace

p=argparse.ArgumentParser()
for name in ('driver','binary','target','drafter','output','binding','patch'):
    p.add_argument('--'+name,required=True)
p.add_argument('--lock-inode',required=True,type=int)
a=p.parse_args()
sys.path.insert(0,a.driver)
import bench_lfm2_dspark as d

root=Path(a.output);root.mkdir(parents=True)
source_binding=json.loads(Path(a.binding).read_text())
for name,digest in source_binding['driver_sha256'].items():
    assert d.sha256_file(Path(a.driver)/name)==digest
settings={'label':'PRIVATE_EXACT_HISTORY_CAUSAL_DIAGNOSTIC_NOT_QUALIFICATION','base_source':'5bed4bd92c7c257f963046b945d83fc931dcbf29','production_source_under_diagnosis':'cf23762368785b470b3bf7485026e2ea4837ce82','base_vs_production':'Only the obsolete test-case deletion differs; production policy and target kernels are identical.','binary_sha256':d.sha256_file(Path(a.binary)),'instrumentation_patch_sha256':d.sha256_file(Path(a.patch)),'controller_sha256':d.sha256_file(Path(__file__)),'context':4096,'parallel':16,'tokens':512,'history':'Original code,T0 scenario at all widths, then original code,T1 warmup and all16 seed oracles, then its width1 warmup and measured case.','source_binding':source_binding,'target_counter_phase_labels':'The raw labels decode/prefill mean recorded/unrecorded forwards; unrecorded also includes queued decode. Sum both for full RPC GPU work.','qualification_throughput':False}
(root/'binding.json').write_text(json.dumps(settings,indent=2)+'\n')

class TraceContext(d.Context):
    def __init__(self,*args,**kwargs):
        self.private_phases=[]
        super().__init__(*args,**kwargs)
    def raw(self,cell,phase,runs):
        super().raw(cell,phase,runs)
        wanted=16 if phase=='warmup' else 512
        for run in runs:
            errors=d.problems(run,wanted,False)
            assert not errors,errors
        self.private_phases.append({'cell':cell,'phase':phase,'requests':len(runs),'max_new':wanted,'sent':min(r['sent'] for r in runs),'completed':max(r['completed'] for r in runs),'rpc_wall_ms':1000*(max(r['completed'] for r in runs)-min(r['sent'] for r in runs)),'activity':d.total_activity(runs),'runs':[{'seed':r['seed'],'token_sha256':r['token_sha256'],'tokens':r['tokens']} for r in runs]})


def sampled_first_cell(ctx,srv,arm,item):
    temp=1.0
    specs=[(item,seed) for seed in d.seeds_for(ctx.seed,ctx.widths,temp)]
    stem=f"r0-{arm}-{item['name']}-T1"
    ctx.raw(stem,'warmup',d.bc.together(ctx.base,d.SERVED_NAME,specs[:1],16,temp))
    seeds=list(dict.fromkeys(seed for _,seed in specs[:max(ctx.widths)]))
    refs=d.oracles(ctx,stem,0,arm,item,temp,seeds)
    srv.check()
    label=stem+'-w1'
    ctx.raw(label,'warmup',d.bc.together(ctx.base,d.SERVED_NAME,specs[:1],16,temp))
    runs=d.bc.together(ctx.base,d.SERVED_NAME,specs[:1],512,temp)
    ctx.raw(label,'measured',runs)
    srv.check()
    for run in runs:
        for kind,reference in refs[run['seed']].items():
            mismatch=d.compare(reference,run,'concurrent_vs_'+kind,run['seed'])
            assert not mismatch,mismatch


def fields(line):
    result={}
    for key,value in re.findall(r'(\w+)=([^\s]+)',line):
        if value in ('true','false'):result[key]=value=='true'
        else:
            try:result[key]=int(value)
            except ValueError:
                try:result[key]=float(value)
                except ValueError:result[key]=value
    return result


def parse(text):
    snapshots=[];current=None;enrollments=[];twins=[];holds=None
    for line in text.splitlines():
        if 'DSPARK_PRIVATE_TWIN ' in line:twins.append(fields(line)['enabled'])
        elif 'DSPARK_PRIVATE_ENROLL ' in line:enrollments.append(fields(line))
        elif 'DSPARK_PRIVATE_PRICE ' in line:
            assert enrollments
            enrollments[-1]['price']=fields(line)
        elif 'DSPARK_PRIVATE_HOLDS ' in line:holds=fields(line)
        elif 'DSPARK_PRIVATE_SNAPSHOT ' in line:
            assert current is None,'incomplete previous snapshot'
            current={'index':fields(line)['index'],'counters':{},'target':{},'rows_by_count':{}}
        elif current is not None:
            if 'DSPARK_PRIVATE_COUNTER ' in line:
                value=fields(line);current['counters'][value.pop('kind')]=value
            elif 'DSPARK_PRIVATE_TARGET ' in line:
                value=fields(line);current['target'][value.pop('phase')]=value
            elif 'DSPARK_PRIVATE_ROW_COUNTS ' in line:
                match=re.search(r'phase=(\w+) bins=(.*)',line);assert match
                bins=[int(v) for v in re.findall(r'\d+',match[2])];assert len(bins)==33
                current['rows_by_count'][match[1]]=bins
                if match[1]=='decode':
                    assert set(current['target'])=={'prefill','decode'}
                    snapshots.append(current);current=None
    assert current is None
    assert twins,'missing actual load-time twin decision'
    return {'snapshots':snapshots,'enrollments':enrollments,'twins':twins,'holds':holds}


def difference(current,previous):
    if isinstance(current,dict):return {k:difference(v,previous.get(k) if previous else None) for k,v in current.items()}
    if isinstance(current,list):return [difference(v,previous[i] if previous else None) for i,v in enumerate(current)]
    return current-(previous or 0)


rows=[];plain_refs={}
with open(d.LOCK,'r+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.fstat(lock.fileno()).st_ino==a.lock_inode
    assert d.pressure_level()==1
    assert not d.competing_processes(Path(a.binary).name)
    for kind in ('target','drafter'):
        for name,digest in source_binding['model_sha256'][kind].items():
            assert d.sha256_file(Path(getattr(a,kind))/name)==digest
    prompt=next(item for item in d.bc.PROMPTS if item['name']=='code')
    for arm in ('plain','bf16','bits8'):
        out=root/arm;out.mkdir()
        args=SimpleNamespace(binary=a.binary,target=a.target,drafter=a.drafter,widths='1,4,8,16',temperatures='0,1',tokens=512,reps=1,seed=1234,port=18194,context=4096,startup_timeout=300)
        ctx=TraceContext(args,out);ctx.logs.mkdir()
        with d.Server(ctx,'exact-history-'+arm,arm) as srv:
            child=srv.proc
            d.scenario(ctx,srv,0,arm,prompt,0.0)
            sampled_first_cell(ctx,srv,arm,prompt)
        assert child.returncode==0
        parsed=parse(srv.log_path.read_text())
        assert len(parsed['snapshots'])==len(ctx.private_phases),(len(parsed['snapshots']),len(ctx.private_phases),'cannot assign per-phase snapshots')
        cursor=0;last=None
        for phase,snapshot in zip(ctx.private_phases,parsed['snapshots'],strict=True):
            n=0 if arm=='plain' or phase['phase']=='draft_false' else phase['requests']
            phase['enrollments']=parsed['enrollments'][cursor:cursor+n];cursor+=n
            assert len(phase['enrollments'])==n and all(r['max_new']==phase['max_new'] for r in phase['enrollments'])
            values={k:snapshot[k] for k in ('counters','target','rows_by_count')}
            phase['private_delta']=difference(values,last);last=values
            for value in phase['private_delta']['target'].values():
                assert value['calls']==value['timed_gpu_calls'],'missing existing GPU duration'
            phase['target_gpu_ms']=sum(value['gpu_ms'] for value in phase['private_delta']['target'].values())
            if phase['phase']=='measured':
                temperature=1 if '-T1-' in phase['cell'] else 0
                for run in phase['runs']:
                    key=(temperature,run['seed'])
                    if arm=='plain':plain_refs[key]=run['token_sha256']
                    else:assert plain_refs[key]==run['token_sha256'],'served target tokens differ'
        assert cursor==len(parsed['enrollments']),'unassigned actual enrollment'
        assert (arm=='plain')==(parsed['holds'] is None)
        row={'arm':arm,'actual_twins':parsed['twins'],'phases':ctx.private_phases,'actual_holds':parsed['holds'],'returncode':child.returncode,'log_sha256':d.sha256_file(srv.log_path)}
        rows.append(row)
        (root/'observed.json').write_text(json.dumps({'binding':settings,'records':rows},indent=2)+'\n')
        print(json.dumps({'arm':arm,'complete':True,'phases':len(ctx.private_phases),'twin':parsed['twins'],'holds':parsed['holds'],'measured':[{'cell':phase['cell'],'activity':phase['activity'],'tracked':[e['tracked'] for e in phase['enrollments']],'target_gpu_ms':phase['target_gpu_ms'],'target_call_shapes':phase['private_delta']['rows_by_count']} for phase in ctx.private_phases if phase['phase']=='measured']}),flush=True)
print(json.dumps({'complete':True,'label':settings['label'],'arms':len(rows)}),flush=True)
