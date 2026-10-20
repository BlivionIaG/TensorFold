#!/usr/bin/env python3
"""Explicit build only; never selects devices or starts GPU jobs."""
import argparse,hashlib,json,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('tree',type=pathlib.Path);p.add_argument('arch',choices=['gfx1030','gfx1100']);p.add_argument('output',type=pathlib.Path);p.add_argument('--hipcc',default='hipcc');a=p.parse_args()
source=pathlib.Path(__file__).with_name('finite.hip');tree=a.tree.resolve();out=a.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
commit=subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True).strip()
dirty=subprocess.check_output(['git','-C',str(tree),'status','--porcelain'],text=True)
tracked=subprocess.check_output(['git','-C',str(tree),'status','--porcelain','--untracked-files=no'],text=True)
if tracked:raise SystemExit('refusing tracked modifications')
untracked=subprocess.check_output(['git','-C',str(tree),'ls-files','--others','--exclude-standard'],text=True).splitlines()
if any(not (x.split('/')[0].startswith('out-')) for x in untracked):raise SystemExit('untracked paths outside declared out-* build outputs')
# Match zig/build/hip.zig common device flags and caps; standalone executable links a host driver.
flags=['-O3','-D__HIP_PLATFORM_AMD__=1','-DUSE_ROCM=1','-DHIPBLAS_V2','-fPIC',
       '-DCUDA_HAS_FP16=1','-DHIP_ENABLE_WARP_SYNC_BUILTINS=1','-std=c++20',
       '-fno-gpu-rdc','-mno-wavefrontsize64','-ffp-contract=off','-DTF_WAVE=32',
       '-DTF_DOT2_F16=1','-DTF_DOT2_BF16='+('1' if a.arch=='gfx1100' else '0'),
       '-DTF_SDOT4=1','-DTF_SDOT8=1','-DTF_MATRIX='+('1' if a.arch=='gfx1100' else '0'),
       '--offload-arch='+a.arch]
args=[a.hipcc,'-v',*flags,'-I'+str(tree/'zig/kernels/hip'),str(source),'-o',str(out)]
compiler_log=out.with_suffix(out.suffix+'.compiler.log')
with compiler_log.open('w') as log:subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
headers={str(f.relative_to(tree)):sha(f) for f in sorted((tree/'zig/kernels/hip').rglob('*.hpp'))}
record={'production_optimization':'O3 as expanded by production hipcc --genco without explicit -O', 'source_commit':commit,'arch':a.arch,'binary_sha256':sha(out),'harness_sha256':sha(source),'header_sha256':headers,'compiler_version':subprocess.check_output([a.hipcc,'--version'],text=True),'flags':flags,'compiler_verbose_log_sha256':sha(compiler_log),'include_root':'zig/kernels/hip','build_mode':'standalone host driver plus production-template device code; no --genco'}
out.with_suffix(out.suffix+'.json').write_text(json.dumps(record,indent=2)+'\n')
