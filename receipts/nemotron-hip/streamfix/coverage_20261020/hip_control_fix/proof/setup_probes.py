from pathlib import Path
import json,struct,subprocess,shutil
D=Path(__file__).parent;W=Path('<SOURCE_TREE>');records=json.loads((D/'fixture_records.json').read_text());HEAD='dc595eeb198c66d3327d7f0ce4a3eb6c9a807ec5'
def pack(v):
 if isinstance(v,str):
  b=v.encode();return (bytes([0xa0+len(b)])if len(b)<32 else b'\xd9'+bytes([len(b)]))+b
 if isinstance(v,int):return bytes([v])if v<128 else b'\xce'+struct.pack('>I',v)
 if isinstance(v,list):return bytes([0x90+len(v)])+b''.join(pack(x)for x in v)
 if isinstance(v,dict):return bytes([0x80+len(v)])+b''.join(pack(k)+pack(x)for k,x in v.items())
 raise ValueError(v)
def elf(desc):
 note_len=20+(len(desc)+3)//4*4;sh=64+note_len;b=bytearray(sh+128);b[:4]=b'\x7fELF';struct.pack_into('<Q',b,0x28,sh);struct.pack_into('<H',b,0x3c,2);struct.pack_into('<III',b,64,7,len(desc),32);b[76:83]=b'AMDGPU\0';b[84:84+len(desc)]=desc;struct.pack_into('<I',b,sh+64+4,7);struct.pack_into('<QQ',b,sh+64+24,64,note_len);return b
fields={'vgprs':'.vgpr_count','sgprs':'.sgpr_count','lds':'.group_segment_fixed_size','private':'.private_segment_fixed_size','bound':'.max_flat_workgroup_size','wave':'.wavefront_size'}
for case in ['normal','partial_green','missing_usage','stale_control']:
 for i,chip in enumerate(['gfx1030','gfx1100','gfx1151','gfx1201']):
  d=D/'fixtures'/case/chip;d.mkdir(parents=True,exist_ok=True)
  for control in [False,True]:
   ks=[]
   for r in records:
    if r['launch']['name'].startswith('control')!=control:continue
    u=r['usage'][i].copy()
    if i==0 and r['launch']['symbol']=='tf_ctl_bounded':
     if case=='partial_green':u['bound']=512
     if case=='missing_usage':u['vgprs']=0
     if case=='stale_control':u['sgprs']+=1
    ks.append({'.name':r['launch']['symbol'],'.symbol':r['launch']['symbol']+'.kd',**{k:u[f]for f,k in fields.items()}})
   payload=elf(pack({'amdhsa.kernels':ks}));(d/('contract_controls.hsaco'if control else'group0.hsaco')).write_bytes(payload)
   if not control:(d/'group1.hsaco').write_bytes(payload)
 for phase in ['before','after']:
  p=D/'probes'/phase/case;p.mkdir(parents=True,exist_ok=True)
  for rel,dest in [('zig/tests/hip/contract_limits.zig','contract_limits.zig'),('zig/src/core/contract/hip_records.zig','records_original.zig'),('zig/src/core/contract/limits.zig','limits.zig'),('zig/src/core/contract/hsaco.zig','hsaco.zig')]:
   b=(W/rel).read_bytes()if phase=='after'else subprocess.check_output(['git','show',HEAD+':'+rel],cwd=W);(p/dest).write_bytes(b)
  # One controlled test fixture binding; original module functions remain actual source bytes.
  stub='const r = @import("records_original.zig");\npub const Record = r.Record;\npub const records = r.records;\npub const check = r.check;\n'
  if phase=='after':stub+='pub const controlProblems = r.controlProblems;\n'
  stub+='pub const controls = blk: { var c = r.controls; '
  stub+= 'c[0].usage[0].bound = 512;'if case=='partial_green'else 'c[0].usage[0].vgprs = 0;'if case=='missing_usage'else '_ = &c;'
  stub+=' break :blk c; };\n';(p/'stub_records.zig').write_text(stub)
  (p/'core.zig').write_text('pub const contract = struct { pub const hip_records = @import("stub_records.zig"); pub const limits = @import("limits.zig"); pub const hsaco = @import("hsaco.zig"); };\n');(p/'hip.zig').write_text('pub const kernels = struct { pub const Group = enum { ops, act }; };\n');(p/'controls.zig').write_text('pub const bytes: [0]u8 = .{};\n')
  expected=case!='normal';tablebad=case in ['partial_green','missing_usage'];(p/'probe.zig').write_text('const std = @import("std");\nconst gate = @import("contract_limits.zig");\ntest "table enforces every control chip" { const n = try gate.table(std.testing.allocator); try std.testing.expect('+('n > 0'if tablebad else'n == 0')+'); }\ntest "objects enforces control chip, usage and metadata" { const n = try gate.objects(std.testing.allocator, std.testing.io, "'+str(D/'fixtures'/case)+'"); try std.testing.expect('+('n > 0'if expected else'n == 0')+'); }\n')
