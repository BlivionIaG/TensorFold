"""Root launcher; dry by default, exact reviewed payload/config pins required to execute."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time,os
S=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--config',type=Path);p.add_argument('--config-sha256');p.add_argument('--payload-sha256');a=p.parse_args();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if not a.execute:
 print(json.dumps({'execute':False,'required':'Exact reviewed payload SHA plus root-authorized config SHA with actual activeVon/release/openingpair pins','order':['v_on_close','v_off_close','w_on_close','w_off_ono']}));raise SystemExit
assert a.config and a.config_sha256 and a.payload_sha256
manifest=S/'payload_sha256.json';assert sha(manifest)==a.payload_sha256
for n,h in json.loads(manifest.read_text()).items():assert sha(S/n)==h
assert sha(a.config)==a.config_sha256;actual=json.loads(a.config.read_text());template=json.loads((S/'config.json').read_text());assert actual['root_authorized']is True
# Only actual permission/proof hashes/timestamp may be hydrated; source/paths/order/settings stay frozen.
normalized=json.loads(json.dumps(actual))
for k in ['root_authorized','root_authorized_at_utc','active_von_state_sha256','collection_release_sha256']:normalized[k]=template[k]
for row,want in zip(normalized['blocks'],template['blocks']):
 if 'opening_pair'in row:row['opening_pair']['state_sha256']=want['opening_pair']['state_sha256']
assert normalized==template
D=S/('root_launch_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));D.mkdir(exist_ok=False);(D/'authorized_config.raw.json').write_bytes(a.config.read_bytes())
with (D/'supervisor.stdout').open('xb')as log:
 child=subprocess.Popen([sys.executable,str(S/'supervisor.py'),'--config',str(a.config.resolve()),'--execute'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 (D/'launched.json').write_text(json.dumps({'pid':child.pid,'parent_pid':os.getpid(),'source_commit':actual['source_commit'],'config_sha256':a.config_sha256,'payload_sha256':a.payload_sha256,'launched_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2)+'\n');rc=child.wait()
(D/'result.json').write_text(json.dumps({'exit_status':rc,'complete':rc==0},indent=2)+'\n');raise SystemExit(rc)
