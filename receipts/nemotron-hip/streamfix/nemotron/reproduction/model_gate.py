"""Validate explicit runtime scheduling release; critical checkpoints never qualify publication."""
import hashlib,json,re
EXPECTED={'NEW':'e0e1c8e97fa14478e60af840942fe28526267bf6','41':'ca40e2dfbf75714b0b0c9f05ca54496c5ecb8546','46':'1596156e21769324b675ef5e0caae0452f361660'}
COUNTS={'NEW':23,'41':43,'46':49}
MEASURED='cc18314dce85b0968496b35333bdd8618b2f9227'
SCOPE='scheduling checkpoints only; remaining exact-head qualification still required before publication'
def load(root):
 release=json.loads((root/'model/release.json').read_text())
 assert release['parent_release'] is True and release['model_checks_all_pass'] is True and release['hip1_released'] is True
 assert release['contract_gate']=='CRITICAL_CONTRACTS_ALL_PASS.json'
 assert (root/'final_models/MODEL_CHECKS_ALL_PASS').read_text().strip()==MEASURED
 path=root/release['contract_gate'];data=path.read_bytes();g=json.loads(data)
 assert g['schema_version']==1 and g['validated'] is True and g['complete_stack_qualified'] is False
 assert g['scope']==SCOPE and g['review_top']==EXPECTED['46'] and g['measured_model_source']==MEASURED
 assert set(g['required_tags'])==set(EXPECTED) and len(g['required_tags'])==3
 receipts=g['receipts'];assert len(receipts)==3 and {str(r['tag']) for r in receipts}==set(EXPECTED)
 for r in receipts:
  assert r['source_commit']==EXPECTED[str(r['tag'])]
  assert set(r['architectures'])=={'gfx1100','gfx1030'}
  n=r['expected_kernel_count'];assert type(n) is int and n==COUNTS[str(r['tag'])]
  for arch,proof in r['architectures'].items():
   wanted=rf'^SUMMARY {arch} {n} kernels: {n} pass gates 1-4, 0 fail, 0 known findings; 5 controls: 5 red as planted, 0 missed; 0 limit problems(?:;|$)'
   assert re.search(wanted,proof['summary']),proof['summary']
   for key in ['raw_log_sha256','binary_sha256']:assert re.fullmatch(r'[0-9a-f]{64}',proof[key])
  assert re.fullmatch(r'[0-9a-f]{64}',r['four_chip_limits_sha256'])
 return {**g,'gate_file_sha256':hashlib.sha256(data).hexdigest(),'release':release,'publication_qualified':False}
