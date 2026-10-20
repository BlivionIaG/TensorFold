"""Write human-readable, source-bound recovery results and bind generated summary files."""
import hashlib,json
from pathlib import Path
S=Path(__file__).resolve().parent;D=S/'public-stage';r=json.loads((D/'summary.json').read_text());m=json.loads((D/'manifest.json').read_text())
text='''# Decode-only inline recovery benchmark

This package retains **21 actual holds** and selects **16 balanced holds**: the original eight drafts-off holds plus eight fresh drafts-on holds. The original attempt stopped on a port-bind error before V620 drafts-on hold 2. Its five completed drafts-on holds are preserved and explicitly excluded; both rigs' drafts-on quartets were rerun in full on lower ports. The interrupted original 16-hold session did not complete.

Measured sources:

- Old baseline: `8649e9858b38e9ef3dcd8d579879c6512a363050`.
- New inline candidate: `729833615595df66c9f9a438ce9a85061843f010`.
- Private review top `60a8da5243ef313fcab23bdb0d85c790a5cda935` has the identical complete Git tree; measurements remain labelled as `729833`, not per-head reruns. See `source_bridge.json`.
- Nemotron checkpoint revision: `d9d758fb83953437f7263256b0d96157e2a348b8`. Cache metadata pins the revision; this package does not claim hashing every weight byte.

Each rig/mode uses **old/new/new/old**, two fixed prompts (`fibonacci-raw`, `gpu-chat-no-think`), 256 completion tokens, three repetitions per hold, temperature 0. W7800 uses TP2 on devices 0 and 1; V620 uses TP1 on device 2. Context 65536 and parallel 4 match the prior recipe. For each prompt, the reported old/new rates are the means of the two per-source hold medians. The original client stores repetition rates rounded to two decimals; every full-precision reported median is crosschecked after rounding against the median of those three stored rates. Full-precision reported medians are used below.

The W7800 TP2 and V620 TP1 quartets ran concurrently on the same physical host using disjoint GPU sets, sharing host/CPU resources. This parallel protocol does not meet the CONTRIBUTING requirement of one model instance per test machine; these measurements retain that scope and are not full performance qualification.

## Actual throughput

'''
for g in r['conditions']:
 for prompt,v in g['prompts'].items():text+=f"- {g['rig']}, drafts {g['mode']}, {prompt}: **{v['old_mean']:.5f} → {v['new_mean']:.5f} tok/s ({v['delta_percent']:+.5f}%)**.\n"
text+='''
All selected receipts pass complete reply and token-hash equality, including across drafts modes. Client exits are zero. Native ranks remained alive through each client run, were intentionally terminated with actual status −15, and their process groups disappeared. Actual executable hashes, loaded HIP/RCCL mappings, checkpoint metadata, client/helper source, raw benchmark arrays and controller logs are retained. Independent audits and exact hold-order values are included.

## Interpretation and limits

The large historical `cc183` slowdown is much smaller in this narrow candidate `729833` run, but several means remain lower. **No parity, stable speedup, statistical significance or performance-acceptance claim is made.** V620 drafts-on chat baseline holds drift 156.61746→133.07299 tok/s; its +5.80109% mean does not establish a stable speedup. W7800 drafts-on chat baseline also drifts 216.02656→204.52731 tok/s. Compare the full hold order, not only mean percentages.

The V620 baseline [first](retry_on/V620tp1/on_1_old.r0.log) and [last](retry_on/V620tp1/on_4_old.r0.log) drafts-on logs retain the same configured policies and matching outputs, while aggregate work counters differ: verify 1011 calls at 11137 µs each versus 993 at 12877 µs; draft 1019 calls at 897 µs versus 1001 at 1038 µs. Round graphs show 14 captured and 1031/1059 rounds replayed versus 16 captured and 1009/1041 replayed. These whole-hold adaptive counters differ alongside timing; they do not isolate the cause of the chat drift.

This is decode-only: **no new cold-prefill result** is claimed. These served holds do not complete separate current host, Qwen/Flash/Nemotron-logit, lifecycle/context/memory/comparator, CONTRIBUTING or upstream-integration qualification. Historical 32-hold 8649→cc183 results remain separate unchanged evidence; the old cold 8k/32k results are not relabelled as candidate `729833` measurements.

`summary.json` names the 16 selected and 5 excluded receipts. `manifest.json` binds original raw bytes to public files; only personal home directories are redacted. All 21 holds and both controller outputs remain available under `attempt1/` and `retry_on/`. The exact historical failure is retained rather than hidden or treated as a completed original protocol.
'''
(D/'README.md').write_text(text)
bridge=S.parent/'private-restack/source_bridge.json';(D/'source_bridge.json').write_bytes(bridge.read_bytes());(D/'write_summary_md.py').write_bytes(Path(__file__).read_bytes())
for name in ['README.md','summary.json','source_bridge.json','write_summary_md.py']:
 b=(D/name).read_bytes();m['files']=[x for x in m['files'] if x['path']!=name];m['files'].append({'path':name,'raw_sha256':hashlib.sha256(b).hexdigest(),'public_sha256':hashlib.sha256(b).hexdigest(),'raw_bytes':len(b),'public_bytes':len(b)})
m['file_count']=len(m['files']);(D/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m['file_count'],'bound files; README + structured summary included')
