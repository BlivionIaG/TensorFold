# Fixed-head CF qualification: both strict speed gates failed

These are the **completed**, unchanged full qualification packets for production source
`cf23762368785b470b3bf7485026e2ea4837ce82` (CF), against pinned base
`6a5048697e0770a464a4093ff6373f6a73f998b6`. The source ID is **not** the evidence
worktree/commit ID. This assembly does not change production code or verdicts.

| Device | Cells OK / expected | Oracle seeds OK / expected | Exactness / scenario failures | Strict speed losses | Clean server exits | Original controller / summary exit |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Apple M4 Pro | 144 / 144 | 306 / 306 | 0 / 0 | **12 / 32** | 9 / 9 | **3 / 3 — FAIL** |
| Apple M5 Ultra | 144 / 144 | 306 / 306 | 0 / 0 | **16 / 32** | 9 / 9 | **3 / 3 — FAIL** |

All 64 speed comparisons are below, including every loss. Complete/exact does not
mean speed-qualified: both original performance and overall verdicts remain `fail`.
There is **no tolerance, pooling, waiver, speed-pass, CI-pass, or re-review claim**.
The nine server shutdowns per device each have `returncode: 0` and
`forced_kill: false`; they do not turn the enclosing driver's exit 3 into a pass.

## Bindings and method

All arms on each device used its one binary; only external-drafter flags differed.
`plain` means no `--drafter`; BF16 adds `--drafter`; 8-bit additionally adds
`--drafter-bits 8`. Each matrix has three balanced arm-order rotations, code/chat,
temperatures 0/1, widths 1/4/8/16, 512 tokens, context 4096, parallel 16,
prompt cache disabled, seed 1234, and diagnostic logging off. Full settings,
environment and startup mode/binary hashes are in each raw metadata/events pair.

| Device | Native binary SHA256 | Environment |
| --- | --- | --- |
| Apple M4 Pro | `20d09fbea8be085a9b9b55d2843517779ac794e914b1bb1cb5d61481cba774ac` | Mac16,7; macOS build 25F80 |
| Apple M5 Ultra | `7a1210390424df60a2610454499f53018fa99bd466cd5a24757ecd7680a541ab` | Mac17,15; macOS build 26A428 |

- Target revision: `667408d494bde191c2cb9dd4f4549827816b9275`;
  config SHA256: `4901707c3cef43e3cf68a0dedc3e39687de43413cab30578a32c63c643984244`.
- Drafter revision: `e052e5b1dd103c38cdd4f8ddc068d7dea4586f55`;
  config SHA256: `5ab0e6e4b2cb052b48d802acc3c095f5ca64d55084ebe8d11986d36b2e9eafeb`.
- Both controllers record identical target/drafter shard, tokenizer and config
  hashes in their raw `binding.json` and `observed.json`; the full model file
  hash maps are also in [assembly-proof.json](assembly-proof.json).
- The seven unchanged frozen driver files are archived in [driver/](driver/).
  Their [SHA256 manifest](driver/SHA256SUMS) matches both controller bindings.
  Original verdict module SHA256:
  `c30d4c902979acaab150440bcf76d95492394e2abe0fe720903cb5f3090881ce`.
- Raw controller [M4](m4/controller/observed.json) / [M5](m5/controller/observed.json)
  receipts, `binding.json`, `run.py`, and completed `1394-full-gate.log` are
  archived verbatim. They preserve the source-bound invocation and original
  exit 3; each gate log's hash matches its controller receipt. These controller
  scripts are historical evidence, **not assembly/replay commands**.
- M5's [bootstrap receipt](m5/controller/bootstrap-observed.json) binds its native
  binary to CF source archive SHA256
  `5c43f984522404e10242ca25f4afc998f8bf6c4a9e505ed7a15ffbb78e564253`.
  Assembly checked these recorded bindings against the packet, not live
  binaries/models or a fresh build.

## All comparisons

Metric: median of the three per-launch `aggregate_tps` samples, in decode tokens/s.
Delta is `(arm median / same-cell plain median - 1) * 100`, not a median of paired
deltas. **Any unrounded negative delta fails**; no tolerance or cross-cell pooling.
Tables round display to six decimal places only; full precision, per-rep samples
and paired diagnostics remain unchanged in each `summary.json`.

### Apple M4 Pro — 12/32 losses, FAIL

Worst: BF16, code, T=1, width 1, **-5.322375682894121%**.

| Prompt | T | Width | Plain median | BF16 median | BF16 delta / strict result | 8-bit median | 8-bit delta / strict result |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| code | 0 | 1 | 176.471779 | 169.692032 | -3.841831% **FAIL** | 177.556641 | +0.614751% PASS |
| code | 0 | 4 | 297.247460 | 292.269037 | -1.674841% **FAIL** | 296.979131 | -0.090271% **FAIL** |
| code | 0 | 8 | 325.301631 | 322.759711 | -0.781404% **FAIL** | 325.344424 | +0.013155% PASS |
| code | 0 | 16 | 360.537666 | 359.093503 | -0.400558% **FAIL** | 360.315934 | -0.061500% **FAIL** |
| code | 1 | 1 | 122.171859 | 115.669414 | -5.322376% **FAIL** | 121.915263 | -0.210029% **FAIL** |
| code | 1 | 4 | 207.489728 | 205.838932 | -0.795604% **FAIL** | 208.841858 | +0.651661% PASS |
| code | 1 | 8 | 236.903082 | 247.210497 | +4.350900% PASS | 248.856404 | +5.045659% PASS |
| code | 1 | 16 | 301.408560 | 303.748464 | +0.776323% PASS | 303.940081 | +0.839897% PASS |
| chat | 0 | 1 | 124.506545 | 124.190966 | -0.253463% **FAIL** | 126.983038 | +1.989047% PASS |
| chat | 0 | 4 | 253.243107 | 253.215899 | -0.010744% **FAIL** | 254.772387 | +0.603879% PASS |
| chat | 0 | 8 | 289.020117 | 289.024672 | +0.001576% PASS | 289.520520 | +0.173138% PASS |
| chat | 0 | 16 | 344.693597 | 346.284433 | +0.461522% PASS | 346.936691 | +0.650750% PASS |
| chat | 1 | 1 | 119.986737 | 124.949788 | +4.136333% PASS | 121.571983 | +1.321184% PASS |
| chat | 1 | 4 | 216.773977 | 219.179060 | +1.109489% PASS | 216.337291 | -0.201448% **FAIL** |
| chat | 1 | 8 | 259.691035 | 261.868219 | +0.838375% PASS | 261.844162 | +0.829111% PASS |
| chat | 1 | 16 | 315.341653 | 317.329789 | +0.630470% PASS | 317.374307 | +0.644588% PASS |

### Apple M5 Ultra — 16/32 losses, FAIL

Worst: 8-bit, code, T=0, width 4, **-1.4341478593879908%**.

| Prompt | T | Width | Plain median | BF16 median | BF16 delta / strict result | 8-bit median | 8-bit delta / strict result |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| code | 0 | 1 | 285.676656 | 288.585667 | +1.018288% PASS | 290.743503 | +1.773630% PASS |
| code | 0 | 4 | 646.799879 | 645.840632 | -0.148307% **FAIL** | 637.523813 | -1.434148% **FAIL** |
| code | 0 | 8 | 855.002490 | 854.667886 | -0.039135% **FAIL** | 853.207706 | -0.209916% **FAIL** |
| code | 0 | 16 | 873.530171 | 872.837177 | -0.079333% **FAIL** | 893.479494 | +2.283759% PASS |
| code | 1 | 1 | 271.984565 | 272.007338 | +0.008373% PASS | 274.819095 | +1.042166% PASS |
| code | 1 | 4 | 553.403244 | 552.807476 | -0.107655% **FAIL** | 552.237769 | -0.210601% **FAIL** |
| code | 1 | 8 | 714.777687 | 713.733706 | -0.146057% **FAIL** | 716.421907 | +0.230032% PASS |
| code | 1 | 16 | 878.078302 | 879.447288 | +0.155907% PASS | 880.427477 | +0.267536% PASS |
| chat | 0 | 1 | 265.561063 | 265.924645 | +0.136911% PASS | 267.688142 | +0.800976% PASS |
| chat | 0 | 4 | 625.639504 | 624.896866 | -0.118701% **FAIL** | 625.446221 | -0.030894% **FAIL** |
| chat | 0 | 8 | 822.731610 | 822.547569 | -0.022370% **FAIL** | 822.557092 | -0.021212% **FAIL** |
| chat | 0 | 16 | 809.380011 | 812.337031 | +0.365344% PASS | 816.620542 | +0.894577% PASS |
| chat | 1 | 1 | 259.795431 | 260.592094 | +0.306650% PASS | 260.453340 | +0.253241% PASS |
| chat | 1 | 4 | 563.508037 | 563.331870 | -0.031263% **FAIL** | 560.732005 | -0.492634% **FAIL** |
| chat | 1 | 8 | 721.988586 | 722.222789 | +0.032439% PASS | 721.133190 | -0.118478% **FAIL** |
| chat | 1 | 16 | 885.648288 | 885.839686 | +0.021611% PASS | 885.564096 | -0.009506% **FAIL** |

## Raw publication and assembly validation

Each [M4 packet](m4/) / [M5 packet](m5/) preserves `metadata.json`, `cells.jsonl`,
`events.jsonl` and `summary.json` **byte-for-byte**. The fifth publishable file,
`requests.jsonl`, is stored as `requests.jsonl.gz`: sequential streaming gzip,
1 MiB chunks, level 9, empty gzip filename, `mtime=0`. Neither request file was
loaded wholesale into RAM. Metadata's explicit `not_publishable: ["private-logs/"]`
is respected; **no `private-logs/` is included**.

The per-device `RAW-SHA256SUMS` manifests describe the five original uncompressed
files. The root [SHA256SUMS](SHA256SUMS) covers every published file except itself,
including compressed requests, raw controllers, frozen driver, this README, and
the machine-readable [assembly proof](assembly-proof.json).

| Device | Original / decompressed request bytes | Original SHA256 = decompressed SHA256 |
| --- | ---: | --- |
| M4 | 37940926 | `5b73b1bcf7d726f01407fa83fe3c86232e711026ed51147df873ec6256cad7d4` |
| M5 | 38817576 | `dc58e91771e32bcf1c68fbbbe6478eb50f48a8f5251ce0d208488bd7e6c4dba3` |

Executed artifact-only validation imported the unchanged archived verdict module
and called `summarize(metadata, cells, events)` for both completed packets. Each
replayed object equals its stored summary after JSON replica-key normalization;
serialization with the **original driver's** `json.dumps(summary, indent=1) + "\n"`
also reproduces the stored summary **byte-for-byte**, including exit 3 and every
loss. The proof records input/output hashes, all 64 comparisons, controller/source/
binary/model/driver bindings, both gzip decompression matches and all 18 clean
shutdown records. No project tests/builds, model/GPU runs, or live-job polling were
performed during assembly.

To reproduce the read-only verdict check from this directory (no serving):

```sh
python3 -B - <<'PY'
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path('driver').resolve()))
from bench_lfm2_dspark_verdict import summarize
for device in ('m4', 'm5'):
    root = Path(device)
    metadata = json.loads((root / 'metadata.json').read_text())
    cells = [json.loads(line) for line in (root / 'cells.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    replay = summarize(metadata, cells, events)
    stored = (root / 'summary.json').read_bytes()
    assert json.loads(json.dumps(replay)) == json.loads(stored)
    assert (json.dumps(replay, indent=1) + '\n').encode() == stored
    print(device, 'unchanged verdict:', replay['verdict'], 'exit:', replay['exit_code'],
          'summary SHA256:', hashlib.sha256(stored).hexdigest())
PY
```
