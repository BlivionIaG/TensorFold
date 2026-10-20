# Interrupted published-source matrix receipts

This is an incomplete measurement, not a completed matrix, balanced quartet, performance qualification, or CONTRIBUTING sign-off. Baseline source is `8649e9858b38e9ef3dcd8d579879c6512a363050`; candidate measured source is `729833615595df66c9f9a438ce9a85061843f010`. Full review-tree equivalence is documented separately at review head `60a8da5243ef313fcab23bdb0d85c790a5cda935`.

87 actual source/raw-data files are retained under `raw/`, including frozen scripts/configuration, source/native/library identities, fixture arrays, request/SSE evidence, historical correctness-attempt archives and logs. `manifest.json` binds each exact original byte count/SHA256 to its public copy. All 90 original files, including three omitted derived Python bytecode caches, remain in the local recovery archive and original inventory. Explicit cache exclusions and their original hashes are listed in the public manifest. The 117,995,520-byte original tar is retained locally, excluded from Git; its SHA256 is `50e8e68c1645f7974fdffe8743c553506396fc766b9a3c725f20dd535170df20`. Every individual file is below 100 MiB.

## Actual retained progress

Only W7800 TP2 drafts OFF ran: first old hold and first new hold finished all four cells; second new hold finished code C1/C8. These 10 complete cells contain 180 requests: 45 warmup and 135 measured. Each complete cell has one warmup plus three measured batches. The eleventh prose-16384/C1 cell is valid JSON with `complete:false`: three completed requests (one warmup plus two measured) survive, preserved separately and excluded from complete-cell comparisons. Total retained request evidence is therefore 183 requests, including 137 measured. Any subsequent in-flight request outcome is unknown. No V620, drafts ON, fourth old hold, or balanced quartet exists in this run. Runtime `state.json` remains incomplete with empty conditions; the analytical index reconstructs only recorded completed client cells without changing runtime state.

The two completed server holds record teardown; the third does not. The host rebooted: current recovery boot ID is `9064ffa5-1d25-4059-a521-5e5d69d03a36`; old controller PIDs and matching native/matrix processes were absent during the read-only recovery observation. This is not a graceful third-hold cleanup receipt and does not identify the reboot cause. No restart was performed.

## Measurement and ownership scope

Exact shapes were 1024 input/512 output code and 16384 input/1024 output prose, C1/C8, T=0, fixed server capacity 8, context 65536, cache/checkpoint/snapshot retention disabled. Complete outputs, exact token counts, finish `length`, cached tokens 0 and full output hashes match across retained requests for each identical fixture. Source/native/config/client/fixture and rank-loaded identities are retained; default admission keys were observed unset for the coordinator and six recorded ranks. Recorded health capacity is 8.

`state.json` records the existing canonical GPU lock held by the coordinator. Owned model groups were serial on assigned HIP0/1; the planned V620 stage never began. Other shared-host activity was permitted, not physical-host isolation. Per-rank memory snapshots actually retained are five in old hold 1, five in new hold 2, and three in unfinished new hold 3. VmHWM is cumulative process high-water memory, including startup/warmup; RSS/PSS/VRAM observations are samples, not complete physical-footprint or pinned-memory peaks.

Raw client/server TTFT, server prefill elapsed, server decode elapsed, delivery TPOT, chunk gaps and aggregate rates are retained. Prefill elapsed is not pure GPU compute; mean TPOT/chunk gaps are not a true per-token ITL distribution. Weighted service rates and wall-time aggregate rates have different denominators. Three measured batch values and hold means/medians are in the reductions. First old/new pair comparisons are unbalanced; no significance or acceptance threshold is inferred. Historical correctness parser failures copied by the runner remain historical failures, not failures of these benchmark requests.

## Reproduce the offline reduction

From this directory:

```sh
python3 tools/independent_matrix_measurement_reduce.py raw/matrix_published729_1/runs/20261020T213915 --partial --derived-index audits/public_analytical_complete_cell_index.json --output reduction.local.json
```

Original raw-file hash references inside runtime records are deliberately preserved. If privacy changed a referenced metadata file, use `manifest.json` to map its original hash to public bytes before checking that reference. Benchmark cell files (including all request/SSE content) are byte-identical to recovered originals. The reducer checks completed-cell hashes, which remain unchanged. `audits/recovery_manifest.json` is the original recovery inventory, not a second public-byte manifest.

Only exact observed private home/recovery-path prefixes were replaced in metadata/logs/tools. Numerical values, timestamps, token arrays, request bodies, emitted output text, SSE data and historical source identities were preserved. Raw local originals remain available; no native executables or weight files are included. Three derived Python bytecode caches are excluded from publication; corresponding Python sources are retained. All original caches remain unchanged in the local archive.
