# LFM2.5-24B-A2B 4-bit, Apple M4 Pro 20-core GPU 48 GB, macOS 26.5.1

Builds: before = 5a6ee034 (#823 head), tip = fd2f8e27 (chain tip #821-#873): bin-egate, mlx-lm 0.32.0 / mlx 0.32.3.

Machine was in interactive use by the user during the whole run (see load/top in progress.log); another agent also built and ran tf-lfm2-exact between steps. Served mlx r1 (12:33-12:36) OVERLAPPED one 8-bit tf-lfm2-exact run by another agent (per coordinator) and is disturbed; mlx r1rerun repeats it at the end.

## Served (port 8091, --context 131072, --prompt-cache-gib 0)

| build | round | decode code tok/s | decode chat tok/s | 4-stream code total / steady / slowest TTFT | 4-stream chat total / steady / slowest TTFT | bench_cache 32k TTFT s | peak footprint GB |
|---|---|---|---|---|---|---|---|
| before | r1 | 125.7 | 122.8 | 190.7 / 200.1 / 0.34 | 186.9 / 198.8 / 0.43 | 33.68 | 16.7 |
| before | r2 | 125.0 | 121.9 | 190.4 / 199.7 / 0.34 | 186.7 / 198.6 / 0.43 | 32.31 | 15.5 |
| tip | r1 | 135.6 | 122.8 | 237.4 / 253.6 / 0.34 | 229.1 / 247.1 / 0.43 | 33.12 | 16.2 |
| tip | r2 | 135.9 | 123.0 | 240.9 / 256.3 / 0.34 | 226.5 / 244.4 / 0.43 | 32.08 | 15.0 |
| mlx | r1 | 78.7 | 310.6 | 90.2 / None / 8.81 | 113.1 / None / 10.19 | - | 13.1 |
| mlx | r2 | 97.2 | 321.6 | 95.0 / None / 8.17 | 113.7 / None / 10.12 | - | 13.1 |
| mlx | r1rerun | 92.0 | 303.1 | 90.2 / None / 8.68 | 107.6 / None / 10.7 | - | 13.1 |

## Served cold prefill tok/s (token arrays)

| build | round | 2048 | 8192 | 32768 | 65536 | 131000 |
|---|---|---|---|---|---|---|
| before | r1 | 1269.0, 1304.5 | 1238.1, 1238.4 | 1035.7 | 853.3 | 600.8 |
| before | r2 | 1248.5, 1110.6 | 1172.6, 1203.8 | 1030.7 | 853.6 | - |
| tip | r1 | 1286.9, 1309.7 | 1165.6, 1199.2 | 1027.9 | 854.9 | 606.8 |
| tip | r2 | 1292.2, 1137.8 | 1193.7, 1230.3 | 1041.8 | 854.6 | - |

## Draft (code prompt, bench_openai 256 tokens, per request rounds / accepted / drafted)

| build | round | code | chat |
|---|---|---|---|
| before | r1 | 204 / 51 / 90; 204 / 51 / 90; 204 / 51 / 90; 204 / 51 / 90 | 255 / 0 / 0; 255 / 0 / 0; 255 / 0 / 0; 255 / 0 / 0 |
| before | r2 | 204 / 51 / 90; 204 / 51 / 90; 204 / 51 / 90; 204 / 51 / 90 | 255 / 0 / 0; 255 / 0 / 0; 255 / 0 / 0; 255 / 0 / 0 |
| tip | r1 | 204 / 51 / 90; 204 / 51 / 90; 204 / 51 / 90; 204 / 51 / 90 | 255 / 0 / 0; 255 / 0 / 0; 255 / 0 / 0; 255 / 0 / 0 |
| tip | r2 | 204 / 51 / 90; 204 / 51 / 90; 204 / 51 / 90; 204 / 51 / 90 | 255 / 0 / 0; 255 / 0 / 0; 255 / 0 / 0; 255 / 0 / 0 |

## Window ms by width (load line)

| build | session | line |
|---|---|---|
| before | served r1 | 1 8.2, 2 12.0, 4 19.7, 8 35.5, 16 67.7 |
| before | served r2 | 1 8.1, 2 11.9, 4 19.8, 8 35.4, 16 67.5 |
| before | window round 1 | 1 8.3, 2 12.2, 4 20.0, 8 36.1, 16 82.1 |
| before | window round 2 | 1 8.3, 2 12.0, 4 20.2, 8 36.2, 16 69.1 |
| before | window round 3 | 1 8.0, 2 11.8, 4 19.7, 8 35.4, 16 68.6 |
| tip | served r1 | 1 8.0, 2 10.6, 4 16.0, 8 27.0, 16 48.8 |
| tip | served r2 | 1 7.9, 2 10.6, 4 15.9, 8 27.0, 16 42.6 |
| tip | window round 1 | 1 7.9, 2 11.0, 4 16.3, 8 27.3, 16 48.3 |
| tip | window round 2 | 1 8.1, 2 10.9, 4 16.2, 8 27.5, 16 43.3 |
| tip | window round 3 | 1 8.2, 2 10.9, 4 16.3, 8 28.5, 16 44.6 |

## In process (tf-lfm2-bench / mlx_cold.py), r1, r2

| length | build | prompt tok/s | decode tok/s |
|---|---|---|---|
| 2048 | before | [1290.0, 1204.0] | [119.6, 117.8] |
| 2048 | tip | [1298.0, 1148.0] | [120.9, 117.5] |
| 2048 | mlx | [1299.2, 1303.9] | [111.2, 111.4] |
| 8192 | before | [1253.0, 1176.0] | [110.1, 110.5] |
| 8192 | tip | [1280.0, 1276.0] | [111.0, 110.3] |
| 8192 | mlx | [1264.5, 1266.1] | [102.2, 102.6] |
| 32768 | before | [1064.0, 1065.0] | [84.8, 85.2] |
| 32768 | tip | [1068.0, 1060.0] | [85.1, 84.9] |
| 32768 | mlx | [1063.0, 1061.9] | [75.2, 76.5] |
| 65536 | before | [766.0, 839.0] | [64.0, 62.9] |
| 65536 | tip | [820.0, 825.0] | [64.9, 64.1] |
| 65536 | mlx | [848.9, 846.5] | [56.7, 55.5] |

## 131000 in process (/usr/bin/time -l)

| build | prompt tok/s | prompt s | decode tok/s | peak footprint GB (time -l) | wall s |
|---|---|---|---|---|---|
| before | 602.0 | 217.46 | 42.6 | 20.23 | 221.34 |
| tip | 592.0 | 221.44 | 43.6 | 16.83 | 226.24 |
| mlx | 574.1 | 228.2 | 36.7 | 19.85 | 233.41 |

## Two-turn resume (prompt cache on)

| build | turn1 TTFT s | turn2 TTFT s | turn2 again TTFT s | turn2 usage | turn2 == again |
|---|---|---|---|---|---|
| tip | 6.564 | 0.222 | 0.062 | {"prompt_tokens": 8009, "completion_tokens": 128, "total_tokens": 8137, "prompt_tokens_details": {"cached_tokens": 7941}, "completion_tokens_details": {"reasoning_tokens": 128}} | True |
| mlx | 6.484 | 0.169 | 0.04 | {} | True |

tip server: prompt=7945 cached=0 tokens=48; prompt=8009 cached=7941 tokens=128; prompt=8009 cached=8005 tokens=128

## token_sha (server log), before vs tip

| request | per session | before == tip |
|---|---|---|
| chat prompt=29 tokens=16 | before_r1: 9f0898be6523; before_r2: 9f0898be6523; tip_r1: 9f0898be6523; tip_r2: 9f0898be6523 | True |
| chat prompt=29 tokens=256 | before_r1: 57752ac6d7f0; before_r2: 57752ac6d7f0; tip_r1: 57752ac6d7f0; tip_r2: 57752ac6d7f0 | True |
| completion prompt=131000 tokens=1 | before_r1: 43c727ee4fc7; tip_r1: 43c727ee4fc7 | True |
| completion prompt=14 tokens=16 | before_r1: 5682c4819504; before_r2: 5682c4819504; tip_r1: 5682c4819504; tip_r2: 5682c4819504 | True |
| completion prompt=14 tokens=256 | before_r1: f66e3fd7e509; before_r2: f66e3fd7e509; tip_r1: f66e3fd7e509; tip_r2: f66e3fd7e509 | True |
| completion prompt=2048 tokens=1 | before_r1: 08490295488a; before_r2: 08490295488a; tip_r1: 08490295488a; tip_r2: 08490295488a | True |
| completion prompt=32768 tokens=1 | before_r1: a6d2a96d80c0; before_r2: a6d2a96d80c0; tip_r1: a6d2a96d80c0; tip_r2: a6d2a96d80c0 | True |
| completion prompt=32768 tokens=64 | before_r1: 6627de0e2fca; before_r2: 6627de0e2fca; tip_r1: 6627de0e2fca; tip_r2: 6627de0e2fca | True |
| completion prompt=65536 tokens=1 | before_r1: c52a05c88e7d; before_r2: c52a05c88e7d; tip_r1: c52a05c88e7d; tip_r2: c52a05c88e7d | True |
| completion prompt=8192 tokens=1 | before_r1: 08490295488a; before_r2: 08490295488a; tip_r1: 08490295488a; tip_r2: 08490295488a | True |

## Concurrent == solo (bench_concurrent --alone --serial)

| session | alone equal / unequal / failed | serial equal / unequal / failed |
|---|---|---|
| before_r1 | 30 / 0 / 0 | 4 / 0 / 0 |
| before_r2 | 30 / 0 / 0 | 4 / 0 / 0 |
| tip_r1 | 30 / 0 / 0 | 4 / 0 / 0 |
| tip_r2 | 30 / 0 / 0 | 4 / 0 / 0 |
| mlx_r1 | 0 / 0 / 30 | 0 / 0 / 4 |
| mlx_r2 | 0 / 0 / 30 | 0 / 0 / 4 |
| mlx_r1rerun | 0 / 0 / 30 | 0 / 0 / 4 |

## Notes

- mlx-lm chat decode (~300+ tok/s) comes with a ~1.9 s TTFT (decode_mlx_*.txt): the rate is measured after the first streamed token, so it is not comparable to TensorFold's chat decode; the code (raw completion) row is the comparable one.
- mlx concurrent 'failed' counts mean the equality check could not run (mlx-lm returns no token_sha; output_sha256 only), not that outputs differed.
- Gate (egate/progress.log, copied as gate_progress.log): host/build/tile exit 0, exact 4-bit 68 passed 0 failed; exact 6-bit and 8-bit crashed (exit 139) in the gate run; the coordinator released the HOLD and the run proceeded on the unchanged fd2f8e27 binaries (binaries_sha256_start/end.txt identical).
- In-process mlx uses mlx_cold.py (mlx-lm default prefill step); decode lengths match TensorFold (128 at 2k/8k, 64 at 32k/64k/131k).
