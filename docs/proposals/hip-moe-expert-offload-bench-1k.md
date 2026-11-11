Vega56 TB4 expert offload, read-through admission (every miss admitted by the stream tile) + prompt-span SDMA staging; 1.5 GiB window, 2334 cache lines (3.85 GiB) + 256-expert scratch, par 8, ctx 2048

greedy, 1 round a cell, per-request figures are mean / p50 / p90 over the requests of all rounds. serve_bench filler prompts (salted, not natural text: hit rate is lower than on prose).

### 1024 input / 512 output tokens

| c | prompt tok | prefill tok/s agg | prefill tok/s per request | decode tok/s agg | decode tok/s per request | total tok/s agg | TTFT s | ITL ms | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 153.9 | 153.9 / 153.9 / 153.9 | 12.1 | 12.1 / 12.1 / 12.1 | 10.5 | 6.65 / 6.65 / 6.65 | 107.4 / 82.2 / 172.6 |  |
| 2 | 1024 | 120.4 | 107.3 / 107.3 / 144.9 | 16.2 | 8.9 / 8.9 / 9.5 | 14.7 | 11.82 / 11.82 / 15.98 | 143.9 / 104.0 / 210.8 |  |
| 4 | 1024 | 108.2 | 69.9 / 49.0 / 126.4 | 20.4 | 6.2 / 6.1 / 7.1 | 19.2 | 22.17 / 22.11 / 34.67 | 207.9 / 136.9 / 274.4 |  |
| 8 | 1024 | 102.7 | 42.7 / 24.2 / 88.3 | 12.6 | 1.8 / 1.8 / 2.0 | 12.4 | 45.10 / 43.01 / 76.12 | 708.4 / 498.1 / 1178.8 |  |

Link (server `experts interval` lines): c=4 hit 86.7 %, 42.6 misses/token, 71.8 MiB/token; c=8 (decode rounds 2015 and 2039 tokens) hit 79.0 -> 67.4 %, 67.7 -> 104.9 misses/token, 114 -> 177 MiB/token over the link.

Before (commit 579a6ffea7, COPIES=32, prompt spans read the window in place):

| c | prefill tok/s agg | decode tok/s agg | TTFT s | ITL ms mean / p50 |
| --- | --- | --- | --- | --- |
| 1 | 30.0 | 11.0 | 34.11 | 119.9 / 90.8 |
| 2 | 27.7 | 8.6 | 51.99 | 241.0 / 150.8 |
| 4 | 25.3 | 8.1 | 96.01 | 457.3 / 242.1 |
| 8 | 24.0 | 7.9 | 193.65 | 883.0 / 419.4 |

Mac alone (mlx_lm.server, M4 Pro, same load): prefill 940-1041 tok/s, decode 72.6 / 117.7 / 161.4 / 178.4 tok/s agg at c = 1 / 2 / 4 / 8, TTFT 1.1-6.9 s.

Story prompts (natural text, `exp.sh`): c=1 decode 20.9 tok/s before -> 26-28.7 after (hit 88 -> 91 %, 44 -> 27 misses/token, ~37 ms/token).
