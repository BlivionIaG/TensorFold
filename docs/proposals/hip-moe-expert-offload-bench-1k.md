Vega56 TB4 expert offload (pageable host tier + 1.5 GiB window + 4.5 GiB VRAM cache), COPIES=32, par 8, ctx 2048

greedy, 1 rounds a cell, per-request figures are mean / p50 / p90 over the requests of all rounds.

### 1024 input / 512 output tokens

| c | prompt tok | prefill tok/s agg | prefill tok/s per request | decode tok/s agg | decode tok/s per request | total tok/s agg | TTFT s | ITL ms | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 30.0 | 30.0 / 30.0 / 30.0 | 11.0 | 11.0 / 11.0 / 11.0 | 6.4 | 34.11 / 34.11 / 34.11 | 119.9 / 90.8 / 193.4 |  |
| 2 | 1024 | 27.7 | 23.9 / 23.9 / 32.0 | 8.6 | 5.6 / 5.6 / 6.6 | 6.9 | 51.99 / 51.99 / 69.50 | 241.0 / 150.8 / 293.1 |  |
| 4 | 1024 | 25.3 | 15.7 / 11.3 / 28.0 | 8.1 | 3.0 / 2.8 / 4.0 | 7.3 | 96.01 / 95.85 / 148.90 | 457.3 / 242.1 / 501.2 |  |
| 8 | 1024 | 24.0 | 9.7 / 5.6 / 19.9 | 7.9 | 1.6 / 1.4 / 2.3 | 7.5 | 193.65 / 184.50 / 326.13 | 883.0 / 419.4 / 879.6 |  |
