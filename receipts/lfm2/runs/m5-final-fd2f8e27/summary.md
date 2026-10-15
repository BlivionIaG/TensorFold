# LFM2.5-24B-A2B 4-bit on Apple M5 Ultra: final benchmark

Apple M5 Ultra, 64-core GPU, 256 GB, macOS 27.0

Builds:
- **before**: #823 head 5a6ee034 (no cheap windows, no tensor units)
- **on**: final chain tip fd2f8e27 (#821-#873 incl. exact 4-bit decode), tensor units on
- **off**: fd2f8e27 with tensor units switched off (local test build, never committed; M1-M4 path on the M5)
- **mlx**: mlx-lm 0.32.0 / mlx 0.32.3 (mlx_lm.server / stream_generate)

## Served (r1 / r2)

| metric | before | on | off | mlx |
|---|---|---|---|---|
| decode code tok/s | 290.9 / 290.2 | 278.8 / 280.5 | 299.5 / 298.3 | 181.2 / 182.0 |
| decode chat tok/s | 261.0 / 260.1 | 263.5 / 264.4 | 265.0 / 264.3 | 508.1 / 511.0 |
| 4-stream code total | 490.1 / 489.8 | 557.0 / 556.4 | 555.0 / 554.3 | 177.2 / 177.9 |
| 4-stream code steady | 509.2 / 509.0 | 573.2 / 572.6 | 580.9 / 580.1 | None / None |
| 4-stream code slowest TTFT s | 0.11 / 0.11 | 0.07 / 0.07 | 0.11 / 0.11 | 4.38 / 4.37 |
| 4-stream chat total | 456.6 / 456.7 | 536.2 / 535.1 | 523.3 / 522.9 | 210.1 / 211.8 |
| 4-stream chat steady | 477.9 / 478.0 | 552.7 / 551.6 | 551.5 / 551.0 | None / None |
| 4-stream chat slowest TTFT s | 0.13 / 0.13 | 0.08 / 0.08 | 0.13 / 0.13 | 5.32 / 5.28 |
| bench_cache 32k cold TTFT s | 8.85 / 8.85 | 3.06 / 3.062 | 8.852 / 8.847 | - / - |
| peak footprint GB | 16.9 / 15.6 | 16.4 / 15.2 | 16.4 / 15.2 | 13.1 / 13.1 |
| prefill 2048 tok/s | [4526.0, 4667.0] / [4521.0, 4667.0] | [8335.0, 13655.0] / [12583.0, 13676.0] | [4438.0, 4656.0] / [4521.0, 4662.0] | - / - |
| prefill 8192 tok/s | [4428.0, 4427.0] / [4426.0, 4428.0] | [13031.0, 13029.0] / [13040.0, 13028.0] | [4430.0, 4430.0] / [4430.0, 4429.0] | - / - |
| prefill 32768 tok/s | [3707.0] / [3707.0] | [10725.0] / [10730.0] | [3707.0] / [3707.0] | - / - |
| prefill 65536 tok/s | [3046.0] / [3046.0] | [8636.0] / [8639.0] | [3047.0] / [3047.0] | - / - |
| prefill 131000 tok/s | [2243.0] / - | [6199.0] / - | [2243.0] / - | - / - |
| window ms width 1 | 3.8 / 3.8 | 3.8 / 3.8 | 3.7 / 3.7 | - / - |
| window ms width 2 | 5.2 / 5.1 | 4.6 / 4.6 | 4.6 / 4.6 | - / - |
| window ms width 4 | 7.4 / 7.3 | 6.2 / 6.2 | 6.2 / 6.2 | - / - |
| window ms width 8 | 11.4 / 11.4 | 10.4 / 10.4 | 10.4 / 10.4 | - / - |
| window ms width 16 | 18.7 / 18.7 | 18.9 / 18.9 | 18.7 / 18.6 | - / - |

## Draft rounds, code prompt (server log, requests in order, run-length)

| build | round | requests | tokens | rounds | accepted | drafted | sha |
|---|---|---|---|---|---|---|---|
| before | r1 | 4 | 256 | 204 | 51 | 90 | f66e3fd7e509 |
| before | r1 | 1 | 16 | 15 | 0 | 0 | 5682c4819504 |
| before | r1 | 1 | 256 | 204 | 51 | 90 | f66e3fd7e509 |
| before | r1 | 1 | 256 | 255 | 0 | 0 | f66e3fd7e509 |
| before | r1 | 3 | 256 | 204 | 51 | 90 | f66e3fd7e509 |
| before | r1 | 12 | 256 | 215 | 40 | 51 | f66e3fd7e509 |
| before | r2 | 4 | 256 | 204 | 51 | 90 | f66e3fd7e509 |
| before | r2 | 1 | 16 | 15 | 0 | 0 | 5682c4819504 |
| before | r2 | 1 | 256 | 204 | 51 | 90 | f66e3fd7e509 |
| before | r2 | 1 | 256 | 255 | 0 | 0 | f66e3fd7e509 |
| before | r2 | 3 | 256 | 204 | 51 | 90 | f66e3fd7e509 |
| before | r2 | 12 | 256 | 215 | 40 | 51 | f66e3fd7e509 |
| on | r1 | 4 | 256 | 216 | 39 | 100 | dcc30ceea29d |
| on | r1 | 1 | 16 | 15 | 0 | 0 | a3362e082033 |
| on | r1 | 1 | 256 | 216 | 39 | 100 | dcc30ceea29d |
| on | r1 | 1 | 256 | 255 | 0 | 0 | dcc30ceea29d |
| on | r1 | 3 | 256 | 216 | 39 | 100 | dcc30ceea29d |
| on | r1 | 2 | 256 | 223 | 32 | 43 | dcc30ceea29d |
| on | r1 | 1 | 256 | 223 | 32 | 42 | dcc30ceea29d |
| on | r1 | 1 | 256 | 224 | 31 | 48 | dcc30ceea29d |
| on | r1 | 1 | 256 | 223 | 32 | 43 | dcc30ceea29d |
| on | r1 | 1 | 256 | 223 | 32 | 42 | dcc30ceea29d |
| on | r1 | 1 | 256 | 223 | 32 | 43 | dcc30ceea29d |
| on | r1 | 1 | 256 | 224 | 31 | 48 | dcc30ceea29d |
| on | r1 | 2 | 256 | 223 | 32 | 43 | dcc30ceea29d |
| on | r1 | 1 | 256 | 223 | 32 | 42 | dcc30ceea29d |
| on | r1 | 1 | 256 | 224 | 31 | 48 | dcc30ceea29d |
| on | r2 | 4 | 256 | 216 | 39 | 100 | dcc30ceea29d |
| on | r2 | 1 | 16 | 15 | 0 | 0 | a3362e082033 |
| on | r2 | 1 | 256 | 216 | 39 | 100 | dcc30ceea29d |
| on | r2 | 1 | 256 | 255 | 0 | 0 | dcc30ceea29d |
| on | r2 | 3 | 256 | 216 | 39 | 100 | dcc30ceea29d |
| on | r2 | 2 | 256 | 223 | 32 | 43 | dcc30ceea29d |
| on | r2 | 1 | 256 | 223 | 32 | 42 | dcc30ceea29d |
| on | r2 | 1 | 256 | 224 | 31 | 48 | dcc30ceea29d |
| on | r2 | 2 | 256 | 223 | 32 | 43 | dcc30ceea29d |
| on | r2 | 1 | 256 | 223 | 32 | 42 | dcc30ceea29d |
| on | r2 | 1 | 256 | 224 | 31 | 48 | dcc30ceea29d |
| on | r2 | 2 | 256 | 223 | 32 | 43 | dcc30ceea29d |
| on | r2 | 1 | 256 | 223 | 32 | 42 | dcc30ceea29d |
| on | r2 | 1 | 256 | 224 | 31 | 48 | dcc30ceea29d |
| off | r1 | 4 | 256 | 204 | 51 | 90 | f66e3fd7e509 |
| off | r1 | 1 | 16 | 15 | 0 | 0 | 5682c4819504 |
| off | r1 | 1 | 256 | 204 | 51 | 90 | f66e3fd7e509 |
| off | r1 | 1 | 256 | 255 | 0 | 0 | f66e3fd7e509 |
| off | r1 | 3 | 256 | 204 | 51 | 90 | f66e3fd7e509 |
| off | r1 | 2 | 256 | 215 | 40 | 52 | f66e3fd7e509 |
| off | r1 | 1 | 256 | 215 | 40 | 51 | f66e3fd7e509 |
| off | r1 | 1 | 256 | 216 | 39 | 54 | f66e3fd7e509 |
| off | r1 | 2 | 256 | 215 | 40 | 52 | f66e3fd7e509 |
| off | r1 | 1 | 256 | 215 | 40 | 51 | f66e3fd7e509 |
| off | r1 | 1 | 256 | 216 | 39 | 54 | f66e3fd7e509 |
| off | r1 | 2 | 256 | 215 | 40 | 52 | f66e3fd7e509 |
| off | r1 | 1 | 256 | 215 | 40 | 51 | f66e3fd7e509 |
| off | r1 | 1 | 256 | 216 | 39 | 54 | f66e3fd7e509 |
| off | r2 | 4 | 256 | 204 | 51 | 90 | f66e3fd7e509 |
| off | r2 | 1 | 16 | 15 | 0 | 0 | 5682c4819504 |
| off | r2 | 1 | 256 | 204 | 51 | 90 | f66e3fd7e509 |
| off | r2 | 1 | 256 | 255 | 0 | 0 | f66e3fd7e509 |
| off | r2 | 3 | 256 | 204 | 51 | 90 | f66e3fd7e509 |
| off | r2 | 2 | 256 | 215 | 40 | 52 | f66e3fd7e509 |
| off | r2 | 1 | 256 | 215 | 40 | 51 | f66e3fd7e509 |
| off | r2 | 1 | 256 | 216 | 39 | 54 | f66e3fd7e509 |
| off | r2 | 2 | 256 | 215 | 40 | 52 | f66e3fd7e509 |
| off | r2 | 1 | 256 | 215 | 40 | 51 | f66e3fd7e509 |
| off | r2 | 1 | 256 | 216 | 39 | 54 | f66e3fd7e509 |
| off | r2 | 2 | 256 | 215 | 40 | 52 | f66e3fd7e509 |
| off | r2 | 1 | 256 | 215 | 40 | 51 | f66e3fd7e509 |
| off | r2 | 1 | 256 | 216 | 39 | 54 | f66e3fd7e509 |

## In process (r1, r2)

| length | metric | before | on | off | mlx |
|---|---|---|---|---|---|
| 2048 | prompt_tps | 4489.0, 4482.0 | 12831.0, 12760.0 | 4542.0, 4540.0 | 12128.4, 12137.3 |
| 2048 | decode_tps | 256.2, 257.1 | 262.2, 260.2 | 262.0, 262.0 | 204.1, 204.2 |
| 8192 | prompt_tps | 4462.0, 4459.0 | 13952.0, 13946.0 | 4565.0, 4562.0 | 13035.8, 13009.1 |
| 8192 | decode_tps | 247.0, 247.1 | 253.1, 252.6 | 251.3, 250.6 | 198.6, 198.1 |
| 32768 | prompt_tps | 3829.0, 3828.0 | 11636.0, 11638.0 | 3835.0, 3834.0 | 11617.7, 11610.2 |
| 32768 | decode_tps | 195.8, 196.5 | 201.1, 200.2 | 199.7, 199.4 | 168.4, 168.4 |
| 65536 | prompt_tps | 3137.0, 3137.0 | 9250.0, 9250.0 | 3134.0, 3134.0 | 9805.0, 9816.0 |
| 65536 | decode_tps | 157.9, 157.7 | 160.0, 160.8 | 159.2, 159.1 | 145.5, 144.8 |

## In process 131,000 tokens (/usr/bin/time -l)

| build | prompt tok/s | decode tok/s | peak GB |
|---|---|---|---|
| before | 2292.0 | 115.7 | 21.93 |
| on | 6511.0 | 117.5 | 18.2 |
| off | 2290.0 | 116.6 | 18.23 |
| mlx | 7234.2 | 113.0 | 21.32 |

## Two-turn resume (cache on)

| server | turn | TTFT s | prompt tokens | cached tokens | processed (mlx log) |
|---|---|---|---|---|---|
| on | turn1 | 0.686 | 7991 | 0 | - |
| on | turn2 | 0.061 | 8055 | 7987 | - |
| on | turn2_again | 0.024 | 8055 | 8051 | - |
| mlx | turn1 | 0.698 | None | None | 7991 |
| mlx | turn2 | 0.065 | None | None | 65 |
| mlx | turn2_again | 0.027 | None | None | 1 |

## token_sha

| build | code | chat | cache32k |
|---|---|---|---|
| before | f66e3fd7e509 | 57752ac6d7f0 | 6627de0e2fca |
| on | dcc30ceea29d | ecb946063060 | acd378159058 |
| off | f66e3fd7e509 | 57752ac6d7f0 | 6627de0e2fca |
| mlx | None | None |  |

## Concurrent == solo (alone equal/unequal, serial equal/unequal)

| build | round | cell | alone | serial |
|---|---|---|---|---|
| before | r1 | code_x1 | 3/0 | 1/0 |
| before | r1 | code_x4 | 12/0 | 1/0 |
| before | r1 | chat_x1 | 3/0 | 1/0 |
| before | r1 | chat_x4 | 12/0 | 1/0 |
| before | r2 | code_x1 | 3/0 | 1/0 |
| before | r2 | code_x4 | 12/0 | 1/0 |
| before | r2 | chat_x1 | 3/0 | 1/0 |
| before | r2 | chat_x4 | 12/0 | 1/0 |
| on | r1 | code_x1 | 3/0 | 1/0 |
| on | r1 | code_x4 | 12/0 | 1/0 |
| on | r1 | chat_x1 | 3/0 | 1/0 |
| on | r1 | chat_x4 | 12/0 | 1/0 |
| on | r2 | code_x1 | 3/0 | 1/0 |
| on | r2 | code_x4 | 12/0 | 1/0 |
| on | r2 | chat_x1 | 3/0 | 1/0 |
| on | r2 | chat_x4 | 12/0 | 1/0 |
| off | r1 | code_x1 | 3/0 | 1/0 |
| off | r1 | code_x4 | 12/0 | 1/0 |
| off | r1 | chat_x1 | 3/0 | 1/0 |
| off | r1 | chat_x4 | 12/0 | 1/0 |
| off | r2 | code_x1 | 3/0 | 1/0 |
| off | r2 | code_x4 | 12/0 | 1/0 |
| off | r2 | chat_x1 | 3/0 | 1/0 |
| off | r2 | chat_x4 | 12/0 | 1/0 |
| mlx | r1 | code_x1 | 0/0 | 0/0 |
| mlx | r1 | code_x4 | 0/0 | 0/0 |
| mlx | r1 | chat_x1 | 0/0 | 0/0 |
| mlx | r1 | chat_x4 | 0/0 | 0/0 |
| mlx | r2 | code_x1 | 0/0 | 0/0 |
| mlx | r2 | code_x4 | 0/0 | 0/0 |
| mlx | r2 | chat_x1 | 0/0 | 0/0 |
| mlx | r2 | chat_x4 | 0/0 | 0/0 |
