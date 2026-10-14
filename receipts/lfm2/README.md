# LFM2.5-24B-A2B receipts: TensorFold and mlx-lm

## Conditions

- Machine: Apple M4 Pro (20-core GPU), 48 GB, macOS 26.5.1. The Mac is a laptop on AC power.
- TensorFold: branch `lfm2-metal` (BlivionIaG/TensorFold). Current run: `c88bd868`, in `runs/c88bd868/`. The first run (`87792521`) is in `runs/87792521/`.
- mlx 0.32.3, mlx-lm 0.32.0.
- Checkpoints (LiquidAI, Hugging Face): 4-bit `667408d494bde191c2cb9dd4f4549827816b9275`, 6-bit `1deec53eb49571e21cdcbf82fd5839248f07ed88`, 8-bit `67ab417dd02d0c9545802a726df668620eae049d`.
- One model process ran at a time. Each run's `versions.txt` records the versions.
- `scripts/receipts_lfm2.sh` ran all steps. Each TensorFold run in process alternates with an mlx-lm run on the same prompt. The token arrays are in `tokens/`.

## Files in a run directory

| Step | Files |
| --- | --- |
| TensorFold server, `--prompt-cache-gib 0`: cold prompts, decode, streams | `tf_prefill.json`, `tf_decode.json`, `tf_concurrent.json`, `tf_cold.log`, `tf_cold_footprint.txt` |
| TensorFold server, prompt cache on: two turns | `tf_two_turn.json`, `tf_two_turn_cold.json` (cache off), `tf_cache_footprint.txt` |
| mlx-lm server: decode, streams, two turns | `mlx_decode.json`, `mlx_concurrent.json`, `mlx_two_turn.json`, `mlx_footprint.txt` |
| In process, alternated, 2k and 8k, 2 rounds | `alt_tf_2k_r*.txt`, `alt_tf_8k_r*.txt`, `alt_mlx_r*.txt` |
| In process, alternated, 32k and 64k, 2 rounds; 131k | `tf_32k_r*.txt`, `mlx_32k_r*.txt`, `tf_64k_r*.txt`, `mlx_64k_r*.txt`, `tf_131k.txt`, `mlx_131k.txt` (c88bd868 only) |
| TensorFold at 131k under `/usr/bin/time -l`, memory only | `tf_131k_mem.txt` (c88bd868 only) |
| mlx-lm at 32k timed by part | `mlx_parts_32k.json` (c88bd868 only) |
| In process, 6-bit and 8-bit | `tf_6bit_*.txt`, `tf_8bit_*.txt`, `mlx_6bit.txt`, `mlx_8bit.txt` |
| Drafted replies against `"draft": false` (server `serve MODEL --port 8091 --context 32768`, `scripts/draft_check.py`) | `tf_draft_check.log` (c88bd868 only): each request's `sha` and drafts accepted |
| Teacher-forced fidelity against mlx-lm | `fidelity_{4,6,8}bit/fidelity.json` (the logit dumps are not included) |

`tile_qmm/` holds the matrices of `tf-tile-check --dump` and `tile_qmm_reference.json` (error against float64, tile_qmm and mlx).

## Results, 4-bit, c88bd868

Cold prompt in process, alternated rounds, tok/s:

| Prompt | TensorFold | mlx-lm |
| --- | --- | --- |
| 2,048 | 1,310 / 1,304 | 1,311 / 1,321 |
| 8,192 | 1,255 / 1,257 | 1,289 / 1,285 |
| 32,768 | 1,054 / 995 | 1,078 / 1,019 |
| 65,536 | 867 / 779 | 884 / 842 |
| 131,000 | 607 | 610 |

Decode, tok/s:

| Case | TensorFold | mlx-lm |
| --- | --- | --- |
| Served, 256 tokens, code | 129.1 | 102.9 |
| Served, 4 streams, total, code / chat | 192.2 / 189.1 | 100.5 / 120.3 |
| In process, after 2,048 tokens | 124.5 / 124.6 | 115.8 / 115.8 |
| In process, after 32,768 tokens | 87.7 / 86.1 | 79.5 / 79.1 |
| In process, after 131,000 tokens | 41.3 | 38.8 |

- Each TensorFold reply at 4 streams was equal to its solo run (12 of 12, code and chat).
- Turn two of a chat with a 7,991-token history: TensorFold 0.22 s with 7,987 cached tokens, mlx-lm 0.17 s. The TensorFold reply with the cache on was equal to the reply with the cache off.
- Peak memory footprint in process at 131,000 tokens (`/usr/bin/time -l`): TensorFold 17.7 GB (`tf_131k_mem.txt`), mlx-lm 21.2 GB (`mlx_131k.txt`). The TensorFold memory run ran while host tests used the machine, so its speed is not a receipt.
- Peak physical footprint of the servers (`vmmap`). These runs are not like for like: TensorFold, cache off, after cold prompts up to 131,000 tokens: 16.7 GB. TensorFold, cache on, two turns at 8k: 14.8 GB. mlx-lm, decode, streams and two turns, no prompt over 8,055 tokens: 14.5 GB.

The mlx-lm server sends the thinking text at the end of a chat reply. Thus its chat decode rate in `mlx_decode.json` is not comparable.

## Served decode with drafts off (c88bd868, a second session)

`runs/c88bd868-drafts-off/`: the same build and `bench_openai.py` command, `--prompt-cache-gib 0`. Order: TensorFold drafts off, on, off, on (`--no-drafts` for off), then mlx-lm, then mlx-lm again, then TensorFold drafts off.

| Run | Code decode, tok/s |
| --- | --- |
| TensorFold, drafts off | 124.3, 122.1, 125.0 |
| TensorFold, drafts on | 123.1, 121.5 |
| mlx-lm server | 93.4, 99.2 |

The adjacent pair (mlx-lm, then TensorFold drafts off): 99.2 and 125.0 tok/s. Every TensorFold reply had the same `token_sha` (f66e3fd7e509) with drafts on and off. At 4 streams with drafts off (`concurrent_off.json`): 186.5 tok/s total for code and 184.9 for chat, each reply equal to its solo run (12 of 12).

## Apple M5 Ultra (5a6ee034)

`runs/m5-ultra-5a6ee034/`: Apple M5 Ultra (64-core GPU), 256 GB, macOS 27.0. TensorFold at 5a6ee034 (the head of the three pull requests, main c7137bb3 merged), built with `-Dcpu=apple_m1`: no M5-specific code, and `tile_qmm` does not use the M5 tensor units. mlx 0.32.3, mlx-lm 0.32.0. 4-bit checkpoint only. One model process at a time. `scripts/` holds the three run scripts.

Exactness: `tf-lfm2-exact` 54 of 54 (4-bit). Served replies have the same `token_sha` as on the M4 Pro (f66e3fd7e509 code, 57752ac6d7f0 chat). At 4 streams, each reply equal to its solo run (12 of 12, code and chat, drafts on and off). Turn two with the cache on equal to the cache-off reply.

Cold prompt, in process, alternated (tok/s, round 1 / round 2):

| Prompt | TensorFold | mlx-lm |
| --- | --- | --- |
| 2,048 | 4,482 / 4,482 | 2,467 / 12,167 (round 1 includes warm-up) |
| 8,192 | 4,394 / 4,396 | 10,825 / 13,322 |
| 32,768 | 3,709 / 3,709 | 11,622 / 11,631 |
| 65,536 | 3,052 / 3,052 | 9,810 / 9,817 |
| 131,000 | 2,247 | 7,226 |

Decode (tok/s):

| Case | TensorFold | mlx-lm |
| --- | --- | --- |
| In process, after 2,048 tokens | 256.7 / 257.1 | 205.0 / 205.1 |
| In process, after 8,192 tokens | 247.6 / 249.3 | 196.8 / 198.4 |
| In process, after 32,768 tokens | 198.0 / 198.3 | 167.8 / 168.1 |
| In process, after 65,536 tokens | 158.7 / 159.1 | 144.7 / 144.9 |
| In process, after 131,000 tokens | 116.3 | 113.4 |
| Served, 256 tokens, code | 290.2 / 290.6 / 289.8 | 178.8 |
| Served, 256 tokens, code, `--no-drafts` | 261.8 / 261.2 | — |
| Served, 4 streams, total, code | 490.3 | 177.8 |
| Slowest first token at 4 streams, code | 0.11 s | 4.38 s |

- Window costs at load: 1 row 3.8 ms, 2 5.2, 4 7.4, 8 11.4, 16 18.7.
- Turn two of the 7,991-token chat: TensorFold 0.082 s (7,987 cached), cache off 1.859 s; mlx-lm 0.175 s.
- Peak memory footprint in process at 131,000 tokens (`/usr/bin/time -l`): TensorFold 17.9 GB, mlx-lm 21.3 GB.
- The mlx-lm server sends the thinking text at the end of a chat reply, so its chat decode rate (511.7) is not comparable, and the concurrent client cannot read its tokens (the "failed" equality counts).

## Cheap windows, M4 Pro control (lfm2/4-cheap-windows with the one-row fix)

`runs/m4-cheap-windows-control/`: one session on the M4 Pro, 4-bit, `--prompt-cache-gib 0`. Order: mlx-lm, PR head 5a6ee034, cheap windows + fix, mlx-lm, cheap windows + fix, PR head, mlx-lm (served), then the same order in process. `ctl_dec_*` = bench_openai, `ctl_conc_*` = bench_concurrent `--levels 1,4 --alone --serial`, `ctl_ip_*` = tf-lfm2-bench / mlx_cold.py; `control.log` lists the run order with times.

| Served, code, tok/s | single stream | 4 streams total (steady) | slowest first token at 4 streams | 16-row window |
| --- | --- | --- | --- | --- |
| mlx-lm a / b / c | 94.4 / 94.5 / 94.2 | 91.6 / 88.5 / 95.4 | 8.5 / 8.9 / 8.2 s | - |
| PR head a / b | 118.6 / 124.8 | 182.6 (192.2) / 189.8 (199.0) | 0.36 / 0.34 s | 71.2 / 67.8 ms |
| cheap windows + fix a / b | 129.9 / 127.8 | 223.9 (237.1) / 223.6 (237.1) | 0.36 / 0.35 s | 47.6 / 47.0 ms |

Every TensorFold reply had token_sha f66e3fd7e509 (code) / 57752ac6d7f0 (chat) and equalled its solo run (12 of 12). Single-stream chat (plain one-row rounds): PR head 118.0 / 123.8, cheap windows + fix 115.2 / 115.9 tok/s. In process (prefill / decode after the prompt): 2k PR head 1,276 / 119.6 and 1,198 / 115.7, cheap windows + fix 1,275 / 120.2 and 1,084 / 115.4, mlx-lm 1,251 / 110.4, 1,278 / 111.6, 1,249 / 109.1; 32k PR head 1,024 / 85.0 and 951 / 81.3, cheap windows + fix 1,023 / 83.8 and 928 / 83.6, mlx-lm 1,051 / 75.8, 939 / 75.9, 991 / 74.8.

## Cheap windows (lfm2/4-core-windows, lfm2/5-lfm2-windows)

`runs/cheap-windows-0b96f2a4/`: gates of the final tree and of the core branch alone, the control run (PR head, the window kernels and mlx-lm, alternated, served and in process) and a drafts-off single-stream A/B against the PR head. Apple M4 Pro only. Its `README.md` lists the files.
