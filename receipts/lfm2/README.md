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
| mlx-lm at 32k timed by part | `mlx_parts_32k.json` (c88bd868 only) |
| In process, 6-bit and 8-bit | `tf_6bit_*.txt`, `tf_8bit_*.txt`, `mlx_6bit.txt`, `mlx_8bit.txt` |
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
- Peak physical footprint: TensorFold server 14.5 GB (cache off) and 14.8 GB (cache on), mlx-lm server 16.7 GB.

The mlx-lm server sends the thinking text at the end of a chat reply. Thus its chat decode rate in `mlx_decode.json` is not comparable.
