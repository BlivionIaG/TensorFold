# LFM2.5-24B-A2B receipts: TensorFold and mlx-lm

## Conditions

- Machine: Apple M4 Pro (20-core GPU), 48 GB, macOS 26.5.1. The Mac is a laptop on AC power.
- TensorFold: branch `lfm2-metal` at `87792521` (BlivionIaG/TensorFold).
- mlx 0.32.3, mlx-lm 0.32.0.
- Checkpoints (LiquidAI, Hugging Face): 4-bit `667408d494bde191c2cb9dd4f4549827816b9275`, 6-bit `1deec53eb49571e21cdcbf82fd5839248f07ed88`, 8-bit `67ab417dd02d0c9545802a726df668620eae049d`.
- One model process ran at a time. `raw/versions.txt` records the versions.
- `scripts/receipts_lfm2.sh` ran all steps. The token arrays are in `tokens/`.

## Files

| Step | Files in `raw/` |
| --- | --- |
| TensorFold server, `--prompt-cache-gib 0`: cold prompts, decode, concurrency | `tf_prefill.json`, `tf_decode.json`, `tf_concurrent.json`, `tf_cold.log`, `tf_cold_footprint.txt` |
| TensorFold server, prompt cache on: two turns | `tf_two_turn.json`, `tf_two_turn_cold.json` (cache off), `tf_cache_footprint.txt` |
| mlx-lm server: decode, concurrency, two turns | `mlx_decode.json`, `mlx_concurrent.json`, `mlx_two_turn.json`, `mlx_footprint.txt` |
| In process, alternated, 4-bit, 2 rounds | `alt_tf_2k_r*.txt`, `alt_tf_8k_r*.txt`, `alt_mlx_r*.txt` |
| In process, long prompts | `tf_32k.txt`, `tf_64k.txt`, `mlx_long.txt` |
| In process, 6-bit and 8-bit | `tf_6bit_*.txt`, `tf_8bit_*.txt`, `mlx_6bit.txt`, `mlx_8bit.txt` |
| Teacher-forced fidelity against mlx-lm | `fidelity_{4,6,8}bit/fidelity.json` (the logit dumps are not included) |

`tile_qmm/` holds the matrices of `tf-tile-check --dump` and `tile_qmm_reference.json` (error against float64, tile_qmm and mlx).

## Results, 4-bit

Cold prompt, tok/s. TensorFold: served, exact token arrays. mlx-lm: in process.

| Prompt | TensorFold | mlx-lm |
| --- | --- | --- |
| 2,048 | 1,300-1,320 | 1,311-1,315 |
| 8,192 | 1,248-1,250 | 1,283-1,287 |
| 32,768 | 1,033 | 1,077 |
| 65,536 | 840 | 876 |
| 131,000 | 588 | 621 |

Decode, tok/s:

| Case | TensorFold | mlx-lm |
| --- | --- | --- |
| Served, 256 tokens, code | 129.2 | 103.1 |
| Served, 4 streams, total | 188-191 | 100-120 |
| In process, 64-token context | 111-122 | 104-116 |
| In process, after 32,768 tokens | 85.0 | 79.5 |
| In process, after 65,536 tokens | 64.0 | 58.9 |

- Each TensorFold reply at 4 streams was equal to its solo run (12 of 12, code and chat).
- Turn two of a chat with a 7,991-token history: TensorFold 0.22 s with 7,987 cached tokens, mlx-lm 0.21 s. The TensorFold reply with the cache on was equal to the reply with the cache off.
- Peak physical footprint: TensorFold server 14.6 GB, mlx-lm server 16.7 GB.

The mlx-lm server sends the thinking text at the end of a chat reply. Thus its chat decode rate in `mlx_decode.json` is not comparable.
