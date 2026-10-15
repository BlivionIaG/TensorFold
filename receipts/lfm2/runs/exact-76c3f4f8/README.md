# tf-lfm2-exact at #873's head 76c3f4f8

76c3f4f8 sets `attn_nax` to null on GPUs without tensor units (at fd2f8e27 it held stack bytes, and `tf-lfm2-exact` crashed at 6 and 8 bits on an M4 Pro).

| File | Machine | Checkpoint |
| --- | --- | --- |
| `m4/exact_fix8-{1,2,3}.txt` | Apple M4 Pro, macOS 26.5.1 | `LFM2.5-24B-A2B-MLX-8bit`, three runs |
| `m4/exact_fix6-{1,2}.txt` | Apple M4 Pro | `LFM2.5-24B-A2B-MLX-6bit`, two runs |
| `m4/exact_fix4-1.txt` | Apple M4 Pro | `LFM2.5-24B-A2B-MLX-4bit` |
| `m5/exact_4bit.txt`, `m5/tile_check.txt` | Apple M5 Ultra, macOS 27.0, tensor units on | `LFM2.5-24B-A2B-MLX-4bit` |

Each `tf-lfm2-exact` run: 68 forward/cache checks passed, 0 failed.
