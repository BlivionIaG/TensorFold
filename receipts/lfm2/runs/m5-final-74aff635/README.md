# M5 Ultra, final code: PR head against all updates and mlx-lm

Apple M5 Ultra (64-core GPU), 256 GB, macOS 27.0. mlx 0.32.3, mlx-lm 0.32.0 (`versions.txt` has the binaries' SHA-256). Checkpoint `LFM2.5-24B-A2B-MLX-4bit` at `667408d4`. One model process at a time; `bench-final.sh` ran everything.

| Name | Build |
| --- | --- |
| `before` | #823 head 5a6ee034 (no cheap windows, no tensor units) |
| `tip` | 74aff635, the head of #873 (all updates) |
| `mlx` | mlx-lm server and `mlx_cold.py` |

Served sessions, alternated: mlx r1, before r1, tip r1, tip r2, before r2, mlx r2. TensorFold servers ran with `--context 131072 --prompt-cache-gib 0`. Each TensorFold session: `prefill_*` (cold served prefill of token arrays, 131k in round 1), `cache_*` (`tools/bench_cache.py --require-cold`, 32,768 tokens), `decode_*` (`bench_openai.py`, 256 tokens, 3 reps), `conc_*` (`bench_concurrent.py --levels 1,4 --alone --serial`), `footprint_*` (`vmmap -summary`), `serve_*.log` (window ms at load, and each request's `rounds` and `accepted` drafts). mlx sessions: decode and concurrent only; mlx-lm sends the thinking text at the end of a chat reply, so its chat rate is not comparable.

In process, alternated: `ip_<build>_<len>_r<k>.txt` (`tf-lfm2-bench` / `mlx_cold.py`), then `ip131_*.txt` once each with `/usr/bin/time -l`.

On the M5 the tip's code reply has other tokens than before's (prompt rows on the tensor units). Its drafts are accepted less often on that text: the code request takes 216 rounds with 39 of 100 drafts accepted, against 204 rounds with 51 of 90 (`serve_*.log`); the time per round is the same.
