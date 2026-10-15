# Cheap windows, parts 4 and 5: final builds

Apple M4 Pro (20-core GPU), 48 GB, macOS 26.5.1. Checkpoint `LFM2.5-24B-A2B-MLX-4bit` at `667408d494bde191c2cb9dd4f4549827816b9275`. Zig 0.17.0, builds with `-Dcpu=apple_m1` from the pushed branches with a clean tree. One model process at a time. Served with `--context 131072 --prompt-cache-gib 0`.

| Name | Build |
| --- | --- |
| `head` | #823 head 5a6ee034 |
| `p4` | part 4, `lfm2/4-core-windows` dd7c516a |
| `p5` | part 5, `lfm2/5-lfm2-windows` 87c3851a |

`run.sh` ran rounds 1 and 2 in the order head r1, p4 r1, p5 r1, p5 r2, p4 r2, head r2. Each session:

- `prefill_*.json`: cold served prefill of token arrays (`prefill_tokens.py`), 2k and 8k twice, 32k and 64k once, 131k once in round 1.
- `cache_*.json`: `tools/bench_cache.py --require-cold` on the 32,768-token prompt, the first request's `ttft_s`.
- `decode_*.json`: `tools/bench_openai.py --tokens 256 --reps 3`, drafts on.
- `conc_*`: `tools/bench_concurrent.py --levels 1,4 --alone --serial`.
- `footprint_*.txt`: `vmmap -summary` at the end of the session.
- `serve_*.log`: the server log; its first line has the LFM2 window ms by width.

`long.sh` repeats the 131k prompt in the order p5, p4, head, head, p4, p5 (rounds 3 and 4). `ctl.sh` then runs mlx-lm, p5, head, mlx-lm (served decode and 4 streams). Their files are added when they finish.
