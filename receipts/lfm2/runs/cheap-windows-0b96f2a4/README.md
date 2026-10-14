# Cheap windows: gates and control run

Apple M4 Pro (20-core GPU), 48 GB, macOS 26.5.1. Checkpoint `LFM2.5-24B-A2B-MLX-4bit` at `667408d494bde191c2cb9dd4f4549827816b9275` (6-bit and 8-bit for `tf-lfm2-exact`). mlx 0.32.3, mlx-lm 0.32.0. One model process at a time. Builds use `-Dcpu=apple_m1`.

| Directory | What | Build |
| --- | --- | --- |
| `control/` | Served decode (`ctl_dec_*.json`), 4 streams (`ctl_conc_*`), server logs with the window ms line (`ctl_*.log`), in process 2k and 32k (`ctl_ip_*`). Order: mlx-lm a, PR head a, fix a, mlx-lm b, fix b, PR head b, mlx-lm c, served then in process. `control.sh` ran it. | PR head 5a6ee034; fix 285779cd (the kernel code before the cleanups) |
| `final/` | `tf-tile-check`, `tf-lfm2-exact` at 4, 6 and 8 bits, host and golden test summary, fmt, lean, one served run with drafts on: `token_sha` and 4 streams equal to solo. | lfm2/4-cheap-windows 0b96f2a4 (tree equal to lfm2/5-lfm2-windows) |
| `core-a/` | The same gates on the core branch alone (its `tf-lfm2-exact` is the 54-check version). | lfm2/4-core-windows dd7c516a |
| `ab/` | Served, `--no-drafts`, `bench_openai.py --tokens 256 --reps 3`, alternated head, final, final, head, head, final (`ab.sh`, `ab.log` has each start's window ms line). | PR head 5a6ee034 and 0b96f2a4 |

The mlx-lm server sends the thinking text at the end of a chat reply, so its chat rate is not comparable and the concurrent client cannot read its tokens (the "failed" equality counts).
