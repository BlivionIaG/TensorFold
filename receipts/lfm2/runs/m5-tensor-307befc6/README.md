# M5 tensor units for LFM2 (parts 6 and 7)

Apple M5 Ultra (64-core GPU), 256 GB, macOS 27.0. mlx 0.32.3, mlx-lm 0.32.0. Checkpoint `LFM2.5-24B-A2B-MLX-4bit` at `667408d4`. Zig 0.17.0 with `-Dcpu=apple_m1 -Dmetal-std=metal4.0`. One model process at a time.

Code: part 6 `lfm2/6-core-tensor` 2ceb6403, part 7 `lfm2/7-lfm2-tensor` 307befc6. 307befc6 differs from the tree measured here only in `zig/tests/lfm2_exact.zig` (`session/exact_4bit_v2.txt` is that test at 307befc6).

| Directory | What |
| --- | --- |
| `tensorfix/` | The night run on the same `zig/src`, `zig/kernels` and `build.zig`: `tf-tile-check`, `tf-lfm2-fp64`, served decode with drafts on and off (`decode_*`, `conc_*`), in process 2k, 32k and 131k (`tf_*.txt`, peak memory from `/usr/bin/time -l`), `mem_new_*` (this code) and `mem_pre_*` (before the scratch sizing). `exact_4bit.txt` is the earlier `tf-lfm2-exact`, which compared prompts in 128-row chunks with one row at a time: its 4 failures (prompts 127, 128, 129, 300) are the tensor-unit bit change. |
| `session/` | One session: `tile_check_6.txt`, `tile_check_7.txt`, `fp64.txt`, `exact_4bit_v2.txt` (68 of 68), in process alternated part 7 and mlx-lm at 2k, 8k, 32k and 64k (`ip_t7_*`, `ip_mlx_*`, two rounds), mlx-lm served decode and 4 streams (`*_mlx_r1`). `run567.sh` and `after7.sh` ran it; the served TensorFold sessions after `mlx r1` were stopped. |
