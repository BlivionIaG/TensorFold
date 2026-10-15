# Exact 4-bit twin decode through half pairs (e9bf19d9 / 0cec9454)

| Directory | Machine | What |
| --- | --- | --- |
| `m4/` | Apple M4 Pro (20-core GPU), 48 GB, macOS 26.5.1 | `window_ms.txt`: the server's window cost at load, #867 builds alternated (the Mac was also in use); `tile_check_E.txt`. |
| `m5/` | Apple M5 Ultra (64-core GPU), 256 GB, macOS 27.0 | `win.log`: window cost at load for every candidate; `served/`: `bench_openai` and `bench_concurrent` for `before` (dd7c516a), `fix` (132bf2a3) and `final` (e9bf19d9), three alternated rounds; `exact_final.txt`: `tf-lfm2-exact` at e9bf19d9; `tc_*.log`: `tf-tile-check`; `vars/`: the candidate kernels. |
