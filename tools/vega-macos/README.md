# tools/vega-macos

Tools for the Vega 10 (gfx900) HIP backend on macOS: the numbers they produce are in [docs/recipes/vega-macos.md](../../docs/recipes/vega-macos.md).

| File | What it is | Language |
|---|---|---|
| `hipprof.zig` (+ `hipprof_table.zig`, `hipprof_fwd.zig`) | A HIP runtime shim that times what an engine launches | Zig, `zig build tf-hipprof` |
| `hipprof_check.zig`, `hipprof_mock.zig`, `hipprof_test.zig` | Its tests: a mock runtime, a driver and the pure parts | Zig, `zig build test-hipprof` |
| `vega_gate.py` | Serves Qwen3.5-4B through a HIP runtime and gates its greedy tokens and logits against mlx_lm; `--prof` runs behind hipprof | Python |
| `serve_bench.py` | Prefill, decode, TTFT and ITL of any OpenAI-compatible streaming server at c = 1..8 | Python |
| `prof_report.py` | A hipprof table as a per-token breakdown | Python |
| `cosine.py` | Per-row cosine and argmax agreement of two logits `.npy` files | Python |
| `mlx_ref.py`, `moe_ref.py` | The mlx_lm references (token ids, logits) the gates compare with | Python (calls mlx_lm) |
| `moe_gate.py` | Greedy gate of a running Qwen3.6-35B-A3B server against `moe_ref.py`'s mlx_lm references | Python |
| `expert_cache_sim.py` | Offline replay of decode routing traces against expert-cache policies (numpy) | Python |
| `fetch-rocm.sh`, `hipcc`, `pins.env` | ROCm headers and device libraries and the macOS `hipcc` wrapper | shell |
| `setup-mlx-env.sh`, `requirements-mlx.txt` | The mlx environment of the references | shell |
| `xcrun-metal-shim/` | A `PATH` shim for a Mac whose Xcode lacks the Metal toolchain | shell |
| `experiments/` | Kernel experiments and their patches | |

## hipprof

TensorFold opens `libamdhip64` through `TENSORFOLD_HIP`. Point that at `libhipprof` and name the real runtime in `HIPPROF_REAL`:

    zig build tf-hipprof                                   # zig-out/lib/libhipprof.dylib
    HIPPROF_REAL=<hipmac>/build/libamdhip64.dylib TENSORFOLD_HIP=$PWD/zig-out/lib/libhipprof.dylib \
        TF_POLICY=graphs=off zig-out/native-hip/bin/tensorfold-native serve ...

Every symbol of `zig/src/hip/abi.zig` (and `hipGetDevicePropertiesR0600`) is forwarded unchanged; `hipModuleGetFunction`,
`hipStreamBeginCapture`, `hipStreamEndCapture`, `hipGraphLaunch` and `hipModuleLaunchKernel` also keep their own books.
`hipModuleLaunchKernel` is timed in one of three ways:

| Mode | Variable | Meaning |
|---|---|---|
| each (default) | `HIPPROF_REP=N` (default 16), `HIPPROF_REP_MATCH` (default `affine_`) | The launch, then `hipStreamSynchronize`, wall clock of both: one kernel plus the runtime's fixed submit/complete cost. After it the same launch N more times back to back with one sync: the pipelined time of a kernel in a stream. Only kernels whose name contains `HIPPROF_REP_MATCH` and that rewrite the same bytes each time (stream and row tiles, not the recurrences). |
| events | `HIPPROF_EVENTS=1` | No sync: `hipEventRecord` before and after every launch, read back later (the GPU's 100 MHz timestamps): the GPU time of each kernel in the pipeline (each record costs about 1.5 us of GPU time) and the gap to the previous kernel. |
| graph | `HIPPROF_GRAPH=1` | `hipGraphLaunch` is followed by a stream sync and timed (a whole decode round, row `GRAPH_LAUNCH`). |

Other variables: `HIPPROF_START=1` (nothing is timed or replayed until the first `SIGUSR1`, so a model load runs at full speed),
`HIPPROF_LOG` (the table, default `hipprof.txt`), `HIPPROF_TRACE=file` and `HIPPROF_TRACE_N=n` (the first n launches, default 3000, as
`name g=gx,gy,gz b=bx,by,bz m= n= k=` lines), `HIPPROF_SLOW_US=us` (a stderr line for every launch slower than that), `HIPPROF_XCHK=1` (the first
60 `affine_dot2_stream` launches' fp16 activations scanned for non-finite values, on stderr). `SIGUSR1` clears the table (and starts the timing),
`SIGUSR2` writes it; it is also written at exit. Launches inside a stream capture are never timed (the sync would end the capture).
For an affine tile the table also gives m, n, k of the launch (the first kernel argument is quant/mlx.hpp's `Affine`; the second, of the stream
tile, its `StreamSides`) and the bytes of weights and group tables that read.

The table is tab separated, one row per kernel and launch shape, the costliest first:
`name count grid block shared m n k sides avg_us min_us total_ms rep_avg_us weight_bytes gap_us` (`prof_report.py` reads it).

`zig build test-hipprof` runs the unit tests of the table, its text format, the argument decoding and the symbol list (every field of
`abi.Api` is forwarded or interposed), then drives the shim over a mock runtime in each mode, with the exit-time dump and a second process
that reads what the first left. The driver takes the shim's path, so it also ran against the C shim this one replaced (`hipprof.c` in the history
before the port): the same checks passed, the exported symbols were identical, and the tables and traces were identical but for the clock columns.

## Language of the tools

TensorFold's own convention is Python for client, gate, reference and fixture tools and Zig for the engine, the server and the `tf-*` developer
executables. `CONTRIBUTING.md` says "Python clients and references under `tools` are development tools" and asks for served-decode receipts from
`python3 tools/bench_openai.py` and `tools/bench_concurrent.py`; `docs/recipes/SKILL.md` repeats it. On upstream `main` `tools/` holds 285 `.py`
files (the `bench_*`, `*_gate`, profile and conformance tools among them) against 3 `.zig`; the Zig developer tools are `tf-*` executables of
`zig/` wired from `zig/build/*.zig`, and `zig build dev-tools` links them. The Python here follows that: it measures requests and compares token
ids, and neither it nor its dependencies (numpy, tokenizers, mlx_lm) is part of native inference or the native build. `hipprof` is different: it
loads into the engine's process, so it is a native library, and it is Zig.

If these tools should be Zig as well, what it takes (nothing below is built):

| Tool | Port | Cost |
|---|---|---|
| `prof_report.py` | A `tf-*` executable over the table format above | about 150 lines; pure, diffable against the Python on any table |
| `cosine.py` | The same, with a `.npy` reader | about 200 lines; pure, diffable on any two files |
| `serve_bench.py` | `std.http` client with SSE parsing and threads; the exact-token prompts need the model's `tokenizer.json`, which `zig/src/core/tokenizer` already reads | about 600 lines and a fake-server test; replaces `uv run --with tokenizers` |
| `vega_gate.py`, `moe_gate.py` | Process spawn, HTTP and sha comparison against `mlx_ref.py`/`moe_ref.py` output, which stay Python | about 400 lines; only the pure parts can be tested without the card |
| `expert_cache_sim.py` | The cache-policy simulator (760 lines, multiprocessing) | about 1000 lines of Zig; diffable on a fixed trace |

The cost of the port is that the fork's `tools/` would then follow a different convention than upstream's, in the tools a pull request to
upstream would carry; that is the reason it is not done here.
