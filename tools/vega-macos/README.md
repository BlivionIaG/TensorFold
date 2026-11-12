# tools/vega-macos

Tools for the Vega 10 (gfx900) HIP backend on macOS: the numbers they produce are in [docs/recipes/vega-macos.md](../../docs/recipes/vega-macos.md).

## Which tool is Zig and which stays Python

The rule: generic TensorFold tooling keeps upstream's convention (Python under `tools/`: `CONTRIBUTING.md` calls the Python clients and references "development tools",
`tools/` of upstream `main` holds 285 `.py` against 3 `.zig`); a tool that belongs to this macOS + Thunderbolt eGPU support stack (the hipmac runtime, the Vega gfx900
HIP backend, the TB4 expert offload) is a Zig `tf-*` executable built with `zig build tf-...` like the other native developer tools of `zig/`, tested offline. The question
asked of each tool: would it still make sense against a backend that is not ours (CUDA, Metal, `mlx_lm.server`)? If yes it is generic and stays; if it only reads our shim's
table, starts our runtime, or models our offload, it is ours.

| Tool | Class | Language | Why |
|---|---|---|---|
| `serve_bench.py` | generic | Python | Measures any OpenAI-compatible streaming server (`tensorfold-native` on any backend, `mlx_lm.server`, a vLLM); knows nothing of hipmac, Vega or TB4; the exact-token prompts need the model's `tokenizer.json` through the `tokenizers` package. Upstream's `tools/bench_*.py` receipts are the same kind of tool. The serving-report numbers of the recipe were produced with it. |
| `cosine.py` | generic | Python | Cosine and argmax agreement of two `[rows, vocab]` logits `.npy` files from any pair of engines. |
| `mlx_ref.py`, `moe_ref.py` | generic reference | Python | The mlx_lm references the gates compare with (Qwen3.5-4B, Qwen3.6-35B-A3B); they call `mlx_lm`, which exists only in Python. |
| `setup-mlx-env.sh`, `requirements-mlx.txt` | generic reference environment | shell | A `uv venv` with the pinned mlx / mlx-lm; nothing to port. |
| `xcrun-metal-shim/xcrun` | generic macOS glue | shell | A `PATH` shim for any Mac whose Xcode lacks the Metal toolchain (it serves the Metal targets of `zig build`, not the HIP stack). |
| `vega_gate.py` -> `vega_gate.zig` | stack | Zig, `zig build tf-vega-gate` | Starts `tensorfold-native serve` on hipmac's `libamdhip64.dylib` (`HIPMAC_BACKEND`, `TENSORFOLD_HIP`), runs it behind hipprof, gates gfx900 tokens and logits; it is the acceptance test of the Vega backend. |
| `prof_report.py` -> `prof_report.zig` | stack | Zig, `zig build tf-prof-report` | Reads the table of `hipprof`, our HIP shim (the format is `hipprof_table.zig`'s, the kernel names are our `tf::rocm` templates, the GB/s are Vega's HBM2 bandwidth). |
| `moe_gate.py` -> `moe_gate.zig` | stack | Zig, `zig build tf-moe-gate` | The gate of the HIP expert-offload server (`expert_offload.zig` is used only by the HIP backend): the greedy tokens of a running 35B-A3B server against `moe_ref.py`'s references, and cache on against `TF_EXPERT_COPIES=0` (`--against`). |
| `expert_cache_sim.py` -> `expert_cache_sim.zig` | stack | Zig, `zig build tf-expert-cache-sim` | Replays decode routing traces against the policies of `zig/src/hip/expert_store.zig` (LRU, SLRU, LFU, 2Q, ARC, TinyLFU ...): it models our offload tier, and the report is what picked SLRU. |
| `hipcc` -> `hipcc.zig` | stack | Zig, `zig build tf-hipcc` | The `hipcc --genco` wrapper over Homebrew LLVM that builds the gfx900 code objects on a Mac without a ROCm install. |
| `fetch-rocm.sh` -> `fetch_rocm.zig` | stack | Zig, `zig build tf-fetch-rocm` | Installs the HIP headers and device libraries of the pinned ROCm wheel for that wrapper (`pins.env` is its data). |
| `hipprof.c` -> `hipprof.zig` (+ `hipprof_table.zig`, `hipprof_fwd.zig`) | stack | Zig, `zig build tf-hipprof` | A HIP runtime shim that loads into the engine's process, so it is native code. |

| File | What it is | Build / test |
|---|---|---|
| `hipprof.zig`, `hipprof_table.zig`, `hipprof_fwd.zig` | A HIP runtime shim that times what an engine launches | `zig build tf-hipprof` |
| `hipprof_check.zig`, `hipprof_mock.zig`, `hipprof_test.zig` | Its tests: a mock runtime, a driver and the pure parts | `zig build test-hipprof` |
| `vega_gate.zig` | Serves Qwen3.5-4B through a HIP runtime and gates its greedy tokens and logits against mlx_lm; `--prof` runs behind hipprof | `zig build tf-vega-gate` |
| `prof_report.zig`, `prof_report_expected.txt` | A hipprof table as a per-token breakdown | `zig build tf-prof-report` |
| `hipcc.zig`, `fetch_rocm.zig` | The macOS `hipcc` wrapper and the ROCm headers / device libraries installer (`pins.env` is embedded in the second) | `zig build tf-hipcc tf-fetch-rocm` |
| `gate_lib.zig`, `cli.zig`, `npy.zig`, `pyfmt.zig`, `pyjson.zig` | What the gates share: one chat request, the server process and its log, the logits comparison, the command line, the `.npy` reader, Python-style number and JSON text | `zig build test-vega-tools` |
| `gate_fake.zig`, `gate_check.zig` | A stand-in for `tensorfold-native serve` / `tf-qwen35-test logits` and the driver that runs the gate against it, passing and failing | `zig build test-vega-tools` |
| `moe_gate.zig` | Greedy gate of a running Qwen3.6-35B-A3B server against `moe_ref.py`'s mlx_lm references; `--against` compares two runs | `zig build tf-moe-gate` (tested with the gate: `test-vega-tools`) |
| `expert_cache_sim.zig` + `expert_cache_sim_{list,policy,freq,config,data,run,report,rng,fmt}.zig`, `npz.zig` | Offline replay of decode routing traces (`.npz`) against expert-cache policies; NumPy's `RandomState` and CPython's `Random` are reproduced for the seeded runs | `zig build tf-expert-cache-sim` |
| `expert_cache_sim_handshake.zig` | `--handshake`: the traces through production's pool (`core/expert_offload.zig`) under the layer handshake's hit/miss split: the share of layers the card passes itself, and the check that the worker's late replay leaves the pool as the always-asking handshake does | `zig-out/bin/tf-expert-cache-sim [TRACES_DIR] --handshake [--cap N]` |
| `expert_cache_sim_test.zig`, `expert_cache_sim_expected.zig` | Its tests, and the Python-generated expected text | `zig build test-expert-cache-sim` |
| `serve_bench.py`, `cosine.py`, `mlx_ref.py`, `moe_ref.py` | Generic Python tools and the references (above) | `uv run --with tokenizers serve_bench.py ...` |
| `pins.env`, `setup-mlx-env.sh`, `requirements-mlx.txt`, `xcrun-metal-shim/` | Pins and shell glue (above) | |
| `experiments/` | Kernel experiments and their patches | |

## hipprof

TensorFold opens `libamdhip64` through `TENSORFOLD_HIP`. Point that at `libhipprof` and name the real runtime in `HIPPROF_REAL`:

    zig build tf-hipprof                                   # zig-out/lib/libhipprof.dylib
    HIPPROF_REAL=<hipmac>/build/libamdhip64.dylib TENSORFOLD_HIP=$PWD/zig-out/lib/libhipprof.dylib \
        TF_POLICY=graphs=off zig-out/native-hip/bin/tensorfold-native serve ...

Every symbol of `zig/src/hip/abi.zig` (and `hipGetDevicePropertiesR0600`) is forwarded unchanged; `hipModuleGetFunction`,
`hipStreamBeginCapture`, `hipStreamEndCapture`, `hipGraphLaunch` and `hipModuleLaunchKernel` also keep their own books. hipmac's signal words
(`hipmacSignalAlloc`, `Store`, `Load`, `Free`; the expert offload's VRAM ack looks them up by name) are exported too and forwarded when the runtime has them, else they answer hipErrorNotSupported (801), so the offload falls back to its host ack.
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
`name count grid block shared m n k sides avg_us min_us total_ms rep_avg_us weight_bytes gap_us` (`tf-prof-report` reads it).

`zig build test-hipprof` runs the unit tests of the table, its text format, the argument decoding and the symbol list (every field of
`abi.Api` is forwarded or interposed), then drives the shim over a mock runtime in each mode, with the exit-time dump and a second process
that reads what the first left. The driver takes the shim's path, so it also ran against the C shim this one replaced (`hipprof.c` in the history
before the port): the same checks passed, the exported symbols were identical, and the tables and traces were identical but for the clock columns.

## Tests, and what the Zig ports were compared with

`zig build test-vega-tools` (offline, no card, no Homebrew) runs the unit tests of the tools and of the modules they share, and `gate_check`: `tf-vega-gate` and `tf-moe-gate` as processes
against `gate_fake` (a stand-in for `tensorfold-native serve` that logs `done <id> ... sha=` like the real server, and for `tf-qwen35-test logits`). `tf-vega-gate` runs passing, with other
token ids (the sha gate fails), with other logits (the cosine / argmax gate fails), with a failing logits program and with a server that dies on start; `tf-moe-gate` against matching and
diverging references, and with `--against`.

The ports keep the command line, the console text and the result files of the Python they replaced (the gate's sha lines and its `results.jsonl` line are what the recipe's numbers
are compared by). That was checked against the Python originals, which were run before deletion (`git log -- tools/vega-macos` has them):

- `prof_report`: 53 hipprof tables of the recorded runs (`ut4g-usbgpu` `artifacts/results/tensorfold-vega-bench/*.prof.txt`), each with the default token count, with 200 and with 1:
  159 runs, output and exit status byte for byte equal. The float text goes through `pyfmt.zig` (C `snprintf`: Zig's own `{d:.1}` rounds the shortest decimal, Python rounds the exact
  binary value, they differ on `30.25`, `0.0905`, `6.05`).
- `vega_gate`: the Python and the Zig gate against the same fake server, on the passing run (with `--speed 2 --logits --policy --env`) and on the four failing ones above:
  stdout equal but for the port, the timings and nothing else, and the `results.jsonl` lines equal (key order, Python float text, `ensure_ascii` escapes, the `tensorfold` dict as
  Python prints it) but for `time`, `load_seconds`, `logits_seconds` and `wall`. The logits comparison is done in the gate (a `.npy` reader and the cosine of `cosine.py`): its line equals
  `cosine.py`'s on seven pairs of random matrices (float16 / 32 / 64, different row and column counts, rows of 150k).
- Not run: the gate against the real server on the card (the card is reserved); `--prof` (it builds `libhipprof` with `zig build tf-hipprof` and signals the server; the pieces are
  the code paths of the Python, unexecuted here).
- `hipcc`: `zig build test-vega-toolchain` runs `tf-hipcc` as a process over `hipcc_fake_clang` (the exact argv of the compile; `--genco` anywhere, overrides, `TF_HIP_COV`, `TF_VEGA_ROOT`, the root derived from argv[0]
  as invoked, the error cases, exit status passed through). By hand, `hipcc_check ... --ab hipcc` ran every case through the bash original too (same exit code, stdout, stderr), and a real `.hip`
  kernel compiled with Homebrew clang / lld through both gave byte-identical `.hsaco` for the default, `-O2`, `gfx906 -DX=1` and `TF_HIP_COV=6` (clang embeds a random CUID, so the comparison
  used `-fuse-cuid=none`).
- `fetch-rocm`: the same step runs `tf-fetch-rocm` over a synthetic wheel (stored and deflated members, members to skip): the file tree, `.hipVersion`, `.pinned`, the `bin/hipcc` link, a quiet second run, a
  reinstall on a stale `.pinned`, the sha mismatch (exit 1, root untouched), a bad CRC, a missing wheel. `diff -r` of the roots made by the shell script and by the Zig tool is empty for the synthetic wheel
  and, for the real pinned wheel (416,591,253 bytes, downloaded in about 12 s, sha256 verified), differs from the script-made `~/.cache/tensorfold-vega/rocm-10.1.0a20260822` only in the
  target of the `bin/hipcc` link (165 files in each); a compile through the downloaded root gives an identical `.hsaco`. A refused connection retries at 1, 2, 4 s; an HTTP error fails at once.
- `moe_gate`: the Python and the Zig gate against the fake server, matching and diverging replies (two rounds), and `--against` a recorded run: stdout equal but for the timings and the result `.json` files
  byte for byte equal, exit status equal (0 and 1).
- `expert_cache_sim`: `zig build test-expert-cache-sim` (number formats against CPython's text, mean / std against NumPy, `RandomState` and `Random(0).shuffle` vectors, a real `np.savez_compressed`
  archive and stored / zip64 / CRC / truncated cases of the `.npz` reader, each of the nine policies on a short sequence of hits and evictions, and the whole report on synthetic traces, equal to the
  Python's, with 1, 3 and 5 workers equal). On the study's real traces (`tf-study-moe-cache/docs/proposals/metal-tb4-split-moe-probe/traces` 34 short and `traces_long` 6 long generations) the Python
  (`uv run --with numpy`, `--jobs 8`) and the Zig stdout were identical, but for the timing lines, for the default run (Python 54 s, Zig 1.4 s), `--cap 1500`, `--long none --cap 4000 --only LRU`, and
  `--only` of each policy group (LRU, SLRU, LFU, 2Q, TinyLFU, W-TinyLFU, ARC, `K=8`, warm, seeded, `prefill replayed`, `own prefill`). The `--only` groups without the LRU baseline crash the Python
  in table 4 (KeyError); the Zig exits 1 as well, with the same stdout up to there.

Differences: the HTTP request of the gate has no client timeout (the Python had 900 s); `--help` and usage errors print the tool's own text (exit 0 / 2); a missing reference file or an HTTP error
ends the run with a one-line message and exit 1 instead of a Python traceback. `tf-hipcc` named without a directory in argv[0] and without `TF_VEGA_ROOT` cannot tell the ROCm root and says so (exit 1; the shell
script used `$0`'s directory). `tf-fetch-rocm` takes `--hipcc PATH` (the wrapper to link, default `tf-hipcc` beside itself; it must exist), `--pins FILE`, `--keep`, prints progress every 32 MiB, and reports a
missing wheel, a non-zip file or a bad CRC in one line. Not covered: Windows (the steps are skipped) and hosts other than macOS.

`tf-expert-cache-sim` ends with a one-line `expert_cache_sim: ...` message and exit 1 instead of a Python traceback (a missing meta entry, a bad archive, a cache smaller than one call's keys, `--jobs 0`),
does not reproduce NumPy's runtime warnings, and expands `~` and `~/...` in the traces path but not `~user`. `tf-moe-gate` ends with a one-line message and exit 1 where the Python gave a traceback.
