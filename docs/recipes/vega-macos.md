# Qwen3.5-4B on an RX Vega 56 (gfx900) from an Apple Silicon Mac

The HIP engine serves Qwen3.5-4B MLX 4-bit on a Vega 10 card (RX Vega 56, gfx900: wave64, 56 CUs, 8 GiB HBM2) attached to an
M4 Pro MacBook Pro over Thunderbolt. There is no ROCm, no Linux and no tinygrad at run time: the engine opens the runtime library
by its macOS name, `libamdhip64.dylib`, which is the native HIP runtime of jkuepker/ut4g-usbgpu `hipmac/` (out of tree;
docs/HIPMAC.md there, <https://github.com/jkuepker/ut4g-usbgpu>). This page is the TensorFold side:
the toolchain that builds the kernels, the port of the kernels to gfx900, the checkpoint, and what has and has not been run.

Status (2026-11-11): everything marked built or tested was run on the Mac; the gfx900 code objects have run on the Vega 56 over Thunderbolt (hipmac) since 2026-11-09:
`hip-affine-test` 30/30, `hip-kernel-test` 157 passed, 14 skipped, 2 failed of 173 (two nemotron epilogue cases that also fail on the rebase base), greedy tokens identical to
`mlx_lm` (sha gate), decode 81.4 tok/s at DPM auto ("Retest on upstream `fb8fdf3d85`" below). `hip-gpu-test` has never been run on the card. The command lines of every
measurement are in "Reproducing the numbers on this page" at the end.

## Toolchain on macOS: `tools/vega-macos/`

`hipcc --genco` needs a ROCm install. On macOS the pieces are taken apart:

| Piece | Source | Pin |
| --- | --- | --- |
| compiler, linker | Homebrew `llvm` and `lld` (clang 23, the version of the ROCm wheel's own clang) | llvm 23.1.3, lld 23.1.3 (`brew install llvm lld`) |
| HIP headers `include/hip/**` | TheRock nightly wheel `rocm_sdk_core`, extracted by `tf-fetch-rocm` | rocm 10.1.0a20260822, HIP 7.16.26332 |
| device libraries `lib/llvm/amdgcn/bitcode/*.bc` (ocml, ockl, oclc_*, hip, opencl) | the same wheel | same |
| libc++ and libm headers the device compile includes | the Xcode macOS SDK (`xcrun --show-sdk-path`), passed as `-isysroot` | SDK 26.5 here |

The wheel is `rocm_sdk_core-10.1.0a20260822-py3-none-linux_x86_64.whl` from `https://rocm.nightlies.amd.com/whl-multi-arch/`
(the index jkuepker/ut4g-usbgpu's `pins.json` pins for its ROCm stack), 416,591,253 bytes, sha256
`cf376d3465cffd0c755a1e1ec08324c519eaf3d79c56778ffede1f6a7a2a5729`. Only 163 files of it are used (6 MB); its Linux x86-64
compiler and libraries are not. `tools/vega-macos/pins.env` holds the pins (`tf-fetch-rocm` has them built in, `--pins FILE` overrides); it checks the sha256.

```sh
brew install llvm lld                                  # clang 23: AMDGPU target, offload bundler
export PATH=$HOME/.local/zig-0.17.0:$PATH              # Zig 0.17.0 (.zig-version), unpacked from ziglang.org; no sudo
zig build tf-hipcc tf-fetch-rocm                       # zig-out/bin/tf-hipcc and tf-fetch-rocm (Zig; they were the shell scripts hipcc and fetch-rocm.sh)
ROCM=$(zig-out/bin/tf-fetch-rocm)                      # ~/.cache/tensorfold-vega/rocm-10.1.0a20260822 (or --wheel PATH)
zig build native-hip tf-qwen35-test -Dhipcc=$ROCM/bin/hipcc -Doptimize=ReleaseFast
```

`$ROCM/bin/hipcc` is a symlink to `tf-hipcc` (`tools/vega-macos/hipcc.zig`, `zig build tf-hipcc`), a wrapper that behaves as `hipcc --genco` does for what
`zig/build/hip.zig` asks of it (one `.hip` file, `--offload-arch`, `-I`, `-D`, `-o`): `-x hip --cuda-device-only
--offload-device-only --no-gpu-bundle-output`, `-O3` unless an `-O` level is given, `-mllvm -amdgpu-early-inline-all=true -mllvm
-amdgpu-function-calls=false` (all three as hipcc 7.x does), `--rocm-path` and `--rocm-device-lib-path` at the fetched root, lld
as the linker, the SDK as the sysroot, and `-mcode-object-version=5` (clang 23's own default is 6; the runtime's loader reads
v4 and v5). The result is an unbundled `ET_DYN` code object, `EM_AMDGPU`, `EF_AMDGPU_MACH_AMDGCN_GFX900`, xnack "any".
`bin/.hipVersion` in the root makes clang add its HIP wrapper header (the device `fmaf`, `expf`, ... next to the host libm);
without it, every kernel fails with "no matching function for call to 'fmaf'".

The default of `-Dhip-arch` on macOS is `gfx900` (elsewhere `gfx1151`, as before).

Known toolchain limits (none touches gfx900 or the RDNA2 objects, which build and are byte-identical to the objects built
before the port):

- Xcode's Metal compiler here predates macOS 27: `zig build test` needs `-Dmetal-std=metal4.0` on this machine.
- clang 23 upstream types `__builtin_amdgcn_fdot2_f32_bf16` and the gfx11 WMMA builtins on `short` vectors where AMD's fork
  takes `__bf16` vectors, so the gfx11 / gfx12 BF16 code objects do not build with Homebrew LLVM. Build those on Linux with ROCm.

## Running

```sh
export DYLD_LIBRARY_PATH=<hipmac>/build         # libamdhip64.dylib (and libut4ghip.dylib), or:  TENSORFOLD_HIP=<path>/libamdhip64.dylib
tensorfold-native serve ~/models/Qwen3.5-4B-MLX-4bit --parallel 1 --context 8192
```

`hip.Runtime.open()` loads `libamdhip64.dylib` on macOS and `libamdhip64.so` on Linux; `TENSORFOLD_HIP` names the file on any
platform (an install outside the loader's search path, or the offline mock: `HIPMAC_BACKEND=mock`).

The engine's HIP calls in a start and one request (measured on the older base the branch had before it was rebased onto upstream main; the mock library behind a recording shim, 9,149 calls): `hipInit`,
`hipDeviceGet`, `hipDevicePrimaryCtxRetain`, `hipCtxSetCurrent`, `hipDeviceGetAttribute`, `hipGetDevicePropertiesR0600`,
`hipMemGetInfo`, `hipModuleLoadData` (10 code objects there; upstream main has 16 groups), `hipModuleGetFunction` (335 kernels there; 559 now), `hipModuleGetGlobal` (1),
`hipStreamCreateWithFlags`, `hipMalloc`, `hipMemcpyHtoD`, `hipMemset`, `hipHostMalloc`, `hipModuleLaunchKernel`,
`hipStreamSynchronize`, `hipStreamBeginCapture`, `hipStreamEndCapture`, `hipGraphInstantiateWithFlags`, `hipGraphUpload`,
`hipGraphLaunch`, `hipGraphDestroy`, `hipMemcpyHtoDAsync`, `hipMemcpyDtoHAsync`, `hipMemcpyDtoDAsync`, `hipFree`,
`hipModuleUnload`, `hipDevicePrimaryCtxRelease`. Everything before the first launch is device and module setup; the GPU does no
work before the first `hipModuleLaunchKernel` apart from the weight uploads (`hipMemcpyHtoD`). `hipRuntimeGetVersion`,
`hipGetErrorName` (errors only), the cooperative-launch, occupancy, `hipGraphAddKernelNode` family and `hipFuncGetAttribute`
are in the symbol table (the 52 symbols of `zig/src/hip/abi.zig`, plus `hipGetDevicePropertiesR0600` and `hipMemGetInfo`, must
resolve, or the runtime is refused) but were not called on this path.

## The checkpoint

`mlx-community/Qwen3.5-4B-MLX-4bit`, revision `32f3e8ecf65426fc3306969496342d504bfa13f3` (2026-03-02): MLX affine 4-bit, group size 64,
tied head, `Qwen3_5ForConditionalGeneration`. `model.safetensors` is 3,034,300,695 bytes, sha256
`5fb9acd0246866381cf8c5c354c6db1019f6498eec4ccb4f5edcc71ffeacb2db` (the Hub's LFS digest); the directory is 2.9 GiB, at
`~/models/Qwen3.5-4B-MLX-4bit`. (`mlx-community/Qwen3.5-4B-4bit` carries the same weights.) Geometry from its config: hidden 2560,
32 layers (24 gated DeltaNet, 8 full attention), 16 query heads, 4 KV heads, head dim 256, 16 key and 32 value linear heads of 128,
intermediate 9216, vocabulary 248320. It has no MTP weights.

The HIP loader reads its geometry from config.json (the Qwen3.5-2B-only constants in `families/qwen3_5/config.zig` are the Metal
admission's), so this size needs no code change: `TF_QWEN_DIR=~/models/Qwen3.5-4B-MLX-4bit zig build test-qwen35` indexes it
("32 layers (8 full attention, 0 MoE), tied head true, MTP false, 1556 MiB of projections and MLPs") and `tensorfold-native info`
reports 2.85 GiB of weights and a 4.85 GiB floor. Against the mock runtime the engine loads it (2.21 GiB of model on the device),
captures and replays its decode graphs and answers a chat request; the Vega's 8 GiB leave room for 3 streams of 32768 tokens.

```sh
uvx --from huggingface_hub hf download mlx-community/Qwen3.5-4B-MLX-4bit --revision 32f3e8ecf65426fc3306969496342d504bfa13f3 --local-dir ~/models/Qwen3.5-4B-MLX-4bit
# or: tensorfold-native pull mlx-community/Qwen3.5-4B-MLX-4bit@32f3e8ecf65426fc3306969496342d504bfa13f3
```

## The port of the kernels to gfx900

gfx900 differs from the RDNA targets in four ways that matter here: wavefronts are 64 lanes (no wave32 mode); it has no
`v_dot2_f32_f16` or `v_dot4` (they start at gfx906), no WMMA matrix cores or bf16 dot (gfx11); it has no `v_fma_mixlo_f16`
(only `v_mad_mixlo_f16`; `llvm-mc -mcpu=gfx900` refuses the first); and a Vega 10 CU has one 64 KiB LDS and 256 VGPRs a lane. The kernels are written in groups of 32
lanes: every shuffle takes width 32 and every "wave" in them is `threadIdx.x >> 5`. On wave64 a group of 32 is half of a
wavefront, and a width-32 shuffle never leaves it, so the tiles keep their lane arithmetic, their reduction trees and their bits
unchanged. What changed is what saw the whole wavefront or used an instruction gfx900 lacks:

| Change | Where | RDNA |
| --- | --- | --- |
| `gfx900` row in the caps table: `wave = 64`, no dot2 / dot4 / matrix, fp16 activations, generation `gcn5`; the `-mno-wavefrontsize64` flag only for wave32 rows; `-Dhip-arch` default gfx900 on macOS | `src/hip/caps.zig`, `build/hip.zig` | unchanged |
| `TF_FDOT2(x, q, acc)`: `__builtin_amdgcn_fdot2` where the instruction exists (a macro, so the objects stay the same bytes); on gfx900 the two products fused one after the other into the accumulator (each product is exact in fp32) | `common/fdot2.hpp`; attention_fa, prefill.hpp, dot2.hip, `DotF16` | byte-identical |
| `DotF16::quad` (x of a quad lane by DPP): the `v_dot2c_f32_f16_dpp` asm stays RDNA2's; others move x by `mov_dpp quad_perm` then `TF_FDOT2` | `common/dot2.hpp` | byte-identical |
| `narrow_product` (silu(gate) * up rounded once to fp16): `v_fma_mixlo_f16` asm stays; on gfx900 the exact product in double narrowed to half by the compiler's correctly rounded `fptrunc` | `tiles/epilogue.hpp` | byte-identical |
| `readfirstlane(tid >> 5)`, a "wave" index held in a scalar register, is wrong on wave64 (the upper group of a wavefront would take the lower's index): `group32(tid)` keeps it on wave32 | `common/wave.hpp`; gemm.hpp | byte-identical |
| top-k's `__ballot` compaction counted lanes `& 31` of a 64-bit mask: lane and wave from `TF_WAVE` | `ops/draw.hpp` (`tf_topk`) | byte-identical |
| `kGen` and `kWave` for gfx900; `static_assert` of the wave width against the gfx9 / RDNA arch | `common/arch.hpp` | unchanged |
| a registry cost table for gfx900 (`src/hip/tuning/gfx900.zon`, generation `gcn5`); `chip()` names the generation `gcn5` for the gates | `src/hip/launches/affine.zig`, `mlx_entries.zig`, `native/hip.zig` | unchanged |

The RDNA2 (gfx1030) code objects of the ten kernel groups of the older base, built by the same wrapper before and after the port, were byte
for byte the same (`cmp`); the comparison on upstream main is in the offline verification below. The gfx11 objects cannot be built with
Homebrew LLVM (above) and were not compared.

All sixteen kernel groups of upstream main (559 kernels: 54, 20, 14, 9, 9, 9, 21, 335, 7, 19, 6, 12, 10, 16, 12 and 6 a group, counted
from the code objects' amdhsa metadata) compile for gfx900 offline (`zig build hip-kernel-build -Dhip-arch=gfx900`, below). Upstream's
RDNA-only lane instructions needed gfx900 forms: `row_xor` (DPP `row_xmask`) and `row_lane` (`v_permlane16`) in
`attention/prefill.hpp`, `row_xor` and `wave_sum_dpp` (`v_permlanex16`) in `decode/plan.hip`. Each is a width-16 or width-32 `__shfl`
with the same value, under `__gfx900__` only; the RDNA code keeps its instruction. The dot2 uses (`TF_FDOT2`) are in `common/dot2.hpp`,
`tiles/gemm.hpp` and `attention/prefill.hpp`; `group32` is in `tiles/gemm.hpp`.

Not on gfx900, by design: the matrix-core GEMM tile (`matrix_gemm.hpp`, WMMA, gfx11) and the BF16 dot2 tiles. They are compiled
out by `TF_MATRIX=0` and `TF_DOT2_BF16=0`, and the registry never selects them for a `matrix = none`, `act = f16` GPU.
gfx900 takes the RDNA2 fp16 family; that family is one the engine already serves on the Deck.

Register pressure and scratch (from each kernel's amdhsa metadata of the offline gfx900 build; no timing): 56 kernels use scratch,
55 of them affine tiles. The 128-row reference tile `affine_gemm_block` takes 428 to 464 bytes of scratch at 256 VGPRs, the K-parallel
`affine_gemm_kp` tiles 64 to 324 bytes, and the stream tiles at 8 columns a lane 156 to 576 bytes; the stream tiles at 2 and 4
columns a lane (the widths `stream_pick.zig` allows on gfx900) use none. Upstream main has no 32- or 64-row dot2 GEMM tiles (the
older base had them, and they spilled little or nothing), so `gfx900.zon` takes the K-parallel tiles for small prompts and `block`
for prompts past 32 rows and decode calls of 64 rows or more; both spill. [INFERENCE] Prompt prefill on a Vega is therefore slower
than on the older base; it is unmeasured. All of the tiles of a family give the same bits, so this changes speed only. The registry's
tests assert the table. Costs are the thresholds gfx1030's table carries, not measurements on a Vega; a measured table replaces them.

Warp-synchronous code that was lock-step on 32 lanes is lock-step on 64, and `__syncwarp()` is a wavefront barrier, so the
`gemm_kp` tile's LDS hand-offs between the two groups of a wavefront stay ordered. A block of 32 threads (`gated_delta`, the
decode attention, the single-wave launches) fills half a wavefront.

Differences in results from RDNA2 that the port can produce, because gfx900 has no dot2 instruction: a dot2's sum rounds once
(the instruction) or twice (two fused steps), so the bits of an fp16 product on gfx900 are the Python reference's, not
necessarily the RDNA2 golden files'. Every tile on gfx900 uses the same fused pair, so the tiles agree with each other (the
engine's own bit rules: solo equals batched, resumed equals fresh).

## Offline verification (Mac, no card)

Run on upstream main (the rebased branch), without the card:

| Check | Result |
| --- | --- |
| `HIPMAC_BACKEND=mock zig build hip-host-test -Dhip=true` | 78 passed (includes the caps, registry, tuning and stream-pick tests for gfx900) |
| `zig build hip-kernel-build -Dhip=true -Dhipcc=<macOS wrapper> -Dhip-arch=gfx900` | all 16 groups compile (about 70 s; the GPU tests compile, not run) |
| the same with `-Dhip-arch=gfx1030`, against a build of upstream main with the same wrapper | the `.text` and `.rodata` (machine code and kernel descriptors) of all 16 groups are byte for byte the same; the whole files differ in the `__hip_cuid_*` symbol, a hash of the source path (and the symbol order after it; looked at in group 3) |

Run on the older base before the rebase, not repeated on upstream main:

| Check | Result |
| --- | --- |
| `zig build test -Dhipcc=... -Dmetal-std=metal4.0` | 443 passed, 1 skipped, 0 failed |
| `TF_QWEN_DIR=~/models/Qwen3.5-4B-MLX-4bit zig build test-qwen35` | the 4B checkpoint indexes (loader tests) |
| `HIPMAC_BACKEND=mock TENSORFOLD_HIP=<hipmac>/build/libamdhip64.dylib tensorfold-native serve ~/models/Qwen3.5-4B-MLX-4bit` | starts, loads, captures 7 graphs, replays 153 of 160 rounds, answers a chat request (the mock does not run kernels, so the text is empty) |

## Status

The branch is on upstream main. What the older base (`vega-macos`) had and what is here:

- x staged in LDS for the one-row decode stream tile, gfx900 only (`stream_stage` in `tiles/stream.hpp`, switched by bit 8 of the launch's
  `lpc_log2` and the launch's dynamic LDS, so no kernel signature and no RDNA code object changes; up to 12 KB, `TF_STREAM_STAGE=0|1`),
  with the one-row tile's code pairs made where they are dotted (`stream_pairs_per_group` for gfx900 at R == 1: the hoisted pairs of upstream's
  tile cost a wave a SIMD). Tests: `kernel_tests/affine/stream.zig` (staged against unstaged bytes at every width, group, row count and
  columns a lane, pinned by hash).
- The 64- and 32-row dot2 GEMM tiles (`affine_gemm_rows`, `gemm_tile`'s `M`), instantiated for gfx900 only (`TF_ROWS` in `dot2_tiles.hip`), in the
  registry behind `Env.gemm_rows` and gfx900's table: 25 to 32 rows `gemm32`, 33 and more `gemm64` (the 128-row tile spills 195+ VGPRs there).
  gfx1030's table does not list them, so RDNA2 keeps upstream's choice.
- `TF_G9_MIX=2` is the default again (the stream tile mixes on fp16 x, a denormal x counts as zero), the kernel tests bound the flushed terms.
- The fused gate/up + silu_mul launch (below), the kernel tests of the commit before it, and the small-kernel tuning below (`tf_gdn_plan_128`,
  `tf_tail`, `tf_conv_split`, `tf_gnorm_out`, `tf_argmax`, the descriptor chain of `tf_plan_apply_f16`; upstream's tail, gnorm and plan_apply
  already had some of the batching, the gfx900 paths are in front of them).
- Offline: all 16 gfx900 groups compile (`hip-kernel-build -Dhip-arch=gfx900`), `hip-host-test` passes (mock), and the gfx1030 `.text` of all
  16 groups is byte for byte upstream main's. On the card: see "On the rebased branch" under "Decode on the card".

## Decode on the card (Vega 56 through hipmac, TinyGPU)

Measured with `vega_gate.py` (now `tools/vega-macos/vega_gate.zig`, built by `zig build tf-vega-gate`: same options, console text and `results.jsonl` line as the Python it replaced, compared on a fake server; the card was not available to rerun the numbers with it) (starts `tensorfold-native serve` on `libamdhip64.dylib`, checks the greedy tokens against
the mlx_lm references of `tools/vega-macos/mlx_ref.py`, then runs story decodes and reads `tensorfold.tokens_per_second`). Every number
in this section was measured on the older base the branch had before it was rebased onto upstream main (`vega-macos`), except the table
"On the rebased branch" below, which was measured on `vega-macos-up`.

### On the rebased branch (`vega-macos-up`, Vega 56, hipmac master 8b38ae3)

Startup: the load-time cost rig times forwards on KV pages nothing wrote, and hipmac's `hipMalloc` does not zero (HIP does not promise it).
NaN keys sent every stream launch down the fp64 replay and the load never ended; the rig now zeroes the pool first (`hip_costs.zig`).

| Check | Result |
| --- | --- |
| sha gate, nothink / think96 | `eb415dcbaf6e` / `bf86edaf9c17`, identical to mlx_lm, in every variant below |
| decode, `vega-macos` (older base) in the same session | 64.5 tok/s |
| decode, `TF_G9_REPAIR=1` (upstream's inline replay) | 62.8 tok/s |
| decode, `TF_G9_REPAIR=2` (one cold site, the default) | 63.8 tok/s (63.76 on the final default) |
| decode, `TF_G9_REPAIR=0` (no replay, measurement only, never shipped) | 64.0 tok/s |
| decode, `TF_STREAM_STAGE=0` / `=1` (before REPAIR=2) | 60.3 / 61.7 tok/s (default 63.0) |
| `hip-affine-test` (the `hip-affine-build` test binary) | 30/30 |
| `hip-kernel-test` | 156 passed, 13 skipped, 2 failed of 171 |

The two failures are `nemotron/epilogue.zig` (the relu^2 layer and the fc1 + shared up pair at one row, hidden 2688). They fail the same way
on the rebase base `4ed9a3e15` (MIX=1, no stage), with `TF_STREAM_STAGE=0` and with `TF_G9_MIX=1`: not from this branch's ports, cause not
diagnosed. `TF_G9_REPAIR` changes no launch's time in the per-kernel profile (down 76.3 us, out_proj 38.3 us in all three) but the decode
round in a graph; the 64 VGPRs of the stream kernel (59 on `vega-macos`) give the same 4 waves a SIMD. What is left of the gap to `vega-macos`
(0.8 tok/s) is the narrow-n stream launches: down (k = 9216) 76 us against about 70.

Prefill (server `prefill_seconds` of a chat prompt, one token out; tok/s):

| Prompt tokens | 50 | 100 | 225 | 526 | 1026 |
| --- | --- | --- | --- | --- | --- |
| `vega-macos-up` | 75 | 76 | 120 | 193 | 190 |
| `vega-macos` | 72 | 76 | 119 | 192 | 189 |
| `vega-macos-up`, `TF_POLICY=gemm=reference` (the block tile) | 54 | 54 | 106 | 168 | 182 |

The 64- and 32-row GEMM tiles are worth 15% at 526 tokens and 40% at 100 over the block tile, and match the older base.

### Retest on upstream `fb8fdf3d85` (2026-11-11; Vega 56, hipmac `c95d708`, DPM auto)

`vega-macos-up` was rebased onto upstream main `fb8fdf3d85` (112 commits after the previous base `21ea8dee4b`; its 29 commits applied without
conflicts). Every row here was measured on the rebased branch (the engine from `native-hip`, `-Dhip-arch=gfx900`, ReleaseFast) with the card at DPM
auto (`hipmac-smi --set-dpm auto`), which holds the sclk at 1200 to 1540 MHz under these loads: the 180 W socket limit is reached (`UT4G_METRICS_LOG=file:1`
samples with activity above 30%: median sclk 1309-1375 MHz, mclk 800 MHz, 179-180 W, edge at most 58 C, hotspot at most 70 C).

| Check | Result |
| --- | --- |
| sha gate, nothink / think96 (`tf-vega-gate`) | `eb415dcbaf6e` / `bf86edaf9c17`, identical to mlx_lm |
| decode N = 1 (`tf-vega-gate`, median of 3 128-token stories) | 81.4 tok/s (81.3-81.5); 80.9 in an earlier gate run with the card at DPM HIGH after a `--force` boot |
| `hip-kernel-test` | 157 passed, 14 skipped, 2 failed of 173; the two failures are the nemotron relu^2 / fc1 + shared up tests of before |
| `hip-affine-test` | 30/30 |
| per-row bit identity (`kernel_tests/affine/stream.zig`, in `hip-kernel-test`) | passes: row r of a several-row launch equals the launch of row r alone, m = 2, 3, 4, 5, 8, 16 |
| `hip-host-test` (mock) | 78 passed |
| gfx1030 code objects, `-Dhip-arch=gfx1030`, 16 groups, same wrapper, against upstream `fb8fdf3d85` | `.text` byte for byte identical in all 16 groups |

Round time (ms, `--parallel N`, N concurrent 128-token requests, "backend verify ... us each" of the server's exit log; DPM auto):

| N | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| before (older upstream base, DPM clocks, `165e94fe4e`) | 12.3 | 17.1 | 25.2 | 44.6 |
| `fb8fdf3d85` | 12.2 | 15.9 | 22.2 | 37.8 |

Aggregate tok/s at N = 1, 2, 4, 8: 81.5, 119.7, 163.9, 186.6. N = 2, 4 and 8 are faster than the old rows (7%, 12%, 15%); the cause is not diagnosed
([INFERENCE]: the old rows were taken in a different thermal and power-limit state, or an upstream change in the engine's scheduling; not checked).

Prefill (server `prefill_seconds` of a chat prompt with a salt in front so no prefix is cached, one token out; tok/s, DPM auto):

| Prompt tokens | 53 | 105 | 225 | 531 | 1024 |
| --- | --- | --- | --- | --- | --- |
| before (852 MHz) | 75 | 76 | 120 | 193 | 190 |
| `fb8fdf3d85`, DPM auto | 141 | 141 | 204 | 301 | 307 |

105 tokens take 748 ms and 53 take 383 ms: about 7.2 ms a token up to 105 tokens, the one-row path the short prompts take. The rows gain 1.9x at 53-105
tokens and 1.6x at 531-1024 over the 852 MHz rows, from 1.5x of clock.

Logits against mlx_lm (`tf-vega-gate --logits`, 87 rows through the decode path): cosine min 0.998616, mean 0.999881, argmax equal 87/87 (the older base
with `TF_G9_MIX=1`: 0.99857 / 0.99988 / 87/87). The gate's own bar is cosine min above 0.999, which this does not reach on either base; the argmax gate passes.
`tf-qwen35-test`, which the logits gate uses, did not compile with the stacked gate/up on gfx900: `zig/tests/qwen35/main.zig`'s `walk` took `h.?` of a
device field (`gate_up`) with no host twin; it skips such a field now (one line).

### The older base (`vega-macos`, before the rebase)

| Step | decode tok/s | Tokens vs mlx_lm |
| --- | --- | --- |
| port as of the first run (two fused fp16 steps a pair, 32 lanes a column, 2 to 8 columns a lane) | 36.5 | identical (France 8 tokens, 96-token thinking) |
| `v_mad_mix_f32` for the stream tile's fp16 pair on fp16 x (`TF_G9_MIX=2` today; the products are exact, the story tokens equal the fused pair's) | 46.7 | identical |
| `v_and_or_b32` builds a code pair in one op (`TF_G9_ANDOR`, bit-identical) | 49.3 | identical |
| lanes a column that leave none idle (80 chunks at k = 2560: 16 lanes, 5 rounds), at most 4 columns a lane; the measured run also staged x in LDS up to k = 4096 (k = 3072 with fp32 x) with the weights requested before the stage, which this branch's tile does not (see Status) | 57.5 (`TF_G9_MIX=2`: 56.4) | identical |
| the same with x widened to fp32 for the mix (`TF_G9_MIX=1`, default: exact, 63/63 tests) | 52.7 | identical |
| `TF_G9_FOLD=1` (`bias - 1024 * scale` in the epilogue: one packed op less a pair; not bit-identical, off by default) | 58.2 | identical |

Cause of the one failing test of the first mad_mix version (`affine ragged m3 n130 k96 b2 g32: mlx.project.decode.stream is 2.3 times
the error an fp32 sum may have against float64`), measured with a 4096-lane micro kernel on the card (fused pair against two
`v_mad_mix_f32`, random fp16 x with and without values below 2^-14): with every fp16 operand normal the two are bit-identical; with
denormal x halves 56 of 4096 dots differ, by up to 5.9e-6 of the terms' magnitude (the bound is 1.6e-6). `v_mad_mix_f32` flushes a denormal
fp16 *source* whatever MODE.FP_DENORM holds (setreg of all four denorm bits changed nothing); an fp32 source is not flushed. So
`TF_G9_MIX=1` (default) keeps x as two fp32 in the stream tile (widened with v_cvt; the measured run staged them in LDS as fp32 pairs) and the dots
are the fused pair's bits: `hip-kernel-test` 63/63, `hip-affine-test` pass, 52.7 tok/s. `TF_G9_MIX=2` keeps x as fp16 pairs and flushes
a denormal x (|x| < 6.1e-5; a term that small is below the fp32 rounding of a sum of terms of ordinary size): 56.4 tok/s, tokens
identical in the gate, but `hip-kernel-test` 62/63 (that case). `-DTF_G9_MIX=0` is the fused pair without the mix (about 36 to 49
tok/s). Logits against mlx_lm (87 rows through the decode path, `tf-vega-gate --logits`, MIX=1): cosine min 0.99857, mean 0.99988,
argmax equal 87/87.

Dropped: reading the codes as fp16 denormals (`TF_G9_RAW`, 51 tok/s): `v_mad_mix_f32` flushes denormal inputs on Vega and the tokens
differ.

Why the stream (gemv) tile is where it is: a read-only kernel with the tile's lane mapping reaches 360 GB/s of the 409.6 GB/s on
big matrices (512 MiB) and 320 GB/s at 25 MB, so the card's memory is not the limit. Ablations at the gate/up shape (26.5 MB,
before the tuning): no scale/bias loads 119 us, no dot arithmetic 110 us, loads only about 80 us, the tile 142 us. After the tuning:
gate/up 114 us (233 GB/s), qkv 83, down 66 (200 GB/s), out 34, attention qkv 72, lm_head 1171 us (305 GB/s).

Switches (all gfx900 only; RDNA code objects are byte-identical to before, checked for gfx1030):

- `-Dhip-cflags="-DTF_G9_MIX=0 -DTF_G9_ANDOR=0 -DTF_G9_FOLD=1"` at build time (variants of `common/fdot2.hpp`).
- `TF_STREAM_CB` and `TF_STREAM_LPC` at run time override the launch rule of `affine.zig` (`streamWith`, `stream_pick.zig`).
- `HIPMAC_IB=0 HIPMAC_BATCH=1` against the defaults: no change in decode speed (49.0 and 49.0 tok/s), the GPU is the limit.

Where a decode token goes (about 393 launches, 15.5 ms of kernel time of the 17.5 ms a token): the stream gemvs 10.5 ms (68%),
`tf_gdn_plan_128` 1.0, `tf_tail` 0.9 (one block a row), `tf_gnorm_out`, `tf_plan_apply_f16`, `tf_conv_split` 0.35 to 0.4 each. Each
dispatch costs the GPU about 4.4 us (barrier and cache acquire), about 1.7 ms a token (8%).

Profiling: `tools/vega-macos/hipprof.zig` is a shim `libamdhip64` that forwards to the runtime and times `hipModuleLaunchKernel`
(`HIPPROF_REP=N` runs N extra back-to-back launches of each kernel for the pipelined time; its modes and variables are in `tools/vega-macos/README.md`);
`tf-vega-gate --prof each` builds it (`zig build tf-hipprof`), `tf-prof-report` (`zig build tf-prof-report`) prints the per-token table. `--no-gate --story-tokens 16` is enough for a profile.

## Not yet verified

- `hip-gpu-test` on the card (never run; against the mock backend six of its runtime tests fail because they need a GPU: 34 pass, 6 fail of 40, 2026-11-11), and the stream test's other recorded pins on a second card.
- Logits cosine against mlx on the card: run (`tf-vega-gate --logits`, 87 rows, "Retest on upstream `fb8fdf3d85`" above): cosine min 0.998616, which does **not** reach the tool's own bar of 0.999 (`--logits` exits 1; the argmax gate passes 87/87). No document accepts or explains the gap.
- The two nemotron `hip-kernel-test` failures (`epilogue.zig`: relu^2, fc1 + shared up) are not diagnosed; they fail on the rebase base too.
- The default `--mtp-drafts 3` of a discrete GPU is irrelevant for this checkpoint (it has no MTP head).

## Dense MLP gate/up + silu_mul in one launch (gfx900)

The loader stacks a dense layer's gate and up rows into one MLX matrix (`quant.fuseRows`, host copy of 12 MB a layer through the scratch arena;
`gate` and `up` of the view are the two halves of the same device bytes, `quant.splitRows`, so prefill and the unfused path read them as before)
and the decode path (`Ops.projectGateUp`) runs the stream tile's pair mode (`Epi::kPair`, the routed experts' one) over it with `limit = -1`: the
epilogue (`swiglu` in `tiles/epilogue.hpp`, gfx900 only) rounds gate and up to fp16, then silu(gate) to fp16, then the product: the bytes of the two
products and `tf_silu_mul`, so the tokens do not change. `TF_FUSE_GU=0` (at load) keeps the separate matrices; `TF_GU_F32=1` runs the fp32 sums into
one rounding (also passes the sha gate; an option, not the default).

| | decode tok/s (median of 3) | launches a token | gate/up kernel |
| --- | --- | --- | --- |
| `TF_FUSE_GU=0` | 57.45 | 393 | 114.5 us + `tf_silu_mul` |
| fused, silu_mul rounding (default) | 57.90 | 363 | 111.8 us |
| fused, fp32 epilogue (`TF_GU_F32=1`) | 57.67 | 363 | 111.8 us |

## Small-kernel tuning of the decode (gfx900)

The gate/up fusion's kernel test is `the stacked gate and up with silu_mul's rounding as the epilogue write the bytes of the two products and
silu_mul` (`zig/src/hip/kernel_tests/decode.zig`; byte for byte against the group kernel and `tf_silu_mul`; `hip-kernel-test` is 64 tests).
Every change below is `#if defined(__gfx900__)` and keeps the arithmetic and its order (the same bits); the gfx1030 code objects' `.text` is
identical to the commit before (only the symbol table's order moves).

Why the small kernels were slow: a wave's loads were issued one dependent step after the other (a descriptor, then a pointer, then the data), with
a `waitcnt` in between, and a 32-thread block used a quarter of a CU's wave slots. The changes issue every independent load first.

| Kernel | Before (us a launch) | After | Change |
| --- | --- | --- | --- |
| `tf_gdn_plan_128` | 43.9 | 25.5 | LDS staging steps 16 -> 4 (3 blocks a CU -> all 512 resident), window loaded ahead of the state, no `threadIdx` bound branches |
| `tf_tail` | 15.6 | 11.1 | `tail_dense`: x, y, weight in one batch, the row in registers |
| `tf_conv_split` | 16.2 | 11.6 | `conv_split_plan`: plan, taps, sample, norm weight in one batch |
| `tf_gnorm_out` | 16.8 | 9.8 | `gnorm_out_small` (width <= 128): y, weight, z in one batch |
| `tf_argmax` | 202 | 140-156 | 16-byte loads, four in flight (the order of a thread's values does not change the result: `better` is a total order) |
| `tf_plan_apply_f16` | 51 | 46 | descriptor chain before the exponentials (eight values loaded together is upstream's own on this branch; the 51 -> 46 us was measured on the older base with both) |

Decode tok/s (median of 3, sha gate passing at each step): 57.7 (`f8904b4af`) -> 58.29 (plan steps only) -> 59.46 (gdn scan) -> 61.5 (tail) -> 62.35 (conv) ->
63.23 (gnorm) -> 64.8 (argmax and plan_apply).

Tried and not kept (all pass the sha gate; tok/s or kernel time not better):

| Experiment | Result |
| --- | --- |
| 3-deep weight prefetch (`stream.hpp`, R == 1) | 58.57 tok/s; gate/up unchanged (113 us), lm_head 1605 us |
| `__launch_bounds__(128, 4)` on the stream tile (<= 64 VGPR) | 53.66 tok/s; gate/up 141 us, down 102 us (spills) |
| `TF_STREAM_STAGE=0`, `TF_STREAM_LPC=8`, `TF_STREAM_CB=4` | gate/up 125 / 132 / 112 us (default 112), down 152 for LPC=8 |
| 8 waves a block (256 threads) | gate/up 111.5, down 66.6, lm_head 1180: no change |
| non-temporal weight loads (`slc`) | the profile shows -4 % on gate/up and qkv, the graph decodes 63.3 tok/s (63.23 before): not kept |
| tail folded into the next GEMV | not built: the residual is updated in place while other blocks read it, and every block would redo the row norm |

Where the stream tile's time goes (ablations of the stream kernel, a profile of the decode with the tile changed; not kept in the tree): with the
dot arithmetic removed gate/up takes 95.6 us (112), down 59, qkv 72.5, lm_head 1035 (1170); with the scale/bias loads removed 91.9 / 66.3 / 64.7 / 906;
with both removed (the weights' loads only) 78.0 / 53.0 / 57.6 / 902 us (302 / 223 / 274 / 352 GB/s of weight bytes). The tile's floor is then 78 us for
gate/up, and the dot arithmetic and the table loads cost 34 us of it: this is what a faster GEMV has to win back, mostly the table loads (an
interleaved scale-and-bias table, one 32-bit load a group, is the idea not tried). Down (k = 9216, 320 blocks) is at 223 GB/s with the loads alone, and
halving its blocks does not move it, so its limit is the access pattern, not the number of waves. Note that `--prof` replays one kernel back to
back (the same weights twice in the L2's reach), so a profile's gain from a load change can be absent in the decode: the tok/s decides.

## Building on this Mac: Xcode license, SDK and the Metal toolchain

2026-11-11: the Xcode license of the Mac these numbers come from **is accepted** (Xcode 27.2, Apple clang 21.0.0), so `clang`, `xcrun` and `make` work and `SDKROOT` and
`DEVELOPER_DIR` are no longer needed for the C build (`make -C hipmac`; the hipprof shim is Zig, `zig build tf-hipprof`). The documented numbers were built with the command line tools' SDK; keep it for byte-identical
objects: `export TF_SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk` (Xcode's default SDK is now 27.x). On a Mac whose license is not accepted (needs sudo) the builds still work
without Xcode:

    export PATH=$HOME/.local/zig-0.17.0:$PATH       # tools in ut4g-usbgpu: tools/macos/fetch_zig.sh pins the tarball (sha256 b607e9b9...c536a)
    export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk TF_SDKROOT=$SDKROOT

and `zig build test` (the Metal tests) and `make -C hipmac` (clang) do not.

**The Metal toolchain** is a separate component that Xcode 27.2 on this Mac does not have (`xcrun metal` fails: "missing Metal Toolchain; use xcodebuild -downloadComponent MetalToolchain").
The macOS asset of it is mounted under `/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin` (metal 32023.883 here) and runs without Xcode.
`tools/vega-macos/xcrun-metal-shim/xcrun` is a script for `PATH` that answers `--show-sdk-path` (and drops `-sdk macosx`) with the command line tools' SDK and runs `metal`, `metallib` and
`metal-ar` from that asset (everything else goes to `/usr/bin/xcrun`):

    PATH=$PWD/tools/vega-macos/xcrun-metal-shim:$PATH zig build test-split -Dmetal-std=metal4.0     # likewise tf-offload-metal, test-qwen27-split, test

`-Dmetal-std=metal4.0` is still needed (Xcode's Metal compiler here predates macOS 27). Untested: `xcodebuild -downloadComponent MetalToolchain` instead of the shim, and whether the
downloaded component still needs the flag. The Vega-only builds (`native-hip`, `hip-*-test`, `tf-qwen35-test`) need neither.

## Several rows a round (N streams), gfx900

A round of N rows costs a + b N on the Vega (measured with `--parallel N` and N concurrent requests, /tmp-free script in the log of this
commit's message): N = 1, 2, 4, 8 take 15.7, 27.9, 43.3, 80.1 ms a round, b about 9 ms a row (Mac Metal: 1.3). The 2 to 16 row products all
run the stream tile (`gfx900.zon`: `stream`, rows <= 16; row tiles of 1, 2 and 4 rows a block); at four rows its launches take 2.1 to 3.5 times
the one-row time (gate/up 130 -> 384 us, down 93 -> 198, qkv 99 -> 249, lm_head 1217 -> 4260), about 6.6 ms of the b, far from the weight read
(26.5 MB of gate/up at 384 us is 69 GB/s), so the weights are read once per row block and the tile is bound by something else. The 2- and 4-row
kernels use 119 and 107 VGPRs (1 and 2 waves a SIMD).
- Kept: the one-row x stage also serves the two-row tile (`stageBytes(rows = 2)`, up to 22 KB): N = 2 goes 27.9 -> 23.6 ms a round.
- Measured and not kept: staging the four-row tile (N = 4 43.3 -> 48.3; `TF_STREAM_STAGE=1` forces it), the one-row tile's two-deep weight prefetch at
  every row count (N = 2, 4, 8: 23.9, 43.8, 81.8: no change, so the weight latency is not what the rows wait for), `TF_STREAM_CB=1|4`, `TF_STREAM_LPC=8`.
- Open: what bounds the 4-row tile (per-wave VALU of the x preparation and the code pairs against LDS and L1 traffic needs counters or a
  kernel with one of them removed); b <= 2 ms is not reached.

### Several rows, redesigned launch shape (gfx900), and the clock

Measured at DPM clocks (boot with `AM_SMU_DPM=1`, sclk 1590 MHz) unless a row says 852: ut4g-boot's default leaves the VBIOS bootup clocks (852 MHz),
and a VALU microbenchmark (8 independent chains of v_fma_f32 / v_mad_mix_f32, 8 waves a SIMD) runs 2.46 T lane-ops/s there, 4.46 T under DPM; the 4B's
one-row decode goes 64.1 -> 81.6 tok/s on the clock alone.

Round time (ms), `--parallel N` with N concurrent 128-token requests, same code at each clock:

| clock | build | N=1 | N=2 | N=4 | N=8 |
| --- | --- | --- | --- | --- | --- |
| 852 MHz | before (d80741d36d) | 15.7 | 27.9 | 43.3 | 80.1 |
| 852 MHz | after | 15.4 | 25.5 | 41.1 | 70.6 |
| 1590 MHz | before (d80741d36d) | 12.4 | 16.6 | 29.5 | 53.0 |
| 1590 MHz | after | 12.3 | 17.1 | 25.2 | 44.6 |

b (a round's extra ms a row): 9 -> about 8.5 at 852 MHz; 5.5 -> 4.3 at 1590 MHz (N = 4 / 8). Target b <= 2 ms is not met.

What the ablations say (the 4-row tile, 18432 x 2560, 852 MHz, 265 us against 109 us at one row): without the stage 250, with LDS reads of x replaced by
constants 242, with no x at all 241, with a quarter of the dots 172 (so the dots are about 125 us, the other VALU and the weight stream the rest), without
x loads at k = 9216 (unstaged) 223 -> 123. The tile is VALU-bound: the FMA count is fixed by keeping every row's sum order (4 rows of 189 M
v_mad_mix is 125 us at 1.5 T/s), 1.5 VALU a FMA with the code pairs, 2 waves a SIMD (forcing 3 or 4 waves: slower or spilled, `amdgpu_waves_per_eu`).
The floor for a row at perfect VALU use is about 1.7 ms a round at 852 MHz.

Kept:
- gfx900 several-row tiles of at most 8 accumulators a lane (R * CB <= 8) run 16 wave groups a block (512 threads, `stream_threads`, launch_bounds) and
  stage x in LDS for 4 rows (k <= 4096, up to 40 KB) as well as 2 and 1: the stage is read by 4 times the columns: 4-row gate/up 396 -> 269 us at 852 MHz.
- Rows 3 and above pick a tile of at most 8 accumulators (4 rows x 4 columns took 195 VGPRs, one wave a SIMD: 8 rows 672 -> 485 us).
- Rows 3 and above whose 4-row stage does not fit (k = 9216: 78 KB) take 2-row blocks, which stage in 39 KB: down at 4 rows 142 -> 94 us (DPM).
- `kernel_tests/affine/stream.zig`: row r of a several-row launch is bit-identical to the launch of row r alone at m = 2, 3, 4, 5, 8, 16 (k = 2560, 4096, 9216);
  `TF_STREAM_BENCH=1` times the tile by rows.
Not kept: the one-row tile's weight prefetch at every row count (no change), `TF_STREAM_CB`, `TF_STREAM_LPC`, forced occupancy.
Open: at DPM the 2-row k = 9216 launch measured 106 us after dropping an `amdgpu_waves_per_eu(1)` experiment attribute (67 us with it; whether that launch
ran at 195 VGPRs and 512 threads is unchecked) and N = 2 is 17.1 against 16.6 ms; the per-row small kernels (gdn plan, conv split, plan apply) are not batched.

### Per-kernel decode cost and the in-SIMD bound (AUTO, ~1290 MHz sclk, 800 MHz HBM, 180 W, edge 36-44 C)

Profiling: `hipprof` in events mode (`HIPPROF_EVENTS=1`) works at `--parallel` > 1. The earlier hang was a dyld segfault: the engine `dlclose`s the
runtime after reading caps and the shim's watcher thread was left in unmapped text; `init()` now pins the image (`RTLD_NODELETE`). Event timestamps
and `s_memrealtime` tick at 27 MHz, not 100 MHz (hipmac fixed in c95d708); tables below are for the older 100 MHz conversion times 4.

Baseline round 12.31 ms at N=1 (81 tok/s), 24.0 ms at N=4. Stream gemv 9.4 ms at N=1, 19.0 ms at N=4 (129 launches a round): about 3.2 ms of b = 4.3.

| kernel | m=1 us | m=4 us | ms per extra sequence |
| --- | --- | --- | --- |
| gate/up | 100 | 215 | 1.23 |
| down (k 9216) | 55 | 113 | 0.62 |
| qkv (12352) | 76 | 131 | 0.49 |
| out (k 4096) | 29 | 67 | 0.40 |
| lm_head | 1115 | 2150 | 0.35 |
| gdn plan (all slots, one launch) | 0.62 ms | 1.82 ms | 0.40 (fp32 state, memory bound, not reducible bit-identically) |

plan_apply ~0.09 ms and gdn_replay ~0.11 ms a sequence; conv_split, tail and the rest ~0. Warmed tile bench, gate/up (n18432 k2560), us at m = 1/2/3/4/8/16:
89/130/158/179/320/604 (down m4 107, qkv 122, out 56, lm_head m1 1037 / m4 2071 / m8 4015).

Finding: at m >= 2 the stream tile is bound inside the SIMDs, not by HBM or latency. With every weight load redirected to chunk 0 (no HBM traffic) gate/up m4
still takes 161 us (190 with the loads), m1 57 us (91), m8 290 us (333). VALU is ~40% busy (440 instrs a round, 256 `v_mad_mix`, ~2.75 ops a code pair);
`v_mad_mix`/`v_fma` chains reach 3.7-4.4 T lane-ops/s in a microbench even at one chain and one wave a SIMD, with no VGPR bank effect (all operands in one bank:
same rate). So neither dependency latency, bank conflicts, nor bytes in flight is the limit; LDS reads or issue stalls remain (unmeasured).

Rejected (gate/up m4 us, 179 baseline; none went through the gate): one-round weight prefetch in the several-row tile (R=4: 127 VGPR + spill, 186; R=2: spills,
lm_head m2 9.2 ms); persistent staged tile, x staged once per block (190; down 113 vs 107; lm_head m4 2158 vs 2071; with prefetch 235); pair-major inner loop,
each column's code pair made once and dotted into all rows, `ds_read_b128` x reads (189.6, still 108 VGPR, occupancy 2).
The code of all three, the timeline instrumentation (`TF_TIMELINE`, block timestamps behind the output rows: at m16 it writes past the buffer and faults the card) and the
warmed bench is in `tools/vega-macos/experiments/multirow-2026-11-11.patch` (`git apply`; against 165e94fe4e).

Next: the ablations that dropped the extraction (`TF_ABL&2`) or the LDS x reads (`&4`) fed garbage to the dots and ran 22-480 ms or wedged the card; redo them with finite data
(x and weights kept in range, e.g. extraction replaced by a cheap pass of valid fp16 pairs) to split the 161 us between LDS reads, extraction and mads. Then m = 8 as one 8-row
block (weights read once) and lm_head (m4 2071 vs m1 1037 us), which depend on that split.

## Serving performance report (`tools/vega-macos/serve_bench.py`)

Reporting standard: prefill (tok/s), decode (tok/s), TTFT (s) and ITL (ms) at c = 1, 2, 4, 8 concurrent requests, per request (mean / p50 / p90) and
aggregate, for two fixed loads: **1024 input / 512 output tokens** and **16384 input / 1024 output tokens**. `serve_bench.py` drives any OpenAI-compatible
streaming server (`/v1/chat/completions`, greedy, thinking off, `max_tokens` = `min_tokens` = the output length with `ignore_eos`): every prompt is
exactly the asked number of tokens as the server counts them (chat template included; the filler is cut at a token boundary of the model's tokenizer,
calibrated against `usage.prompt_tokens`), with an eight-digit salt in front, unique to each request, so no prefix cache serves it. Prefill tok/s =
prompt tokens / TTFT (so a request that waits behind others' prefill shows a lower per-request rate; "agg" = all the round's prompt tokens over the
time to the last first token); decode tok/s = (out - 1) / (last chunk - first chunk); total tok/s agg = output tokens / the round's wall time
(prefill included). One untimed warm-up round of every c precedes the cells. Run it with `uv run --with tokenizers tools/vega-macos/serve_bench.py
--url http://127.0.0.1:PORT --model NAME --tokenizer MODEL_DIR --json out.json --label "server, model, hardware, clocks"`.

### Mac alone (M4 Pro), measured 2026-11-11 offline, card not in use

`mlx_lm.server` (mlx-lm from `uvx`, `--decode-concurrency 8 --prompt-concurrency 8 --prompt-cache-size 1 --temp 0`), Qwen3.5-4B-MLX-4bit, 2 rounds a cell
(raw: `serve_bench` JSON of that run). Nothing else was running on the Mac during these rows except the benchmark client.

| load | c | prefill tok/s agg | prefill tok/s per request (mean) | decode tok/s agg | decode tok/s per request (mean) | total tok/s agg | TTFT s (mean / p90) | ITL ms (mean / p50 / p90) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1k / 512 | 1 | 764.8 | 764.8 | 57.2 | 57.2 | 49.6 | 1.34 / 1.35 | 18.1 / 15.1 / 26.3 |
| 1k / 512 | 2 | 647.3 | 323.9 | 104.2 | 52.1 | 78.6 | 3.16 / 3.24 | 19.6 / 19.4 / 24.2 |
| 1k / 512 | 4 | 776.8 | 194.2 | 181.3 | 45.3 | 123.7 | 5.27 / 5.35 | 22.1 / 22.1 / 23.0 |
| 1k / 512 | 8 | 431.6 | 54.1 | 203.1 | 25.4 | 104.8 | 18.95 / 19.43 | 39.4 / 39.1 / 40.7 |
| 16k / 1k | 1 | 664.4 | 664.4 | 56.5 | 56.5 | 23.9 | 24.69 / 25.39 | 17.7 / 17.7 / 18.4 |
| 16k / 1k | 2 | 670.1 | 335.1 | 90.7 | 45.4 | 28.7 | 48.93 / 50.29 | 22.0 / 21.9 / 23.2 |
| 16k / 1k | 4 | 690.8 | 172.7 | 127.0 | 31.7 | 32.2 | 95.02 / 98.88 | 31.5 / 31.2 / 32.5 |
| 16k / 1k | 8 | 697.4 | 87.2 | 141.1 | 17.6 | 33.3 | 187.96 / 189.32 | 56.7 / 55.9 / 56.9 |

`mlx_lm.server`, Qwen3.6-35B-A3B-4bit, 1 round a cell (a pass of `moe_ref`, an mlx run of another job, shared the Mac during the first minutes of this run;
its client had no connection retry yet: one of the eight 1k c = 8 requests failed with a socket error, those figures are the other seven):

| load | c | prefill tok/s agg | prefill tok/s per request | decode tok/s agg | decode tok/s per request | total tok/s agg | TTFT s | ITL ms (mean / p50 / p90) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1k / 512 | 1 | 939.5 | 939.5 | 72.6 | 72.6 | 63.0 | 1.09 | 13.8 / 13.7 / 14.1 |
| 1k / 512 | 2 | 990.2 | 495.3 | 117.7 | 58.9 | 95.3 | 2.07 | 17.0 / 16.9 / 17.2 |
| 1k / 512 | 4 | 1023.3 | 255.9 | 161.4 | 40.3 | 122.9 | 4.00 | 24.8 / 24.7 / 25.0 |
| 1k / 512 | 8 (7 of 8 requests) | 1041.4 | 148.8 | 178.4 | 25.5 | 133.1 | 6.88 | 39.2 / 39.2 / 39.6 |
| 16k / 1k | 1 | 850.5 | 850.5 | 63.2 | 63.2 | 28.9 | 19.26 | 15.8 / 15.7 / 16.2 |
| 16k / 1k | 2 | 851.3 | 425.6 | 91.6 | 45.8 | 33.7 | 38.49 | 22.0 / 21.7 / 22.4 |
| 16k / 1k | 4 | 857.8 | 214.5 | 119.0 | 29.7 | 37.0 | 76.40 | 33.6 / 33.6 / 34.0 |
| 16k / 1k | 8 | does not fit | | | | | | the server's Metal command buffer died with `Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)`; all 8 requests failed |

TensorFold's Metal server cannot serve the 4B: `tensorfold-native serve` (the Metal build of this branch) stops at start with `the native Qwen engine cannot
load this checkpoint (UnsupportedQwenGeometry)` (upstream's Metal Qwen3.5 backend is the pinned 2B; the 4B runs on the Mac only through the `qwen27` layer-range
engine of the offload branch, which has no HTTP server). So there is no TensorFold-Metal row for the 4B.

### RX Vega 56 through hipmac (`c95d708`), `tensorfold-native` from `native-hip` on `vega-macos-up` `c14179e9e4`, 2026-11-11

Qwen3.5-4B-MLX-4bit, DPM auto (`UT4G_METRICS_LOG=file:1`, samples with activity above 30%: sclk median 1361-1369 MHz, range 1207-1501, mclk 800 MHz, 180 W of the
180 W limit, edge at most 69 C, hotspot at most 81 C, HBM at most 72 C), 2 rounds a cell. **16k / 1k at c = 8 does not fit**: the server refuses to start with
`--parallel 8 --context 17600`: "`--parallel 8 needs 5.51 GiB at a 17600-token window, and the HIP memory budget leaves 4.52 GiB: serve --parallel 6, or lower --context`"
(2.21 GiB of weights, 1.0 GiB reserve, 0.59 GiB a stream at that window; 6 streams fit). The 16k rows therefore ran on `--parallel 6 --context 17600` (c = 1, 2, 4;
the load is not shrunk); the 1k rows ran on `--parallel 8 --context 2048` (the 1.5k tokens of a request fit: c = 8 is served, with the 17.6k window the same c would not be).

| load | c | prefill tok/s agg | prefill tok/s per request (mean) | decode tok/s agg | decode tok/s per request (mean) | total tok/s agg | TTFT s (mean / p90) | ITL ms (mean / p50 / p90) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1k / 512 | 1 | 305.6 | 305.6 | 76.8 | 76.8 | 51.2 | 3.35 / 3.35 | 13.0 / 13.0 / 13.2 |
| 1k / 512 | 2 | 301.5 | 228.4 | 83.8 | 50.2 | 65.9 | 5.07 / 6.79 | 20.5 / 17.1 / 17.4 |
| 1k / 512 | 4 | 298.6 | 158.0 | 89.9 | 30.8 | 78.5 | 8.52 / 13.72 | 34.2 / 24.3 / 24.5 |
| 1k / 512 | 8 | 295.9 | 96.8 | 89.9 | 17.7 | 83.9 | 17.51 / 26.75 | 60.7 / 41.7 / 42.1 |
| 16k / 1k | 1 | 300.6 | 300.6 | 57.7 | 57.7 | 14.2 | 54.50 / 54.53 | 17.3 / 17.3 / 17.7 |
| 16k / 1k | 2 | 291.3 | 215.1 | 25.3 | 26.0 | 14.8 | 85.03 / 112.52 | 52.0 / 25.6 / 26.0 |
| 16k / 1k | 4 | 286.5 | 125.2 | 20.8 | 12.7 | 15.2 | 157.22 / 228.71 | 109.0 / 41.1 / 41.6 |
| 16k / 1k | 8 | does not fit (above) | | | | | | |

Reading the rows. The card's prefill is one stream at a time at about 300 tok/s whatever c is (the aggregate is flat 286-306), so a request's TTFT is its
queue of prefills; the per-request prefill figure divides the prompt by that wait. The "decode agg" column is the sum of (out - 1) over the span from the
earliest first token to the latest last token, which includes the other requests' prefills (a later request's prefill stalls the earlier ones' decode:
ITL mean far above p50, p90 about the stall's decode round), so at 16k it is low (20-25 tok/s) while the steady decode with every stream running is what the
ITL p50 gives: 17.3 ms at c = 1, about 25.6 ms at c = 2 and 41 ms at c = 4 (`prompt cache: kept nothing at 16320 tokens: 255 pages pass the limit of 92` warnings:
the cache is 0.33 GiB, so a 16k prompt is not kept, which is wanted here). Against the Mac's mlx_lm.server on the same loads the Vega decodes faster at c = 1 on 1k
(76.8 against 57.2 tok/s, ITL 13.0 against 18.1 ms), level at c = 1 on 16k (57.7 against 56.5), and is behind in total tok/s at 16k (14-15 against 24-32); its
prefill (about 300 tok/s) is below mlx's 660-760. That the card prefills one prompt at a time is [INFERENCE] from the flat aggregate and the TTFT queue, not from the engine's code.

## Qwen3.6-35B-A3B MLX 4-bit by expert offload (`feat/moe-expert-offload`)

The 19.0 GiB checkpoint (16.9 GiB of routed experts) does not fit the card's 7.9 GiB; `tensorfold-native serve ~/models/Qwen3.6-35B-A3B-4bit` offloads the routed
experts automatically (`TF_EXPERT_OFFLOAD=0|1|auto`, `--ssd-experts GIB` sets the cache). Design, the 1.96 GiB DMA-window finding and results:
`docs/proposals/hip-moe-expert-offload.md`. Environment: `HIPMAC_BACKEND=ut4g TENSORFOLD_HIP=<hipmac>/build/libamdhip64.dylib`; `TF_EXPERT_COPIES` (misses the cache admits a round; the default 4096
admits every miss: a miss's one trip over the link is also its admission, the stream tile stores what it reads into the cache line), `TF_EXPERT_PROTECT` (percent of lines an SLRU protects; 0 = LRU),
`TF_EXPERT_WINDOW_MIB` (pinned window, default 1536; the card maps ~2047 MiB of host memory in all, the Thunderbolt PCIe DART aperture, see `ut4g-usbgpu/docs/DMA_LIMIT.md`). The Mac must have ~19 GiB of free, unwired memory
(an mlx process holding the same model, or several GiB wired, shrinks what `hipHostMalloc` can pin). Decode runs eager (value waits are not capturable); a prompt span copies each expert it uses
from the window into a one-layer VRAM scratch by SDMA before its GEMMs (a 466-token prompt prefills in 4.5 s instead of 23.8 s by reading the window in place).
The server log carries `experts interval/total` lines (hit rate, misses/token and MiB/token over the link, window fill, admissions, stage ms/token, round ms) and an `experts span` line a prompt span.
Checks: `tools/vega-macos/moe_ref.py` (mlx_lm greedy references, 8 prompts incl. a 466- and a ~1100-token one) and `tf-moe-gate` (`zig build tf-moe-gate`; shas of a running server, `--against` for cache on vs `TF_EXPERT_COPIES=0`; the Zig port of `moe_gate.py`, which produced the recorded runs).
Recovery after a wedge ("GC TLB invalidation did not acknowledge", "not a GPU this engine supports"): `ut4g-boot --force`, then DPM auto (`hipmac-smi`).

## Pipelined greedy rounds, the host token word and the in-graph top-k (gfx900, `LATENCY.md` L8)

Card result so far (`ut4g-usbgpu` `artifacts/latency/partA`, A5): the gates pass, but `tf-vega-gate --speed 3` at c = 1 is 80.5 tok/s against 81.1 before (0.7 % slower, 2 A/B rounds); everything below is on by default and the default of the costly piece flips to off if the card confirms it. `TF_LATENCY_OFF` (comma list, read at start-up) turns the pieces off one at a time: `pipe` (a lone greedy stream waits for each token, as before), `mirror` (the token returns by the device-to-host copy; implies `pipe`) and `topk` (sampled rows' top-k runs eagerly); `TF_LATENCY_OFF=pipe,mirror` is the previous decode path for greedy rounds (the mock backend gives the old sha with it). Still to be taken: ITL at c = 4, and the gate shas (`tf-vega-gate`: `eb415dcbaf6e` nothink, `bf86edaf9c17` think96) with each switch.

- **Pipelined rounds (a).** A lone greedy one-token stream runs through the lane core's `gpu_tokens` path: `Hip.queueRound` launches round N+1 while N runs. The replayed graph of such a round (`Shape.piped`) has no plan copy inside it; its plan goes up beforehand from a four-slot pinned ring (`plan.Buffer.sendQueued`), and the argmax of the round before wrote its token to the plan's first word, which the embed reads (the partial copy skips that word). The first round after a prompt, or any round whose token the host holds, takes a full copy. Two rounds are in flight at most (the core lands one before queueing the one after next). EOS needs no rollback: the one speculative round computes a row the stream's `release` discards after syncing; `Caches.total` has the slot. Sampled and penalized streams, several streams, drafting and tensor parallel keep verify windows (`Model.pipe_sampled = false`, `Engine.pipes`).
- **Token store (b).** The argmax kernels also store the pick to a pinned mapped host word (`Drawer.mirror`, eight slots), named by a plan word so a round can pick its slot without a graph variant; the host reads the word after the fence (the stream sync of a plain round, an event behind a queued one) and the 4-byte device-to-host copy node is gone.
- **Top-k in the graph (c).** A round with sampled rows runs its top-k launch inside the graph (shape key: a candidate stride of 32..512 and whether the argmax runs; per-row counts travel in the plan); the host draw reads the candidate lists with one copy. Tensor-parallel groups keep the eager launch.
- **Launch fusion (e), a list, nothing fused yet.** A decode round of the 4B launches 335 kernels (`HIPPROF_TRACE` with `TF_POLICY=graphs=off` on the mock backend): 129 affine GEMV (4 a layer + head), 64 `tf_tail` (two a layer), and per linear layer (24) `tf_conv_split`, `tf_gdn_plan_128`, `tf_gnorm_out`; per full layer (8) 2 `tf_qk_rope`, `tf_plan_kv_write`, `tf_plan_score2_f16`, `tf_softmax_stats`, `tf_plan_apply_f16`, `tf_sum_partials`, `tf_attn_gate`; once `tf_plan_keep`, `tf_gdn_replay_128`, `tf_embed_rows`, the first rms, `tf_argmax_part`, `tf_argmax_join`. Candidates, each bit-exact in principle and each needing the card's sha gate: the two `tf_qk_rope` into one launch (8 fewer); `tf_softmax_stats` and `tf_plan_apply_f16` into `tf_sum_partials` with `tf_attn_gate` as its epilogue (24 fewer); `tf_plan_kv_write` into the k rope (8 fewer); `tf_conv_split` into `tf_gdn_plan_128` (24 fewer, the conv recomputed by the value-chunk blocks of a head); the argmax join into the last block of the part kernel (1 fewer); the tails into the last block of the GEMV before them (64 fewer, the largest and the riskiest: a ticket and a fence per GEMV). At about 4.4 us a launch the first five are about 65 launches, 0.3 ms (2 %) `[INFERENCE]`, which is the lower part of `LATENCY.md`'s T5 range; none is implemented.
- **With expert offload (`feat/moe-expert-offload`).** A model whose routed experts are offloaded decodes eager with value waits, which a queued round cannot do, so `facts` leaves `gpu_tokens` off there and every round is a verify window as before; the token store and the in-graph top-k apply to its replayed rounds only.

## Reproducing the numbers on this page (command lines as run)

Everything below ran on one Mac (M4 Pro, macOS 27.2) with the card booted by `ut4g-usbgpu`'s `hipmac/tools/ut4g-boot` and `hipmac-smi --set-dpm auto`. Each line is marked **V** (copied from the
shell call that produced the numbers) or **R** (reconstructed from the pieces). The recorded run scripts and the raw outputs (`serve_bench` JSON, server logs, metrics CSVs) are kept in
`ut4g-usbgpu`, `artifacts/results/` (index: `artifacts/results/README.md`); `docs/REPRODUCE.md` there walks through all of it from a clean Mac.

Environment of the builds and runs (**V**, `/tmp/up_all.sh`, `/tmp/k2/env_up.sh`; adapt the paths):

    export PATH=$PWD/tools/vega-macos/xcrun-metal-shim:$HOME/.local/zig-0.17.0:$PATH           # the shim only for the Metal targets
    ROCM=$HOME/.cache/tensorfold-vega/rocm-10.1.0a20260822                                      # zig-out/bin/tf-fetch-rocm
    export HIPMAC_BACKEND=ut4g TENSORFOLD_HIP=<ut4g-usbgpu>/hipmac/build/libamdhip64.dylib      # card; HIPMAC_BACKEND=mock for the offline tests
    export UT4G_METRICS_LOG=/tmp/metrics.csv:1                                                  # optional: one telemetry sample a second

Card state before a run (**V**): `cd ut4g-usbgpu && AM_FW_DIR=$PWD/work/fw python3 hipmac/tools/ut4g-boot` (boots when needed; `--force` after a wedge, "GC TLB invalidation did not
acknowledge" or a killed test binary), then `hipmac/dev/build/hipmac-smi --set-dpm auto`. Since ut4g-usbgpu `16dde97` the boot leaves the DPM range AUTO by itself; the first gate of
the rebase ran at the forced-HIGH state of a `--force` boot (80.9 tok/s). `AM_SMU_DPM=1` (powerplay on) was the KernelTune2 boot variable before that commit; without it the card stays at
the VBIOS bootup clocks (852 MHz: 64.1 tok/s against 81.6 with DPM).

Build and the test suites (**V**; `-Dhip-arch=gfx900` is the default on macOS):

    zig build -Dhip=true -Dhipcc=$ROCM/bin/hipcc -Doptimize=ReleaseFast -Dhip-arch=gfx900 native-hip tf-qwen35-test hip-kernel-build hip-affine-build hip-host-test      # about 3 min; 78/78 host tests
    zig build -Dhip=true -Dhipcc=$ROCM/bin/hipcc -Doptimize=ReleaseFast -Dhip-arch=gfx900 hip-kernel-test --summary all    # card: 157 passed, 14 skipped, 2 failed of 173
    zig build -Dhip=true -Dhipcc=$ROCM/bin/hipcc -Doptimize=ReleaseFast -Dhip-arch=gfx900 hip-affine-test --summary all    # card: All 30 tests passed
    HIPMAC_BACKEND=mock zig build hip-host-test -Dhip=true                                                                # offline: 78 passed

A cached result prints `run test cached` and does not touch the card; the runs also executed the cached test binary directly (`.zig-cache/o/<hash>/test`, cache-busted with
`-Dhip-cflags="-DTF_RERUN=1"`), or with the KernelTune2 helper that runs the `zig test` command of `hip-{kernel,affine}-build --verbose` with a `--test-filter`. A test binary that hangs and is
killed leaves the card needing `ut4g-boot --force`. `hip-gpu-test` was never run on the card. The gfx1030 identity check: build `hip-kernel-build -Dhip-arch=gfx1030` in this tree and in an
`upstream/main` checkout with the same wrapper (`--cache-dir`/`-p` apart), `llvm-objcopy -O binary --only-section=.text groupN.hsaco`, `cmp` for the 16 groups (**V**: all identical).

Gate, rounds, prefill (**V**, as `python3 tools/vega-macos/vega_gate.py` with these options, before that tool became `tf-vega-gate` (`zig build tf-vega-gate`, installed to `zig-out/bin`); each tag leaves `~/.cache/tensorfold-vega/bench/<tag>.server.log` and a row in `results.jsonl`):

    tools/vega-macos/setup-mlx-env.sh                                                           # once; or `uvx --from mlx-lm ...`
    ~/.cache/tensorfold-vega/mlxenv/bin/python tools/vega-macos/mlx_ref.py ~/models/Qwen3.5-4B-MLX-4bit ~/.cache/tensorfold-vega/ref      # references (R: no run line was recorded)
    UT4G_METRICS_LOG=/tmp/up_a_gate.csv:1 zig-out/bin/tf-vega-gate --tag up-auto-gate --speed 3                         # sha gate + decode median of 3: 81.39 tok/s
    zig-out/bin/tf-vega-gate --tag up-auto-logits --speed 0 --logits                                                    # logits: cosine min 0.998616, FAILS the 0.999 bar

The round-time and prefill tables ran with a throw-away client (`up_bench.py rounds BIN MODEL N TAG` and `prefill BIN MODEL 1 TAG`: N concurrent 128-token story requests, then unique-salted
prompts of 53/105/225/531/1024 tokens; kept in `ut4g-usbgpu` `artifacts/results/scripts/vega-4b/`); `tf-vega-gate` itself spawns `tensorfold-native serve MODEL --parallel 1 --context 4096 --port
18099 --no-thinking --temperature 0 --no-drafts`.

Serving report, Vega rows (**V**, script `up_vega_sb.sh`; port 18210; `$M` = `~/models/Qwen3.5-4B-MLX-4bit`; one server per load, stopped with SIGINT after the client):

    UT4G_METRICS_LOG=/tmp/up_sb/vega_1k.csv:1  zig-out/native-hip/bin/tensorfold-native serve $M --parallel 8 --context 2048  --port 18210 --no-thinking --temperature 0 --no-drafts > vega_1k.server.log
    uv run --quiet --with tokenizers tools/vega-macos/serve_bench.py --url http://127.0.0.1:18210 --model $M --tokenizer $M --loads 1024:512 --concurrency 1,2,4,8 --reps 2 \
       --json /tmp/up_sb/vega_1k.json --label "tensorfold-native serve (HIP, gfx900) --parallel 8 --context 2048, Qwen3.5-4B-MLX-4bit, RX Vega 56, DPM auto" > vega_1k.md 2> vega_1k.err
    # 16k: --parallel 6 --context 17600, --loads 16384:1024 --concurrency 1,2,4, files vega_16k.*  (--parallel 8 --context 17600 is refused: needs 5.51 GiB, the HIP budget leaves 4.52 GiB)

Mac-alone rows (**V**; ports 18200 / 18201; `uvx --from mlx-lm` was unpinned: the environment was mlx 0.32.4 / mlx-lm 0.32.2, from `tools/vega-macos/requirements-mlx.txt` [R]):

    (uvx --from mlx-lm mlx_lm.server --model $M --port 18200 --decode-concurrency 8 --prompt-concurrency 8 --prompt-cache-size 1 --temp 0 --max-tokens 4096 > /tmp/up_mlx4b.log 2>&1 &)
    uv run --quiet --with tokenizers tools/vega-macos/serve_bench.py --url http://127.0.0.1:18200 --model $M --tokenizer $M --reps 2 --json /tmp/up_sb/mlx4b.json --label "mlx_lm.server, Qwen3.5-4B-MLX-4bit, M4 Pro"
    # 35B-A3B: M=$HOME/models/Qwen3.6-35B-A3B-4bit, --port 18201, --reps 1, default loads and concurrency, files mlx35b.*; 16k c=8 ran out of memory on Metal (kIOGPUCommandBufferCallbackErrorOutOfMemory in the server log)

The 4B model: `uvx --from huggingface_hub hf download mlx-community/Qwen3.5-4B-MLX-4bit --revision 32f3e8ecf65426fc3306969496342d504bfa13f3 --local-dir ~/models/Qwen3.5-4B-MLX-4bit`; the 35B-A3B:
`hf download mlx-community/Qwen3.6-35B-A3B-4bit --revision 38740b847e4cb78f352aba30aa41c76e08e6eb46 --local-dir ~/models/Qwen3.6-35B-A3B-4bit` (17 files, 20.40 GB, 2 min 8 s).

## Reproducing the 35B-A3B numbers (command lines as run)

**V** = copied from the shell call that produced the numbers, **R** = reconstructed. The scripts are kept verbatim in `ut4g-usbgpu` `artifacts/results/scripts/moe/` and the raw outputs (`serve_bench` JSON, server logs with the
`experts interval` lines, gate replies) in `artifacts/results/moe-offload-bench/` and `artifacts/results/tensorfold-vega-bench/` (index `artifacts/results/README.md`).

Preconditions: card booted (`ut4g-boot`, DPM auto: since ut4g-usbgpu `16dde97` the boot leaves it there; `ut4g-boot --force` was run once at the start of the session), ~19 GiB of free unwired Mac memory
(no `mlx` process holding the 35B while the card server runs: the 17 GiB pinned host tier plus 19 GiB of an mlx process exceeds the Mac's 37.9 GiB wire limit, `hipHostMalloc` then fails at 2-3 GiB), and a fresh
`TinyGPU server` per `tensorfold-native serve` start (`pkill -f 'tensorfold-native serve'; sleep 3; pkill -f 'TinyGPU server'; sleep 1`: the dext keeps the host-memory mappings until the connection closes).
Model: `hf download mlx-community/Qwen3.6-35B-A3B-4bit --revision 38740b847e4cb78f352aba30aa41c76e08e6eb46 --local-dir ~/models/Qwen3.6-35B-A3B-4bit` (**V**, 17 files, 20.40 GB, 2 min 8 s).

Build (**V**; `/tmp/moe_env.sh` = Zig 0.17.0 and the Metal shim on `PATH`, `ROCM=$HOME/.cache/tensorfold-vega/rocm-10.1.0a20260822`):

    zig build native-hip -Dhip=true -Dhip-arch=gfx900 -Dmetal-std=metal4.0 -Dhipcc=$ROCM/bin/hipcc -Doptimize=ReleaseFast        # zig-out/native-hip/bin/tensorfold-native

Server (**V**, `/tmp/moe_serve.sh`; `PAR` and `CTX` are the only per-run knobs besides `TF_EXPERT_*`; no `--ssd-experts` was ever passed, the offload turns on by itself because 19.0 GiB do not fit the 6.9 GiB the HIP budget grants,
and the cache is what the plan leaves: 2876 lines = 4.74 GiB at `--parallel 1 --context 4096`, 2334 lines = 3.85 GiB at `PAR=8 CTX=2048`, 2192 lines = 3.61 GiB at `PAR=2 CTX=17408`; `TF_EXPERT_WINDOW_MIB` was never set, 1536 MiB):

    HIPMAC_BACKEND=ut4g TENSORFOLD_HIP=<ut4g-usbgpu>/hipmac/build/libamdhip64.dylib UT4G_METRICS_LOG=/tmp/moe_metrics.csv:1 \
      zig-out/native-hip/bin/tensorfold-native serve ~/models/Qwen3.6-35B-A3B-4bit --parallel $PAR --context $CTX --port 18099 --no-drafts --temperature 0 --no-thinking

Runs (the binary of each row is **R**: built from the working tree before the commit; the transcripts never printed the hash next to a run):

| Output (`/tmp/moe_bench/` unless named) | Server | Client | Binary |
| --- | --- | --- | --- |
| `l1k.{md,json,err}` (first 1k/512 table, COPIES 32) | `PAR=8 CTX=2048` | `uv run --quiet --with tokenizers python tools/vega-macos/serve_bench.py --url http://127.0.0.1:18099 --model Qwen3.6-35B-A3B-4bit --tokenizer ~/models/Qwen3.6-35B-A3B-4bit --concurrency 1,2,4,8 --loads 1024:512 --reps 1 --timeout 3000 --json l1k.json --label ...` (**V**) | `579a6ffea7` |
| `b1k_a.*` (current 1k/512 table, plain LRU) | `PAR=8 CTX=2048`, COPIES default 4096, `TF_EXPERT_PROTECT` unset | `/tmp/bench_run.sh b1k_a 1024:512 1,2,4,8 8 2048` (**V**; the script starts the server above and runs `serve_bench.py ... --concurrency 1,2,4,8 --loads 1024:512 --reps 1 --json /tmp/moe_bench/b1k_a.json --label b1k_a`) | `4c671bbcc8` (R) |
| `ab_lru.*`, `ab_slru.*` (filler 1k/256, SLRU A/B) | `PAR=8 CTX=2048` + `TF_EXPERT_PROTECT=0` / `75` | `/tmp/bench_run.sh ab_lru 1024:256 1,4,8 8 2048 TF_EXPERT_PROTECT=0`, `... ab_slru ... TF_EXPERT_PROTECT=75` (**V**) | `07b09d3709` (R) |
| `b16k.*` (16k/1k, SLRU 75 %) | `PAR=2 CTX=17408` | `/tmp/bench_run.sh b16k 16384:1024 1,2 2 17408` (**V**) | `07b09d3709` |
| `l16k_c1.{md,err}` (empty: this run failed, nothing was recorded; c=4/8 at 16k were never run) | `PAR=1 CTX=17408` | `serve_bench.py ... --concurrency 1 --loads 16384:...` | |

Gate (`tools/vega-macos/moe_ref.py`, `tf-moe-gate`; the mlx environment `tools/vega-macos/setup-mlx-env.sh`): references **V** `~/.cache/tensorfold-vega/mlxenv/bin/python tools/vega-macos/moe_ref.py` (the first run had to wait until no
card server held host memory: "Insufficient Memory"; 8 prompts since the 466- and ~1100-token Merrowick prompts, shas `eb415dcbaf6e 188a33e44468 be83e20c3349 dba6cb89062a ed065dc57de4 554c5b65193a ff333359d187 037a3176d0c5`), then
`/tmp/gate_pair.sh TAG` (**V**, with `moe_gate.py`, whose output and result file `tf-moe-gate` reproduces: server with the cache on, `tf-moe-gate --log SERVER.LOG --tag TAG_on` against the references; server with `TF_EXPERT_COPIES=0`, `--tag TAG_off`; then a python one-liner compares the two replies: all True).
Result at every build (`g1`, `g3`): the cache-on and cache-off replies are identical for all 8 prompts, and 3 of 8 equal the mlx references; the five others diverge at a near tie of the mlx logprobs (`/tmp/margin.py`: ' the' -0.876 vs
' shorter' -0.876, ' due' -0.596 vs ' in' -0.846, ' meet' -0.973 vs ' marry' -1.348, ...), [INFERENCE: different arithmetic, not diagnosed]: the gate that must hold is cache on = cache off (`--against`).

Story benchmark (**V**, `/tmp/exp.sh TAG [ENV=VAL]`): start the server, three decodes of "Write a long story N about a dragon who learns to code." (N = 1..3, `max_tokens` 200 (300 for the 26-28.7 rows), `temperature 0`, thinking off) through `curl` on
`/v1/chat/completions`, the number is the server's own `tensorfold.tokens_per_second`: `exp.sh c32 TF_EXPERT_COPIES=32` gave 11.14 / 22.06 / 20.79 tok/s; before read-through admission 20.89 (hipprof run), after it 26.12 / 28.73 / 26.94.
Per-kernel table of a decode (**V**, `/tmp/prof_run.sh TAG`): `hipprof` shim with `HIPPROF_EVENTS=1`, `TF_POLICY=graphs=off`, `kill -USR1/-USR2` around a profiled 200-token decode, then `tf-prof-report TABLE` (`zig build tf-prof-report`; it was `prof_report.py` for the recorded runs).

Cache simulator (**V**, with `expert_cache_sim.py`; now `zig build tf-expert-cache-sim`, `zig-out/bin/tf-expert-cache-sim [TRACES_DIR] [--long DIR] [--cap N] [--jobs N] [--only STR]`, same options and the same report, compared on the study's traces: 1.4 s on 8 workers against 54 s for the Python): the default is the study's traces (`~/Projects/Infra/tf-study-moe-cache/docs/proposals/metal-tb4-split-moe-probe/traces` and `../traces_long`; the original run took 12-17 s on 14 workers);
the output of the last run is `ut4g-usbgpu` `artifacts/results/moe-offload-bench/expert_cache_sim_output.txt` (LRU 82.9 %, SLRU 75 % 83.6 %, LFU aged 83.7 %). The traces themselves come from the study (`routing_trace.py`, see its proposal "Reproduce").

Mac-alone 35B (**V**): `uvx --from mlx-lm mlx_lm.server --model ~/models/Qwen3.6-35B-A3B-4bit --port 18201 --decode-concurrency 8 --prompt-concurrency 8 --prompt-cache-size 1 --temp 0 --max-tokens 4096`, then `serve_bench.py --url http://127.0.0.1:18201
--model $M --tokenizer $M --reps 1 --json mlx35b.json` with the default loads and concurrency (decode 72.6 / 117.7 / 161.4 / 178.4 tok/s at c = 1/2/4/8 on 1k/512; 16k c = 1 63.2; 16k c = 8 out of memory). The other Mac-alone figure, 88.8 / 135.8 / 162.4 / 176.4 / 206.9 tok/s,
is `mlx_lm` `batch_generate` through `docs/proposals/metal-tb4-split-moe-probe/mlx_batched_decode.py MODEL 1,2,3,4,8 1 40` (distinct 9-token prompts, 40 tokens, best of 2): a different harness, not reconciled with the server figure.
