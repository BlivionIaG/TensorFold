# Qwen3.5-4B on an RX Vega 56 (gfx900) from an Apple Silicon Mac

The HIP engine serves Qwen3.5-4B MLX 4-bit on a Vega 10 card (RX Vega 56, gfx900: wave64, 56 CUs, 8 GiB HBM2) attached to an
M4 Pro MacBook Pro over Thunderbolt. There is no ROCm, no Linux and no tinygrad at run time: the engine opens the runtime library
by its macOS name, `libamdhip64.dylib`, which is the native HIP runtime of jkuepker/ut4g-usbgpu `hipmac/` (out of tree;
docs/HIPMAC.md there, <https://github.com/jkuepker/ut4g-usbgpu>). This page is the TensorFold side:
the toolchain that builds the kernels, the port of the kernels to gfx900, the checkpoint, and what has and has not been run.

Status: everything below that is marked built or tested was run on the Mac without the card. No kernel of the
gfx900 code objects has run on a GPU yet; the first run on the card is the next step (see "Not yet verified").

## Toolchain on macOS: `tools/vega-macos/`

`hipcc --genco` needs a ROCm install. On macOS the pieces are taken apart:

| Piece | Source | Pin |
| --- | --- | --- |
| compiler, linker | Homebrew `llvm` and `lld` (clang 23, the version of the ROCm wheel's own clang) | llvm 23.1.3, lld 23.1.3 (`brew install llvm lld`) |
| HIP headers `include/hip/**` | TheRock nightly wheel `rocm_sdk_core`, extracted by `fetch-rocm.sh` | rocm 10.1.0a20260822, HIP 7.16.26332 |
| device libraries `lib/llvm/amdgcn/bitcode/*.bc` (ocml, ockl, oclc_*, hip, opencl) | the same wheel | same |
| libc++ and libm headers the device compile includes | the Xcode macOS SDK (`xcrun --show-sdk-path`), passed as `-isysroot` | SDK 26.5 here |

The wheel is `rocm_sdk_core-10.1.0a20260822-py3-none-linux_x86_64.whl` from `https://rocm.nightlies.amd.com/whl-multi-arch/`
(the index jkuepker/ut4g-usbgpu's `pins.json` pins for its ROCm stack), 416,591,253 bytes, sha256
`cf376d3465cffd0c755a1e1ec08324c519eaf3d79c56778ffede1f6a7a2a5729`. Only 163 files of it are used (6 MB); its Linux x86-64
compiler and libraries are not. `tools/vega-macos/pins.env` holds the pins; `fetch-rocm.sh` checks the sha256.

```sh
brew install llvm lld                                  # clang 23: AMDGPU target, offload bundler
ROCM=$(tools/vega-macos/fetch-rocm.sh)                 # ~/.cache/tensorfold-vega/rocm-10.1.0a20260822 (or --wheel PATH)
export PATH=$HOME/.local/zig-0.17.0:$PATH              # Zig 0.17.0 (.zig-version), unpacked from ziglang.org; no sudo
zig build native-hip tf-qwen35-test -Dhipcc=$ROCM/bin/hipcc -Doptimize=ReleaseFast
```

`$ROCM/bin/hipcc` is a symlink to `tools/vega-macos/hipcc`, a bash wrapper that behaves as `hipcc --genco` does for what
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

Measured with `tools/vega-macos/vega_gate.py` (starts `tensorfold-native serve` on `libamdhip64.dylib`, checks the greedy tokens against
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
| sha gate, nothink / think96 (`vega_gate.py`) | `eb415dcbaf6e` / `bf86edaf9c17`, identical to mlx_lm |
| decode N = 1 (`vega_gate.py`, median of 3 128-token stories) | 81.4 tok/s (81.3-81.5); 80.9 in an earlier gate run with the card at DPM HIGH after a `--force` boot |
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

Logits against mlx_lm (`vega_gate.py --logits`, 87 rows through the decode path): cosine min 0.998616, mean 0.999881, argmax equal 87/87 (the older base
with `TF_G9_MIX=1`: 0.99857 / 0.99988 / 87/87). The script's own bar is cosine min above 0.999, which this does not reach on either base; the argmax gate passes.
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
tok/s). Logits against mlx_lm (87 rows through the decode path, `vega_gate.py --logits`, MIX=1): cosine min 0.99857, mean 0.99988,
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

Profiling: `tools/vega-macos/hipprof.c` is a shim `libamdhip64` that forwards to the runtime and times `hipModuleLaunchKernel`
(`HIPPROF_REP=N` runs N extra back-to-back launches of each kernel for the pipelined time); `vega_gate.py --prof each` builds it,
`prof_report.py` prints the per-token table. `--no-gate --story-tokens 16` is enough for a profile.

## Not yet verified

- Logits cosine against mlx on the card (`vega_gate.py --logits` exists; the greedy tokens match for the checked prompts). `zig build
  tf-qwen35-test` does not compile on upstream main (`zig/tests/qwen35/main.zig:170`), so `--logits` cannot run on this branch.
- The default `--mtp-drafts 3` of a discrete GPU is irrelevant for this checkpoint (it has no MTP head).
- `hip-gpu-test`, and the stream test's other recorded pins on a second card.

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

## Building on a Mac whose Xcode license is not accepted

`clang`, `xcrun` and `make` stop with "You have not agreed to the Xcode license agreements" until the license is accepted (needs sudo). The builds
here do not need Xcode, only the command line tools' SDK and Zig's C compiler:

    export PATH=$HOME/.local/zig-0.17.0:$PATH
    export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk TF_SDKROOT=$SDKROOT
    export TF_CC="zig cc"                      # tools/vega-macos/vega_gate.py builds the hipprof shim with it (default clang)

`zig build test` (the Metal tests) and `make -C hipmac` (clang) do not work without it.

The Metal compiler of Xcode 27 is a downloadable component (`xcodebuild -downloadComponent MetalToolchain`, which needs the license too). If it was
downloaded before, it sits in `/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin` and runs
without Xcode: a script named `xcrun` first on `PATH` that drops `-sdk macosx`, answers `--show-sdk-path` with the command line tools' SDK and
execs `metal`, `metallib` and `metal-ar` from that directory makes `zig build test-split`, `tf-offload-metal` and `test-qwen27-split` build with
`-Dmetal-std=metal4.0`. The script is not part of the tree.

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
