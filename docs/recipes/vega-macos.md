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
- Offline here: all 16 gfx900 groups compile (`hip-kernel-build -Dhip-arch=gfx900`), `hip-host-test` passes (mock), and the gfx1030 `.text` of all
  16 groups is byte for byte upstream main's. Not run: any kernel on the card, `hip-kernel-test`, `hip-affine-test`, the sha gate, tok/s,
  prefill tok/s on this branch.

## Decode on the card (Vega 56 through hipmac, TinyGPU)

Measured with `tools/vega-macos/vega_gate.py` (starts `tensorfold-native serve` on `libamdhip64.dylib`, checks the greedy tokens against
the mlx_lm references of `tools/vega-macos/mlx_ref.py`, then runs story decodes and reads `tensorfold.tokens_per_second`). Every number
in this section was measured on the older base the branch had before it was rebased onto upstream main; none was re-measured on the
rebased branch, which differs in the ways listed under Status.

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

- Logits cosine against mlx on the card (`vega_gate.py --logits` exists; the greedy tokens match for the checked prompts).
- The default `--mtp-drafts 3` of a discrete GPU is irrelevant for this checkpoint (it has no MTP head).
- Everything above on the card with this branch: tokens against mlx_lm (the sha gate), decode tok/s (the older base's 64.8), prefill tok/s
  (the older base's ~200), the hardware tests (`hip-kernel-test`, `hip-affine-test`, `hip-gpu-test`), and the recorded hash pins of the stream
  test (`pinned_f16_gfx900_flush` was recorded on the older base's tile).

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
