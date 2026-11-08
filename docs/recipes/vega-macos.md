# Qwen3.5-4B on an RX Vega 56 (gfx900) from an Apple Silicon Mac, fork branch `vega-macos`

The HIP engine serves Qwen3.5-4B MLX 4-bit on a Vega 10 card (RX Vega 56, gfx900: wave64, 56 CUs, 8 GiB HBM2) attached to an
M4 Pro MacBook Pro over Thunderbolt. There is no ROCm, no Linux and no tinygrad at run time: the engine opens the runtime library
by its macOS name, `libamdhip64.dylib`, which is the native HIP runtime of
[ut4g-usbgpu `hipmac/`](https://github.com/jkuepker/ut4g-usbgpu) (docs/HIPMAC.md there). This page is the TensorFold side:
the toolchain that builds the kernels, the port of the kernels to gfx900, the checkpoint, and what has and has not been run.

Status of this branch: everything below that is marked built or tested was run on the Mac without the card. No kernel of the
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
(the index the ut4g-usbgpu `pins.json` pins for its ROCm stack), 416,591,253 bytes, sha256
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
export DYLD_LIBRARY_PATH=<hipmac>/build         # libamdhip64.dylib (and libut4ghip.dylib), or:  TF_HIP_LIB=<path>/libamdhip64.dylib
tensorfold-native serve ~/models/Qwen3.5-4B-MLX-4bit --parallel 1 --context 8192
```

`hip.Runtime.open()` loads `libamdhip64.dylib` on macOS and `libamdhip64.so` elsewhere; `TF_HIP_LIB` names the file on either
(an install outside the loader's search path, or the offline mock: `HIPMAC_BACKEND=mock`).

The engine's HIP calls in a start and one request (the mock library behind a recording shim, 9,149 calls): `hipInit`,
`hipDeviceGet`, `hipDevicePrimaryCtxRetain`, `hipCtxSetCurrent`, `hipDeviceGetAttribute`, `hipGetDevicePropertiesR0600`,
`hipMemGetInfo`, `hipModuleLoadData` (10 code objects), `hipModuleGetFunction` (335 kernels), `hipModuleGetGlobal` (1),
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
| `TF_FDOT2(x, q, acc)`: `__builtin_amdgcn_fdot2` where the instruction exists (a macro, so the objects stay the same bytes); on gfx900 the two products fused one after the other into the accumulator (each product is exact in fp32) | `common/fdot2.hpp`; attention_fa, prefill, gdn_prefill, dot2.hip, `DotF16` | byte-identical |
| `DotF16::quad` (x of a quad lane by DPP): the `v_dot2c_f32_f16_dpp` asm stays RDNA2's; others move x by `mov_dpp quad_perm` then `TF_FDOT2` | `common/dot2.hpp` | byte-identical |
| `narrow_product` (silu(gate) * up rounded once to fp16): `v_fma_mixlo_f16` asm stays; on gfx900 the exact product in double narrowed to half by the compiler's correctly rounded `fptrunc` | `tiles/epilogue.hpp` | byte-identical |
| `readfirstlane(tid >> 5)`, a "wave" index held in a scalar register, is wrong on wave64 (the upper group of a wavefront would take the lower's index): `group32(tid)` keeps it on wave32 | `common/wave.hpp`; gemm.hpp, gdn_prefill | byte-identical |
| top-k's `__ballot` compaction counted lanes `& 31` of a 64-bit mask: lane and wave from `TF_WAVE` | `ops/draw.hpp` (`tf_topk`) | byte-identical |
| `kGen` and `kWave` for gfx900; `static_assert` of the wave width against the gfx9 / RDNA arch | `common/arch.hpp` | unchanged |
| a registry cost table for gfx900 (`src/hip/tuning/gfx900.zon`, generation `gcn5`); `chip()` names the generation `gcn5` for the gates | `src/hip/launches/affine.zig`, `mlx_entries.zig`, `native/hip.zig` | unchanged |

The RDNA2 (gfx1030) code objects of all ten kernel groups, built by the same wrapper before and after the port, are byte for byte
the same (`cmp`). The gfx11 objects cannot be built with Homebrew LLVM (above) and were not compared.

All ten kernel groups (335 kernels) compile for gfx900 with no unported kernel left:

| Group | Kernels | gfx900 |
| --- | --- | --- |
| `ops/ops.hip` (norms, rope, elementwise, argmax, top-k, MoE router, TP) | 30 | ported (draw.hpp) |
| `ops/act.hip` | 17 | wave-32-group code, unchanged |
| `attention/attention.hip` (decode attention, FA2 prefill) | 14 | `TF_FDOT2` |
| `recurrence/gated_delta.hip` | 8 | unchanged |
| `attention/prefill.hip` (64-row prefill attention) | 4 | `TF_FDOT2` |
| `recurrence/gdn_prefill.hip` (chunked DeltaNet) | 5 | `TF_FDOT2`, `group32` |
| `decode/decode.hip`, `decode/plan.hip` | 7 + 9 | unchanged |
| `tiles/dot2_tiles.hip` (stream, row, lanes, GEMM, K-parallel tiles; 2-8 bits) | 234 | `DotF16`, epilogue, `group32` |
| `tiles/dot2.hip` (reference, wide, tiled, split) | 7 | `TF_FDOT2` |

Not on gfx900, by design: the matrix-core GEMM tile (`matrix_gemm.hpp`, WMMA, gfx11) and the BF16 dot2 tiles. They are compiled
out by `TF_MATRIX=0` and `TF_DOT2_BF16=0`, and the registry never selects them for a `matrix = none`, `act = f16` GPU.
gfx900 takes the RDNA2 fp16 family; that family is one the engine already serves on the Deck.

Register pressure and scratch (from each kernel's amdhsa metadata; no timing yet): the decode stream tile, the row and lanes
tiles and every non-GEMM kernel fit without spills, except `tf_gdn_h`, the DeltaNet prefill state kernel, which uses 68 bytes
of scratch a lane (so the runtime must provide scratch: a private segment with the kernel's `private_segment_fixed_size`).
The prompt GEMM tiles spill: the 128-row tile (`affine_gemm_block`) 195 to 225 of 256 VGPRs (428 to 460 bytes of scratch), the 64-row
tile 44 to 60, the K-parallel tiles 15 to 79; the 32-row tile spills nothing. `gfx900.zon` therefore never selects the 128-row
tile: a prompt takes the K-parallel tiles to 24 rows, the 32-row tile to 32 and the 64-row tile past that; a decode call of 64
rows or more takes the reference block tile. All of the tiles of a family give the same bits, so this changes speed only.
The registry's tests assert it. Costs are the thresholds gfx1030's table carries, not measurements on a Vega; a measured table
replaces them.

Warp-synchronous code that was lock-step on 32 lanes is lock-step on 64, and `__syncwarp()` is a wavefront barrier, so the
`gemm_kp` tile's LDS hand-offs between the two groups of a wavefront stay ordered. A block of 32 threads (`gated_delta`, the
decode attention, the single-wave launches) fills half a wavefront.

Differences in results from RDNA2 that the port can produce, because gfx900 has no dot2 instruction: a dot2's sum rounds once
(the instruction) or twice (two fused steps), so the bits of an fp16 product on gfx900 are the Python reference's, not
necessarily the RDNA2 golden files'. Every tile on gfx900 uses the same fused pair, so the tiles agree with each other (the
engine's own bit rules: solo equals batched, resumed equals fresh).

## Offline verification (Mac, no card)

| Check | Result |
| --- | --- |
| `zig build native-hip tf-qwen35-test -Dhipcc=$ROCM/bin/hipcc` | builds; Mach-O arm64 executables |
| `zig build hip-kernel-build hip-affine-build hip-gpu-build` | build (GPU tests compile, not run) |
| `zig build test -Dhipcc=... -Dmetal-std=metal4.0` | 443 passed, 1 skipped, 0 failed (includes the caps, registry and tuning tests for gfx900) |
| `TF_QWEN_DIR=~/models/Qwen3.5-4B-MLX-4bit zig build test-qwen35` | the 4B checkpoint indexes (loader tests) |
| ten code objects for gfx900, all 335 kernels | build; every kernel resolves through the mock runtime's `hipModuleGetFunction` |
| `HIPMAC_BACKEND=mock TF_HIP_LIB=<hipmac>/build/libamdhip64.dylib tensorfold-native serve ~/models/Qwen3.5-4B-MLX-4bit` | starts, loads, captures 7 graphs, replays 153 of 160 rounds, answers a chat request (the mock does not run kernels, so the text is empty) |

## Not yet verified

- Any kernel on the card; every number in `gfx900.zon` is gfx1030's or a spill-driven guess.
- `hip-gpu-test`, `hip-affine-test`, `hip-kernel-test`, `tf-qwen35-test check` on a Vega: run
  `zig build hip-kernel-test -Dhipcc=$ROCM/bin/hipcc` and the others with `HIPMAC_BACKEND` unset once the card is the runtime's.
  [INFERENCE] The affine golden test runs `affine.hip`, which has no dot2 and only width-32 shuffles, so it should pass as on RDNA.
- The runtime's code-object parser (shared by the mock and the card backend) accepts all ten objects; that the card's device layer
  runs them, including the scratch of `tf_gdn_h` and the xnack-"any" flag, is not tested.
- The default `--mtp-drafts 3` of a discrete GPU is irrelevant for this checkpoint (it has no MTP head).
