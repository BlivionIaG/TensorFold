# Issue draft: model family, backend or arithmetic proposal

Draft for `.github/ISSUE_TEMPLATE/proposal.yml`. Not posted. Title: `Proposal: TB4 external-GPU offload for Metal hosts
(layer split, MoE expert split, computed placement)`. Full design:
[metal-tb4-split-moe.md](metal-tb4-split-moe.md).

## Kind of proposal

- [x] Distributed execution
- [x] Other

## Decision requested

Should a Mac host be able to serve one model across Metal and an external GPU reached over Thunderbolt 4, inside the
native engine, under these terms?

1. A device-agnostic offload layer inside the native engine (the host stage in the Mac's process, the card's runtime in a second
   process over a unix socket, the window shared memory both map; one binary cannot link both engines today), that reuses the cluster planner, budget, admission and cost code and the
   fabric framing, but not the cluster transport or SSH launch.
2. A boundary rule between stages of different activation types (bf16 Metal, fp16 gfx900 HIP; bf16 on RDNA3 and newer
   and on NVIDIA): fp32 hidden row across the boundary, round-to-nearest-even into the consumer's type, same-engine
   equality as the hard gate, and an FP64-referenced fidelity gate for mixed-type placements.
3. The gate under "Proposed gate" and the pull-request order: (1) transport, offload interface and dense layer split;
   (2) MoE layer split and expert split; (3) hot-expert cache and adaptive placement.

## Scope and benefit

What changes. A placement planner computes, from the model config, the Mac's memory and measured Metal bandwidth, the
external GPU's VRAM and bandwidth, and a startup link probe, where each layer, expert class, KV cache and state live,
and refuses with the missing quantity when the budget does not close. Placements: all-Metal, fully resident on the
external GPU (Mac is host and overflow), layer split (external GPU runs layers `[0,k)` plus the embedding, Metal the
rest, one hop each way), expert split and reverse expert split for MoE (two hops a split layer). The external side is
any TensorFold backend that supplies a pinned host window, a device-side wait and write on a 32-bit word, a launch plan
and a timestamp; HIP through the out-of-tree `hipmac` runtime on gfx900 today, RDNA3/RDNA4 and NVIDIA later through the
same interface. Metal side gains a `HostWindow` (`newBufferWithBytesNoCopy` over the memory the device DMAs) and a
`Relay` that pre-commits event-gated command buffers.

Who gains. Mac owners with an external GPU: capacity (a 24 GiB Mac cannot hold the 19.7 GB peak of Qwen3.6-35B-A3B
4-bit but can hold about 12 GiB of it beside a Vega 56's 7.5 GiB), aggregate throughput with two or more streams, and,
with a stronger card, speed (modelled 158 tok/s resident on a 24 GiB RDNA3-class card and 299 on a 32 GiB NVIDIA-class
card against 91.6 all-Metal [INFERENCE]).

What the native engine supports today. HIP serves Qwen3.5 dense and MoE on gfx900 (Qwen3.5-4B 52-57 tok/s documented);
upstream `main` serves `qwen3_5_moe` on Metal. Cluster supports Metal and CUDA nodes over network fabrics only and
refuses heterogeneous execution; nothing splits one model across two devices of one host.

Honest limits. Measured on this M4 Pro, all-Metal decodes the 35B-A3B at 91.6 tok/s. On the Vega card the single-stream
split is modelled at or below that (84 tok/s layer split, 52 expert split); the gain there is capacity and 2+ streams
(131 tok/s modelled). The speed case needs a faster card. A Metal kernel cannot poll a flag written by another agent
(measured), so every handoff costs a command-buffer wake of about 65 us on an idle Mac and 149 us (p50) with the card in
use; the layer split pays it once a token, the expert split forty times. First card run (Qwen3.5-4B, 32 layers): a split
at any of five cuts, Metal first or card first, gives the unsplit run's tokens (40 of 40) and logits cosine 0.9996-0.99999;
one stream takes 15-24 ms a position against 14 ms all-Metal and 19 ms all-card, as the one-stream model says. Two sequences pipelined over the split decode 120 tok/s at the planner's cut (113 predicted) and 129-130 at the best cuts, every
sequence token-identical; the fair baseline, the Mac alone batching two sequences in one engine, decodes 116 tok/s (155 at three, 198 at four, where the
split stays at 120), so on the 4B the split is level at two sequences and behind beyond, and its value is capacity. Chunked prefill through the split is 3x the
one-position-a-step rate (156-251 tok/s) but the Mac alone prefills at 415-494. The 35B-A3B on the Mac alone (mlx_lm; the native Metal engine does not admit its shape) decodes 88.8 / 135.8 / 176.4 / 206.9 tok/s at
1 / 2 / 4 / 8 sequences; the written split (one row a step) cannot beat that beyond two sequences, and the model says a split whose stages batch
rows of several sequences would be 1.1-1.2x ahead on the Vega and that a card holding the model should run it resident (needs a whole-model one-row decode above 89 tok/s).
The 4B's Mac baseline is also higher with mlx_lm (154.6 at two sequences against the split's 120-130).

## Checkpoint, revision and licence

- `mlx-community/Qwen3.6-35B-A3B-4bit` at `38740b847e4cb78f352aba30aa41c76e08e6eb46` (MLX affine 4-bit, group 64,
  routers 8-bit, bf16 activations; 40 layers, 30 Gated DeltaNet + 10 full attention, hidden 2048, 256 experts top-8,
  expert width 512, one shared expert). Apache-2.0 (base `Qwen/Qwen3.6-35B-A3B`, Apache-2.0). The recipe's checkpoint
  `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` at `81169a9bc511a27c1b4eedb77a2cd98ced431847` adds the MTP head.
- First slice (dense): `mlx-community/Qwen3.5-4B-MLX-4bit` at `32f3e8ecf65426fc3306969496342d504bfa13f3`.
- The feature itself is model-generic; no weights are converted or re-quantized.

## Backend

- [x] Metal
- [x] CUDA (later; the interface is shaped for it, no CUDA work proposed first)
- [x] HIP

## Chips, precision and stored arithmetic

Chips: Apple M4 Pro (tested); any Apple Silicon Mac is unverified. External: RX Vega 56 (gfx900) tested at the
repository level only; RDNA3/RDNA4 and NVIDIA unverified. Checkpoint precision unchanged; no stored weight or cache is
converted. Each stage keeps its own backend's activation type (bf16 on Metal, fp16 on gfx900 where the engine has no
bf16 dot). The stage boundary carries the hidden row as fp32, which widens bf16 and fp16 exactly; the consumer rounds
into its own type with round-to-nearest-even, and an fp16 consumer refuses a request whose residual exceeds 65,504 with a
named error. With both stages bf16 the boundary is the identity. Slot sums across the boundary stay the sequential fp32
sum in slot order. No lossy KV, no FP8 activations, no token-changing option.

## Shared modules to extend

- `zig/src/cluster/plan.zig`, `budget.zig`, `admission.zig`, `cost.zig` and `zig/src/fabric/cost.zig` (labelled link
  constants; add a measured PCIe-tunnel link kind and correct the Apple-GPU polled-handoff entry, which this machine
  contradicts); `Backend.hip`; role classification for Qwen tensor names.
- Fabric channel framing and the fakes (`fake.zig`, `toy.zig`) for the window protocol tests.
- `zig/src/metal/` (`Device.bufferNoCopy`, `SharedEvent`, residency) for `HostWindow` and `Relay`.
- `zig/src/hip/` and `families/qwen3_5/backend/hip/forward/` (a layer range; later the expert seam beside the existing
  expert-parallel partial-sum hooks in `moe.zig`).
- New `zig/src/core/split/` only for what has no home: the window protocol, the `ExternalDevice` vtable and the planner
  glue. Cluster transports are not reused because they are host-to-host (MCDMA over SSH-launched ranks), and round
  messages carry token ids rather than activations.

## Smallest first slice

PR 1: `core/split` (window protocol, `ExternalDevice`, fakes), Metal `HostWindow` and `Relay`, `tensorfold offload probe`,
the planner for dense models with a strict `placement.json` (`tensorfold-split/1`), a layer range in the HIP and Metal
engines, `--device-split auto|PATH` for dense checkpoints, `Backend.hip` and the link kind in cluster. It leaves out MoE,
the expert split, hot-expert caching, vision and any non-gfx900 device. It depends on `hipmac` for the card receipt:
device-side wait/write on a host word (`hipStreamWaitValue32`/`WriteValue32`-class; the window allocation exists). The
mock device and no-GPU tests do not depend on it. Stacked pull requests link the issue with `Refs #N`.

## Proposed gate

Proposed here, not agreed.

- Exactness: for a fixed placement and every admitted stream count, drafted equals no-drafts, resumed equals fresh,
  concurrent equals solo, a row's bits do not depend on the window; a toy dense and toy MoE family run split on fake
  devices token-for-token equal to unsplit; a cut link ends in a named error inside the timeout and the next request is
  valid.
- Fidelity for mixed activation types (Vega with Metal): per projection class and at the boundary, native error against
  an FP64 reference no worse than the worse single-device engine; teacher-forced histories with every top-token
  difference and margin reported against all-Metal. Same-type stages: split equals unsplit token for token on fixtures
  that fit one device.
- Performance: repository clients on the same machine, revision and prompts as the unsplit baseline; decode and cold
  prefill; the planner's prediction within 15%; `auto` never selects a placement it predicts slower than all-Metal.
  Targets for the first card receipt: link round trip at 8 KiB p50 <= 250 us and p99 <= 500 us; Qwen3.5-4B split with
  two streams at least level with the Mac alone batching two sequences (measured: 120-130 against 116); single stream within 5% of all-Metal (not met, 16-24 ms against 14).
- Memory: peak on both devices, window bytes, RAM class and context; a run under a 24 GiB-class budget.
- Platforms: M4 Pro run; other Apple chips unrun. gfx900 run; RDNA3/4 and NVIDIA unrun.
- Kernel-contract records for any new kernel; `zig build test test-golden`, `dev-tools`, lean base and head counts.

## Gate status

Proposed here

## Link to the gate decision

(none yet)

## Hardware available

One MacBook Pro with M4 Pro (14 CPU, 20 GPU cores, 48 GiB, macOS 27.2) and one RX Vega 56 (8 GiB) on a Thunderbolt 4
enclosure through TinyGPU's signed driver extension and the out-of-tree `hipmac` runtime. Needs other testers: other
Apple chips and RAM classes (especially 24 GiB, where capacity matters), RDNA3/RDNA4 and NVIDIA cards through TinyGPU,
Thunderbolt 5 hosts. Offline probes ran without the card: Metal window wrap, command-buffer and event wake
latency, memory-visibility rules, MLX expert throughput, all-Metal decode; see the probe directory next to the design.
On the card: the startup probe and the 4B layer split (design, First card run).

## Related issues, pull requests and earlier attempts

None found for same-host Metal plus external GPU execution. Context: the cluster and fabric documentation
(`app/docs/clustering.md`, which refuses heterogeneous Metal/CUDA execution until an engine profile exists), the remote
engine notes, `docs/recipes/qwen3.6-moe.md` and `docs/recipes/vega-macos.md`. The earlier idea of streaming cold experts
to a small GPU over the link is rejected in the design with numbers (177 ms a token if all cold).

## Privacy

- [x] This proposal holds no private paths, hostnames, tokens, keys or confidential prompts.
