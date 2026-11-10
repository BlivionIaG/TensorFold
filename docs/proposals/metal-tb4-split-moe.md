# TB4 external-GPU offload for Metal hosts: layer split, expert split, computed placement

Status: proposal. The offline part of the first pull request is implemented on this branch (see
[Implementation status](#implementation-status-pull-request-1-offline)), including the engines' layer range; the
MoE models are not, and the card ran once, for the Qwen3.5-4B layer split (see First card run). The measurements are real and reproducible
(commands and raw output under [metal-tb4-split-moe-probe](metal-tb4-split-moe-probe/)); everything marked
[INFERENCE] is a model or an assumption that a named milestone replaces with a measurement.
The companion [issue draft](metal-tb4-split-moe.issue.md) follows `.github/ISSUE_TEMPLATE/proposal.yml`.
Upstream keeps design notes in `docs/design/`; this file lives in `docs/proposals/` on the working branch and moves
there if the proposal is agreed.

## Summary

A Mac with Apple Silicon can serve a model across two devices on the same host: Metal on the SoC, and any TensorFold
backend on an external GPU that reaches the Mac through Thunderbolt 4 (a PCIe tunnel; today an RX Vega 56 under the
HIP backend and the out-of-tree `hipmac` runtime, later AMD RDNA3/RDNA4 through the same runtime, and NVIDIA through
TinyGPU's NVIDIA path). The feature is one generic transport, one device-agnostic offload interface, and a planner
that computes where layers, experts and the KV cache live from measured hardware numbers, then three placements built
on them:

1. **Layer split** (dense and MoE models): the external GPU runs layers `[0, k)` plus the embedding, Metal runs
   `[k, L)` plus the head, or the reverse. One hop each way per token. This is the simplest placement, the first
   pull request, and the one the numbers favour on today's card.
2. **Resident**: when the external GPU holds the whole checkpoint plus its caches, it runs everything and the Mac is
   host, tokenizer and overflow.
3. **Expert split** (MoE): attention, mixers, shared expert, router, KV and some expert layers on the external GPU;
   routed experts of the other layers on Metal (or the reverse when the external GPU is much faster). Two hops per
   split layer, so it needs the most careful synchronization.

What the measurements say, on this M4 Pro (48 GiB) and the Vega card:

| Question | Answer |
| --- | --- |
| Does the offload beat plain Metal for one stream on the Vega card? | No. All-Metal decodes the 35B-A3B at 91.6 tok/s; the best single-stream split is a few percent below it (layer split, 84 tok/s modelled) or well below it (expert split, 52 tok/s modelled). Vega's whole-token decode bandwidth is below the M4 Pro's. |
| What does the Vega card buy? | Capacity (a 24 GiB Mac holds 12 GiB instead of 19 GiB of weights) and, on the 4B, no aggregate throughput: two pipelined streams decode 120-130 tok/s, the Mac alone batching two sequences in one round 116 (one engine, same prompts), and the Mac alone is ahead at three and four (155 and 198 against the split's 120). For the 35B-A3B the Mac alone batches 88.8 / 135.8 / 176.4 / 206.9 tok/s at 1 / 2 / 4 / 8 sequences (measured, mlx_lm); the implemented split (single-row steps) cannot beat it beyond two sequences, and a split whose stages batch rows of several sequences is modelled 1.1-1.2x ahead on the Vega [INFERENCE] (see M4 groundwork). |
| Where does the speed come from? | A faster external GPU. Modelled 7900 XTX-class: 158 tok/s resident; 5090-class: 299 tok/s resident, 3.5x all-Metal [INFERENCE]. |
| Can Metal wait on a flag written by the external GPU (or the CPU)? | No. A running kernel never reliably sees a store from another agent; GPU stores are not visible mid command buffer either. Every handoff is a command-buffer boundary. Measured wake cost 65 us. |
| Can Metal wrap the memory the external GPU DMAs into? | Yes. `newBufferWithBytesNoCopy` accepted 16 KiB to 1 GiB shm-backed mappings, including `MAP_FIXED` inside a `PROT_NONE` arena the way `hipmac` lays out its sysmem. |

The design therefore does not depend on GPU-to-GPU polling. A host relay thread is the single place where both
devices' ordering rules meet, and the planner prefers placements with few crossings.

## Decision requested

Maintainers are asked to agree:

1. That an in-host, device-agnostic offload layer belongs in the native engine, reusing the cluster planner, budget and
   cost code and the fabric framing, not the cluster transport and SSH launch.
2. The gate in [Proposed gate](#proposed-gate), in particular the arithmetic rule for a stage boundary between a bf16
   backend and an fp16 backend, and the fidelity gate used until both stages share a byte-equal contract.
3. The pull-request order in [Pull requests](#pull-requests): transport and interface with the dense layer split first,
   then the MoE expert split, then the hot-expert cache and adaptive placement.

## Target checkpoints

### Qwen3.6-35B-A3B (MoE)

Verified against the Hugging Face API and `config.json` on 2026-11-10; Qwen3.6 exists, so no nearest-checkpoint
substitution is needed.

| Item | Value |
| --- | --- |
| Repository | `mlx-community/Qwen3.6-35B-A3B-4bit`, revision `38740b847e4cb78f352aba30aa41c76e08e6eb46` (2026-04-16) |
| Same weights | `lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit` `0c4a20a6...`; `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` adds `mtp-4bit.safetensors` (the recipe's tested revision is `81169a9bc511a27c1b4eedb77a2cd98ced431847`, 20.9 GB) |
| Licence | Apache-2.0 (also the base `Qwen/Qwen3.6-35B-A3B`) |
| Format | MLX affine 4-bit, group 64; routers and `shared_expert_gate` at 8 bits; `model_type` `qwen3_5_moe` |
| Layers | 40: 30 `linear_attention` (Gated DeltaNet) and 10 `full_attention`, every fourth layer (`full_attention_interval` 4) |
| Hidden / vocab | 2048 / 248,320, embeddings and `lm_head` untied |
| Attention | 16 query heads, 2 KV heads, head_dim 256, output gate, rotary 25% (mrope), theta 1e7, window 262,144 |
| DeltaNet | 16 key heads x 128, 32 value heads x 128, conv kernel 4, fp32 state |
| MoE | 256 routed experts, top-8, `moe_intermediate_size` 512, one shared expert (512) with sigmoid gate, router fp32 softmax, weights renormalized over the eight |
| MTP | `mtp_num_hidden_layers` 1 (in the TensorFold checkpoint only) |
| Vision | 27-layer tower, 0.83 GiB, not used here; image parts stay refused as the recipe requires |
| Size | 20.40 GB (19.00 GiB) on disk, 18.17 GiB without the vision tower |

Tensor bytes, read from the four safetensors headers (`switch_mlp` experts are stacked `[256, N, K]`):

| Tensor class | Per layer | Layers | Total |
| --- | ---: | ---: | ---: |
| Routed experts, 3 projections x 256 | 432.0 MiB (1.6875 MiB an expert) | 40 | 16.875 GiB |
| DeltaNet mixer (`in_proj_qkv`, `z`, `a`, `b`, conv, `out_proj`) | 18.13 MiB | 30 | 544.0 MiB |
| Full attention (`q` with gate, `k`, `v`, `o`) | 14.63 MiB | 10 | 146.3 MiB |
| Shared expert and its gate | 1.69 MiB | 40 | 67.6 MiB |
| Router (8-bit) | 0.53 MiB | 40 | 21.3 MiB |
| Norms | | | 0.3 MiB |
| `embed_tokens` | | | 272.8 MiB |
| `lm_head` | | | 272.8 MiB |
| **Everything but routed experts** | | | **1,325 MiB (1.29 GiB)** |

Per-stream state: DeltaNet 61.4 MiB (30 layers x (2 MiB state + 48 KiB conv)); KV 20,480 B a token at fp16
(10 layers x 2 heads x 256 x 2 x 2 B): 160 MiB at 8k, 640 MiB at 32k, 2.5 GiB at 128k, 5.0 GiB at 262k.
Bytes read for one decode token: 1.56 GiB (1.29 GiB dense side incl. embedding row, 0.53 GiB for 8 experts x 40
layers), plus 120 MiB of DeltaNet state read and written.

### Qwen3.5-4B (dense), the first-slice model

`mlx-community/Qwen3.5-4B-MLX-4bit`, revision `32f3e8ecf65426fc3306969496342d504bfa13f3`: 32 layers (24 DeltaNet, 8
attention), hidden 2560, MLP 9216, tied head, 2.9 GB. It is the checkpoint `docs/recipes/vega-macos.md` already serves
on the card (52-57 tok/s), so a dense layer split can be validated against two existing single-device receipts.

## What was measured

All offline on the M4 Pro (macOS 27.2, 48 GiB, page size 16 KiB); no external GPU touched. Source and raw output:
[metal_probe.m](metal-tb4-split-moe-probe/metal_probe.m), [mlx_expert_probe.py](metal-tb4-split-moe-probe/mlx_expert_probe.py),
[results.txt](metal-tb4-split-moe-probe/results.txt), [placement_model.py](metal-tb4-split-moe-probe/placement_model.py). Other
builds ran on the Mac meanwhile (load average about 5), so tails are pessimistic; medians were stable across runs.

### Memory handoff

| Probe | Result |
| --- | --- |
| `newBufferWithBytesNoCopy` over `shm_open` + `MAP_SHARED` memory, 16 KiB, 1 MiB, 64 MiB, 1 GiB | All succeeded, `contents == ptr`, 12-54 us to wrap even 1 GiB (pages are not wired up front), GPU read/write round trip correct |
| Pointer 4 KiB past a page boundary; length 20 KiB | Also accepted, `contents == ptr`, CPU stores visible. Do not rely on it: Apple documents page alignment; keep every window 16 KiB aligned |
| 16 MiB `MAP_FIXED` shm mapping inside a 64 MiB `PROT_NONE` reservation (how `hipmac` places a sysmem block at its GPU VA) | Wrapped |
| Second mapping of the same shm object in one process | CPU store through one visible through the wrapped buffer |
| `MTLResourceHazardTrackingModeUntracked` | Accepted |
| `hipmac` sysmem blocks | 64 MiB minimum, at most 128 a connection, a DMA table at the head of the mapping, the CPU pointer equals the GPU VA. One process holds both runtimes, so Metal wraps the pointer `hipmac` already returned and no fd passing is needed |

### Ordering and waking (what dictates the protocol)

| Probe | Result |
| --- | --- |
| Command buffer, one dispatch, `commit` + `waitUntilCompleted` | p50 91 us, p99 153 us |
| Same with a completion handler | p50 90 us |
| `commit` until a CPU poll of the kernel's store sees it | p50 67 us (the memory is visible about 25 us before the completion callback) |
| Pre-committed buffer waiting on an `MTLSharedEvent`; CPU sets `signaledValue`; kernel store visible to the CPU | p50 65 us, p90 80 us |
| Dependent 1-thread dispatches in one encoder | 1-3.4 us each once the GPU is awake |
| One MoE layer's Metal half: event wake, 13.5 MiB expert stream in 2 dispatches, ack seen | p50 188 us, p99 490 us |
| CPU store to a flag, a Metal thread resident in a spin loop (plain load, `atomic_fetch_add`, `volatile`, `coherent(device)`; with and without `dc civac`) | Never seen in 1 s in most cells; in other runs seen after 0.2-340 ms. Unusable |
| CPU store, then 15,000 later dependent 1 us poll kernels in the same command buffer | Never seen |
| GPU store (marker), same command buffer keeps running 50 ms | CPU sees it only when the buffer finishes (51-54 ms) |
| CPU data stored before the event that a pre-committed buffer waits on; or before `commit` | 0 stale words in 6,000 trials: command-buffer start is a coherence point |
| Flag stored after 32 KiB of data (same kernel after `threadgroup_barrier(mem_device)`, or a later dispatch), CPU reads the data when it sees the flag | 16-624 stale words of 24.6 M in 1-4 of 3,000 trials. The flag can outrun the data |

Consequences. (a) Metal cannot poll. It is woken per command buffer, by `commit` or by an event, at about 65 us, and its
results are safely visible only after the buffer completes. (b) Data and flag cannot be trusted to arrive in order
from a Metal kernel, so a reply carries its own validity stamps (see [Replies](#replies)). (c) The `Polled handoff: 2 us
plus 3 us, assumed, unverified on Apple GPUs` entry in `zig/src/fabric/cost.zig` is false on this machine; the `Event
handoff` entry (10 us GPU to host, 100 us host to GPU, probed on an M5) is the right order of magnitude. The external GPU
side has no such limit: `hipmac` already polls sysmem fences from the CPU and the CP can `WAIT_REG_MEM` on sysmem.

### Metal throughput

| Probe | Result |
| --- | --- |
| All-Metal decode, `mlx_lm` 0.32.2 / mlx 0.32.4, the 35B-A3B above, greedy, 256 tokens, twice | **91.2 and 92.0 tok/s**, peak 19.7 GB (a 164 GB/s effective rate over the 1.56 GiB plus state read a token: 60% of the 273 GB/s peak) |
| Same for Qwen3.5-4B 4-bit | 65.4 and 66.0 tok/s, peak 2.57 GB |
| `gather_qmm` on the real expert shapes (random weights, 40-layer serial chain, 8 experts a layer, cold picks) | 3.74-4.49 ms a token, 94-112 us a layer, 126-151 GB/s |
| One layer alone, `eval()` each time | 205 us (graph build and submit included) |
| 4-row window, 32 (row, expert) pairs a layer | 10.4-13.6 ms a token, 261-339 us a layer |

### Card numbers taken from the repository (not re-measured)

TB4 host to device 3.19 GB/s, device to host 3.35 GB/s (256 MiB staged), SDMA 2.9/3.4 GB/s, GPU-side dispatch 4.4 us,
19-20 us for a synced single dispatch, MMIO read round trip about 12 us, posted write about 1 us, D2D blit 89-123 GB/s.
Qwen3.5-4B decode 52-57 tok/s, 393 dispatches a token, GEMV kernels 170-305 GB/s of a 409.6 GB/s peak; the 4B's whole
token reads about 135 GB/s. Source: `docs/HIPMAC.md` and `docs/recipes/vega-macos.md`.

## Offload interface

The external side is any TensorFold backend that can provide the following. Nothing is Vega-specific; the Vega card is
the first implementation and the one the numbers below start from.

```text
ExternalDevice (sketch, not a final API)
  caps          name, arch, vram bytes, peak bandwidth, activation type (f16 | bf16), launch cost, max host windows,
                has_flag_wait, has_flag_write, has_timestamps
  alloc/upload  device memory and bulk copy (SDMA or copy engine)
  window(n)     pinned host memory the device can read and write, n a multiple of 16 KiB, CPU pointer stable
  plan          a recorded launch plan (HIP graph / indirect buffer / CUDA graph) with patchable arguments
  wait(w, off, v)    enqueue a device-side wait until the 32-bit word at w+off equals v (stream wait value)
  write(w, off, v)   enqueue a device-side ordered write of v after all earlier work in the stream (release write)
  timestamp()   device clock, for the startup probe and for `tf-*` receipts
```

| Backend | `window` | `wait` / `write` | Status |
| --- | --- | --- | --- |
| HIP on `hipmac`, gfx900 | `hipHostMalloc(mapped)` (sysmem blocks; the ABI binds it already, `zig/src/hip/memory.zig`) | `hipStreamWaitValue32`/`hipStreamWriteValue32` are not in the 52 symbols the engine binds; `libut4gdev` emits `WAIT_REG_MEM` and `RELEASE_MEM` internally for fences and engine switches, so a runtime addition is small. It lives out of tree | needs a `hipmac` change |
| HIP, RDNA3/RDNA4 over TinyGPU with `hipmac` | same | same; `gfx1100` tuning tables already exist | later, same code path |
| NVIDIA over TinyGPU's NVIDIA path | `cuMemHostAlloc`/`cuMemHostRegister` or the runtime's equivalent | `cuStreamWaitValue32`/`cuStreamWriteValue32` | [INFERENCE] depends on the runtime TinyGPU's NVIDIA path offers; the CUDA backend is the consumer |
| Mock | heap memory plus a thread | atomics | for no-GPU tests |

Metal side, in `zig/src/metal`: `Device.bufferNoCopy` and `SharedEvent` exist already (the survey found both, plus
`icb.zig`, `residency.zig`). New: a `HostWindow` wrapper that checks 16 KiB alignment and holds a residency set, and a
`Relay` that pre-commits event-gated command buffers, owns the one polling thread, and exposes `post(ticket)` and
`done(ticket)`.

The family code keeps asking its own backend for everything else. A stage runs a layer range on its backend; the stage
boundary is a fp32 hidden row (below). No new scheduler: the lane core, drafting and cache machinery stay in the engine,
as `docs/recipes/SKILL.md` requires.

## Transport

One window per direction pair, in one 64 MiB `hipmac` block (only about 2 MiB is used), 16 KiB aligned, wrapped by Metal
once at startup and registered with the external device once.

```text
slot (per microbatch, per hop)        all offsets 64-byte aligned
  header   : seq u64 | kind u16 | layer u16 | rows u16 | status u16        -- written last by the producer
  rows     : rows x hidden x f32  (residual stream; exact in both directions, 8 KiB a row for hidden 2048)
  aux      : ids u32 x top_k | weights f32 x top_k   (expert split only)
  reply    : rows x hidden x f32, interleaved with stamps (below)
```

The boundary tensor is fp32: bf16 and fp16 both widen exactly, the consumer rounds into its own activation type with
round-to-nearest-even. The hidden row, 8 KiB, crosses the link in 2.5 us at 3.3 GB/s plus latency, so the doubled
width costs nothing that matters at decode. At prefill a 2048-row chunk is 16 MiB, 5 ms a hop.

### Directions

* **External GPU to Metal.** The external GPU writes the slot, then performs an ordered `write` of `seq` to the header
  word (`RELEASE_MEM` on `hipmac`). The relay thread polls header words with plain CPU loads (coherent with PCIe
  writes; the card's own fence polling relies on it today). On a new `seq` it sets `signaledValue` on the event that the
  next pre-committed Metal command buffer waits on. The CPU stored the data before the signal, so the buffer sees it
  (0 stale in 6,000 trials). Wake cost 65 us.
* **Metal to external GPU.** Metal's buffer ends; the data is final once the buffer completes, and visible to a CPU
  poll about 25 us earlier. The relay thread then writes the header word with a plain store and the external GPU's
  `wait` releases (a CP poll of host memory, a few microseconds over TB4 [INFERENCE]). A faster option is to stamp the
  data and let the external GPU consume without the CPU, below.

### Replies

A Metal kernel can publish its flag before its data is visible (measured, rare, about 1 buffer in 1,000). The slot
therefore never relies on a flag for Metal-produced data:

* **Level A (default, safe).** Wait for command-buffer completion on the CPU, then store the header word. Costs the
  25-50 us between memory visibility and the completion callback.
* **Level B (milestone-gated).** Each 16-byte group of the reply carries 12 data bytes and a 4-byte `seq` stamp; the
  external GPU's consume kernel polls and re-reads groups whose stamp is not current, bounded. This removes the CPU
  from the return path. It needs 16-byte stores to stay one unit through Apple's write-back and the card's PCIe reads;
  that is [INFERENCE] and is what milestone M2 measures. The reply grows by one third.

### Startup probe

`tensorfold offload probe` measures, once per device and cable, and writes a probe receipt that the planner reads
(each number carries `Source = measured`, as `fabric/cost.zig` already labels them): bulk H2D/D2H at 4 KiB, 64 KiB and
16 MiB; external dispatch latency; a flag round trip in each direction through the relay; Metal event wake; and the
device's sustained read bandwidth on a streaming kernel at 25 MB and 512 MiB. The probe is the only place link
constants come from; the planner never carries hard-coded link numbers.

## Placement

Placement is computed, not configured. `--device-split auto` (the default when `--device-split` is given) runs the
planner; `--device-split placement.json` pins a result for reproducibility.

### Inputs and outputs

* Model: per layer, resident weight bytes (dense, routed experts), bytes read per decode token, KV bytes a context
  token, state bytes, launches a token; embedding and head bytes. All derive from `config.json` and tensor headers;
  the loader already reads them.
* Mac: unified-memory budget (`recommendedMaxWorkingSetSize`; 37.4 GiB of 48 here), measured decode bandwidth (the
  planner may reuse the load-time timing the Mac engine already does).
* External GPU: usable VRAM after reserved regions, peak and measured bandwidth, activation type, launch cost.
* Link: probe receipt (bandwidth, hop latencies each way).
* Request: context length, stream count, drafting depth.

Output: for each layer, device and which tensor classes (mixer, shared expert, router, routed experts, KV, state) live
there; embedding and head owner; prefill owner; resident-expert layers or hot-expert set; predicted decode tok/s and
the cost each term contributes. It is printed by `tensorfold info` and refused with the missing quantity when the
budget does not close, using the existing `budget.zig` categories (attention, shared, router, experts, embed, head,
norms, state) and `admission.zig` (`fitsAt`, `maxStreams`).

### Modes the planner enumerates

| Mode | Where things live | Hops a token |
| --- | --- | --- |
| all-Metal | everything on the Mac | 0 |
| resident | everything on the external GPU; the Mac keeps the tokenizer, sampling bookkeeping, overflow | 0 (a token id back, 4 B) |
| layer split | external `[0,k)` + embedding, Metal `[k,L)` + head (or the reverse); k from the VRAM budget | 1 each way |
| expert split | external: mixers, shared expert, router, KV, state, embedding, head, expert layers it can hold; Metal: routed experts of the rest | 2 a split layer |
| reverse expert split | Metal: mixers, KV, state, head; external: routed experts of as many layers as fit | 2 a offloaded layer |

Prefill is planned separately. Decode is bandwidth-bound; prefill is compute-bound and the Mac's GPU is the stronger
prefill device today (the card prefills the 4B at about 200 tok/s). When the Mac holds the full checkpoint, prefill
runs on Metal and the external GPU's KV and state are filled by a bulk copy afterwards: 3 attention layers plus 12
DeltaNet layers on the card is 72 MiB at 8k (23 ms) and 0.8 GiB at 128k (0.24 s). When the Mac cannot hold it (a 24 GiB
Mac), prefill runs through the layer split in 2048-row chunks, hops overlapped with compute.

### Model of decode time

For one stream, `T = sum over devices (bytes read / (peak * eff) + launches * t_launch) + hops`, with `eff` the
fraction of peak the decode kernels reach end to end. Calibration against this Mac and card:

| Case | Model | Measured |
| --- | ---: | ---: |
| 35B-A3B, all-Metal, `eff` 0.62 | 94 tok/s | 91.6 |
| Qwen3.5-4B, all-Metal | 68 tok/s | 66.0 |
| Qwen3.5-4B on the Vega card, `eff` 0.37, 4.4 us a launch | 55 tok/s | 52-57 |

Hop costs: eGPU to Metal 90 us (flag visible 5, event wake 65, flush 20), Metal to eGPU 15 us [INFERENCE: not yet
measured on a card]. The two other profiles use `eff` 0.50 (7900 XTX class) and 0.55 (the fraction of peak the GB10
reaches at the recipe's 88 tok/s) with 2.5 and 1.5 us launches; those numbers are assumptions.

### Predicted placements for the 35B-A3B

Output of `placement_model.py` (a sketch of the planner: same inputs, same arithmetic, no tensors). tok/s, decode,
one stream unless stated; "2+ streams" is the pipeline bound `1 / max(stage)` with two microbatches in flight.

**RX Vega 56, 8 GiB (today)**, usable 7.9 GiB, 410 GB/s peak:

| Mode | 8k context | 128k context | Placement |
| --- | ---: | ---: | --- |
| all-Metal | 86 | 38 | Mac 18.8 / 21.1 GiB |
| resident | does not fit | does not fit | 18.8 GiB > 7.9 |
| layer split, best serial | 84 | 37 | card 1.1 GiB (embedding), Mac the rest: the card is slower than Metal, so the planner gives it almost nothing |
| layer split, 2+ streams | **131** | **54** | card layers `[0,16)` at 8k (7.8 GiB), `[0,14)` at 128k (7.6 GiB); Mac 11.4 / 13.9 GiB; stages 6.0 + 7.6 ms |
| expert split | 52 | 27 | card 7.8 GiB with 14 expert layers resident, 26 crossings |
| reverse expert split | 85 | 38 | not chosen (the card is slower than Metal) |

At 8k: 14 resident expert layers fit in the expert split; 13 at 32k, 8 at 128k, 2 at 262k (7.9 GiB minus 1.29 GiB dense,
0.4 GiB scratch, KV and state).

**24 GiB Mac with the Vega card** (Mac budget 16 GiB, ctx 32k): all-Metal does not fit (19.2 GiB); layer split 64 tok/s
single stream, 100 tok/s with 2+ streams, card 7.5 GiB, Mac 12.2 GiB. This is the case the card enables.

**RX 7900 XTX, 24 GiB (960 GB/s) [INFERENCE]**, usable 23 GiB:

| Mode | 8k | 128k |
| --- | ---: | ---: |
| all-Metal | 86 | 38 |
| **resident** | **158** | **86** |
| layer split, best serial | 131 | 72 |
| layer split, 2+ streams | 234 | 119 |
| expert split | 159 (all 40 expert layers resident, 0 crossings: the same as resident) | 87 |
| reverse | 85 | 38 |

The planner picks resident until KV no longer fits (the 262k window at 5 GiB plus 18.2 GiB of weights), then a layer
split with the tail on Metal.

**RTX 5090, 32 GiB (1.8 TB/s) [INFERENCE]**, usable 31 GiB: resident 299 tok/s at 8k and 169 at 128k (3.5x and 4.4x
all-Metal); 2+ streams on a layer split 367 and 199. The Mac is host and overflow.

**Reverse split.** Metal keeps mixers and KV and the card runs routed experts. It pays only when the experts' read-time
saving exceeds two crossings: routed bytes read a layer a token of at least 40 MiB for 7900 XTX class and 30 MiB for
5090 class (never for Vega: it is slower than Metal). Qwen3.6-A3B reads 13.5 MiB a layer, so it is not chosen for this
model; it is the placement for an MoE with roughly twice the active expert bytes a layer, or a Mac whose memory cannot
hold the dense part and cache but whose experts fit the card.

**Dense Qwen3.5-4B** (the first slice), 8k: all-Metal 62 tok/s (66 measured); Vega layer split 61 single stream,
109 with 2+ streams; 7900 XTX class 149 resident, 207 pipelined; 5090 class 295 resident.

Numbers for the 7900 XTX and 5090 profiles are model outputs from published peaks and one assumed efficiency each.
They set the target and the stop criteria; they are not claims.

### Placement is data

`placement.json` (`"schema": "tensorfold-split/1"`, strict like `cluster.schema.json`): `device` (backend, index,
`window_mib`), `layers` (per-layer owner; or `{"external": [0, 14], "metal": [14, 40]}`), `expert_layers` (resident on
the external device), `embedding`, `head`, `prefill` (`metal` | `external` | `split`), `fallback` (`none` | `metal`),
`timeouts_ms`. `auto` writes the file it chose beside the probe receipt, so a run can be repeated.

## Behaviour by phase

* **Decode, one stream.** Layer split: one hop each way (105 us); expert split: two a split layer
  (`TO_MAC + TO_GPU` plus Metal's compute); the planner prefers the placement with fewer crossings unless the
  bandwidth saving beats them.
* **Several streams or a drafting window.** Two or more microbatches keep both devices busy: while the Metal stage
  works on microbatch A the external stage works on B. Rows of one stream stay together; the lane core keeps its
  one-round contract and the pipeline is inside a round. A window of R rows crosses as R x 8 KiB.
* **MTP and drafts.** The MTP layer is a 41st layer (attention plus 256 experts, 0.47 GB) owned by the head side.
  Each verify window reads about twice the bytes of a serial step (recipe), which raises the expert share and
  favours placements that keep experts on the faster device. Drafts only choose which rows a round verifies, so
  split and unsplit replies stay equal for the same stage arithmetic.
* **Prefill.** As above. Hop volume is rows x 8 KiB a boundary.
* **Streaming routed experts to the card per token.** Rejected below; the cold path never moves weights per token.

## Alternatives considered

| Alternative | Numbers | Verdict |
| --- | --- | --- |
| Stream cold experts to VRAM over TB4 | One expert is 1.6875 MiB, 0.55 ms at 3.2 GB/s; a token needs 320 expert reads (8 x 40): 177 ms if all cold, 5.6 tok/s. At a 90% hit rate 17.7 ms a token, a 56 tok/s ceiling before any compute; 6 GiB of cache is 3,640 experts, 36% of 10,240, and Qwen's routers train with a balance loss (coefficient 0.001), so a flat distribution is more likely than the skew 90% needs [INFERENCE]. Router look-ahead gives one layer of warning at best (the next router depends on this layer's output) | Rejected as the primary path; the hot-expert cache (PR 3) is the measured-hit-rate follow-up |
| All-Vega, experts in host memory read zero-copy | The 540 MiB of routed experts a token reads, at 3.3 GB/s: 170 ms, 5.9 tok/s | Rejected |
| All-Metal | 91.6 tok/s on 48 GiB | The baseline every mode must beat or justify (capacity, streams) |
| Expert split on today's card as the main placement | 52 tok/s single stream at 8k, 40 crossings of about 195-215 us (65 wake, 94-112 Metal, 20 flush, 15 return) | Kept as PR 2 for faster cards and for capacity; not the default on Vega |
| Per-expert LRU or frequency cache | A layer needs no crossing only if all 8 picks hit: hit rate^8 is 0.43 at 90% and 0.004 at 50%. Resident layers remove crossings deterministically | Layers, not experts, are the placement unit until PR 3 measures real routing |
| Spin-wait kernels on a shared flag | CPU store seen after 0.2-340 ms or never; GPU store invisible until the command buffer ends | Rejected, measured |
| Reuse the cluster transport (MCDMA, SSH launch) for the card | `node.Backend` is `metal | cuda` and links are `tb5 | cx5 | cx7`; the round protocol sends token ids, not activations; a node is a host | Reuse planner, budget, admission, cost, fake-test patterns; add a local `HostWindow` exchange |
| Run the card as a separate server process | Adds a hop through sockets and a second KV owner | Rejected |

## TensorFold architecture fit

**New abstraction or extend cluster?** Extend cluster's planning, not its transport. `plan.zig` already places every
tensor "under one layout (pipeline x tensor, experts spread or split)"; `budget.zig` has the per-category need,
`admission.zig` the `fitsAt` and `maxStreams` searches, `cost.zig` the round cost, `fabric/cost.zig` the labelled link
constants (it even has a `local` link kind, `assumed`). Gaps found in the vega-macos checkout at `be3b2804a`:
`node.Backend` has no `hip`; `LinkKind` has no PCIe-tunnel kind; cluster nodes have one memory-bandwidth figure and no
per-node efficiency; role classification names DeepSeek, Kimi and GLM tensors, not Qwen's. A layer split is cluster's
`pipeline = 2` with the two "nodes" being devices of one host, so PR 1 adds those values and a
`HostWindow`-backed `Exchange`; it does not touch SSH launch, membership or MCDMA. The in-layer expert split is a
different seam (an `ExpertHost` the HIP family calls per layer), added in PR 2 beside the existing HIP expert-parallel
partial-sum hooks (`remap`, `moeZeroForeign`).

Base. The checkout at `be3b2804a` merges from `c240bb96a` (2026-11-03). Upstream `main` (checked 2026-11-10) already
registers `qwen3_5_moe` on Metal through the lane host (`families/qwen27/moe_gpu.zig`, `kernels/metal/qwen27/moe*.metal`,
contract tests `metal_qwen27_moe*.zig`: router, sort, gather, SwiGLU, combine, row kernels with the shared expert as one
more matrix). Every pull request here is written against `main`, not against this branch.

**CLI and config.** `--device-split auto|PATH`, `--device-split-window-mib N` (default 64, `hipmac` block minimum),
`--no-device-split-fallback`; `tensorfold info` prints the placement and the prediction; `tensorfold offload probe`
writes the receipt. Settings are not read from the environment unless a production caller needs one, and then
through `platform/environment.zig` rules as the pull-request skill requires.

**Interfaces touched.**

| Area | Change |
| --- | --- |
| `zig/src/core/split/` (new) | placement planner (shares `cluster/budget`, `admission`, `cost` types), window protocol, `ExternalDevice` vtable, fake devices |
| `zig/src/metal/` | `HostWindow`, `Relay` |
| `zig/src/hip/` | optional `hipStreamWaitValue32`/`WriteValue32` symbols in `abi.zig`, window allocation helper; mock support |
| `zig/src/families/qwen3_5/backend/hip/forward/` | a layer range (`span` over `[lo,hi)`), `Stage` with boundary hooks, later `routed = .external` in `moe.zig` |
| `zig/src/native/` | one new adapter that builds both backends (`-Dsplit`), nothing selected implicitly |
| `zig/src/cluster/` | `Backend.hip`, link kind, role classification for Qwen names |
| docs | recipe page for the split on Mac, README table row after the receipts exist |

### One process or two

**Decision: two processes over a unix socket (`split/remote.zig`) are the design, not a stopgap.** The pipeline runs in the
process that owns the card; the host stage is served by the process that owns the Mac's GPU; both map the window
(the card's pinned memory, passed as a shared-memory descriptor); only the calls that start a step and wait for it cross
the socket, and rows, tokens and stamps never do.

Why not one binary that links both engines. Zig compiles a source file into one module of a program, so the Metal engine
(root `tensorfold.zig`, 674 files reachable by `@import`) and the HIP engine (root `core/root.zig` as module `core`, 360
files) cannot both be linked while they share files: the build refuses with `file exists in modules 'core' and 'core0'`.
Counted over `@import("*.zig")` reachability from `zig/` on `21ea8dee4`:

| Pair of roots | Files in both |
| --- | --- |
| `core/root.zig` and `tensorfold.zig` | 124 |
| `core/root.zig` and `qwen27_core.zig` (the Metal lane core) | 78 (checkpoint reading, GGUF, safetensors, NPY, quantization, segments, KV pool, direct I/O) |
| `core/root.zig` and the `families/qwen27` tree | 0 |

Making the sets disjoint means a shared `checkpoint_io`-style module that both builds import and rewriting every
relative importer of those files: a build-graph refactor across upstream code, reviewed on its own merits and for
reasons that have nothing to do with this feature. The socket keeps the change inside `split/` and the two engines'
range hooks.

What the socket costs and buys.

* Cost: a unix socket request and reply between two threads (40 B out, 8 B back; `cc -O2`, M4 Pro, 20,000 round trips)
  is 7.0 us at the median and 13-14 us at the 99th percentile; a step makes two calls (`step`, `finish`), so about
  14 us of the 15-24 ms a position takes on the card (measured positions: First card run). Between two processes
  rather than threads it is [INFERENCE] the same order.
* Buys: an eGPU unplug or a wedged card takes down the card process, not the address space that holds the Mac's model
  and KV; the Mac side sees a closed socket and returns from `serve` (today the tool then exits; keeping the model
  loaded for a new client is [INFERENCE] a small change, not made); one client per card is a property
  of the dext, and the card process is the one that owns it; hipmac stays out of tree and out of the Mac engine's
  link line; each side builds with only its own backend.
* Launch: a convenience driver (`tensorfold serve --device-split PATH`) spawns the card process as a child, hands it
  the socket path, and waits for `serving`; the user sees one command. It is not in pull request 1.

In-process linking stays possible later and is conditional on disjoint file sets for the two roots. If a future
refactor gets there, `Peer` already is the seam: `LayerPeer` is called directly instead of through `Client`, nothing
else in `pipeline.zig` changes.

### Exactness, in the repository's terms

Two engines with different activation types meet at a boundary: Metal runs bf16, the gfx900 path runs fp16 (it has no
bf16 dot; `docs/recipes/vega-macos.md`), RDNA3 and newer and NVIDIA run bf16. CONTRIBUTING keeps the checkpoint's
activation precision and forbids a mode that changes tokens against the serial reference of the same engine, so the
rules are:

1. **Same-engine equality (hard).** For a fixed placement, drafted equals `"draft": false`; resumed equals fresh;
   concurrent equals solo at every admitted stream count; partial acceptance, cancel and reset leave the next request
   valid; the window of rows does not change a row's bits. The split engine is its own engine and `--no-drafts` on it is
   the oracle. This holds by construction when each stage's kernels already satisfy their family's row-independence
   contract and the boundary is deterministic (fp32 widening, RNE narrowing, no reductions across the boundary).
2. **Boundary arithmetic is documented, not hidden.** A bf16 stage feeding an fp16 stage narrows fp32 to fp16 (overflow
   above 65,504 is checked and refuses the request with a named error); an fp16 stage feeding a bf16 stage rounds to
   bf16. With both stages bf16 (RDNA3 and newer, NVIDIA, Metal) the boundary is the identity.
3. **Independent fidelity (hard when the stage types differ).** Per projection class and for the boundary, native error
   against an FP64 reference no worse than the worse of the two single-device engines; teacher-forced histories with
   every top-token difference and margin reported against the all-Metal engine, using the documented tie policy only for
   the same two contenders. This is the gate on the Vega card.
4. **Byte equality with the single-device engine (target for same-type stages).** For a dense layer split of one
   backend type, or two bf16 backends with a shared reduction contract, a split run equals the unsplit run token for
   token. Tested on a synthetic tiny MoE and a tiny dense model whose weights fit one device (the contract-test
   pattern already used by `tests/qwen35/` synthetic fixtures); the real checkpoint is gated by items 1 and 3.
5. **No shortcuts.** No lossy KV, no FP8 activations, no reduced-precision boundary, no mode that changes tokens
   behind an option.

Kernel contract: the expert-split Metal service reuses the upstream routed kernels; any new kernel (an fp16-activation
instantiation of the row kernel, a combine that returns the prefix of the slot-ordered sum) registers in
`metal_main.zig` and `registry.zig` with a planted control, and its `SUMMARY` goes in the pull request. The slot sum
across the boundary stays the sequential fp32 sum in slot order: Metal returns the first 8 slots' prefix, the external
side adds the shared slot, which is bitwise the sequential sum (the HIP combine puts the shared expert at slot 9).

### Tests

* No-GPU: protocol property tests on the fake window (random interleavings, stamped and unstamped replies, injected
  stale words, late and duplicate `seq`, timeouts, cancel, reset, short reads); the planner on the real
  `config.json` numbers (memory table above is a fixture) and on the synthetic profiles, including refusals;
  `placement.json` schema tests with planted-bad files; the boundary narrow/widen arithmetic against a scalar reference.
* Fake-device engine test: a toy dense family and a toy MoE family (as `cluster/toy.zig`) run unsplit and split with
  in-process fake devices; tokens must be identical, and a cut link must end in a named peer-lost error within the timeout with the
  next request valid.
* Device tests skip cleanly without hardware: Metal `HostWindow` and `Relay` contract tests (the probes ported to Zig
  with thresholds), the HIP window and wait/write test on a card, the contract entries.
* Gates in the pull request: `zig build test test-golden`, `zig build dev-tools`, lean checker base and head counts,
  `kernel-contract` and `hip-kernel-contract` where kernels change, token hashes from `tools/bench_openai.py` and
  `tools/bench_concurrent.py`.

### Failure handling

* A device that stops answering (cable pulled, card wedged): every wait is bounded by `timeouts_ms`; on expiry the relay
  signals all pending events with an error value so Metal buffers drain, the engine fails the in-flight requests with
  a named error and refuses new ones. No silent fallback.
* `"fallback": "metal"` is explicit and only valid when the Mac can hold the model; the engine then restarts the
  affected requests from their prompts (KV on the card is gone).
* Process death: `hipmac`'s guardian quiesces SDMA and queues on SIGKILL (it measured 70-103 ms to reopen); the split
  adds nothing to its hot path and must not hold an SDMA copy across a possible kill without it.
* One client per card: the split process takes the card's lock the way the HIP engine does and refuses to start behind
  another model process.

### Portability

The offload layer names a window, a wait, a write and a launch plan, not a card. The same code serves RDNA3/4 once
`hipmac` drives them, and NVIDIA when TinyGPU's NVIDIA path supplies a CUDA-driver-equivalent; the planner changes
numbers, not code. A Mac without a card is unaffected: nothing is built unless `-Dsplit` is set.

### Licence

TensorFold is Apache-2.0. `hipmac` is out of tree (`ut4g-usbgpu`, the repository the recipe already cites); the engine
reaches it through the HIP runtime ABI; TinyGPU's licence applies to its dext and server. The model is Apache-2.0. No code is copied from
other engines; the probes and the model are written for this proposal.

## Milestones and measurements

Each milestone has a stop criterion; none needs more than one Mac and one card.

| | Measurement | Go if |
| --- | --- | --- |
| M0 (done offline) | the probes above | window wrap works, wake cost known: yes, 65 us idle (149 us with the card in use) |
| M1 (receipt measured; relay round trips at 4 KiB / 64 KiB not separately) | on the card: flag round trip each way through the relay at 4 KiB and 64 KiB, `hipStreamWaitValue32`-class wait latency, GPU read of a 64 KiB host window; 4B dispatch counts with fused boundary kernels | eGPU to Metal to eGPU at 8 KiB p50 <= 250 us, p99 <= 500 us |
| M2 | Level B stamped replies: stale-stamp rate and retry cost against the card's PCIe reads | zero accepted stale words in 10^6 replies, else stay at Level A |
| M3 (done on the 4B: tokens identical, cosine >= 0.9996; 2 streams 120-130 tok/s: 1.6-1.8x ONE Mac stream but only 1.03-1.13x the Mac alone batching two sequences (116), and below it at 3-4 sequences (155, 198): the gate as worded (against one Metal queue) is met, the honest comparison is not a win on this model; one stream 16-24 ms against 14 ms all-Metal, the 5% single-stream clause not met, as the model says) | Qwen3.5-4B layer split Metal + Vega, tokens against the unsplit runs (gate 4) and fidelity (gate 3); single stream and 2 streams | 2-stream throughput >= 1.5x one Metal queue (model 1.7x); single stream within 5% of all-Metal |
| M4 (Mac-alone baseline measured: 88.8 / 135.8 / 176.4 / 206.9 tok/s at 1 / 2 / 4 / 8 sequences, mlx_lm; native Metal cannot run the shape) | 35B-A3B layer split with stages that batch rows of several sequences, KV hand-off after Metal prefill, 24 GiB-class memory cap emulated with a lower budget | fits the cap; tok/s at 2, 4 and 8 sequences above the Mac alone's, within 15% of the model |
| M5 | expert split on the card with resident layers | 14 crossings fewer than all-split, tok/s within 15% of the model; if single-stream is below 50 tok/s it stays opt-in |
| M6 | hot-expert cache: per-layer routing histograms on real prompts (code, prose), hit rate versus cache size | an all-hit layer rate that pays the cache's copy cost; otherwise drop PR 3 |
| M7 | a stronger card (7900 XTX class, 5090 class) through the same interface | resident decode within 15% of the planner's prediction |

## Implementation status (pull request 1, offline)

Implemented on `feat/tb4-offload-pr1-up` (on upstream main) and tested without the card: the Mac's GPU is real, the
external device is the CPU mock or hipmac's mock backend, which executes streams in order but does not compute. Run on
macOS arm64 only (the `split` module has no platform imports, so [INFERENCE] it builds elsewhere; it was not built
elsewhere). The card ran this work once (2026-11-10, [First card run](#first-card-run-rx-vega-56-over-thunderbolt-4-2026-11-10)):
the probe and the Qwen3.5-4B split. That run used `tb4-offload-card`, this branch merged with the gfx900 port
`vega-macos-up` (below), with the two-process driver (`remote.zig`, `reference.zig`, `tf-offload-card`, `tf-offload-metal`).

| Piece | Where | State |
|---|---|---|
| `ExternalDevice` (alloc/upload/download, host windows, stream wait/write on window words, bounded sync, timestamp), `Caps` | `zig/src/split/device.zig` | tested through the mock and the HIP adapter on hipmac's mock |
| Window protocol: 64-byte header with the sequence stamp written last, 16-byte stamped payload rows | `wire.zig` | tested, including stale and late stamps |
| CPU mock device (own stream thread, injectable latency, `kill`) | `mock.zig` | tested |
| Layer pipeline (`external_first`, `metal_first`, `external_only`; level A and level B; steps of several rows; one pipeline a sequence over one device, `streams.zig`) | `pipeline.zig`, `streams.zig` | tested on the toy model and on the real engine (Metal+Metal), chunked prompts and two sequences included; the overlap itself only on the mock with injected delays |
| Exact integer toy model, split at every layer, either order, both levels | `toy.zig`, `metal_toy.zig` | token-identical to the unsplit run, on the CPU peer and on a real Metal GPU peer |
| Metal `HostWindow` (`newBufferWithBytesNoCopy`, 16 KiB alignment refused otherwise), `Relay` (event-gated handoff), wake probe | `zig/src/metal/window.zig`, `relay.zig`, `wake_probe.zig` | tested on an M4 Pro |
| `tf-offload probe` (receipt `tensorfold-offload-probe/1`) | `probe.zig`, `metal_probe.zig`, `cli.zig` | runs against the mock; against hipmac's mock through the HIP adapter; on the card: run once (First card run) |
| `tf-offload plan` (placement `tensorfold-split/1`) | `geometry.zig`, `planner.zig`, `placement.zig`, `cli.zig` | reproduces `placement_results.txt` within 1%; dense configs executable, MoE only `all_metal` |
| HIP adapter: `HipExternal` over `hipStreamWaitValue32`/`WriteValue32`, optional shared engine stream | `zig/src/hip/offload.zig` | passes on hipmac's mock backend (with the gfx900 caps row); on the card it ran the probe and the split runs below |
| Metal engine runs the tied-head Qwen3.5-4B and a layer range `[first,last)` (embed first, final norm/head/argmax last, KV and linear state only for owned layers); stamped-row pack kernels | `families/qwen27/` (`range.zig`, `stage_pass.zig`, `stage_rows.zig`, `split_peer.zig`) | real 4B on the M4 Pro: whole-model decode 74-75 tok/s; Metal+Metal split through the pipeline token-identical over 32 tokens at cuts 1, 8, 16, 31, both orders, both levels (16 runs); synthetic-weights version of the same test; 13.9 ms/position split against 13.4 unsplit (two engine instances in one process) |
| HIP engine layer range (`Range`, ranged weights/caches, prompt span and decode step, `HipLayerStage` as `ExternalStage`) | `families/qwen3_5/backend/hip/` | plumbing tests offline; on the card the 4B decodes through it (whole range and every cut tried, both orders): 47/48 top-1 and logits cosine >= 0.9996 against the unsplit Metal run (First card run) |
| Boundary kernels on HIP: `importRow`, `exportRow`, `consumeStamped` (level B: 16-byte stamped groups, bounded re-read, status word) | `zig/kernels/hip/split/rows.hip`, `zig/src/hip/ops/split_rows.zig` | host references; compiled to code objects for gfx900, gfx1030, gfx1100, gfx1151, gfx1201; contract records (gate 5) identical on four chips; run on gfx900 through the split (level A and B, status word 0); the standalone gates 1-4 on a GPU not run |
| `Backend.hip`, `LinkKind.tb_gpu`, hop costs in `fabric/cost.zig` | `zig/src/cluster`, `zig/src/fabric` | tested; a hip node is planned but never launched |

Commands (no card; the machine this was developed on has no working Metal compiler, so a local `xcrun` shim stood in for it when building; the shim is not part of the change):

```sh
zig build test-split            # protocol, planner, mock, pipeline, CLI, Metal window/relay/toy, Metal host
zig build tf-offload            # zig-out/bin/tf-offload
tf-offload demo --order metal_first --level b --split 5   # toy model across the mock device and Metal
tf-offload probe --device mock --out receipt.json
tf-offload plan --config config.json --receipt receipt.json --context 8192 --streams 2 --out placement.json
zig build tf-qwen27-split-synthetic && zig-out/bin/tf-qwen27-split-synthetic   # synthetic weights, Metal+Metal split
zig build tf-qwen27-split && TF_SPLIT_QWEN_DIR=<Qwen3.5-4B-MLX-4bit> zig-out/bin/tf-qwen27-split   # the real 4B
zig build -Dhip=true test-qwen35-split-host      # HIP layer range and HipLayerStage plumbing (no device)
zig build -Dhip=true tf-offload-hip   # same CLI with `--device hip`, which opens the card
```

On the M4 Pro, `plan` on the Qwen3.5-4B config with a mock receipt picks the pipelined layer split (115 tok/s
predicted at two streams, 62 all-Metal) and on the 35B-A3B config picks all-Metal (87 tok/s predicted); the mock
receipt's link numbers are the mock's, not a card's (the card's receipt gives the table in First card run).

### First card run (RX Vega 56 over Thunderbolt 4, 2026-11-10)

The first end-to-end tokens through the pipeline with one half on the Mac's GPU and one on the card: Qwen3.5-4B MLX
4-bit (32 layers), a 49-position sequence (the prompt and the unsplit Metal run's continuation, teacher forced), an M4
Pro and one RX Vega 56 (gfx900, 8 GiB). Everything below was measured; the Mac was shared with other builds, which moves
the Metal-side times by tens of percent from run to run.

**Two processes, not one.** One program cannot link both engines: Zig allows a source file in one module of a program, and
the Metal engine and the HIP engine both include the `core` files (`error: file exists in modules 'core' and 'core0'`).
So the card's process (`tf-offload-card`: HIP engine, `HipExternal`, the pipeline) and the Mac's (`tf-offload-metal serve`:
Metal engine, the host stage) are joined by `split/remote.zig`: a unix socket carries only `step`, `finish` and `reset`
calls, and the window is the same memory in both. hipmac's `hipmacHostExport` hands out a duplicate of the pinned block's
shared-memory descriptor (sent with `SCM_RIGHTS`); the Mac's process maps it and wraps the mapping for Metal
(`newBufferWithBytesNoCopy`). Rows and tokens never cross the socket; a step pays the socket's round trips on top of the
window's.

**Probe receipt** (`tf-offload-hip probe --device hip`, 400 iterations): flag store to device wait release (hop to device)
p50 10.5 us, p99 28.5 us; card flag to Metal command buffer completion (hop to host) p50 159 us, p99 479 us, of which
Metal's wake p50 149 us, p99 450 us (65 us on an idle Mac in M0: the wake is slower while the card is in use); H2D 3.17 GB/s,
D2H 3.34 GB/s. `fabric/cost.zig`'s `metal_event_handoff` now carries the measured 149 us end to end.

**Planner with that receipt** (context 8192, `tf-offload plan`):

| Model | Placement | Predicted tok/s |
|---|---|---|
| 4B | all Metal | 64.0 |
| 4B | resident on the card | 50.5 |
| 4B | layer split, one stream | 62.8 |
| 4B | layer split, 2 streams, 16 layers on the card | 113.2 |
| 35B-A3B | all Metal (chosen) | 89.1 |
| 35B-A3B | layer split, 2 streams, 15 layers on the card (not executable in this change) | 130.5 |
| 35B-A3B | expert split / reverse expert split | 47.4 / 87.3 |

At one stream `auto` picks all-Metal for both models: a sequential split cannot beat the faster device. The gain is at two
or more streams (and in capacity), as the model said before the card was used; the slower wake costs the expert split
(40 hops) more than the layer split (one hop).

**Fidelity** (top-1 and logits against the unsplit Metal run, teacher forced, 48 positions, vocabulary 248,320):

| Run | Top-1 equal | Logits cosine min / mean | Free-running tokens (40) |
|---|---|---|---|
| unsplit on the card | 47 / 48 | 0.99968 / 0.99994 | identical to Metal |
| Metal `[0,4)` + card `[4,32)`, level A and B | 47 / 48 | 0.99958 / 0.99995 | identical |
| Metal `[0,8)` + card, A and B | 47 / 48 | 0.99974 / 0.99995 | identical |
| Metal `[0,16)` + card, A and B | 47 / 48 | 0.99983 / 0.99997 | identical |
| Metal `[0,24)` + card, A and B | 47 / 48 | 0.99988 / 0.99998 | identical |
| Metal `[0,28)` + card, A and B | 47 / 48 | 0.99996 / 0.99999 | identical |
| card `[0,k)` + Metal `[k,32)`, k = 4, 8, 16, 24, 28, level A | 47 / 48 | not read (the head is on the Mac) | identical |

The single disagreement is the same position (2) in every run, and there the Metal logits themselves tie: the two
leading bf16 logits are equal (13.125); Metal picks the lower index, the card's fp16 head the other. The drift grows with
the card's share of the layers, as expected. Level B (stamped groups consumed on the card) gave the same logits as level
A in every run and its status word read 0 after the last step; a stale stamp in an earlier step would have been
overwritten, so [INFERENCE] the 10^6-reply accounting (M2) is still to do.

**Time per position** (from the pipeline's phases, level A, Metal `[0,k)` first, 48 positions):

| Cut k | Total p50 | Metal stage | Card wait |
|---|---|---|---|
| 4 | 20.2 ms | 4.4 ms | 16.8 ms |
| 8 | 18.9 ms | 5.3 ms | 14.7 ms |
| 16 | 18.8 ms | 9.8 ms | 10.5 ms |
| 24 | 21.2 ms | 15.0 ms | 6.4 ms |
| 28 | 21.2 ms | 16.8 ms | 4.3 ms |

Card first, Metal `[k,32)` and the head second: 15.3, 15.8, 17.1, 20.6 and 23.6 ms at k = 4, 8, 16, 24 and 28. References
on the same runs: unsplit Metal 13.7-14.0 ms per position (token by token through the same engine), unsplit card through
the window pipeline 19.1-19.2 ms (eager step, no graph; 52 tok/s, the recipe's graph-replayed figure). From the phases
the card costs about 0.52 ms a layer plus 2.1 ms for the head (4.3 ms for 4 layers, 6.4 for 8, 16.8 for 28), and Metal
0.25 to 0.55 ms a layer in a burst plus a fixed 3-4 ms (embedding, command buffer, wake, socket). The halves never overlap
in one stream, so a split is slower than the Mac alone by about the card's share of the work: the one-stream model. Recording
the card's step takes 0.04-0.13 ms. The crossing itself (hop to device 10 us, hop to host 160 us) is under 2% of a
position. The 2-stream pipeline that overlaps the halves was written after this run (below, [Streams and chunked
prefill](#streams-and-chunked-prefill)) and measured on the card: 120 tok/s at the planner's cut against 113 predicted.

**A first-launch hang the pipeline cannot see.** The first split run hung in the card's runtime: the first launch of a
kernel that needs more scratch than the queue holds makes `hipModuleLaunchKernel` wait for the queue to drain
(`scratch_grow` -> `wait_seq_locked`), and the queue was waiting on the window. `HipLayerStage.warm` records one plain and
one stamped step on zero rows before the first wait. Any stream stage of another backend needs the same rule: no launch
that can wait on the queue after the first window wait is queued.

Reproduce (the card to oneself; `<lib>` is the hipmac runtime, out of tree):

```sh
zig build tf-offload-metal && zig build -Dhip=true tf-offload-card -Dhipcc=<gfx900 hipcc wrapper> -Doptimize=ReleaseFast
M=<Qwen3.5-4B-MLX-4bit directory>
TF_SPLIT_QWEN_DIR=$M zig-out/bin/tf-offload-metal ref ref.bin 40            # unsplit Metal: tokens, predictions, logits
export HIPMAC_BACKEND=ut4g TENSORFOLD_HIP=<lib>
zig-out/bin/tf-offload-card $M ref.bin unsplit                                 # the whole model on the card
TF_SPLIT_QWEN_DIR=$M zig-out/bin/tf-offload-metal serve s.sock 0 16 &          # Metal [0,16)
zig-out/bin/tf-offload-card $M ref.bin s.sock metal_first 16                   # card [16,32); levels A then B
TF_SPLIT_QWEN_DIR=$M zig-out/bin/tf-offload-metal serve s.sock 16 32 &         # Metal [16,32)
zig-out/bin/tf-offload-card $M ref.bin s.sock external_first 16                # card [0,16)
```

### Streams and chunked prefill

Written after the first card run, tested offline (CPU mock device, hipmac's mock backend, Metal+Metal on the 4B), then run
on the card (results below).

* **A step carries rows.** `StepIo.rows` and `Peer.step(.., rows, ..)`: a step is one decode position or up to
  `Config.max_rows` (at most 256) consecutive positions of one sequence, a prompt chunk. A token step carries `rows` u32;
  a hidden step `rows` plain f32 rows or, at level B, `rows` stamped rows, each on its own 16-byte groups (`wire.packStampedRows`;
  the stamp is per row so a consumer re-reads only the row that is late). A stage that ends the model answers with one
  token, the greedy pick after the last row. The tokens a host stage reads are staged in the window, behind the room
  the device's token reply has, so the socket protocol carries only offsets, a row count and a stream number.
* **Sequences share a device.** One `Pipeline` a sequence, over its own two slots of one window (`Config.window_offset`,
  `Pipeline.windowBytes(cfg, n)`), its own stage on each side (a stage owns the state of its layers for one sequence).
  `Pipeline.step` is `submit` then `collect`; `split/streams.zig` runs the skewed schedule (collect A, submit A's next
  step, collect B, submit B's next step), so a token costs about the slower side instead of the sum: with 3 ms
  per side the mock run of two sequences takes under 1.8x one (serial would be 2x, ideal 1x). The device queue is in order, so
  publication order is device order; no new device primitive.
* **Metal side.** `stage_pass.launch` takes the ids of `rows` positions: one is the decode kernels, more are the prompt
  kernels (`Session.promptChunk`'s path), output stored as stamped rows by the existing row-store kernel; the head draws
  on the last row. Stages of one model share its frame and prompt scratch and are serialized (`Serial`), so two sequences
  on one Mac model take turns on the GPU while the other half runs.
* **HIP side.** `HipLayerStage` takes `rows`: decode round for one, the engine's prompt span for more, one
  `consumeStamped` launch a row, one export or head projection of the last row. Each stage owns its caches and a ring of
  plan buffers (the decode step's plan words are copied asynchronously from pinned host memory, so a second recorded step must not
  overwrite the first's), over one engine and one stream; `warm` covers the row-count ladder, because the first launch of a
  new shape can wait for the queue to drain.
* **Remote.** `Request` carries `rows` and `stream`; `Host.peers` is one stage a sequence.

Tests (all offline): the toy model, every order and level, prompt chunks of 2..16 and three sequences on one device, token-identical
to the reference; the Metal+Metal real 4B at cuts 1, 8, 16, 31, both orders, both levels: decode, chunked prefill (17-token prompts
in chunks of 5) and two sequences all identical to the unsplit run (and chunked equal to token by token); HIP launch
sequences for rows 3 and 4 and for two sequences over one engine (disjoint plan buffers, per-row stamp words).

**Second card run (2026-11-10; Qwen3.5-4B MLX 4-bit, the same card, M4 Pro; `tb4-offload-card` = this branch merged with
the gate-passing `vega-macos-up`).** `tf-offload-card MODEL REF SOCKET ORDER CUT streams=N chunk=C repeat=R` with
`tf-offload-metal serve SOCKET FIRST LAST N` (FIRST..LAST the Mac's layer range; raw lines in [card_streams.txt](metal-tb4-split-moe-probe/card_streams.txt)). One prompt of 9 tokens
(or 72, the same prompt repeated 8 times) and 40 generated tokens a sequence; decode time is the run with the tokens minus the run with the
prompt only, so a rate is good to a few tok/s (repeats of the same run vary by that much). Every run's tokens were checked: 9-token prompts
against the reference file, 72-token prompts against the same run one position a step. Every sequence of every run was token-identical.
The card layers are `[cut,32)` and the head for `metal_first` (Metal `[0,cut)`), `[0,cut)` for `external_first`.

Decode, aggregate tokens per second over all sequences (a stream's share in brackets):

| Order, cut (card layers) | 1 stream | 2 streams | 3 streams | 4 streams |
|---|---|---|---|---|
| metal_first 8 (24) | 48.5 | 84.6 (42.3) | | |
| metal_first 12 (20) | | 98.0 (49.0) | | |
| metal_first 16 (16) | 50.7 | **119.9** (60.0) | 120.3 (40.1) | 120.3 (30.1) |
| metal_first 20 (12) | | **129.0** (64.5) | | |
| metal_first 24 (8) | 44.4 | 105.8 (52.9) | | |
| external_first 8 (8) | | 93.2 | | |
| external_first 12 (12) | | 106.0 | | |
| external_first 16 (16) | 52.4 | **130.1** (65.0) | | |
| external_first 20 (20) | | 113.2 | | |
| external_first 24 (24) | | 94.7 | | |

Level B gives the same figures within 1 tok/s on `metal_first`. Against the planner (4B, receipt of the first run): 113.2 tok/s predicted
for two streams, 62.8 for one; measured 119.9 at the planner's cut 16 and 129-130 at the best cuts (`metal_first` 20, `external_first` 16),
so the model is within 6% at its own cut and slightly pessimistic at the best; one stream measures 44-53 (50.7 at cut 16 against the
62.8 predicted, because the eager step costs more than the model's Vega figure, 19 ms a position all-card in the first run). Three and four streams do not add throughput at cut 16: the slower half is saturated at two. At 120 tok/s
a position leaves the pipeline every 8.3 ms, so the slower side's work for one position is about 8 ms there.

**The fair baseline: the Mac alone, same prompts, same tokens, batched.** `tf-offload-metal parallel REF N [CHUNK [REPEAT]]` runs
N sequences of the reference's prompt on the Mac's GPU in one engine, prefill in chunks, then decode in shared rounds of one row a sequence
(the engine's batched decode; best of two passes after a warm-up; three full runs agreed within 1%, the third with nothing else building; the first two overlapped a HIP kernel compile on another core). Every sequence token-identical to the reference.

| Sequences | Mac alone, decode tok/s (a stream) | Split, best of the table above | Split at the planner's cut 16 |
|---|---|---|---|
| 1 | 70.1-71.1 (70-71) | 52.4 | 50.7 |
| 2 | 115.5-116.5 (58) | 129-130 | 119.9 |
| 3 | 154.7-154.8 (51.6) | | 120.3 |
| 4 | 197.2-199.4 (49.3-49.8) | | 120.3 |

So, for this checkpoint on this Mac, the split is not faster than the Mac alone: at two sequences it is level (1.03-1.13x at the best cuts,
1.04x at the planner's), at three and four the Mac alone is ahead by 1.3x and 1.65x because a batched round reads the weights once for
all rows and the 4B's rows are cheap, while the split's slower half is saturated at two. The earlier "1.6-1.8x" compared the split with one
Mac stream; that is not the Mac's best use of its own GPU and is withdrawn as a speed claim. What the split buys on the 4B is only the
memory it frees (the Mac holds 16 of 32 layers at cut 16).

Chunked prefill (prompt 72 positions, `metal_first` unless stated; tok/s over the prefill only):

| Cut | Chunk 1 | Chunk 8 | Chunk 24 | Chunk 72 |
|---|---|---|---|---|
| 16 | 53.1 | 111.3 | 159.1 | 156.3 |
| 24 | | | | 251.2 |
| external_first 16 | | 116.4 | | 150.4 |

The Mac alone prefills the same 72 positions at 78.6 tok/s one position a step, 365.9 at chunk 8 and 415.4 at chunk 72 (two sequences, chunk 72:
493.8 aggregate): 1.7-2.7x the split's 150-251 at the same chunk, because the card's half of the prefill is the slow part. Two streams at chunk 8: 157.3 (`metal_first` 16) and 160.9 (`external_first` 16) tok/s aggregate. Chunked prefill is 3x the
token-by-token rate and saturates at about 156 tok/s for 16 layers on the card (the first run's whole-model figure on the card
was about 200 tok/s, so 16 layers alone would be about 400 [INFERENCE]); the Mac's half, at the Metal stage's speed, bounds the
rest. The decode rate after a chunked prefill is unchanged. The single-position fidelity check of the first run (cut 16, both
levels) passes after the merge: 47 of 48 top-1 (the same bf16 tie at position 2), logits cosine min 0.99983, mean 0.99997, status
word 0.

What this does not show: sequences longer than 112 positions (the Mac side was opened for 128), the 35B-A3B, any prompt other than
one 9-token prompt, and drift beyond one 4B checkpoint's 40 generated tokens.

### M4 groundwork: the Mac alone on the 35B-A3B (no card)

Raw lines: [mac_alone_batched.txt](metal-tb4-split-moe-probe/mac_alone_batched.txt); the model:
[batched_split_model.py](metal-tb4-split-moe-probe/batched_split_model.py), output in `batched_split_model_output.txt`.

**Download.** `mlx-community/Qwen3.6-35B-A3B-4bit` at `38740b847e4cb78f352aba30aa41c76e08e6eb46` in `~/models/Qwen3.6-35B-A3B-4bit`:
17 files, 20.40 GB (19.0 GiB) on disk (four safetensors of 5.29, 5.37, 5.37 and 4.38 GB, the tokenizer and configs); 2 min 8 s over this
link. This Mac has 48 GiB (`hw.memsize` 51,539,607,552), so the model fits with 28 GiB spare (mlx peak 19.6 GB for one sequence, 20.3 GB for eight).

**TensorFold's native Metal engine cannot run it in this checkout.** `tf-offload-metal ref` stops at `error.UnqualifiedQwenShape`:
`families/qwen27/model.zig` `supported` admits the dense 27B, 9B and 4B sizes and one routed shape (3,072 wide, 48 layers, the
122B), not the 35B-A3B's 2,048 x 40. The recipe's Metal commands for this model describe the earlier Python implementation. So
the baseline is mlx_lm 0.32.2 (mlx 0.32.4) `batch_generate`, greedy, the nine-token prompts below, 40 tokens, best of two passes after a warm-up.

**Mac alone, decode, aggregate tok/s** (different prompts a sequence: a mixture of experts reads the union of the rows' experts, and the
same prompt N times picks the same experts and flatters batching: 139.7 / 189.7 / 237.4 at 2 / 4 / 8):

| Sequences | 1 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|
| tok/s | 88.8 (88.2-89.7) | 135.8 (134.9-137.1) | 162.4 | 176.4 (173.9-178.0) | 206.9 |
| per sequence | 88.8 | 67.9 | 54.1 | 44.1 | 25.9 |

A round of N rows costs `a + b N` with a = 7.1 ms (what every round reads once) and b = 3.9 ms a row (eight experts of 40 layers, 540 MiB,
at 144 GB/s: a row brings almost no experts another row already read), which fits the four points within 2.5%. Prefill (the library's rate over
the batch): 9-token prompts 250 tok/s, 72 tokens 452 (one sequence) to 731 (four), 216 tokens 820 (one) and 799 (two). Context
length does not change decode here (88.6 tok/s at 216 tokens).

**The 4B's Mac baseline was low.** The same script on the 4B gives 84.6 / 154.6 / 205.7 / 220.9 tok/s for 1 / 2 / 3 / 4 sequences,
against TensorFold Metal's 70.1 / 116.5 / 154.7 / 199.0 (table above). The Mac's best use of itself on the 4B is mlx_lm's, so
at two sequences the card split's 120-130 tok/s is 16-22% *below* the Mac alone, not level. The conclusion above stands and is stronger.

**What the split does today, and what it would have to do.** The implemented streams run one row of one sequence a step (a step's `rows` are
consecutive positions of one sequence). The slower side's one-row round therefore bounds the whole pipeline however many sequences
there are: 4B measured 120 tok/s at two, three and four sequences. The model (`batched_split_model.py`; `a`, `b` fitted above; the card's speed
`r` the planner's whole-model one-row figure over the Mac's [INFERENCE]; validated on the 4B: model 117 tok/s against 120-130 measured at two to four
sequences, 67 against 53 at one) gives for the 35B-A3B:

| Card | r | Holds | Mac alone N=2 / 4 / 8 | Resident | Split, single-row steps (implemented) N=2 / 4 / 8 | Split, stages that batch rows of different sequences (not written) N=2 / 4 / 8 |
|---|---|---|---|---|---|---|
| RX Vega 56, 8 GiB | 0.68 | 13-16 of 40 layers | 134 / 175 / 207 | does not fit | 148 / 142 / 132 | 148 / 210 / 255 |
| 24 GiB class (7900 XTX modelled) | 1.79 | the model | 134 / 175 / 207 | 237 / 311 / 369 | 246 / 246 / 246 | 246 / 362 / 480 |
| 32 GiB class (5090 modelled) | 3.38 | the model | 134 / 175 / 207 | 443 / 584 / 695 | 378 / 378 / 378 | 378 / 559 / 745 |

**What a 24-32 GiB card needs.**
* To beat the Mac alone *resident* (the whole model on the card; only tokens cross, about 100 us a token): r > 1, that is a whole-model one-row
  decode above the Mac's 88.8 tok/s (about 140 GB/s effective for the 1.56 GiB read a token). Any card with 24 GiB or more and a decode path at
  half its peak bandwidth clears it from about 300 GB/s of peak. A card that holds the model can run it resident, which is the simplest design; in the model a batched split with the Mac beats
  resident by 1.04-1.30x at 24 GiB (r 1.8) and by 0.85-1.07x at 32 GiB (r 3.4), before the crossings' engineering and the second engine.
* To beat it as a *split* (the card does not hold the model, as the Vega): the stages must batch rows of different sequences. With
  batched stages the card needs r above 0.50 at two sequences, 0.32 at four and 0.19 at eight (free crossings), and the Vega at r 0.68
  clears all three, by 1.11x, 1.20x and 1.23x [INFERENCE]; with single-row steps (what is written) it wins only at two sequences (1.11x) and
  loses at four and eight.
* Capacity, not speed, is the case for a card below 20 GiB or a model above the Mac's budget (a 24 GiB Mac holds 12 GiB of weights).

The implication for the pull-request plan: before the MoE layer split (pull request 2) there is a stage that steps *rows of several sequences*
(the engines already have shared rounds: Metal's `Runner` slots, HIP's `batch_rows`), and its value is bounded by the table, not by the
2-stream measurement. Not measured: the card on the 35B (the HIP MoE layer range does not exist), any native-Metal 35B number, routing overlap between
real rows beyond eight sequences, and the mlx and TensorFold Metal engines' difference on the 4B (84.6 against 70.1 tok/s one sequence), which says the
TensorFold Metal baseline above is not the Mac's best.

### Rebase onto upstream main

The first version of this work sat on a base 6,207 commits behind `ashhart/TensorFold` main. The pull-request commits
were cherry-picked onto upstream main (21ea8dee4 at the time) and the gfx900 work of the Vega branch was ported
separately (`vega-macos-up`: seven commits, the conflicts and the dropped commits are in its commit messages).
Conflicts in this series and how they were resolved:

* `cluster/launch.zig`, `launch_test.zig`: upstream added CUDA probes and a per-node `bin`; both kept, and the
  `launchable()` refusal added in front of them. `build.zig`: upstream moved the Metal targets to `zig/build/`; the
  split steps live in the new `zig/build/split.zig`, called from `metal_targets.zig`. `hip/root.zig`, `docs/recipes/README.md`:
  both sides kept.
* New compile errors after the rebase, fixed at the source: `bringup.Live.init` (a new upstream caller of
  `Membership.init`, now `try`) and `fabric/mcdma.zig` (an exhaustive switch over `LinkKind`).
* Upstream's HIP runtime reads `TENSORFOLD_HIP` (not the fork's `TF_HIP_LIB`) and its caps table has no gfx900 row: the
  adapter's mock test skips on a base without the Vega port, runs on `tb4-offload-card`.
* Upstream's Metal Qwen3.5 backend is still the pinned 2B's; dense geometry and untied heads go through the `qwen27`
  engine, which could not run a tied head. This series adds the tied head and the layer range there instead of in the 2B
  engine.

Cluster tests (`zig test` on the cluster root with stand-in modules, because `test-cluster` needs the Metal compiler):
115 pass, 6 skip, 0 fail on the rebased tree.

### Not in this pull request, and why

* **Metal-prefill-then-state-handoff.** Chunked prefill through the split exists (rows > 1 per step, above) but both sides
  prefill their own layers; having the Mac prefill the whole model and ship the card's layers' KV and linear state
  across is the next change, and the one that matters for long prompts [INFERENCE].
* **Prefill speed.** Chunked prefill is tested for exactness only; no rate is measured, and the head draws on every chunk on
  the Metal side (one extra head pass per intermediate chunk).
* **Tokens on the card beyond one prompt.** One 49-position sequence on one 4B checkpoint (First card run) bounds the
  Metal-to-HIP drift at logits cosine >= 0.9996 and one tie-break; more prompts, long contexts and the 35B-A3B are not
  measured.
* **Graph replay of the ranged HIP step**: the stage is eager (a graph would bake the sequence number and window
  addresses in); `seq` through a device-side word is the follow-up.
* **Level B on the Metal side as consumer**: Metal reads stamped rows on the CPU (`wire.readStamped`), the GPU consume
  kernel exists only for HIP.
* **A serving driver**: a command that spawns the card process as a child, loads a tokenizer and runs a prompt end to end
  (`tensorfold serve --device-split`) is not in this change. The two-process design is deliberate ([One process or
  two](#one-process-or-two)); the tests and the card programs drive the pipeline directly.
* Milestone M2 (stale stamps over 10^6 replies) is not done; the receipt's `hop_*` fields are the card's numbers from the
  first run. The 2-stream runs of M3 are done on the 4B (Streams and chunked prefill).

## Pull requests

One behaviour each, stacked, each against `main`, each linking the issue with `Refs #N`. Items marked *hipmac* depend
on out-of-tree runtime work; everything else is generic.

1. **Transport, interface, dense placements: resident, layer split.** (Offline half implemented, see above; card runs and the
   `--device-split` serving driver remain.) `core/split` (window protocol, `ExternalDevice`, fakes), Metal `HostWindow` and
   `Relay`, `tensorfold offload probe`, the planner for dense models and `placement.json`, a layer range in the HIP and Metal
   engines, `--device-split` for dense checkpoints, cluster `Backend.hip` and the link kind. **Resident on the external device,
   tokens over Thunderbolt, is a first-class placement here, not a corner of the split**: for a 24-32 GiB card that holds the model
   it is the primary win (it decodes at the card's speed and moves 4 B a token; see M4 groundwork), the planner picks it when it
   fits and is faster, and `tf-offload-card MODEL REF unsplit` is its end-to-end smoke test (the pipeline order `external_only`).
   Tests: protocol, planner, toy dense family split/unsplit, resident on the toy and on the real engine. *hipmac*: window
   allocation (exists) and optional wait/write symbols with mock support; the card gate (M3) needs the runtime change.
2. **Stages that step rows of several sequences.** A step names its sequences (`StepIo.slots`): one slot is a prompt chunk or a
   decode position, several slots are a decode batch, one row of each sequence at its own position, in one round on each side
   (the engines' shared rounds: Metal `Runner` slots, HIP `batch_rows`). A lane serves up to eight sequences; lanes pipeline over
   one device (`streams.zig`); the resident placement batches the same way. Why it comes before the MoE split: the Mac alone
   batches, and a split of single-row steps cannot beat it beyond two sequences (M4 groundwork). Tests: token-identical per
   sequence against its own unsplit run for 1 to 8 sequences (Metal+Metal on the real 4B, the toy model, HIP plumbing on the mock).
3. **MoE.** Layer split for `qwen3_5_moe` (stage hooks in `moe.zig`, KV/state hand-off after Metal prefill) on top of PR 2's
   batched stages, then the expert split and reverse split behind the same planner; Metal expert service reusing upstream
   kernels; boundary kernels with contract entries. Needs the native Metal engine to admit the 35B-A3B's shape (it does
   not today) or a baseline outside it.
4. **Hot-expert cache and adaptive placement.** Routing histograms, resident-expert sets chosen from measured hit rates,
   re-planning between requests (never inside a round), MTP head placement. Only after M6.

Documentation (recipe page, README row) lands with the first pull request that has receipts for it. The optimization
ledger cell for the family is updated by the pull request that adds or drops an optimization.

## Proposed gate

* **Exactness.** Items 1-5 above, for every admitted stream count.
* **Performance.** Decode and cold prefill with the repository clients on the same machine, checkpoint revision and
  prompts as the unsplit baseline; split no slower than all-Metal at the widest admitted stream count on the card where
  the planner says it should win, and the planner's prediction within 15%. A placement the planner predicts slower than
  all-Metal is never selected by `auto`.
* **Memory.** Peak footprint on both devices, window bytes, with RAM class and context stated; a 24 GiB-class budget run.
* **Platforms.** Metal: M4 Pro run; M1-M3 and M5 unrun until someone with those chips posts numbers. Card: gfx900 run.
  RDNA3/4 and NVIDIA unrun.
* The gate is proposed here and is not agreed until a maintainer says so.

## ATC flow (described, not run)

Before any pull request: `atc doctor --json` once for the clone, `atc check <issue> --json` (proceed only on `free` or
`yours`; on `proposeInstead` offer pair, split or review), `atc claim <issue> --json` once the issue exists and work is
authorized, keep the returned session number, `atc pulse <issue> <session> working` while working, `atc who <file>
--json` after edits to shared files (`moe.zig`, `plan.zig`), `atc overlap --json` after the last rebase and before
each pull request, resolve every `"conflicts"` entry, `atc land <issue> <session>` when the session stops, `atc release
<issue>` when stopping. Nothing was posted or claimed while writing this document.

## Open risks

* **The card is slower than Metal for decode.** Single-stream gains need a faster external GPU; today the case is
  capacity and 2+ streams, and on the 4B the 2-stream split is level with the Mac alone batching two sequences (120-130 against 116 tok/s) and slower at three and four; its value there is capacity, and whether a MoE's batched round on the Mac scales worse (more experts read) than the dense 4B's is the open question for M4 [INFERENCE].
* **Wake cost of a Metal hop** is a property of this macOS and GPU (65 us on an idle Mac, 149 us p50 and 450 us p99
  with the card in use); it dominates the expert split (40 hops) and not the layer split (1). A change in a macOS release moves the answer; the probe receipt carries the OS build.
* **Visibility of Metal output to PCIe readers** (flag before data). Measured toward the CPU only; Level A avoids it.
* **Mixed activation types** (fp16 stage, bf16 stage): fp16 overflow on large residual outliers; fidelity gate, not
  bit equality.
* **hipmac extensions** (wait/write value, window) are outside this repository; the interface has a mock, but the card
  receipt needs the runtime.
* **One client per card, 128 sysmem blocks, SDMA wedge on SIGKILL** shape the window design and the failure rules.
* **TB4 sharing.** A TB4 port shared with storage or a display changes the link numbers; the probe measures the real
  link at startup, and the planner uses the receipt, not a constant.
* **Hardware coverage.** Only an M4 Pro and a Vega 56 are available; everything about stronger cards and other chips
  needs other testers.
* **Router skew** (hot-expert cache) is unmeasured; PR 3 is conditional on M6.
* **Upstream drift.** The local base predates upstream's Metal MoE; first work is the rebase.

## Reproduce

```sh
cd docs/proposals/metal-tb4-split-moe-probe
DEVELOPER_DIR=/Library/Developer/CommandLineTools clang -O2 -fobjc-arc -framework Metal -framework Foundation metal_probe.m -o metal_probe
./metal_probe wrap; ./metal_probe latency; ./metal_probe poll; ./metal_probe vis   # vis spins about a minute
uv venv /tmp/mlxenv && uv pip install --python /tmp/mlxenv/bin/python mlx mlx-lm
/tmp/mlxenv/bin/python mlx_expert_probe.py 40
/tmp/mlxenv/bin/python -m mlx_lm generate --model <35B-A3B dir> --max-tokens 256 --temp 0 --ignore-chat-template
python3 placement_model.py
```

The probes are development tools in the sense of CONTRIBUTING (Objective-C and Python are not part of the native build);
PR 1 ports them to Zig with thresholds.
