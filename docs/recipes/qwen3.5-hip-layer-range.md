# Qwen3.5 / Qwen3.6 on native HIP: a layer range of the model

The HIP engine in `zig/src/families/qwen3_5/backend/hip/` can run a contiguous range of a dense model's layers, so two
devices can each hold part of one model and pass the hidden row between them through a host window. This page covers
the range itself and the stage that drives it. Status: offline only. Nothing below has run on a card, and no token has
been compared between a split and an unsplit run.

| Piece | Where |
| --- | --- |
| `Range{first, last}` (half open; `all`, `first:last`, `first:`), validation and refusals | `model/view.zig` |
| Weights uploaded for the range only (embedding when `first == 0`; final norm and head when it ends the model, the tied head an upload of its own) | `model/weights.zig` (`fromCheckpointRange`), `model/bridge.zig` |
| KV pages and linear state for owned layers only (`LayerCache.absent` elsewhere) | `forward/pages.zig`, `forward/state.zig` |
| Prompt span over the range | `forward/forward.zig` (`spanWith`, `Handoff`), `Engine.spanStage` |
| Decode step over the range (one row) | `forward/window.zig` (`forward`), `engine/round.zig` (`stage`), `Engine.stage` |
| Boundary kernels (import, stamped consume, export) | `zig/src/hip/ops/split_rows.zig` |

`Engine.Options.range` selects the range (default: every layer, with the launches and bits unchanged). The range that
holds layer 0 embeds a u32 token; the range that ends the model applies the final norm, the head and the greedy argmax.
The row that crosses is the residual sum after the range's last layer, before the next input norm, as `hidden` fp32
values whatever the activation dtype. A range's last layer ends in a plain residual add where the whole model's decode
fuses that add with the next layer's norm; the next range starts with that norm. [INFERENCE] the fused and the unfused
launches give the same bits, as the prompt path's formulas do; only a run on a device can show it.

Refused by name, before any device is opened: a range outside the model or empty (`BadLayerRange`), with `--vision`
(`RangeWithVision`), tensor parallelism (`RangeWithTensorParallel`), a bf16 checkpoint (`RangeWithBf16Checkpoint`), a
drafter's taps or the lane host, which drafts and verifies whole-model rounds (`RangeWithDraft`). A ranged engine's
`prefill`, `spanRows`, `verify` and `choose` return `RangedEngine`.

Limits: a prompt runs the one-token decode step once per position (rows = 1), and KV and linear state advance one
position per step on the side that owns the layer; prefill in batches across the window is not here. The decode step is
recorded eagerly each token, not replayed from a graph, because a graph would bake in the stamp's sequence number and the
window addresses [INFERENCE: not measured]. The launches of a stage and the waits and writes of whatever drives it must
be on one stream, and the engine opened with `batch_rows = 1`.

Tests, no device: `zig build -Dhip=true test-qwen35-split-host` runs a recording stand-in for the HIP runtime on a
launcher that names gfx1030, over the 4B's shape with fake weights. It compares a ranged decode step's and a ranged prompt
span's launches with the unsplit ones' slice (the embedding only first, the head only last, import only non-first, export
only non-last). The stand-in computes nothing: this checks plumbing, not tokens. `test-qwen35-hip-host` runs the range's
validation and parse tests.

## As the external stage of a split model

`HipLayerStage` (`backend/hip/split_stage.zig`) records one ranged decode step per pipeline step between the wait on the
input slot and the release write of the output slot, reading and writing the window's slots in place by GPU address.
`HipExternal` (`zig/src/hip/offload.zig`) is the card as the split layer's `ExternalDevice`: pinned host windows mapped
for the card, `hipStreamWaitValue32` / `hipStreamWriteValue32` on the window's words, a bounded `sync`, and device memory
by address. It needs a runtime that exports those two calls; it refuses an architecture the kernel caps table does not
know. A caller that owns the engine's stream passes it as `Options.stream`, so the waits, the launches and the writes
share one stream. `zig build -Dhip=true tf-offload-hip` builds `tf-offload` with a `hip` device next to `mock`;
`probe --device hip` needs a card and a runtime from `TENSORFOLD_HIP` or the search path.

Tests, no device: `zig build -Dhip=true test-hip-offload-host` runs `HipExternal` on a stand-in runtime whose stream is
a queue run in order by a thread, so a wait holds later operations until the host writes the word; it also checks the
stamped row's layout against `split.wire`. `test-qwen35-split-host` also runs the real `Pipeline` over a `HipLayerStage`
in every order and checks the stage's launches sit between the pipeline's wait and write. Not run: a stage on a card.
