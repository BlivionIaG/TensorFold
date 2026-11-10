# Qwen3.5 / Qwen3.6 on native HIP: a layer range as one stage of a split model

The HIP engine in `zig/src/families/qwen3_5/backend/hip/` can run a contiguous range of a dense model's layers, so a
Mac's Metal GPU and an external HIP GPU can each run part of one model for decode, the hidden row crossing a host
window (the transport and planner are in [the split-offload proposal](../proposals/metal-tb4-split-moe.md)). Status:
offline only. Nothing below has run on a card, and no token has been compared between devices.

| Piece | Where |
| --- | --- |
| `Range{first, last}` (half open; `all`, `first:last`, `first:`), validation and refusals | `model/view.zig` |
| Weights uploaded for the range only (embedding when `first == 0`; final norm and head when it ends the model, the tied head an upload of its own) | `model/weights.zig` (`fromCheckpointRange`), `model/bridge.zig` |
| KV pages and linear state for owned layers only (`LayerCache.absent` elsewhere) | `forward/pages.zig`, `forward/state.zig` |
| Prompt span over the range | `forward/forward.zig` (`spanWith`, `Handoff`), `Engine.spanStage` |
| Decode step over the range (one row) | `forward/window.zig` (`forward`), `engine/round.zig` (`stage`), `Engine.stage` |
| The external stage for `split.pipeline` | `split_stage.zig` (`HipLayerStage`) |
| Boundary kernels (import, stamped consume, export) | `zig/src/hip/ops/split_rows.zig` |

`Engine.Options.range` selects the range (default: every layer, with the launches and bits unchanged). The stage holding
layer 0 embeds a u32 token read in place from the window; the stage ending the model applies the final norm, the head
and the greedy argmax and writes the token into the window. The row that crosses is the residual sum after the range's
last layer, before the next input norm, as `hidden` fp32 values whatever the activation dtype. A range's last layer ends
in a plain residual add where the whole model's decode fuses that add with the next layer's norm; the next stage starts
with that norm. [INFERENCE] the fused and the unfused launches give the same bits, as the prompt path's formulas do;
only a run on a device can show it.

Refused by name, before any device is opened: a range outside the model or empty (`BadLayerRange`), with `--vision`
(`RangeWithVision`), tensor parallelism (`RangeWithTensorParallel`), a bf16 checkpoint (`RangeWithBf16Checkpoint`), a
drafter's taps or the lane host, which drafts and verifies whole-model rounds (`RangeWithDraft`). A ranged engine's
`prefill`, `spanRows`, `verify` and `choose` return `RangedEngine`.

Limits of this change: a prompt runs the one-token decode step once per position (rows = 1), and KV and linear state
advance one position per step on the side that owns the layer; prefill in batches across the window is not here. The
decode step is recorded eagerly each token, not replayed from a graph: a graph would bake in the stamp's sequence number
and the window addresses. The stage's launches and the pipeline's waits and writes must be on one stream: open the
adapter (`hip.offload.Options.stream`) on the engine's stream, and open the engine with `batch_rows = 1`.

Tests, no device: `zig build -Dhip=true test-qwen35-split-host` runs a recording stand-in for the HIP runtime on a
launcher that names gfx1030, over the 4B's shape with fake weights. It compares a ranged decode step's launches with the
unsplit step's slice (the embedding only first, the head only last, import only non-first, export only non-last), the
same for a prompt span, the allocations of a ranged engine against its own plan, and the stage's recorded operations
between the pipeline's wait and write in every order. They compute nothing: they check plumbing, not tokens.
