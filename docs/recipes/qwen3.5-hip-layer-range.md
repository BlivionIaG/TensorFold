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
| Prompt span over the range (`rows` positions) | `forward/forward.zig` (`spanWith`, `Handoff`), `engine/stage.zig` (`span`), `Engine.spanStage` |
| Decode round over the range (one row, or one row of each of several sequences), the plan buffers a stage owns | `forward/window.zig` (`forward`), `engine/stage.zig` (`decode`, `decodeRows`, `newPlan`), `Engine.stage`, `Engine.stageRows` |
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

A step carries `rows` positions of one sequence: one row runs the decode round, several run the prompt span (the prompt
kernels), so a prompt chunk costs one forward and one hand-off pair for all its rows. [INFERENCE] a chunk of one row
runs the decode kernels, not the prompt's, so its bits can differ from an unsplit prefill's for that position; only a run
on a device can show whether a token ever differs. A stamped input is consumed row by row (one launch and one status
word each: `HipLayerStage.stampStatus` reports the worst); plain rows cross in one import and one export. The stage
warms every kernel and shape it can take (`warm`: the decode step and the span at each row count up to `max_rows`),
because a kernel that needs more scratch makes the runtime wait for the stream to drain, which a stream waiting on the
window never does. Steps are recorded eagerly, not replayed from a graph: a graph would bake in the stamp's sequence
number and the window addresses. The stage's launches and the pipeline's waits and writes must be on one stream: open
the adapter (`hip.offload.Options.stream`) on the engine's stream, and open the engine with `batch_rows >= slots` (below)
and `prompt_rows >= max_rows`.

A stage serves `Options.slots` sequences (a pipeline lane's, at most 8 and at most `batch_rows`), each with its own caches
and position. A step names its slots: one slot, and its rows are consecutive positions of that sequence; `rows` slots,
and it is a decode batch, row i the next position of sequence `slots[i]` at its own position. A batch is one decode round
over `rows` windows of different caches (the engine's lane-round machinery: one plan, one forward), so the weights are
read once for all of them; each row's stamped input is its own consume launch and the head writes a token a row. Only the
named slots advance, by one, and a sequence that skips a step keeps its position. The round runs exactly `rows` rows, not
padded to the lane rounds' buckets: a stage's rows come from the window, and padding rows would read what lies behind
them. [INFERENCE] the plan kernels take any row count and every row is bitwise serial, as the lane rounds' padding
relies on; only a run on a device can show it. `warm` runs the batches of every size from 2 to `slots`.

Several lanes share one engine and one stream through one `HipLayerStage` each (own caches, status words and plan
buffers; the pool must hold the caches of every slot of every stage). A decode step's plan words go up from pinned host
words by an asynchronous copy that runs when the stream reaches it, so each stage has `Options.plans` (default 2) buffers
it takes in turn, and relies on one invariant: at most `plans - 1` earlier steps of a lane are recorded and not collected
when it records another. The pipeline collects a step's tokens before it queues the lane's next step, so one buffer would
do and two are slack. The arenas and the shared round's host rows are used while recording or ordered by the stream.

Tests, no device: `zig build -Dhip=true test-qwen35-split-host` runs a recording stand-in for the HIP runtime on a
launcher that names gfx1030, over the 4B's shape with fake weights. It compares a ranged decode step's launches with the
unsplit step's slice (the embedding only first, the head only last, import only non-first, export only non-last), the
same for a prompt span (plain and stamped), the allocations of a ranged engine against its own plan, the stage's recorded
operations between the pipeline's wait and write in every order, the boundary launches of a multi-row step (consume
count, row offsets, status words), decode batches of 1, 2, 3 and 8 slots in every order and level (one round, a consume a
row, only the named slots advance, a skipped slot keeps its position), two lanes' disjoint plan buffers over one engine
and the warm-up's coverage. They compute nothing: they check plumbing, not tokens.
