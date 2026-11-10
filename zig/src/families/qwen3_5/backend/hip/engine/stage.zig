//! The steps of an engine that runs a layer range (a stage of a model split across devices): one decode row, or a prompt
//! span of several rows, over a sequence's caches, recorded on the engine's stream and not waited for. The boundary rows
//! cross through a `fwd.Handoff`; every launch is eager (a graph would bake in the hand-off's stamp and addresses).

const hip = @import("hip");
const state = @import("../forward/state.zig");
const win = @import("../forward/window.zig");
const fwd = @import("../forward/forward.zig");
const plan = @import("../forward/plan.zig");
const round = @import("round.zig");
const Engine = @import("engine.zig").Engine;

/// A device plan the engine's own rounds would hold, for a stage that keeps several (see `decode`).
pub fn newPlan(e: *Engine) !plan.Buffer {
    return plan.Buffer.init(&e.driver, e.o.batch_rows, e.model().spec.n_layers, e.pool.count);
}

/// One decode step (one row, one slot) of the layers of `e`'s range over `caches` at slot `pos`: the plan words go up
/// in the stream, then the forward. `tokens` is the device address of the u32 token id (layer 0's stage); `hand` the
/// boundary rows of the stages that have one; `argmax_at` the device address of the i32 the head's greedy pick lands in
/// (the last stage). Returns the head's logits row, or null short of the head.
///
/// The plan's words go up from the pinned host words of `buf` by an asynchronous copy that runs when the stream reaches
/// it, so `buf` must not be filled again before that copy ran. Every sequence's stage owns its buffers (`newPlan`) and
/// takes them in turn: with n buffers a stage may have n - 1 earlier steps recorded and not collected when it records
/// one more. The rest a step touches is written while recording (the shared round's host rows) or ordered by the stream.
pub fn decode(e: *Engine, buf: *plan.Buffer, caches: *state.Caches, pos: usize, tokens: u64, hand: fwd.Handoff, argmax_at: u64) !?hip.ops.Tensor {
    const st = &e.round;
    const m = e.model();
    if (e.stopped) return error.GroupStopped;
    if (pos >= caches.covered()) return error.ContextFull;
    const none = [1]u32{0};
    const rows = [1]round.Rows{.{ .caches = caches, .pos = pos, .tokens = &none }};
    const shape = try round.shapeOf(e, &rows);
    st.wins[0] = .{ .desc = caches.desc.base(), .pos = pos, .tokens = &none };
    e.rounds.reset();
    const o = e.ops(&e.rounds);
    try win.kept(o, m, shape.rows, st.snaps);
    const l = plan.Layout.of(shape.rows, shape.slots, m.spec.n_layers);
    buf.fill(l, shape, st.wins[0..1], st.scratch.desc.base(), st.snaps);
    const words = buf.dev.base();
    const p: hip.plan_ops.Plan = .{ .args = buf.args(l), .rows = shape.rows, .slots = shape.slots };
    const r: win.Round = .{ .plan = p, .tokens = tokens, .span = shape.span, .inputs = st.snaps, .positions3 = words + 4 * l.rope, .rotary = words + 4 * l.rotary, .hand = hand };
    try buf.send(e.stream.handle, l);
    return (try round.forwardOn(o, m, r, null, argmax_at)).y;
}

/// A prompt span of `len` rows from position `pos0` over the layers of `e`'s range, with the prompt's kernels (the bits
/// of `Engine.prefill`'s rows). The first stage reads `len` u32 token ids at the device address `input`; the others read
/// their `len` rows through `hand`, one consume launch a stamped row. Short of the last layer the residual rows go out
/// through `hand.dst` as f32 and the result is null. The last stage projects the span's last row only and writes its
/// greedy pick (an i32) at `argmax_at`; the result is that logits row. The plan words are not used, so the span owns no
/// buffer, and the arena it works in is the engine's prompt arena (`len` at most `Options.prompt_rows`).
pub fn span(e: *Engine, caches: *state.Caches, input: u64, len: usize, pos0: usize, hand: fwd.Handoff, argmax_at: u64) !?hip.ops.Tensor {
    if (e.stopped) return error.GroupStopped;
    if (len == 0 or pos0 + len > caches.total or len > e.o.prompt_rows) return error.PromptTooLong;
    e.prompts.reset();
    const o = e.ops(&e.prompts);
    const m = e.model();
    const hidden = try fwd.spanWith(o, m, caches, input, len, pos0, null, hand);
    const head = m.head orelse return null;
    const y = try o.project(fwd.at(hidden, (len - 1) * m.spec.hidden), head, 1, false);
    try o.argmaxRows(y, 1, head.n, argmax_at);
    return y;
}
