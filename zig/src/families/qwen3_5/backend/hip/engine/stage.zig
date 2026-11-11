//! The steps of an engine that runs a layer range (a stage of a model split across devices): one decode row, or a prompt
//! span of several rows, over a sequence's caches, recorded on the engine's stream and not waited for. The boundary rows
//! cross through a `fwd.Handoff`; every launch is eager (a graph would bake in the hand-off's stamp and addresses).

const std = @import("std");
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

/// A sequence a decode round takes a row of: its caches and the position the row is.
pub const Row = struct { caches: *state.Caches, pos: usize };

/// One decode step (one row, one slot) of the layers of `e`'s range over `caches` at slot `pos`: `decodeRows` of one.
pub fn decode(e: *Engine, buf: *plan.Buffer, caches: *state.Caches, pos: usize, tokens: u64, hand: fwd.Handoff, argmax_at: u64) !?hip.ops.Tensor {
    return decodeRows(e, buf, &.{.{ .caches = caches, .pos = pos }}, tokens, hand, argmax_at);
}

/// One decode round of the layers of `e`'s range over `seqs.len` rows, row i the next position of the sequence `seqs[i]`
/// (its own caches, its own position, no sequence twice): the plan words go up in the stream, then one forward, so the
/// weights are read once for all the rows. `tokens` is the device address of the u32 token ids, one a row (layer 0's
/// stage); `hand` the boundary rows of the stages that have one, `seqs.len` of them; `argmax_at` the device address of
/// the i32 each row's greedy pick lands in (the last stage), one a row. Returns the head's logits rows, or null short of
/// the head. The round runs exactly `seqs.len` rows, never padded to a bucket: a stage's rows come from the window,
/// and padding rows would read what lies behind them [INFERENCE: the plan kernels take any row count and every row is
/// bitwise serial, as the lane rounds' padding relies on; only a run on a device can show it].
///
/// The plan's words go up from the pinned host words of `buf` by an asynchronous copy that runs when the stream reaches
/// it, so `buf` must not be filled again before that copy ran. Every sequence's stage owns its buffers (`newPlan`) and
/// takes them in turn: with n buffers a stage may have n - 1 earlier steps recorded and not collected when it records
/// one more. The rest a step touches is written while recording (the shared round's host rows) or ordered by the stream.
pub fn decodeRows(e: *Engine, buf: *plan.Buffer, seqs: []const Row, tokens: u64, hand: fwd.Handoff, argmax_at: u64) !?hip.ops.Tensor {
    const st = &e.round;
    const m = e.model();
    if (e.stopped) return error.GroupStopped;
    if (seqs.len == 0 or seqs.len > e.o.batch_rows) return error.WindowTooWide;
    var visible: usize = 0;
    for (seqs, 0..) |q, i| {
        if (q.pos >= q.caches.covered()) return error.ContextFull;
        for (seqs[0..i]) |p| if (p.caches == q.caches) return error.WindowTooWide;
        visible = @max(visible, q.pos + 1);
    }
    const cap = std.mem.alignForward(usize, e.o.capacity, plan.min_span);
    const shape: plan.Shape = .{ .rows = @intCast(seqs.len), .slots = @intCast(seqs.len + 1), .span = @intCast(@min(plan.spanOf(visible), cap)) };
    const none = [1]u32{0};
    for (seqs, st.wins[0..seqs.len]) |q, *w| w.* = .{ .desc = q.caches.desc.base(), .pos = q.pos, .tokens = &none };
    e.rounds.reset();
    const o = e.ops(&e.rounds);
    try win.kept(o, m, shape.rows, st.snaps);
    const l = plan.Layout.of(shape.rows, shape.slots, m.spec.n_layers);
    buf.fill(l, shape, st.wins[0..seqs.len], st.scratch.desc.base(), st.snaps);
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
