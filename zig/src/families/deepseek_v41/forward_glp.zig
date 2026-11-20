//! The EXL3 forward's GLP steering edits (spec v3 hc_fold_pre_engram): where the plan's two edits sit in a layer.
//!
//! Direction N edits backbone block N twice: its attention writer (before the hc post) and the four-copy stream its FFN
//! leaves (before the next Engram or the head). The fused boundary defers a sublayer's hc post into the next launch, so
//! an edited fold first materialises the pending post (the unfused kernel the Engram flush already uses) and edits the
//! result in place; the lagged pre mixes come from the pre-FFN stream and are the model's own, so they are not touched.
//! With no plan nothing here runs.
const std = @import("std");
const mtl = @import("metal");
const model = @import("model.zig");

const Model = model.Model;

/// Block `index`'s attention writer [rows][5120] bf16, edited in place after the attention's full output projection.
pub fn writer(m: *Model, enc: mtl.ComputeEncoder, index: usize, rows: u32) void {
    const plan = m.glp orelse return;
    if (!plan.covers(@intCast(index))) return;
    const b = m.workspace.view(.branch);
    plan.writer(enc, @intCast(index), b.buffer, b.offset, rows);
    enc.barrier();
}

/// Block `index`'s FFN fold: the pending hc post made real, then the four-copy stream edited in place.
pub fn fold(m: *Model, enc: mtl.ComputeEncoder, index: usize, rows: u32) void {
    const plan = m.glp orelse return;
    if (!plan.covers(@intCast(index))) return;
    const ws = &m.workspace;
    if (m.pending_expand) {
        const cur = m.streamView();
        const nxt = if (m.stream_alt) ws.view(.stream) else ws.view(.stream_next);
        model.mhc.encodePost(enc, &m.pipelines.mhc, ws.view(.branch), cur, ws.view(.post), ws.view(.comb), nxt, .{ .rows = rows, .width = 5120, .branch_stride = 5120, .residual_stride = 20480, .y_stride = 20480 });
        m.stream_alt = !m.stream_alt;
        m.pending_expand = false;
        enc.barrier();
    }
    const s = m.streamView();
    plan.fold(enc, @intCast(index), s.buffer, s.offset, rows);
    enc.barrier();
}

/// Whether the state entering layer `lo` (the cone's stash) holds a materialised stream: the layer before it is edited.
pub fn expandedBefore(m: *const Model, lo: u32) bool {
    const plan = m.glp orelse return false;
    return lo > 0 and plan.covers(lo - 1);
}
