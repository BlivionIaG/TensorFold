//! Native EXL3 GLPv3 paired edits; materialize deferred FFN post before editing, then recompute the next boundary.
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

/// The block's pending hyper-connection post made real in the other stream buffer, if one is pending.
fn expand(m: *Model, enc: mtl.ComputeEncoder, rows: u32) void {
    if (!m.pending_expand) return;
    const ws = &m.workspace;
    const cur = m.streamView();
    const nxt = if (m.stream_alt) ws.view(.stream) else ws.view(.stream_next);
    model.mhc.encodePost(enc, &m.pipelines.mhc, ws.view(.branch), cur, ws.view(.post), ws.view(.comb), nxt, .{ .rows = rows, .width = 5120, .branch_stride = 5120, .residual_stride = 20480, .y_stride = 20480 });
    m.stream_alt = !m.stream_alt;
    m.pending_expand = false;
    enc.barrier();
}

/// Block `index`'s FFN fold: the pending hc post made real, then the four-copy stream edited in place.
/// A capture in force reads that same post-fold stream and leaves it as it is: the two are never set together.
pub fn fold(m: *Model, enc: mtl.ComputeEncoder, index: usize, rows: u32) void {
    const plan = m.glp;
    const capture = m.glp_capture;
    const steered = if (plan) |p| p.covers(@intCast(index)) else false;
    if (!steered and capture == null) return;
    std.debug.assert(plan == null or capture == null);
    expand(m, enc, rows);
    const s = m.streamView();
    if (capture) |c| {
        c.record(enc, @intCast(index), s.buffer, s.offset, rows);
        enc.barrier();
    }
    if (steered) {
        plan.?.fold(enc, @intCast(index), s.buffer, s.offset, rows);
        enc.barrier();
    }
}

/// Whether the state entering layer `lo` (the cone's stash) holds a materialised stream: the layer before it is edited.
pub fn expandedBefore(m: *const Model, lo: u32) bool {
    if (lo == 0) return false;
    if (m.glp) |plan| return plan.covers(lo - 1);
    // A capture run materialises every fold it reads, so the stream entering block lo>0 is real.
    return m.glp_capture != null;
}
