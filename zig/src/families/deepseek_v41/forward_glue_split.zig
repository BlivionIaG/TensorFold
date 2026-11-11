//! The layer-split producer's glue: a chunk's boundary rows (the cut layer's residual and the next site's pre) to the
//! pinned ring, copied asynchronously (rank 0). Nothing here waits for the stream.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const run = @import("run.zig");
const fwd = @import("forward.zig");
const pf = @import("forward_prefill.zig");
const split = @import("handoff/split.zig");
const Copy = @import("split_copy.zig").Copy;
const Sink = @import("split_copy.zig").Sink;
const Forward = fwd.Forward;

/// Rank 0: the pinned ring for a split's chunks of up to `rows_max` rows, rows published to `sink`. Other ranks hold none.
pub fn begin(f: *Forward, sink: Sink, rows_max: u32) !void {
    if (f.comm.rank() != 0) return;
    const st = pf.state(f);
    if (st.split_copy != null) return error.BadGlue;
    st.split_copy = try Copy.create(f.gpa, f.runner.d, sink, rows_max);
}

/// Publishes the chunks whose copies have finished, in order, without waiting for any.
pub fn poll(f: *Forward) !void {
    const c = pf.state(f).split_copy orelse return;
    try c.ring.poll();
}

/// The split is over: `ok` awaits and publishes every chunk still in flight; either way the ring is freed.
pub fn end(f: *Forward, ok: bool) !void {
    const st = pf.state(f);
    const c = st.split_copy orelse return;
    st.split_copy = null;
    defer c.destroy();
    // a failed split may leave the compute stream's pre copies queued: the slots are not freed under them
    errdefer f.runner.stream.synchronize() catch {};
    if (ok) try c.ring.drain() else try f.runner.stream.synchronize();
}

/// glue split_hold (): the boundary buffer is about to be rewritten, so the compute stream waits (on the device) for
/// the previous chunk's copy out of it
pub fn hold(f: *Forward, r: *run.Runner) !void {
    const c = pf.state(f).split_copy orelse return;
    if (c.last) |s| try r.stream.wait(c.done[s]);
}

/// glue split_rows (x [n, 4 * 5120] bf16, pre [n, 4] f32, layer): queues the copies of this chunk's boundary rows
pub fn rows(f: *Forward, r: *run.Runner, c: *const calls.Call) !void {
    const cp = pf.state(f).split_copy orelse return;
    const xt = c.args[0].arg.t;
    const pt = c.args[1].arg.t;
    const n: usize = @intCast(xt.shape[0]);
    const xrow: usize = @intCast(xt.shape[1] * 2);
    const prow: usize = @intCast(pt.shape[1] * 4);
    if (xrow != split.stream_bytes or prow != split.pre_bytes or pt.shape[0] != xt.shape[0]) return error.BadGlue;
    if (xt.stride[0] != xt.shape[1] or pt.stride[0] != pt.shape[1]) return error.BadGlue;
    if (n > cp.rows_max) return error.BadGlue;
    const row0 = (f.slot.pending orelse return error.BadGlue).start;
    const slot = try cp.ring.acquire();
    // the pre is rewritten by the next chunk's first site: it leaves on the compute stream, behind this chunk's last use
    try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(cp.pre[slot].bytes.ptr, try r.tensorAddr(pt), n * prow, r.stream.handle), "cuMemcpyDtoHAsync");
    try cp.ready.record(r.stream);
    // the residual copies on its own stream while the next chunk computes; `hold` keeps that chunk off it until done
    try cp.stream.wait(cp.ready);
    try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(cp.x[slot].bytes.ptr, try r.tensorAddr(xt), n * xrow, cp.stream.handle), "cuMemcpyDtoHAsync");
    try cp.done[slot].record(cp.stream);
    cp.last = slot;
    try cp.ring.submit(slot, row0, n);
}
