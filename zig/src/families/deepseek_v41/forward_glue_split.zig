//! The layer-split producer's glue: a chunk's boundary rows (the cut layer's residual and the next site's pre) to the host.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const run = @import("run.zig");
const fwd = @import("forward.zig");
const pf = @import("forward_prefill.zig");
const split = @import("handoff/split.zig");
const Forward = fwd.Forward;

/// The most rows a host copy takes at once: its scratch is `block_rows` rows of the stream (40 KB each).
const block_rows: usize = 256;

/// glue split_rows (x [n, 4 * 5120] bf16, pre [n, 4] f32, layer): rank 0 writes row i of the chunk as the stream's
/// 40,960 bytes then the pre's 16 into the stage's rows (`split.row_bytes` each); the other ranks hold the same
/// values and write nothing. The stream is synchronised first and the copy is synchronous: the rows are in the stage
/// when this step returns, and the chunk's `publish` follows the program.
pub fn rows(f: *Forward, r: *run.Runner, c: *const calls.Call) !void {
    if (f.comm.rank() != 0) return;
    const dst = pf.state(f).split_dst;
    const xt = c.args[0].arg.t;
    const pt = c.args[1].arg.t;
    const n: usize = @intCast(xt.shape[0]);
    const xrow: usize = @intCast(xt.shape[1] * 2);
    const prow: usize = @intCast(pt.shape[1] * 4);
    if (xrow != split.stream_bytes or prow != split.pre_bytes or pt.shape[0] != xt.shape[0]) return error.BadGlue;
    if (xt.stride[0] != xt.shape[1] or pt.stride[0] != pt.shape[1]) return error.BadGlue;
    if (dst.len != n * split.row_bytes) return error.BadGlue;
    try r.stream.synchronize();
    const stride: usize = @intCast(split.row_bytes);
    const scratch = try f.gpa.alloc(u8, @min(n, block_rows) * xrow);
    defer f.gpa.free(scratch);
    const x_addr = try r.tensorAddr(xt);
    var at: usize = 0;
    while (at < n) {
        const m = @min(block_rows, n - at);
        const src: cuda.DeviceBuffer = .{ .d = r.d, .ptr = x_addr + at * xrow, .len = m * xrow };
        try src.download(0, scratch[0 .. m * xrow]);
        for (0..m) |i| @memcpy(dst[(at + i) * stride ..][0..xrow], scratch[i * xrow ..][0..xrow]);
        at += m;
    }
    // the pre's rows are 16 bytes: all of them at once, then each behind its row's stream
    const pre = try f.gpa.alloc(u8, n * prow);
    defer f.gpa.free(pre);
    const p_src: cuda.DeviceBuffer = .{ .d = r.d, .ptr = try r.tensorAddr(pt), .len = n * prow };
    try p_src.download(0, pre);
    for (0..n) |i| @memcpy(dst[i * stride + xrow ..][0..prow], pre[i * prow ..][0..prow]);
}
