//! Replays the GLP steering calls (tf_dsv41_glp_v1, glp.cu): the fold's edit, the attention writer's, the final's.

const glp = @import("glp_ops.zig");
const replay = @import("kernels_replay.zig");

const Call = replay.Call;
const Tensor = replay.Tensor;
const Arg = replay.Arg;
const Kw = replay.Kw;
const numel = replay.numel;
const dim = replay.dim;
const is = replay.is;

fn tp(t: Tensor) u64 {
    return if (numel(t) == 0) 0 else t.ptr;
}

fn stride0(t: Tensor) i64 {
    return if (numel(t) == 0 or t.stride.len == 0) 0 else t.stride[0];
}

/// The kernels read fixed element types: a tensor of another type is refused, never reinterpreted.
fn typed(t: Tensor, want: replay.DType) !void {
    if (t.dtype != want) return error.WrongArgument;
}

/// stream (x, tap, dir, alpha, mode, R, width), writer (g, dir, alpha, mode, R, width),
/// final (x, pre, nw, out, eps, R, width).
pub fn glpExt(o: glp.Ops, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    if (is(func, "stream")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "tap", "dir", "alpha", "mode", "R", "width" } };
        const x = try c.tensor(0);
        const tap = try c.tensor(1);
        const dir = try c.tensor(2);
        try typed(x, .bf16);
        try typed(dir, .f32);
        if (numel(tap) != 0) try typed(tap, .bf16);
        try o.stream(.{ .x = x.ptr, .xs = x.stride[0], .tap = tp(tap), .ts = stride0(tap), .dir = dir.ptr, .alpha = @floatCast(try c.float(3)), .mode = @intCast(try c.int(4)), .R = @intCast(try c.int(5)), .width = @intCast(try c.int(6)) });
        return true;
    }
    if (is(func, "writer")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "g", "dir", "alpha", "mode", "R", "width" } };
        const g = try c.tensor(0);
        const dir = try c.tensor(1);
        try typed(g, .bf16);
        try typed(dir, .f32);
        if (g.shape.len != 3 or g.stride[2] != 1) return error.WrongArgument;
        try o.writer(.{ .g = g.ptr, .gr = g.stride[0], .gs = g.stride[1], .dir = dir.ptr, .alpha = @floatCast(try c.float(2)), .mode = @intCast(try c.int(3)), .R = @intCast(try c.int(4)), .width = @intCast(try c.int(5)), .world = @intCast(try dim(g, 0)) });
        return true;
    }
    if (is(func, "final")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "pre", "nw", "out", "eps", "R", "width" } };
        const x = try c.tensor(0);
        const pre = try c.tensor(1);
        const nw = try c.tensor(2);
        const out = try c.tensor(3);
        try typed(x, .bf16);
        try typed(pre, .f32);
        try typed(out, .bf16);
        if (nw.dtype != .f32 and nw.dtype != .bf16) return error.WrongArgument;
        try o.final(.{ .x = x.ptr, .xs = x.stride[0], .pre = pre.ptr, .nw = nw.ptr, .out = out.ptr, .eps = @floatCast(try c.float(4)), .nwbf16 = @intFromBool(nw.dtype == .bf16), .R = @intCast(try c.int(5)), .width = @intCast(try c.int(6)) });
        return true;
    }
    return false;
}
