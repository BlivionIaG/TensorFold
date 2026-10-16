//! Scalar CPU GEMV for one EXL3 matrix. No SIMD and no allocation.
//! The matrix is `exl3_format.dequantize`. y[j] = sum_i x[i] * W[i*N + j].

const std = @import("std");
const format = @import("exl3_format.zig");

/// fp32 SwiGLU from the v0.6.6 gate-up epilogue (`act_mode` 1).
/// `gate` is clamped to `limit` from above. `up` is clamped to `[-limit, limit]`.
/// `out[i] = silu(gate_i) * up_i`. A `limit` of +inf leaves both sides unclamped.
/// `error.Length` is returned before any store to `out`.
pub fn swiglu(gate: []const f64, up: []const f64, out: []f64, limit: f64) error{Length}!void {
    if (gate.len != up.len or gate.len != out.len) return error.Length;
    for (gate, up, out) |g, u, *y| {
        const gg = @min(g, limit);
        const uu = @min(@max(u, -limit), limit);
        y.* = (gg / (1.0 + @exp(-gg))) * uu;
    }
}

/// Weighted sum of `weights.len` expert rows. `rows` is row-major `[topk][n]`.
/// `out[j] = sum_s weights[s] * rows[s * n + j]`. Length is checked before any store.
pub fn combine(rows: []const f64, weights: []const f64, n: usize, out: []f64) error{Length}!void {
    if (n == 0 or out.len != n or weights.len == 0 or rows.len != weights.len * n) return error.Length;
    @memset(out, 0);
    for (weights, 0..) |w, s| {
        const row = rows[s * n ..][0..n];
        for (out, row) |*y, v| y.* += w * v;
    }
}

/// K = tiles_k * 16 and N = tiles_n * 16, both multiples of 128 (one Hadamard block).
/// `scratch` holds 3*K*N f64 values: decoded W in `[0, K*N)`, then `dequantize`'s 2*K*N workspace.
/// `error.Length` is returned before any store to `y`.
pub fn gemv(
    w: format.Width,
    codebook: format.Codebook,
    trellis: []const i16,
    tiles_k: usize,
    tiles_n: usize,
    suh: []const f16,
    svh: []const f16,
    x: []const f64,
    y: []f64,
    scratch: []f64,
) error{Length}!void {
    const k = tiles_k * 16;
    const n = tiles_n * 16;
    if (k % format.hadamard_dim != 0 or n % format.hadamard_dim != 0) return error.Length;
    if (x.len != k or y.len != n or suh.len != k or svh.len != n) return error.Length;
    const cells = k * n;
    if (scratch.len < 3 * cells) return error.Length;

    const weight = scratch[0..cells];
    format.dequantize(w, codebook, trellis, tiles_k, tiles_n, suh, svh, weight, scratch[cells .. 3 * cells]);

    for (0..n) |j| {
        var sum: f64 = 0;
        for (0..k) |i| sum += x[i] * weight[i * n + j];
        y[j] = sum;
    }
}

const sentinel: f64 = @bitCast(@as(u64, 0x7FF8_0000_0000_0001));

/// 128x128 mul1, width 2 halves. Trellis length is 8*8*Width.tileWords().
const Square = struct {
    allocator: std.mem.Allocator,
    w: format.Width,
    tiles_k: usize,
    tiles_n: usize,
    trellis: []i16,
    suh: []f16,
    svh: []f16,
    x: []f64,
    y: []f64,
    scratch: []f64,

    fn init(allocator: std.mem.Allocator) !Square {
        const w: format.Width = .{ .halves = 2 };
        const tiles_k: usize = 8;
        const tiles_n: usize = 8;
        const k = tiles_k * 16;
        const n = tiles_n * 16;
        const trellis = try allocator.alloc(i16, tiles_k * tiles_n * w.tileWords());
        errdefer allocator.free(trellis);
        for (trellis, 0..) |*word, i| word.* = @bitCast(@as(u16, @truncate(i *% 4099 + 7)));

        const suh = try allocator.alloc(f16, k);
        errdefer allocator.free(suh);
        for (suh) |*v| v.* = 1.0;
        const svh = try allocator.alloc(f16, n);
        errdefer allocator.free(svh);
        for (svh) |*v| v.* = 1.0;

        const x = try allocator.alloc(f64, k);
        errdefer allocator.free(x);
        for (x, 0..) |*v, i| v.* = @floatFromInt(i + 1);

        const y = try allocator.alloc(f64, n);
        errdefer allocator.free(y);
        for (y) |*v| v.* = sentinel;

        const scratch = try allocator.alloc(f64, 3 * k * n);
        errdefer allocator.free(scratch);
        return .{
            .allocator = allocator,
            .w = w,
            .tiles_k = tiles_k,
            .tiles_n = tiles_n,
            .trellis = trellis,
            .suh = suh,
            .svh = svh,
            .x = x,
            .y = y,
            .scratch = scratch,
        };
    }

    fn deinit(self: Square) void {
        self.allocator.free(self.trellis);
        self.allocator.free(self.suh);
        self.allocator.free(self.svh);
        self.allocator.free(self.x);
        self.allocator.free(self.y);
        self.allocator.free(self.scratch);
    }
};

test "128x128 mul1 gemv matches dequantize then a naive matvec" {
    const gpa = std.testing.allocator;
    const s = try Square.init(gpa);
    defer s.deinit();
    const k = s.tiles_k * 16;
    const n = s.tiles_n * 16;
    try std.testing.expectEqual(@as(usize, 128), k);
    try std.testing.expectEqual(@as(usize, 128), n);
    try std.testing.expectEqual(@as(usize, 8 * 8 * s.w.tileWords()), s.trellis.len);

    try gemv(s.w, .mul1, s.trellis, s.tiles_k, s.tiles_n, s.suh, s.svh, s.x, s.y, s.scratch);

    const weight = try gpa.alloc(f64, k * n);
    defer gpa.free(weight);
    const dq_scratch = try gpa.alloc(f64, 2 * k * n);
    defer gpa.free(dq_scratch);
    format.dequantize(s.w, .mul1, s.trellis, s.tiles_k, s.tiles_n, s.suh, s.svh, weight, dq_scratch);

    for (0..n) |j| {
        var sum: f64 = 0;
        for (0..k) |i| sum += s.x[i] * weight[i * n + j];
        try std.testing.expectEqual(@as(u64, @bitCast(sum)), @as(u64, @bitCast(s.y[j])));
    }
}

test "scratch one f64 short returns Length and leaves y untouched" {
    const gpa = std.testing.allocator;
    const s = try Square.init(gpa);
    defer s.deinit();
    for (s.y) |*v| v.* = sentinel;
    const short = s.scratch[0 .. s.scratch.len - 1];
    try std.testing.expectError(error.Length, gemv(s.w, .mul1, s.trellis, s.tiles_k, s.tiles_n, s.suh, s.svh, s.x, s.y, short));
    for (s.y) |v| try std.testing.expectEqual(@as(u64, @bitCast(sentinel)), @as(u64, @bitCast(v)));
}

test "swiglu is silu(gate) times up, and a finite limit clamps both sides" {
    var out = [_]f64{ 9, 9 };
    const gate = [_]f64{ 0, 2 };
    const up = [_]f64{ 3, 4 };
    try swiglu(&gate, &up, &out, std.math.inf(f64));
    try std.testing.expectEqual(@as(f64, 0), out[0]);
    const silu2 = 2.0 / (1.0 + @exp(-2.0));
    try std.testing.expectApproxEqAbs(silu2 * 4.0, out[1], 1e-12);

    const wide_gate = [_]f64{2};
    const wide_up = [_]f64{4};
    var clamped = [_]f64{9};
    try swiglu(&wide_gate, &wide_up, &clamped, 1);
    const silu1 = 1.0 / (1.0 + @exp(-1.0));
    try std.testing.expectApproxEqAbs(silu1, clamped[0], 1e-12);

    var held = [_]f64{9};
    try std.testing.expectError(error.Length, swiglu(wide_gate[0..], wide_up[0..], held[0..0], 1));
    try std.testing.expectEqual(@as(f64, 9), held[0]);
}

test "combine is the weighted sum of expert rows" {
    const rows = [_]f64{ 1, 2, 3, 4 };
    const weights = [_]f64{ 0.25, 0.75 };
    var out = [_]f64{ 9, 9 };
    try combine(&rows, &weights, 2, &out);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), out[1], 1e-12);
    try std.testing.expectError(error.Length, combine(rows[0..3], &weights, 2, &out));
}

test "x shorter than K returns Length" {
    const gpa = std.testing.allocator;
    const s = try Square.init(gpa);
    defer s.deinit();
    for (s.y) |*v| v.* = sentinel;
    const short_x = s.x[0 .. s.x.len - 1];
    try std.testing.expectError(error.Length, gemv(s.w, .mul1, s.trellis, s.tiles_k, s.tiles_n, s.suh, s.svh, short_x, s.y, s.scratch));
    for (s.y) |v| try std.testing.expectEqual(@as(u64, @bitCast(sentinel)), @as(u64, @bitCast(v)));
}
