//! Host side of the unquantized hc_lowrank split.
//! Each rank already owns 160 of the 320-wide axis and produces a hidden partial.
//! Those partials meet here, rank 0 then rank 1. No CUDA.

const std = @import("std");
const shape = @import("exl3_shape.zig");
const tp = @import("exl3_tp.zig");

/// One rank's dense product. `w` is row-major `[hc_lowrank/2][hidden]`: column `j` is strided by `hidden`,
/// matching `exl3_cpu.gemv`, so `out[j]` is the dot of `x` with that column.
/// A length other than those three sizes returns `error.Length` before any store. Does not allocate.
pub fn partial(w: []const f32, x: []const f32, out: []f32) error{Length}!void {
    const part = shape.hc_lowrank / 2;
    const n = shape.hidden;
    if (w.len != part * n or x.len != part or out.len != n) return error.Length;
    for (0..n) |j| {
        var sum: f32 = 0;
        for (0..part) |i| sum += x[i] * w[i * n + j];
        out[j] = sum;
    }
}

/// Hidden-sized sum of the two partials. Rank 0, then rank 1. Not inside `partial`.
/// A length other than `hidden` returns `error.Length` before any store.
pub fn add(partial0: []const f32, partial1: []const f32, out: []f32) error{Length}!void {
    const n = shape.hidden;
    if (partial0.len != n or partial1.len != n or out.len != n) return error.Length;
    for (partial0, partial1, out) |a, b, *y| y.* = a + b;
}

/// `exl3_bounce_bytes` for hc_lowrank only.
/// D2H of this partial (2560 values, 5120 bytes) plus H2D of the summed hidden, one layer.
/// Not multiplied by 48. MoE and GDN are not included.
pub fn bounceBytes(tokens: usize) u64 {
    return @as(u64, @intCast(tokens)) * 5120 * 2;
}

test "one-hot x picks the row of the hc_lowrank part" {
    const cut0 = try tp.cut(.hc_lowrank, 0, 2);
    try std.testing.expectEqual(@as(usize, 160), cut0.part);
    const r0 = try tp.range(.hc_lowrank, 0, 2);
    const r1 = try tp.range(.hc_lowrank, 1, 2);
    try std.testing.expectEqual(@as(usize, 0), r0.lo);
    try std.testing.expectEqual(@as(usize, 160), r0.hi);
    try std.testing.expectEqual(@as(usize, 160), r1.lo);
    try std.testing.expectEqual(@as(usize, 320), r1.hi);
    try std.testing.expectEqual(cut0.part, r0.hi - r0.lo);
    try std.testing.expectEqual(cut0.part, r1.hi - r1.lo);

    const part = cut0.part;
    const n = shape.hidden;
    try std.testing.expectEqual(@as(usize, 160 * 2560), part * n);

    const gpa = std.testing.allocator;
    const w = try gpa.alloc(f32, part * n);
    defer gpa.free(w);
    @memset(w, 0);

    var x: [shape.hc_lowrank / 2]f32 = @splat(0);
    try std.testing.expectEqual(part, x.len);

    var rank0: [shape.hidden]f32 = @splat(42);
    const kept0 = rank0;
    try std.testing.expectError(error.Length, partial(w[0..part], &x, &rank0));
    try std.testing.expectError(error.Length, partial(w, x[0 .. part - 1], &rank0));
    try std.testing.expectError(error.Length, partial(w, &x, rank0[0 .. n - 1]));
    try std.testing.expectEqualSlices(f32, &kept0, &rank0);

    // Rank 0 owns global [0, 160). A one-hot at local 0 is that first row, not the poisoned neighbor.
    const local0: usize = 0;
    try std.testing.expectEqual(r0.lo + local0, local0);
    for (0..n) |j| w[local0 * n + j] = @floatFromInt(j + 3);
    for (0..n) |j| w[1 * n + j] = -1000;
    x[local0] = 1;
    try partial(w, &x, &rank0);
    try std.testing.expectEqualSlices(f32, w[local0 * n .. local0 * n + n], &rank0);

    // Rank 1 owns global [160, 320). A one-hot at the last local row is global 319.
    var rank1: [shape.hidden]f32 = @splat(-7);
    @memset(w, 0);
    const local1 = part - 1;
    try std.testing.expectEqual(@as(usize, 319), r1.lo + local1);
    try std.testing.expect(r1.lo + local1 < r1.hi);
    for (0..n) |j| w[0 * n + j] = 80;
    for (0..n) |j| w[local1 * n + j] = -@as(f32, @floatFromInt(j)) - 1;
    @memset(&x, 0);
    x[local1] = 1;
    try partial(w, &x, &rank1);
    try std.testing.expectEqualSlices(f32, w[local1 * n .. local1 * n + n], &rank1);

    // Two nonzeros so the product is a dot, not a copy of the hot row.
    @memset(w, 0);
    @memset(&x, 0);
    const row_a: usize = 2;
    const row_b: usize = 5;
    for (0..n) |j| w[row_a * n + j] = 1;
    for (0..n) |j| w[row_b * n + j] = 4;
    x[row_a] = 2;
    x[row_b] = 0.5;
    var dotted: [shape.hidden]f32 = @splat(0);
    try partial(w, &x, &dotted);
    const fours: [shape.hidden]f32 = @splat(4);
    try std.testing.expectEqualSlices(f32, &fours, &dotted);

    var out: [shape.hidden]f32 = @splat(7);
    const kept = out;
    try std.testing.expectError(error.Length, add(rank0[0 .. n - 1], &rank1, &out));
    try std.testing.expectError(error.Length, add(&rank0, rank1[0..0], &out));
    try std.testing.expectError(error.Length, add(&rank0, &rank1, out[0 .. n - 1]));
    try std.testing.expectEqualSlices(f32, &kept, &out);

    // (j + 3) + (-j - 1) = 2. Rank 0, then rank 1.
    try add(&rank0, &rank1, &out);
    const twos: [shape.hidden]f32 = @splat(2);
    try std.testing.expectEqualSlices(f32, &twos, &out);
}

test "bounceBytes is the hc_lowrank exchange for one layer" {
    try std.testing.expectEqual(@as(u64, 10240), bounceBytes(1));
    try std.testing.expectEqual(@as(u64, 0), bounceBytes(0));
    try std.testing.expectEqual(@as(u64, 10240) * 3, bounceBytes(3));
}
