//! Host add of one partial hidden. Not a GDN, attention, or router program.
//! No CUDA and no NCCL.
//! Rank order is rank 0, then rank 1, the same order `exl3_tp.zig` states.
//! Rank 0's slice is that rank's routed partial plus its shared output.
//! Rank 1's slice is routed only. Its shared output is not an argument.
//! The sum is fp32. `out` stays f32, like `exl3_route`, after one bf16 round
//! of that sum. Neighbors do not store a u16 hidden.
//! This pack's hidden width is `shape.hidden` (2560). Callers pass slices.

const std = @import("std");
const shape = @import("exl3_shape.zig");

/// `out[i] = bf16(rank0[i] + rank1_routed[i])`.
/// `rank0` is already routed plus shared. Adding rank 1's shared output
/// would count the replicated shared expert twice.
/// `error.Length` is returned before any store to `out`.
pub fn addHidden(rank0: []const f32, rank1_routed: []const f32, out: []f32) error{Length}!void {
    if (rank0.len != rank1_routed.len or rank0.len != out.len) return error.Length;
    for (rank0, rank1_routed, out) |a, b, *y| {
        y.* = bf16(a + b);
    }
}

/// One round, fp32 through bf16 and back. NaN stays a NaN.
/// Same tie-to-even as `exl3_route`. That helper is private, and `zig test`
/// in this directory cannot import `../`.
fn bf16(v: f32) f32 {
    const u: u32 = @bitCast(v);
    const h: u16 = if (std.math.isNan(v))
        @truncate((u >> 16) | 0x40)
    else
        @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
    return @bitCast(@as(u32, h) << 16);
}

fn checksum(v: []const f32) u32 {
    var s: u32 = 0;
    for (v) |x| s ^= @as(u32, @bitCast(x));
    return s;
}

test "a length mismatch stores nothing" {
    const rank0 = [_]f32{ 1, 2, 3 };
    const rank1 = [_]f32{ 4, 5 };
    var out = [_]f32{ 9, 9, 9 };
    try std.testing.expectError(error.Length, addHidden(&rank0, &rank1, &out));
    try std.testing.expectEqual(@as(f32, 9), out[0]);
    try std.testing.expectEqual(@as(f32, 9), out[2]);
    try std.testing.expectError(error.Length, addHidden(rank0[0..2], &rank1, out[0..3]));
    try std.testing.expectEqual(@as(f32, 9), out[0]);
}

test "the fp32 sum rounds to bf16 once" {
    // 4097 * 2^-20 is eight f32 ulps (2^-20, not 2^-23) above the bf16
    // midpoint between 1 and 1 + 2^-7. One round of the sum steps up.
    // Rounding each partial first lands on that midpoint and ties back to 1.
    const rank0 = [_]f32{1};
    const rank1 = [_]f32{@as(f32, 4097) * 0x1p-20};
    var out = [_]f32{0};
    try addHidden(&rank0, &rank1, &out);
    try std.testing.expectEqual(@as(f32, 1.0078125), out[0]);
    try std.testing.expect(out[0] != bf16(bf16(rank0[0]) + bf16(rank1[0])));
}

test "rank 1's shared vector does not change the sum" {
    const n = shape.hidden;
    try std.testing.expectEqual(@as(usize, 2560), n);

    var routed0: [n]f32 = undefined;
    var shared0: [n]f32 = undefined;
    var routed1: [n]f32 = undefined;
    var shared1_a: [n]f32 = undefined;
    var shared1_b: [n]f32 = undefined;
    var rank0: [n]f32 = undefined;
    var out: [n]f32 = undefined;
    var again: [n]f32 = undefined;
    var with_a: [n]f32 = undefined;
    var with_b: [n]f32 = undefined;

    for (0..n) |i| {
        const t: f32 = @floatFromInt(i);
        routed0[i] = t * 0.015625;
        shared0[i] = 0.25;
        routed1[i] = 0.5;
        // Two nonzero shared outputs. Neither is passed to addHidden.
        shared1_a[i] = 1 + t * 0.015625;
        shared1_b[i] = -2 - t * 0.03125;
        // Rank 0's partial is routed plus shared, then rank 1's routed partial.
        rank0[i] = routed0[i] + shared0[i];
    }
    rank0[0] = 1;
    routed1[0] = @as(f32, 4097) * 0x1p-20;

    try addHidden(&rank0, &routed1, &out);
    try addHidden(&rank0, &routed1, &again);

    for (0..n) |i| {
        try std.testing.expectEqual(bf16(rank0[i] + routed1[i]), out[i]);
        try std.testing.expectEqual(out[i], again[i]);
        with_a[i] = bf16(rank0[i] + routed1[i] + shared1_a[i]);
        with_b[i] = bf16(rank0[i] + routed1[i] + shared1_b[i]);
    }

    const clean = checksum(&out);
    const dirty_a = checksum(&with_a);
    const dirty_b = checksum(&with_b);
    // Folding in rank 1's shared output moves the checksum, and a different
    // shared vector moves it again. addHidden must match neither.
    try std.testing.expect(shared1_a[0] != 0 and shared1_b[0] != 0);
    try std.testing.expect(clean != dirty_a);
    try std.testing.expect(clean != dirty_b);
    try std.testing.expect(dirty_a != dirty_b);
    try std.testing.expectEqual(@as(f32, 1.0078125), out[0]);
}
