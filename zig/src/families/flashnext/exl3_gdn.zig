//! Host reduction for one GDN out-projection.
//! `pinRank` walks pinned bytes. A K-split is that walker's `tp.legal` test on
//! the out_proj axis the caller passes: more than one rank, a non-zero axis,
//! and a Hadamard cut. `exl3_tp.cut` has no GDN role.
//! No kernel, conv1d, or recurrent scan.

const std = @import("std");
const tp = @import("exl3_tp.zig");

/// One token of bf16 hidden, 2560 * 2. Both directions of the exchange use this width.
const hidden_bytes: u64 = 5120;

/// True when this out_proj was K-split, so the hidden partials have to be reduced.
/// `axis` is the header axis `pinRank` measures for that tensor (the K side).
/// Both ranks get the same answer: a partial each, or a full replica each.
pub fn outProjSplit(axis: usize, ranks: usize) bool {
    return ranks > 1 and axis != 0 and tp.legal(axis, ranks);
}

/// Copy `partial0` when the out_proj was replicated. Sum rank 0, then rank 1, when it was K-split.
/// Lengths are checked before any store. The sum stays fp32; there is no bf16 round.
pub fn reduceOut(partial0: []const f32, partial1: []const f32, split: bool, out: []f32) error{Length}!void {
    if (partial0.len != partial1.len or partial0.len != out.len) return error.Length;
    if (!split) {
        @memcpy(out, partial0);
        return;
    }
    for (partial0, partial1, out) |a, b, *y| {
        y.* = a + b;
    }
}

/// `exl3_bounce_bytes` for the GDN out-projection only.
/// A replica exchanges nothing. A K-split adds the D2H of this partial and the
/// H2D of the summed hidden, one layer (`tokens * 5120 * 2`). The caller counts
/// layers. The MoE hop is not included.
pub fn bounceBytes(split: bool, tokens: usize) u64 {
    if (!split) return 0;
    const n: u64 = @intCast(tokens);
    return n * hidden_bytes * 2;
}

test "a replicated out_proj copies partial0 and bounces nothing" {
    const partial0 = [_]f32{ 1, 2, 3, 4 };
    const partial1: [4]f32 = @splat(7);
    var out: [4]f32 = @splat(0);
    try reduceOut(&partial0, &partial1, false, &out);
    try std.testing.expectEqualSlices(f32, &partial0, &out);
    try std.testing.expectEqual(@as(u64, 0), bounceBytes(false, 0));
    try std.testing.expectEqual(@as(u64, 0), bounceBytes(false, 1));
    try std.testing.expectEqual(@as(u64, 0), bounceBytes(false, 8192));
}

test "a K-split out_proj sums rank 0 then rank 1 and counts the exchange" {
    const partial0 = [_]f32{ 1, -2, 0.5, 8 };
    const partial1 = [_]f32{ 3, 4, 0.25, -1 };
    var out: [4]f32 = @splat(0);
    try reduceOut(&partial0, &partial1, true, &out);
    for (partial0, partial1, out) |a, b, y| {
        try std.testing.expectEqual(a + b, y);
    }
    try std.testing.expectEqual(@as(u64, 0), bounceBytes(true, 0));
    try std.testing.expectEqual(@as(u64, 10240), bounceBytes(true, 1));
    try std.testing.expectEqual(@as(u64, 8192) * 10240, bounceBytes(true, 8192));
}

test "outProjSplit follows legal on the out_proj header axis" {
    const ranks: usize = 2;
    var bad: usize = 0;
    var good: usize = 0;
    var axis: usize = 1;
    while (axis < 8192 and (bad == 0 or good == 0)) : (axis += 1) {
        if (tp.legal(axis, ranks)) {
            if (good == 0) good = axis;
        } else if (bad == 0) {
            bad = axis;
        }
    }
    try std.testing.expect(bad != 0 and good != 0);
    try std.testing.expect(!tp.legal(bad, ranks));
    try std.testing.expect(tp.legal(good, ranks));
    try std.testing.expect(!outProjSplit(bad, ranks));
    try std.testing.expect(outProjSplit(good, ranks));
    // One rank keeps the whole span. A zero axis is not a measured K, so pinRank replicates it.
    try std.testing.expect(tp.legal(good, 1));
    try std.testing.expect(!outProjSplit(good, 1));
    try std.testing.expect(!outProjSplit(0, ranks));
}

test "a length mismatch writes nothing" {
    var out = [_]f32{ 4, 5, 6 };
    const same = [_]f32{ 1, 2, 3 };
    const short = [_]f32{ 9, 9 };
    const before = out;
    try std.testing.expectError(error.Length, reduceOut(&short, &same, true, &out));
    try std.testing.expectEqualSlices(f32, &before, &out);
    try std.testing.expectError(error.Length, reduceOut(&same, &short, false, &out));
    try std.testing.expectEqualSlices(f32, &before, &out);
    try std.testing.expectError(error.Length, reduceOut(&same, &same, true, out[0..2]));
    try std.testing.expectEqualSlices(f32, &before, &out);
}
