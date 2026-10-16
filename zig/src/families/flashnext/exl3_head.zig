//! Host concat of the two-rank lm_head logit split.
//! Rank 0 owns the low vocab half and rank 1 the high half. The halves are copied,
//! not added. This is not the embedding table.

const std = @import("std");
const shape = @import("exl3_shape.zig");
const tp = @import("exl3_tp.zig");

/// Join rank 0's logits in front of rank 1's. A reduction would add the halves; this does not.
/// `error.Length` is returned before any store to `out`.
pub fn concat(rank0: []const f32, rank1: []const f32, out: []f32) error{Length}!void {
    const low = tp.range(.head, 0, 2) catch unreachable;
    const high = tp.range(.head, 1, 2) catch unreachable;
    const w0 = low.hi - low.lo;
    const w1 = high.hi - high.lo;
    if (rank0.len != w0 or rank1.len != w1 or out.len != shape.vocab or w0 + w1 != out.len) return error.Length;
    @memcpy(out[0..w0], rank0);
    @memcpy(out[w0..], rank1);
}

/// fp32 bytes of the concatenated lm_head logits for `tokens` steps, both ranks' halves together.
/// This is the lm_head D2H, not an MoE exchange and not 5120.
pub fn bounceBytes(tokens: usize) u64 {
    return @as(u64, tokens) * @as(u64, shape.vocab) * 4;
}

test "head ranges abut at half vocab and concat does not add" {
    const low = try tp.range(.head, 0, 2);
    const high = try tp.range(.head, 1, 2);
    try std.testing.expectEqual(@as(usize, 0), low.lo);
    try std.testing.expectEqual(@as(usize, 124160), low.hi);
    try std.testing.expectEqual(@as(usize, 124160), high.lo);
    try std.testing.expectEqual(@as(usize, 248320), high.hi);
    try std.testing.expectEqual(shape.vocab, high.hi);

    const w0 = low.hi - low.lo;
    const w1 = high.hi - high.lo;
    const gpa = std.testing.allocator;
    const rank0 = try gpa.alloc(f32, w0);
    defer gpa.free(rank0);
    const rank1 = try gpa.alloc(f32, w1);
    defer gpa.free(rank1);
    @memset(rank0, 1);
    @memset(rank1, 2);

    const short = try gpa.alloc(f32, w0);
    defer gpa.free(short);
    @memset(short, 9);
    try std.testing.expectError(error.Length, concat(rank0, rank1, short));
    try std.testing.expect(std.mem.allEqual(f32, short, 9));

    const out = try gpa.alloc(f32, shape.vocab);
    defer gpa.free(out);
    try concat(rank0, rank1, out);
    try std.testing.expectEqual(@as(f32, 1), out[0]);
    try std.testing.expectEqual(@as(f32, 1), out[low.hi - 1]);
    try std.testing.expectEqual(@as(f32, 2), out[high.lo]);
    try std.testing.expect(out[0] != 3);

    try std.testing.expectEqual(@as(u64, shape.vocab) * 4, bounceBytes(1));
    try std.testing.expectEqual(@as(u64, shape.vocab) * 8, bounceBytes(2));
}
