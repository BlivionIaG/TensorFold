//! Host greedy pick after the lm_head concat.
//! Ties keep the lower index, the same `>` comparison as `exl3_route.select`.
//! NaN and infinity are not candidates. The logit buffer is not written.

const std = @import("std");
const shape = @import("exl3_shape.zig");

/// Index of the largest finite logit.
/// `logits.len` must be `shape.vocab`. A wrong length, or a row with no finite
/// value, is `error.Length` and is not an index.
pub fn sample(logits: []const f32) error{Length}!u32 {
    if (logits.len != shape.vocab) return error.Length;
    var best_i: u32 = 0;
    var best: f32 = 0;
    var have = false;
    for (logits, 0..) |v, i| {
        if (!std.math.isFinite(v)) continue;
        // Strictly greater so an equal logit keeps the earlier index.
        if (!have or v > best) {
            best = v;
            best_i = @intCast(i);
            have = true;
        }
    }
    if (!have) return error.Length;
    return best_i;
}

test "a one at rank 1's first index is that index" {
    const tp = @import("exl3_tp.zig");
    const high = try tp.range(.head, 1, 2);
    try std.testing.expectEqual(@as(usize, 124160), high.lo);

    const gpa = std.testing.allocator;
    const logits = try gpa.alloc(f32, shape.vocab);
    defer gpa.free(logits);
    @memset(logits, 0);
    logits[124160] = 1;
    try std.testing.expectEqual(@as(u32, 124160), try sample(logits));
    try std.testing.expectEqual(@as(f32, 1), logits[124160]);
}

test "equal peaks keep the lower index" {
    const gpa = std.testing.allocator;
    const logits = try gpa.alloc(f32, shape.vocab);
    defer gpa.free(logits);
    @memset(logits, 0);
    logits[3] = 4;
    logits[9] = 4;
    try std.testing.expectEqual(@as(u32, 3), try sample(logits));
    try std.testing.expectEqual(@as(f32, 4), logits[3]);
    try std.testing.expectEqual(@as(f32, 4), logits[9]);
}

test "a short buffer is refused" {
    const logits = [_]f32{ 1, 4, 0 };
    try std.testing.expectError(error.Length, sample(&logits));
}

test "non-finite logits are not a pick" {
    const gpa = std.testing.allocator;
    const logits = try gpa.alloc(f32, shape.vocab);
    defer gpa.free(logits);
    @memset(logits, std.math.nan(f32));
    try std.testing.expectError(error.Length, sample(logits));

    logits[0] = std.math.inf(f32);
    logits[3] = std.math.nan(f32);
    logits[9] = 4;
    try std.testing.expectEqual(@as(u32, 9), try sample(logits));
}
