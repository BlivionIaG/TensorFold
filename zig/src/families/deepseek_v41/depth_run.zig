//! The acceptance-run depth controller (omlx patches/deepseek_v41/mtp.py AcceptanceDepthController): counts only.
const std = @import("std");

/// The next window's drafts from a round that verified `used` drafts and kept `kept` rows (the pending row included):
/// a rejection keeps one lookahead past the accepted prefix, a fully accepted window grows by one, at most `cap`.
pub fn runDepth(used: u32, kept: u32, cap: u32) u32 {
    const accepted = @min(kept -| 1, used);
    return @max(1, @min(cap, accepted + 1));
}

test "run depth: a rejection keeps one lookahead, a whole window grows by one, capped" {
    try std.testing.expectEqual(@as(u32, 1), runDepth(3, 1, 4)); // nothing accepted
    try std.testing.expectEqual(@as(u32, 3), runDepth(3, 3, 4)); // two drafts accepted
    try std.testing.expectEqual(@as(u32, 4), runDepth(3, 4, 4)); // the whole window landed: one deeper
    try std.testing.expectEqual(@as(u32, 4), runDepth(4, 5, 4)); // and stops at the cap
    try std.testing.expectEqual(@as(u32, 1), runDepth(0, 1, 4)); // a plain round
    try std.testing.expectEqual(@as(u32, 2), runDepth(1, 9, 4)); // kept never counts past what was verified
}
