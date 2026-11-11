//! The round planner's fixed decode slice per prompt piece (TF_DSV41_PREFILL_INTERLEAVE_MS).

const std = @import("std");
const Fairness = @import("sched.zig").Fairness;

test "a fixed decode slice per piece owes the same decode time whatever the piece took" {
    var f: Fairness = .{ .share = 0.5, .slice_s = 0.25 };
    f.after(1.0, 0.0, true); // a share of 0.5 would owe a second here
    try std.testing.expect(!f.allow(true));
    f.after(0, 0.125, true);
    try std.testing.expect(!f.allow(true));
    f.after(0, 0.125, true);
    try std.testing.expect(f.allow(true));
    f.after(0.1, 0.0, true); // a short piece owes the same slice
    try std.testing.expectEqual(@as(f64, 0.25), f.debt);
    f.after(0, 0.0625, false); // nothing prefilling: the debt is gone
    try std.testing.expect(f.allow(true));
}
