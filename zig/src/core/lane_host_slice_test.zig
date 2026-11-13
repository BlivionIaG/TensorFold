//! Prompt share and time-slice arithmetic (lane_host_fill.zig, lane_host_slice.zig).
const std = @import("std");
const fill = @import("lane_host_fill.zig");
const Share = fill.Share;
const Slice = @import("lane_host_slice.zig").Slice;

test "a share of 0.5 owes the decoders a step's time; 1 owes nothing; rounds pay it down" {
    var s: Share = .{ .p = 0.5 };
    s.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 1000), s.debt_ns);
    s.stepped(1000, false);
    try std.testing.expectEqual(@as(i64, 1000), s.debt_ns);
    s.rounded(400);
    s.rounded(800);
    try std.testing.expectEqual(@as(i64, 0), s.debt_ns);
    var one: Share = .{};
    one.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 0), one.debt_ns);
    var q: Share = .{ .p = 0.25 };
    q.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 3000), q.debt_ns);
}

test "a share below the floor, NaN and inf fall back to 1; the floor and 1 stand" {
    for ([_][]const u8{ "1e-12", "0.0099", "0", "-0.5", "nan", "inf", "-inf", "2", "x", "" }) |text| {
        try std.testing.expectEqual(@as(f64, 1), Share.fromText(text).p);
    }
    try std.testing.expectEqual(@as(f64, 0.01), Share.fromText("0.01").p);
    try std.testing.expectEqual(@as(f64, 1), Share.fromText("1").p);
}

test "prefill service: TENSORFOLD_PREFILL_BUDGET is whole rows; unset, 0 or anything else is none, main's order" {
    for ([_][]const u8{ "", "0", "x", "-512", "+512", "5e2", "4294967296" }) |text| {
        try std.testing.expectEqual(@as(u64, 0), Share.budget(text));
    }
    try std.testing.expectEqual(@as(u64, 512), Share.budget("512"));
    try std.testing.expectEqual(@as(u64, 1), Share.budget(" 1\t"));
    try std.testing.expectEqual(@as(u64, 0), (Share{}).rows);
}

test "a step's debt saturates instead of overflowing" {
    var s: Share = .{ .p = Share.min_share };
    s.stepped(std.math.maxInt(i64), true);
    try std.testing.expect(s.debt_ns > 0);
    s.stepped(std.math.maxInt(i64), true);
    try std.testing.expectEqual(@as(i64, std.math.maxInt(i64)), s.debt_ns);
    s.rounded(1000);
    try std.testing.expect(s.debt_ns < std.math.maxInt(i64));
}

test "a slice of 250 ms at 0.45 ms a row is 512 rows; the row time follows the pieces; off is zero" {
    var s = Slice.of("250");
    try std.testing.expectEqual(@as(u32, 512), s.rows());
    s.sampled(512, 512 * 900_000);
    try std.testing.expectEqual(@as(u32, Slice.min_rows), s.rows());
    var big = Slice.of("2000");
    try std.testing.expect(big.rows() > 3000 and big.rows() % 64 == 0);
    for (0..5) |_| big.sampled(1024, 1024 * 900_000);
    try std.testing.expect(big.rows() < 3000 and big.rows() >= Slice.min_rows);
    s.sampled(10, 1);
    try std.testing.expect(!Slice.of("x").on() and !Slice.of("-5").on() and !Slice.of("0").on());
    try std.testing.expectEqual(@as(u32, Slice.min_rows), Slice.of("0.001").rows());
}
