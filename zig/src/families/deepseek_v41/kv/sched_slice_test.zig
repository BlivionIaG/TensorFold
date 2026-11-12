//! The round planner's fixed decode slice per prompt piece (TF_DSV41_PREFILL_INTERLEAVE_MS).

const std = @import("std");
const sched = @import("sched.zig");
const Fairness = sched.Fairness;

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

fn firstPiece(decode_rows: u64, decoding: bool) !u64 {
    var p = sched.Planner.init(std.testing.allocator, .{ .rows = 4096, .decode_rows = decode_rows });
    defer p.deinit();
    try p.seqs.append(p.gpa, .{ .key = 1, .n = 20000, .submitted = 0, .done = 0, .save_at = null });
    try p.seqs.append(p.gpa, .{ .key = 2, .n = 100, .submitted = 0, .done = 99, .save_at = null, .decoding = decoding });
    var out: std.ArrayList(sched.Piece) = .empty;
    defer out.deinit(p.gpa);
    var finals: std.ArrayList(usize) = .empty;
    defer finals.deinit(p.gpa);
    try p.plan(null, &out, &finals);
    try std.testing.expectEqual(@as(usize, 1), out.items.len);
    return out.items[0].end - out.items[0].start;
}

test "a prompt piece is cut to decode_rows only while a stream decodes" {
    try std.testing.expectEqual(@as(u64, 2048), try firstPiece(2048, true));
    try std.testing.expectEqual(@as(u64, 1024), try firstPiece(1024, true));
    try std.testing.expectEqual(@as(u64, 4096), try firstPiece(2048, false)); // nobody waits: the full piece
    try std.testing.expectEqual(@as(u64, 4096), try firstPiece(0, true)); // off: `rows`
    try std.testing.expectEqual(@as(u64, 4096), try firstPiece(8192, true)); // never above `rows`
}
