//! A remote prefix one token short of the prompt: the pass is that one row; the landed prefix is the state kept.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const fixture = @import("lane_remote_test.zig");
const t = std.testing;

/// Copies of nothing: the planned marks are only read, never restored.
const Blank = struct {
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 64;
    }
    fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!pc.Saved {
        return try t.allocator.create(u8);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
    fn drop(_: *anyopaque, saved: pc.Saved) void {
        t.allocator.destroy(@as(*u8, @ptrCast(saved)));
    }
};

fn store() pc.Store {
    var none: u8 = 0;
    return pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Blank.bytes, .save = Blank.save, .restore = Blank.restore, .drop = Blank.drop } }, .{ .planned = true, .grid = 128, .tail = true, .tail_align = 128, .min_prompt = 0, .min_gap = 64 }, 1 << 20);
}

test "remote prefix of all but the last token: restored at len-1, kept as a remote state only with TF_PREFILL_KEEP, all reported pulled" {
    var prompt: [1000]u32 = undefined; // an odd length off every grid: the cut is where the producer's export stands
    for (&prompt, 0..) |*token, i| token.* = @intCast(i + 1);
    inline for (.{ true, false }) |keep| {
        var s = store();
        defer s.deinit();
        var peer: fixture.Peer = .{ .at = 999 };
        defer peer.kept.deinit(t.allocator);
        const seen = try fixture.runKeep(&peer, &.{ .prompt = &prompt, .max_tokens = 2 }, null, &s, .{ .keep = keep });
        try t.expect(seen.restoring and seen.remote);
        try t.expectEqual(@as(u32, 999), seen.at);
        try t.expectEqual(@as(usize, @intFromBool(keep)), seen.marks); // the landed prefix is the one mark
        try t.expectEqual(@as(u32, 0), seen.snap_at);
        try t.expectEqual(@as(?u32, 999), seen.remote_prefix);
        try t.expectEqual(@as(?u32, 999), seen.pulled_prefix);
        try t.expectEqual(@as(usize, 2), seen.tokens);
        try t.expectEqual(@as(u32, 1), peer.fills);
        try t.expectEqual(@as(u32, 1), peer.drops.load(.acquire));
        try t.expectEqual(@as(u64, @intFromBool(keep)), s.counts.kept);
        if (keep) try t.expect(s.entries.items[0].remote and s.entries.items[0].at == 999);
    }
}

test "remote prefix of all but the last token, kept: the same prompt again resumes it and asks the pair for nothing" {
    var prompt: [1000]u32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(i + 1);
    var s = store();
    defer s.deinit();
    var peer: fixture.Peer = .{ .at = 999 };
    defer peer.kept.deinit(t.allocator);
    _ = try fixture.runKeep(&peer, &.{ .prompt = &prompt, .max_tokens = 1 }, null, &s, .{ .keep = true });
    const again = try fixture.runKeep(&peer, &.{ .prompt = &prompt, .max_tokens = 2 }, null, &s, .{ .keep = true });
    try t.expect(again.restoring and again.remote); // the kept state: restored here, still reported remote
    try t.expectEqual(@as(u32, 999), again.at);
    try t.expectEqual(@as(u32, 1), peer.fills); // the cache's state beat a new fill
    try t.expectEqual(@as(?u32, 0), again.pulled_prefix); // nothing pulled this time
    try t.expectEqual(@as(u64, 1), s.counts.misses); // only the first run missed
}

test "remote prefix 64 rows short of the prompt, kept: the same prompt again resumes a kept state and asks the pair for nothing" {
    var prompt: [1000]u32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(i + 1);
    var s = store();
    defer s.deinit();
    var peer: fixture.Peer = .{ .at = 936 };
    defer peer.kept.deinit(t.allocator);
    const first = try fixture.runKeep(&peer, &.{ .prompt = &prompt, .max_tokens = 1 }, null, &s, .{ .keep = true });
    try t.expectEqual(@as(usize, 1), first.marks); // the pull's own end is the mark kept: nothing else lies past it
    try t.expectEqual(@as(u32, 936), s.entries.items[0].at);
    _ = try fixture.runKeep(&peer, &.{ .prompt = &prompt, .max_tokens = 2 }, null, &s, .{ .keep = true });
    try t.expectEqual(@as(u32, 1), peer.fills);
}

test "remote prefix 64 rows short, a family that keeps ends: the landed prefix is kept beside the end state and a repeat resumes it" {
    var prompt: [1000]u32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(i + 1);
    var none: u8 = 0;
    var s = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Blank.bytes, .save = Blank.save, .restore = Blank.restore, .drop = Blank.drop } }, .{ .planned = true, .grid = 128, .tail = true, .tail_align = 128, .min_prompt = 0, .min_gap = 64, .end_mark = true }, 1 << 20);
    defer s.deinit();
    var peer: fixture.Peer = .{ .at = 936 };
    defer peer.kept.deinit(t.allocator);
    _ = try fixture.runKeep(&peer, &.{ .prompt = &prompt, .max_tokens = 1 }, null, &s, .{ .keep = true });
    var landed = false;
    for (s.entries.items) |e| landed = landed or e.at == 936;
    try t.expect(landed);
    _ = try fixture.runKeep(&peer, &.{ .prompt = &prompt, .max_tokens = 2 }, null, &s, .{ .keep = true });
    try t.expectEqual(@as(u32, 1), peer.fills);
}
