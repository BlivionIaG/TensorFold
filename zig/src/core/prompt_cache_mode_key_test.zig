//! A family's mode key (`Rules.key_salt`): a state kept under one mode never resumes another.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Fake = @import("prompt_cache_test.zig").Fake;
const fresh = @import("prompt_cache_test.zig").fresh;
const Store = pc.Store;

test "prompt cache: a family's mode key keeps another mode's kept states from resuming" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer s.deinit();
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var p = try s.begin(a, &ids, 8, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqual(fresh(&ids), f.pass(&s, &ids, p));
    p = try s.begin(a, &ids, 8, &.{}, &.{}, null, &.{});
    const resumed = p.from;
    try std.testing.expect(resumed > 0);
    try std.testing.expectEqual(fresh(&ids), f.pass(&s, &ids, p));
    // Another mode's guidance feeds the same prompt's coda KV: the first mode's states are not its to resume.
    s.rules.key_salt = 0x5a17;
    p = try s.begin(a, &ids, 8, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqual(fresh(&ids), f.pass(&s, &ids, p));
    p = try s.begin(a, &ids, 8, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(resumed, p.from);
    try std.testing.expectEqual(fresh(&ids), f.pass(&s, &ids, p));
    // Each mode keeps resuming its own states.
    s.rules.key_salt = 0;
    p = try s.begin(a, &ids, 8, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(resumed, p.from);
    try std.testing.expectEqual(fresh(&ids), f.pass(&s, &ids, p));
}
