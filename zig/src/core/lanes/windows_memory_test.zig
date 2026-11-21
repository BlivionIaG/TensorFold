//! Probability planning must fit its final output into a bounded round arena.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const win = @import("windows.zig");

pub fn check(comptime node: anytype, comptime hybrid: anytype) !void {
    const a = std.testing.allocator;
    var cfg = try @import("config.zig").Config.init(a, .{ .exact_width = 16, .drafts = 3, .mtp = true, .speculate = true, .draft_probabilities = true, .draft_prior = &.{ 0.8, 0.5 } }, 16, 3);
    defer cfg.deinit(a);
    var target: @import("fake.zig").Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: @import("fake.zig").FixedClock = .{};
    var backend = target.backend();
    var vtable = backend.vtable.*;
    vtable.probabilities = struct {
        fn decline(_: *anyopaque, _: *Stream, out: []f64) !bool {
            @memset(out, 99);
            return false;
        }
    }.decline;
    backend.vtable = &vtable;
    var limited = std.testing.FailingAllocator.init(a, .{});
    var e = Engine.init(a, &cfg, backend, clock.clock());
    defer e.deinit();
    var s = try Stream.init(a, .{ .id = "bounded", .prompt = &.{ 1, 2 }, .max_new = 64 });
    defer s.deinit(a);
    _ = try e.rule.rates(win.who(&s));
    // The round gets only enough arena space for its final probability buffer.
    e.arena.deinit();
    e.arena = std.heap.ArenaAllocator.init(limited.allocator());
    _ = try e.arena.allocator().alloc(f64, 32);
    const capacity = e.arena.queryCapacity();
    limited.fail_index = limited.alloc_index;
    limited.resize_fail_index = limited.resize_index;
    for (0..2) |mode| {
        try std.testing.expect(e.arena.reset(.retain_capacity));
        _ = try e.arena.allocator().alloc(u8, capacity - 16 * @sizeOf(f64));
        var parents: [16]i32 = undefined;
        for (&parents, 0..) |*q, i| q.* = @as(i32, @intCast(i)) - 1;
        const p: win.Plan = .{ .stream = &s, .position = 2, .kind = .head, .tokens = &.{}, .parents = &parents, .branch_from = 4, .context_copy = true, .copy_trusted = false };
        const out = if (mode == 0) try node(&e, &s, 16) else try hybrid(&e, p, 16);
        // the chain's chances fall with depth; a copy row's is its parent's times the copy rate
        try std.testing.expect(out[0] > out[1] and out[1] > out[2] and out[2] >= out[3] and out[3] > 0);
    }
    try std.testing.expect(!limited.has_induced_failure);
}
