//! A drafter that asks for the shared-round guard (Facts.shared_guard): which shared rounds draft, replies unchanged.
const std = @import("std");
const Config = @import("config.zig").Config;
const Model = @import("config.zig").Model;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const Drafted = @import("drafted.zig").Drafted;
const dt = @import("drafted_test.zig");

const gpa = std.testing.allocator;

test "a stream joining shared rounds that the guard runs plain holds no first drafts; a lone one does" {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var head: dt.FakeDrafter = .{ .facts_ = .{ .depth = 4, .step_ms = 0.5, .batched = true, .shared_guard = true } };
    defer head.deinit();
    var wrapped = try Drafted.init(gpa, target.backend(), head.drafter());
    defer wrapped.deinit();
    const m: Model = .{ .exact_width = 16, .gpu_tokens = true, .hidden_rows = true, .batch_rows = 32, .max_streams = 4 };
    var cfg = try Config.init(gpa, wrapped.facts(m), 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    var streams: [2]sm.Stream = undefined;
    for (&streams, 0..) |*s, i| s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = dt.long[0 .. 12 + i], .max_new = 24, .eos = &.{96} });
    defer for (&streams) |*s| s.deinit(gpa);
    try engine.addStream(&streams[0]);
    try std.testing.expectEqual(@as(u32, 1), head.holds); // alone: its first drafts are held
    for (0..3) |_| try engine.step();
    const before = head.holds;
    try std.testing.expect(!try engine.guard.drafts(2)); // the guard measures plain shared rounds first
    try engine.addStream(&streams[1]);
    try std.testing.expectEqual(before, head.holds); // joining: none held
    while (engine.activeCount() > 0) try engine.step();
}

test "without the guard's flag a stream joining shared rounds holds its first drafts as before" {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var head: dt.FakeDrafter = .{ .facts_ = .{ .depth = 4, .step_ms = 0.5, .batched = true } };
    defer head.deinit();
    var wrapped = try Drafted.init(gpa, target.backend(), head.drafter());
    defer wrapped.deinit();
    const m: Model = .{ .exact_width = 16, .gpu_tokens = true, .hidden_rows = true, .batch_rows = 32, .max_streams = 4 };
    var cfg = try Config.init(gpa, wrapped.facts(m), 16, 15);
    defer cfg.deinit(gpa);
    try std.testing.expect(!cfg.shared_guard);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    var streams: [2]sm.Stream = undefined;
    for (&streams, 0..) |*s, i| s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = dt.long[0 .. 12 + i], .max_new = 24, .eos = &.{96} });
    defer for (&streams) |*s| s.deinit(gpa);
    try engine.addStream(&streams[0]);
    for (0..3) |_| try engine.step();
    const before = head.holds;
    try engine.addStream(&streams[1]);
    try std.testing.expectEqual(before + 1, head.holds); // joining: its first drafts held
    while (engine.activeCount() > 0) try engine.step();
}

const Run = struct { tokens: [3][64]u32 = undefined, lens: [3]usize = @splat(0), holds: u32 = 0 };

/// Three streams decoded together: drafts off, or the fake drafter with or without the guard.
fn run(drafts: bool, guard: bool) !Run {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var head: dt.FakeDrafter = .{ .facts_ = .{ .depth = 4, .step_ms = 0.5, .batched = true, .shared_guard = guard } };
    defer head.deinit();
    var wrapped = try Drafted.init(gpa, target.backend(), head.drafter());
    defer wrapped.deinit();
    const m: Model = .{ .exact_width = 16, .gpu_tokens = true, .hidden_rows = true, .batch_rows = 32, .max_streams = 4 };
    var cfg = try Config.init(gpa, if (drafts) wrapped.facts(m) else m, 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, if (drafts) wrapped.backend() else target.backend(), clock.clock());
    defer engine.deinit();
    var streams: [3]sm.Stream = undefined;
    for (&streams, 0..) |*s, i| s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = dt.long[0 .. 11 + i], .max_new = 48, .eos = &.{96}, .drafts = drafts });
    defer for (&streams) |*s| s.deinit(gpa);
    for (&streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    var out: Run = .{ .holds = head.holds };
    for (&streams, 0..) |*s, i| {
        out.lens[i] = s.emitted().len;
        @memcpy(out.tokens[i][0..out.lens[i]], s.emitted());
    }
    return out;
}

test "the guard changes only which shared rounds draft: replies equal one-token decode with it on and off" {
    const plain = try run(false, false);
    const off = try run(true, false);
    const on = try run(true, true);
    for (0..3) |i| {
        try std.testing.expectEqualSlices(u32, plain.tokens[i][0..plain.lens[i]], off.tokens[i][0..off.lens[i]]);
        try std.testing.expectEqualSlices(u32, plain.tokens[i][0..plain.lens[i]], on.tokens[i][0..on.lens[i]]);
    }
    try std.testing.expect(on.holds < off.holds); // its plain stretch holds no drafts
}

/// A clock that reads the same ms for every span: two of them with different ms stand for a fast and a slow machine.
const FlatClock = struct {
    ms: f64,

    fn clock(c: *FlatClock) @import("backend.zig").Clock {
        return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(_: *anyopaque) void {}

    fn elapsed(ptr: *anyopaque, _: @import("backend.zig").Mark) f64 {
        const c: *FlatClock = @ptrCast(@alignCast(ptr));
        return c.ms;
    }
};

/// Three streams, the guard asked for or not, timed or not, on a clock reading `ms` for every span.
fn runClock(timed: bool, guard: bool, ms: f64) !Run {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var head: dt.FakeDrafter = .{ .facts_ = .{ .depth = 4, .step_ms = 0.5, .batched = true, .shared_guard = guard } };
    defer head.deinit();
    var wrapped = try Drafted.init(gpa, target.backend(), head.drafter());
    defer wrapped.deinit();
    const m: Model = .{ .exact_width = 16, .gpu_tokens = true, .hidden_rows = true, .batch_rows = 32, .max_streams = 4 };
    var cfg = try Config.init(gpa, wrapped.facts(m), 16, 15);
    defer cfg.deinit(gpa);
    cfg.timed = timed;
    var clock: FlatClock = .{ .ms = ms };
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    var streams: [3]sm.Stream = undefined;
    for (&streams, 0..) |*s, i| s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = dt.long[0 .. 11 + i], .max_new = 48, .eos = &.{96}, .drafts = true });
    defer for (&streams) |*s| s.deinit(gpa);
    for (&streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    var out: Run = .{ .holds = head.holds };
    for (&streams, 0..) |*s, i| {
        out.lens[i] = s.emitted().len;
        @memcpy(out.tokens[i][0..out.lens[i]], s.emitted());
    }
    return out;
}

test "with timed off the shared guard keeps no plain stretch: the same schedule on a fast and a slow clock" {
    const fast = try runClock(false, true, 0.5);
    const slow = try runClock(false, true, 400.0);
    try std.testing.expectEqual(fast.holds, slow.holds);
    for (0..3) |i| try std.testing.expectEqualSlices(u32, fast.tokens[i][0..fast.lens[i]], slow.tokens[i][0..slow.lens[i]]);
    const unasked = try runClock(false, false, 0.5);
    try std.testing.expectEqual(unasked.holds, fast.holds); // as if the guard were not asked for
}
