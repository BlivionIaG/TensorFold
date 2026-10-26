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
