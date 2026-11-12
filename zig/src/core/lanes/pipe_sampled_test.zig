//! `Model.pipe_sampled`: a backend that draws sampled rows on the host keeps those streams in verify windows while greedy ones run a step ahead.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const Sampling = @import("sampling.zig").Sampling;
const Penalties = @import("sampling.zig").Penalties;
const sm = @import("stream.zig");
const fake = @import("fake.zig");

const gpa = std.testing.allocator;

const Run = struct { queued: usize, reply: []u32 };

fn run(pipe_sampled: bool, sampling: ?Sampling, penalties: Penalties) !Run {
    var costs: [16]Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .pipe_sampled = pipe_sampled, .mtp = false, .speculate = false, .speculate_early = false, .drafts = 0, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = false }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, .max_new = 24, .eos = &.{96}, .sampling = sampling, .penalties = penalties, .drafts = false });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    return .{ .queued = target.queued, .reply = try gpa.dupe(u32, s.emitted()) };
}

test "greedy streams run a step ahead whatever pipe_sampled says, sampled and penalized ones only when it holds" {
    const sampled: Sampling = .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 };
    const penalized: Penalties = .{ .presence = 0.5 };
    const cases = [_]struct { sampling: ?Sampling, penalties: Penalties, windowed: bool }{
        .{ .sampling = null, .penalties = .{}, .windowed = false },
        .{ .sampling = .{ .seed = 1, .temperature = 0 }, .penalties = .{}, .windowed = false },
        .{ .sampling = sampled, .penalties = .{}, .windowed = true },
        .{ .sampling = null, .penalties = penalized, .windowed = true },
    };
    for (cases) |c| {
        const open = try run(true, c.sampling, c.penalties);
        defer gpa.free(open.reply);
        const shut = try run(false, c.sampling, c.penalties);
        defer gpa.free(shut.reply);
        try std.testing.expect(open.queued > 0);
        if (c.windowed) try std.testing.expectEqual(@as(usize, 0), shut.queued) else try std.testing.expect(shut.queued > 0);
        // pipelined or windowed, the reply is the same
        try std.testing.expectEqualSlices(u32, open.reply, shut.reply);
    }
}
