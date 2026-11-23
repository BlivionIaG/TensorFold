//! drive.zig on the CPU twin: lanes with oracle-corrupted drafts == the serial decode; drafts off == serial.
const std = @import("std");
const twin = @import("twin.zig");
const drive = @import("drive.zig");

const gpa = std.testing.allocator;

test "oracle drafts and no drafts both give the serial decode" {
    const dims: twin.Dims = .{};
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    const max_new = 90;
    const want = blk: {
        const t = try twin.Twin.init(gpa, dims, 1);
        defer t.deinit();
        break :blk try t.serial(0, &prompt, max_new, null, null);
    };
    defer gpa.free(want);
    const shape: @import("dspark.zig").Shape = .{ .block = dims.block, .window = dims.window, .hidden = dims.dim, .candidates = 8 };
    for ([_]bool{ false, true }) |drafts| {
        const t = try twin.Twin.init(gpa, dims, 2);
        defer t.deinit();
        var nd: drive.NoDraft = .{};
        var o: drive.Oracle = .{ .serial = want, .prompt_len = prompt.len, .vocab = dims.vocab };
        const r = try drive.generate(gpa, t.target(), if (drafts) o.pass() else nd.pass(), shape, &prompt, max_new, drafts);
        defer gpa.free(r.tokens);
        try std.testing.expectEqualSlices(u32, want, r.tokens);
        if (drafts) {
            try std.testing.expectEqual(@as(u64, 0), o.off_anchor);
            try std.testing.expect(r.accepted > 0 and r.accepted < r.drafted); // some kept, some rejected
            try std.testing.expect(o.round >= 8); // every corruption kind came round twice
        } else try std.testing.expectEqual(@as(u64, 0), r.drafted);
    }
}

const lanes = @import("lanes");

/// One greedy stream through lanes over the fake target; `hold` is how many drafts its head keeps whatever the ask.
fn heldRun(hold: ?usize, drafts: bool) ![]u32 {
    const Config = lanes.Config;
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .hold = hold, .head_accepts = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    var s = try lanes.Stream.init(gpa, .{ .id = "held", .prompt = &prompt, .max_new = 60, .eos = &.{96}, .drafts = drafts });
    defer s.deinit(gpa);
    try e.addStream(&s);
    while (e.activeCount() > 0) try e.step();
    return gpa.dupe(u32, s.emitted());
}

test "a head holding fewer drafts than the budget asked verifies only the ones it holds" {
    const want = try heldRun(null, false);
    defer gpa.free(want);
    // 0 is a head whose own policy declines the round; 1 and 2 are shallower than the budget's four
    for ([_]usize{ 0, 1, 2 }) |held| {
        const got = try heldRun(held, true);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u32, want, got);
    }
}
