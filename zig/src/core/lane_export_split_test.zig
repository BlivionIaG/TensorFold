//! A layer-split handoff request is run by the exporter's `split.run` on the engine thread, not by a prompt pass.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const rt = @import("lane_remote_test.zig");
const exporting = @import("lane_export.zig");
const t = std.testing;

/// The split a test runs (the exporter's own pointer is the fake exporter's); tests run one at a time.
var current: *Split = undefined;

/// The fake family's split: what `run` was given, and what it answers.
const Split = struct {
    accepts: bool = true,
    result: anyerror!u32 = 77,
    runs: u32 = 0,
    cut: u32 = 0,
    prompt_len: usize = 0,
    cancel_seen: bool = false,
    thread: ?std.Thread.Id = null,

    fn ok(_: *anyopaque, _: u32) bool {
        return current.accepts;
    }

    fn run(_: *anyopaque, h: exporting.Handoff, prompt: []const u32, cancel: ?exporting.Cancel) anyerror!u32 {
        const s = current;
        s.runs += 1;
        s.cut = h.split_cut;
        s.prompt_len = prompt.len;
        s.cancel_seen = if (cancel) |c| !c.cancelled() else false; // present, and not cancelled
        s.thread = std.Thread.getCurrentId();
        return s.result;
    }
};

const Seen = struct {
    done: std.atomic.Value(bool) = .init(false),
    reason: ?api.Reason = null,
    message: []const u8 = "",
    tokens: [4]u32 = undefined,
    n: usize = 0,
    prefilled: u32 = 99,

    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        switch (ev.*) {
            .prefilled => |p| s.prefilled = p,
            .tokens => |ts| for (ts) |tk| {
                s.tokens[s.n] = tk;
                s.n += 1;
            },
            .finished => |f| {
                s.reason = f.reason;
                s.message = f.message;
                s.done.store(true, .release);
            },
            else => {},
        }
    }
};

/// One handoff request (`cut` 0: a plain handoff) through a host whose exporter has `split` (null: none).
fn serve(e: *rt.Exp, split: ?*Split, cut: u32, seen: *Seen) !void {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
    e.fake = &fake;
    host.exporter = e.exporter();
    if (split) |s| {
        current = s;
        host.exporter.?.split = .{ .ok = Split.ok, .run = Split.run };
    }
    try host.start();
    defer host.stop();
    var h = rt.id;
    h.split_cut = cut;
    try host.engine().submit(1, &.{ .prompt = &rt.six, .max_tokens = 1, .handoff = h }, .{ .ctx = seen, .event = Seen.event });
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (!seen.done.load(.acquire)) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() > until) return error.ExportFixtureTimeout;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
}

test "layer split: the exporter runs the cut once and the request ends on its token, with no prompt pass or export" {
    var e: rt.Exp = .{};
    var sp: Split = .{};
    var seen: Seen = .{};
    try serve(&e, &sp, 14, &seen);
    try t.expectEqual(@as(u32, 1), sp.runs);
    try t.expectEqual(@as(u32, 14), sp.cut);
    try t.expectEqual(rt.six.len, sp.prompt_len);
    try t.expect(sp.cancel_seen);
    try t.expect(sp.thread.? != std.Thread.getCurrentId()); // the engine's thread
    try t.expectEqual(api.Reason.length, seen.reason.?);
    try t.expectEqualSlices(u32, &.{77}, seen.tokens[0..seen.n]);
    try t.expectEqual(@as(u32, 0), seen.prefilled);
    try t.expectEqual(@as(u32, 0), e.calls); // the split reports its own state: no export, no failure from the host
}

test "layer split: a failed run tells the decoder and ends the request failed" {
    var e: rt.Exp = .{};
    var sp: Split = .{ .result = error.SplitTooShort };
    var seen: Seen = .{};
    try serve(&e, &sp, 14, &seen);
    try t.expectEqual(@as(u32, 1), sp.runs);
    try t.expectEqual(api.Reason.failed, seen.reason.?);
    try t.expectEqual(@as(usize, 0), seen.n);
    try t.expectEqual(@as(u32, 1), e.calls);
    try t.expectEqualStrings("SplitTooShort", e.why.?);
}

test "layer split: a run cancelled mid-way ends the request cancelled and the decoder hears so" {
    var e: rt.Exp = .{};
    var sp: Split = .{ .result = error.Cancelled };
    var seen: Seen = .{};
    try serve(&e, &sp, 14, &seen);
    try t.expectEqual(api.Reason.cancelled, seen.reason.?);
    try t.expectEqual(@as(u32, 1), e.calls);
    try t.expectEqualStrings("Cancelled", e.why.?);
}

test "layer split: a plain handoff never reaches the split and exports at its cut as before" {
    var e: rt.Exp = .{};
    var sp: Split = .{};
    var seen: Seen = .{};
    try serve(&e, &sp, 0, &seen);
    try t.expectEqual(@as(u32, 0), sp.runs);
    try t.expectEqual(@as(?usize, 5), e.tokens);
    try t.expectEqual(api.Reason.length, seen.reason.?);
}

test "layer split: a cut the exporter does not take, or an exporter with no split, is refused at submit" {
    for ([_]?bool{ false, null }) |accepts| {
        var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
        defer cfg.deinit(t.allocator);
        var fake: lanes.fake.Fake = .{ .gpa = t.allocator };
        defer fake.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
        defer core.deinit();
        var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
        var e: rt.Exp = .{ .fake = &fake };
        var sp: Split = .{ .accepts = accepts orelse true };
        host.exporter = e.exporter();
        if (accepts != null) {
            current = &sp;
            host.exporter.?.split = .{ .ok = Split.ok, .run = Split.run };
        }
        try host.start();
        defer host.stop();
        var h = rt.id;
        h.split_cut = 14;
        var seen: Seen = .{};
        const sink: api.Sink = .{ .ctx = &seen, .event = Seen.event };
        try t.expectError(error.InvalidRequest, host.engine().submit(1, &.{ .prompt = &rt.six, .max_tokens = 1, .handoff = h }, sink));
        try t.expectEqual(@as(u32, 0), sp.runs);
    }
}
