//! A layer split on the real lane host with a fake backend: the job is admitted when the stream is announced
const std = @import("std");
const api = @import("engine_api.zig");
const lanes = @import("lanes");
const remote = @import("lane_remote.zig");
const t = std.testing;

/// One fake split source: `wants` says split; `split` announces `feed` after `delay_ms` (or fails before)
const Peer = struct {
    feed: u8 = 0,
    delay_ms: u64 = 0,
    fail: bool = false,
    no_split: bool = false,
    announced: std.atomic.Value(bool) = .init(false),
    finished: std.atomic.Value(bool) = .init(false),
    admitted_early: std.atomic.Value(bool) = .init(false),
    fills: std.atomic.Value(u32) = .init(0),
    running: std.atomic.Value(u32) = .init(0),

    fn wants(_: *anyopaque, _: remote.Ask) remote.Verdict {
        return .split;
    }

    fn fill(ptr: *anyopaque, _: []const u32, _: u64, _: remote.Directive, _: *const std.atomic.Value(bool)) anyerror!?remote.Filled {
        const p: *Peer = @ptrCast(@alignCast(ptr));
        _ = p.fills.fetchAdd(1, .acq_rel);
        return null;
    }

    fn drop(_: *anyopaque, _: *anyopaque) void {}

    fn split(ptr: *anyopaque, _: []const u32, _: u64, _: remote.Directive, gone: *const std.atomic.Value(bool), announce: remote.Announce) anyerror!void {
        const p: *Peer = @ptrCast(@alignCast(ptr));
        std.Io.sleep(t.io, .fromMilliseconds(@intCast(p.delay_ms)), .awake) catch {};
        if (p.fail) return error.LinkDown;
        announce.feed(&p.feed);
        p.announced.store(true, .release);
        const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
        while (!p.finished.load(.acquire) and !gone.load(.acquire)) {
            if (std.Io.Clock.awake.now(t.io).toNanoseconds() > until) return error.EngineStalled;
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
    }

    fn source(p: *Peer) remote.Source {
        return .{ .ptr = p, .wants = wants, .fill = fill, .drop = drop, .running = &p.running, .split = if (p.no_split) null else split };
    }
};

/// What the stream held when its pass began, and what the request heard.
const Seen = struct {
    peer: *Peer,
    feed: ?*anyopaque = null,
    remote: bool = false,
    pulled: bool = false,
    stats: api.Stats = .{},
    done: std.atomic.Value(bool) = .init(false),

    /// The backend's side of the seam: take the feed, write the report, end the feed once.
    fn chunk(ctx: *anyopaque, s: *lanes.Stream, n: usize) void {
        if (n != 0) return;
        const x: *Seen = @ptrCast(@alignCast(ctx));
        x.feed = s.reuse.split;
        x.remote = s.reuse.remote;
        x.pulled = s.reuse.pulled;
        if (s.reuse.split == null) return;
        s.reuse.split = null;
        s.split_report = .{ .cut = 14, .rows = s.prompt_len - 1, .passes = 3, .first_row_ns = 600_000_000, .rows_ns = 9_000_000_000, .state_ns = 9_500_000_000, .wait_ns = 400_000_000, .compute_ns = 8_000_000_000, .state_wait_ns = 100_000_000, .restore_ns = 50_000_000, .bytes = 1 << 20 };
        x.peer.finished.store(true, .release);
    }

    pub fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const x: *Seen = @ptrCast(@alignCast(ctx));
        switch (ev.*) {
            .finished => |f| {
                x.stats = f.stats;
                x.done.store(true, .release);
            },
            else => {},
        }
    }
};

fn run(peer: *Peer, tokens: []const u32) !Seen {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var seen: Seen = .{ .peer = peer };
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator, .prefill_hook = Seen.chunk, .prefill_hook_ctx = &seen };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
    host.remote = peer.source();
    try host.start();
    defer host.stop();
    try host.engine().submit(1, &.{ .prompt = tokens, .max_tokens = 1 }, .{ .ctx = &seen, .event = Seen.event });
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (!seen.done.load(.acquire) or peer.running.load(.acquire) > 0) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() > until) return error.SplitFixtureTimeout;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
    return seen;
}

const prompt = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18 };

test "split: the stream takes the announced feed as remote provenance, and the reply reports the split" {
    var peer: Peer = .{ .delay_ms = 20 };
    const seen = try run(&peer, &prompt);
    try t.expect(seen.feed == @as(?*anyopaque, &peer.feed));
    try t.expect(seen.remote and seen.pulled);
    try t.expectEqualStrings("split", seen.stats.remote.route);
    try t.expectEqual(@as(u32, 14), seen.stats.remote.split_cut);
    try t.expectEqual(@as(u64, 7), seen.stats.remote.split_rows);
    try t.expectEqual(@as(u32, 3), seen.stats.remote.split_passes);
    try t.expectEqual(@as(f64, 0.6), seen.stats.remote.split_first_row_s);
    try t.expectEqual(@as(f64, 8.0), seen.stats.remote.split_compute_s);
    try t.expectEqual(@as(u64, 1 << 20), seen.stats.remote.bytes);
    try t.expectEqual(@as(u32, 7), seen.stats.remote_prefix); // the reply says the prefix was another machine's
    try t.expectEqual(@as(u32, 0), seen.stats.pulled_prefix);
    try t.expectEqual(@as(u32, 0), peer.fills.load(.acquire)); // a split is not a pull
}

test "split: a split that fails before its stream opens leaves the prompt to this machine, as a failed pull does" {
    var peer: Peer = .{ .fail = true, .delay_ms = 5 };
    const seen = try run(&peer, &prompt);
    try t.expect(seen.feed == null and !seen.remote);
    try t.expectEqual(@as(u32, 0), seen.stats.remote_prefix);
    try t.expectEqual(@as(u32, 0), seen.stats.remote.split_cut);
    try t.expect(!peer.announced.load(.acquire));
}

test "split: a source without a split falls back to its pull for a split verdict" {
    var peer: Peer = .{ .no_split = true };
    const seen = try run(&peer, &prompt);
    try t.expect(seen.feed == null);
    try t.expectEqual(@as(u32, 1), peer.fills.load(.acquire));
}
