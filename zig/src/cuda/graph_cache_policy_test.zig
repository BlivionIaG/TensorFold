//! The keyed graph cache's policy over a fake engine: capture, replay, LRU, fingerprints, rank votes, the memory floor.

const std = @import("std");
const gc = @import("graph_cache.zig");
const registry = @import("graph_registry.zig");
const Stream = @import("stream.zig").Stream;
const Body = gc.Body;
const Agree = gc.Agree;
const Engine = gc.Engine;
const Outcome = gc.Outcome;
const Settings = gc.Settings;
const Cache = gc.Cache;
const Snapshot = gc.Snapshot;
const snapshots = gc.snapshots;
const report_every = gc.report_every;
const fingerprintOf = gc.fingerprintOf;

const Fake = struct {
    captures: u32 = 0,
    launches: u32 = 0,
    live: u32 = 0,
    fail_capture: bool = false,
    slots: [64]u8 = undefined,

    fn engine(f: *Fake) Engine {
        return .{ .ctx = f, .capture = cap, .launch = lau, .free = fre };
    }
    fn cap(ctx: *anyopaque, stream: Stream, body: Body) anyerror!*anyopaque {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.fail_capture) return error.CaptureFailed;
        _ = stream;
        _ = body; // a capture records the body; nothing runs
        f.captures += 1;
        f.live += 1;
        return &f.slots[f.captures % f.slots.len];
    }
    fn lau(ctx: *anyopaque, _: *anyopaque, _: Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.launches += 1;
    }
    fn fre(ctx: *anyopaque, _: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.live -= 1;
    }
};

const Ran = struct {
    n: u32 = 0,
    fn body(r: *Ran) Body {
        return .{ .ctx = r, .run = go };
    }
    fn go(ctx: *anyopaque, _: Stream) anyerror!void {
        const r: *Ran = @ptrCast(@alignCast(ctx));
        r.n += 1;
    }
};

const no_stream: Stream = .{ .d = undefined, .handle = null };

test "graph cache: capture on first use, replay after, LRU at the cap" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 2 });
    defer c.deinit();
    try std.testing.expectEqual(Outcome.captured, try c.run(1, no_stream, 7, r.body()));
    try std.testing.expectEqual(Outcome.replayed, try c.run(1, no_stream, 7, r.body()));
    try std.testing.expectEqual(Outcome.captured, try c.run(2, no_stream, 7, r.body()));
    _ = try c.run(1, no_stream, 7, r.body()); // 1 is now the most recent
    try std.testing.expectEqual(Outcome.captured, try c.run(3, no_stream, 7, r.body())); // evicts 2
    try std.testing.expect(c.has(1) and c.has(3) and !c.has(2));
    try std.testing.expectEqual(@as(u32, 0), r.n); // the body never ran eagerly
    try std.testing.expectEqual(@as(u32, 2), f.live);
    try std.testing.expectEqual(@as(u32, 5), f.launches);
}

test "graph cache: a new fingerprint drops every graph" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{});
    defer c.deinit();
    _ = try c.run(1, no_stream, 7, r.body());
    _ = try c.run(2, no_stream, 7, r.body());
    try std.testing.expectEqual(Outcome.captured, try c.run(1, no_stream, 8, r.body()));
    try std.testing.expectEqual(@as(usize, 1), c.count());
    try std.testing.expectEqual(@as(u32, 1), f.live);
}

const Peer = struct {
    theirs: []const bool, // the other rank's verdicts, in call order
    at: usize = 0,
    fn agree(p: *Peer) Agree {
        return .{ .ctx = p, .all = all };
    }
    fn all(ctx: *anyopaque, ok: bool) anyerror!bool {
        const p: *Peer = @ptrCast(@alignCast(ctx));
        const t = p.theirs[p.at];
        p.at += 1;
        return ok and t;
    }
};

test "graph cache: a capture failing on another rank turns graphs off here too" {
    var f: Fake = .{};
    var r: Ran = .{};
    // per miss: room, then the capture's verdict; the peer's capture fails on the second key
    var peer: Peer = .{ .theirs = &.{ true, true, true, false } };
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{});
    c.agree = peer.agree();
    defer c.deinit();
    try std.testing.expectEqual(Outcome.captured, try c.run(1, no_stream, 7, r.body()));
    try std.testing.expectEqual(Outcome.eager, try c.run(2, no_stream, 7, r.body()));
    try std.testing.expect(c.failed);
    try std.testing.expectEqual(@as(u32, 0), f.live); // this rank's good graphs are freed as well
    try std.testing.expectEqual(Outcome.eager, try c.run(1, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(u32, 2), r.n);
}

var room_flag = true;
fn roomFn() bool {
    return room_flag;
}

test "graph cache: below the memory floor the least recent quarter goes, the cap follows, the step runs eagerly" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 48, .min_keep = 2 });
    c.room = roomFn;
    defer c.deinit();
    room_flag = true;
    for (0..8) |k| _ = try c.run(@intCast(k), no_stream, 7, r.body());
    room_flag = false;
    try std.testing.expectEqual(Outcome.eager, try c.run(100, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(usize, 6), c.count());
    try std.testing.expectEqual(@as(u32, 6), c.cap);
    try std.testing.expect(!c.has(0) and !c.has(1) and c.has(7));
    room_flag = true;
    // room again: the cap is back to max, the miss captures without evicting
    try std.testing.expectEqual(Outcome.captured, try c.run(100, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(usize, 7), c.count());
    try std.testing.expectEqual(@as(u32, 48), c.cap);
    try std.testing.expect(c.has(7) and c.has(2));
    try std.testing.expectEqual(@as(u64, 1), c.stats.floor_hits);
}

test "graph cache: a dip under the floor does not leave later keys churning once memory is back" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 16, .min_keep = 2 });
    c.room = roomFn;
    defer c.deinit();
    room_flag = true;
    for (0..12) |k| _ = try c.run(@intCast(k), no_stream, 7, r.body());
    room_flag = false; // a long prompt's workspace held
    _ = try c.run(200, no_stream, 7, r.body());
    room_flag = true; // freed: the decode after it meets 12 new keys (a new context bucket), then replays them
    const before = c.stats.evicted;
    for (0..12) |k| _ = try c.run(@intCast(300 + k), no_stream, 7, r.body());
    for (0..12) |k| try std.testing.expectEqual(Outcome.replayed, try c.run(@intCast(300 + k), no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(u32, 16), c.cap);
    // only what the cap of 16 forces: 9 left after the dip + 12 new = 21 -> 5 evictions, none during the replays
    try std.testing.expectEqual(before + 5, c.stats.evicted);
}

test "graph cache: hold - below the floor a miss runs eagerly, nothing is evicted, the held keys replay" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 48, .min_keep = 2, .hold = true });
    c.room = roomFn;
    defer c.deinit();
    room_flag = true;
    for (0..8) |k| _ = try c.run(@intCast(k), no_stream, 7, r.body());
    room_flag = false;
    try std.testing.expectEqual(Outcome.eager, try c.run(100, no_stream, 7, r.body()));
    try std.testing.expectEqual(Outcome.eager, try c.run(101, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(usize, 8), c.count());
    try std.testing.expectEqual(@as(u32, 48), c.cap);
    for (0..8) |k| try std.testing.expectEqual(Outcome.replayed, try c.run(@intCast(k), no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(u64, 0), c.stats.evicted);
    try std.testing.expectEqual(@as(u64, 2), c.stats.floor_hits);
    room_flag = true;
    try std.testing.expectEqual(Outcome.captured, try c.run(100, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(usize, 9), c.count());
}

test "graph cache: the floor's enter / under / leave state and its counts, as /health reads them" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 48, .hold = true });
    c.room = roomFn;
    c.name = "test floor";
    defer c.deinit();
    room_flag = true;
    for (0..4) |k| _ = try c.run(@intCast(k), no_stream, 7, r.body());
    try std.testing.expect(!c.under);
    room_flag = false;
    for (0..report_every + 3) |k| _ = try c.run(@intCast(1000 + k), no_stream, 7, r.body());
    try std.testing.expect(c.under);
    try std.testing.expectEqual(@as(u64, report_every + 3), c.under_eager);
    // the held keys replay under the floor and leave the state alone
    try std.testing.expectEqual(Outcome.replayed, try c.run(2, no_stream, 7, r.body()));
    try std.testing.expect(c.under);
    var out: [8]Snapshot = undefined;
    const mine = for (out[0..snapshots(&out)]) |x| {
        if (std.mem.eql(u8, x.name(), "test floor")) break x;
    } else return error.NotPublished;
    const want: registry.Counts = .{ .held = 4, .captured = 4, .replayed = 1, .eager = report_every + 3, .floor_hits = report_every + 3, .under = true, .misses = 4 + report_every + 3, .request_capture_ns = mine.counts.request_capture_ns };
    try std.testing.expectEqual(want, mine.counts);
    // room again: the next miss captures and the floor is left; the counts keep their totals
    room_flag = true;
    try std.testing.expectEqual(Outcome.captured, try c.run(2000, no_stream, 7, r.body()));
    try std.testing.expect(!c.under);
    try std.testing.expectEqual(@as(u64, report_every + 3), c.stats.floor_hits);
    // a second dip starts its own count
    room_flag = false;
    _ = try c.run(3000, no_stream, 7, r.body());
    try std.testing.expectEqual(@as(u64, 1), c.under_eager);
    room_flag = true;
}

var meter_gib: f64 = 100;
var meter_reads: u32 = 0;
fn meterFn() ?f64 {
    meter_reads += 1;
    defer meter_gib -= 0.25; // each read sees a quarter GiB less: a capture's before and after differ by that
    return meter_gib;
}

test "graph cache: the meter prices a capture, replays never read it, a freed cache leaves the snapshots" {
    var f: Fake = .{};
    var r: Ran = .{};
    var out: [8]Snapshot = undefined;
    const others = snapshots(&out);
    {
        var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 4 });
        defer c.deinit();
        c.meter = meterFn;
        c.name = "metered";
        meter_reads = 0;
        try std.testing.expectEqual(Outcome.captured, try c.run(1, no_stream, 7, r.body()));
        try std.testing.expectEqual(@as(f64, 256), c.stats.capture_mib);
        try std.testing.expectEqual(@as(u32, 2), meter_reads);
        try std.testing.expectEqual(Outcome.replayed, try c.run(1, no_stream, 7, r.body()));
        try std.testing.expectEqual(@as(u32, 2), meter_reads);
        try std.testing.expectEqual(others + 1, snapshots(&out));
        const mine = for (out[0 .. others + 1]) |x| {
            if (std.mem.eql(u8, x.name(), "metered")) break x;
        } else return error.NotPublished;
        try std.testing.expectEqual(@as(u64, 1), mine.counts.replayed);
    }
    try std.testing.expectEqual(others, snapshots(&out));
}

test "graph cache: off runs every step eagerly" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .on = false });
    defer c.deinit();
    try std.testing.expectEqual(Outcome.eager, try c.run(1, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(u32, 0), f.captures);
}

test "graph cache: TF_GRAPH_CACHE=0 or off turns a family's cache off without a rebuild; anything else leaves it" {
    const on: Settings = .{};
    for ([_][]const u8{ "0", "off", "OFF" }) |v| try std.testing.expect(!on.operator(v).on);
    for ([_]?[]const u8{ null, "1", "on", "" }) |v| try std.testing.expect(on.operator(v).on);
    try std.testing.expect(!(Settings{ .on = false }).operator("1").on); // the operator only turns a cache off
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), on.operator("off"));
    defer c.deinit();
    for (0..3) |_| try std.testing.expectEqual(Outcome.eager, try c.run(1, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(u32, 0), f.captures);
}

test "graph cache: fingerprint" {
    try std.testing.expect(fingerprintOf(&.{ 1, 2 }) != fingerprintOf(&.{ 2, 1 }));
}

test "graph cache: warm captures without running or launching the step, and the first run replays it" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{});
    defer c.deinit();
    try std.testing.expectEqual(gc.Warmed.captured, try c.warm(1, no_stream, 7, r.body(), true, 0));
    try std.testing.expectEqual(gc.Warmed.held, try c.warm(1, no_stream, 7, r.body(), true, 0));
    try std.testing.expectEqual(@as(u32, 1), f.captures);
    try std.testing.expectEqual(@as(u32, 0), f.launches);
    try std.testing.expectEqual(@as(u32, 0), r.n);
    try std.testing.expectEqual(Outcome.replayed, try c.run(1, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(u32, 1), f.captures);
    const s = c.stats;
    try std.testing.expectEqual(@as(u64, 1), s.captured);
    try std.testing.expectEqual(@as(u64, 1), s.precaptured);
    try std.testing.expectEqual(@as(u64, 0), s.misses);
    try std.testing.expectEqual(@as(u64, 0), s.request_capture_ns);
    try std.testing.expectEqual(@as(u64, 1), s.replayed);
}

test "graph cache: warm leaves the reserve free, so a request-path capture evicts nothing" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 8 });
    defer c.deinit();
    for (0..6) |k| try std.testing.expectEqual(gc.Warmed.captured, try c.warm(@intCast(k), no_stream, 7, r.body(), true, 2));
    try std.testing.expectEqual(gc.Warmed.full, try c.warm(6, no_stream, 7, r.body(), true, 2));
    try std.testing.expectEqual(gc.Warmed.held, try c.warm(3, no_stream, 7, r.body(), true, 2)); // held keys are held at the limit
    try std.testing.expectEqual(Outcome.captured, try c.run(100, no_stream, 7, r.body()));
    try std.testing.expectEqual(Outcome.captured, try c.run(101, no_stream, 7, r.body()));
    try std.testing.expectEqual(@as(u64, 0), c.stats.evicted);
    try std.testing.expectEqual(@as(u64, 2), c.stats.misses);
}

test "graph cache: warm stops below the memory floor on this rank or another, and leaves the floor state alone" {
    var f: Fake = .{};
    var r: Ran = .{};
    var peer: Peer = .{ .theirs = &.{ true, true, false } }; // room, capture, then the peer's room verdict is no
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{});
    c.agree = peer.agree();
    c.room = roomFn;
    defer c.deinit();
    room_flag = true;
    try std.testing.expectEqual(gc.Warmed.captured, try c.warm(1, no_stream, 7, r.body(), true, 0));
    try std.testing.expectEqual(gc.Warmed.no_room, try c.warm(2, no_stream, 7, r.body(), true, 0)); // the peer's no
    peer.theirs = &.{true};
    peer.at = 0;
    try std.testing.expectEqual(gc.Warmed.no_room, try c.warm(3, no_stream, 7, r.body(), false, 0)); // the caller's price check
    room_flag = false;
    peer.at = 0;
    try std.testing.expectEqual(gc.Warmed.no_room, try c.warm(4, no_stream, 7, r.body(), true, 0)); // the cache's own room rule
    room_flag = true;
    try std.testing.expectEqual(@as(usize, 1), c.count());
    try std.testing.expect(!c.under and c.stats.floor_hits == 0 and c.stats.eager == 0);
}

test "graph cache: a capture failing on a peer during warm turns graphs off here too" {
    var f: Fake = .{};
    var r: Ran = .{};
    var peer: Peer = .{ .theirs = &.{ true, true, true, false } };
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{});
    c.agree = peer.agree();
    defer c.deinit();
    try std.testing.expectEqual(gc.Warmed.captured, try c.warm(1, no_stream, 7, r.body(), true, 0));
    try std.testing.expectEqual(gc.Warmed.off, try c.warm(2, no_stream, 7, r.body(), true, 0));
    try std.testing.expect(c.failed);
    try std.testing.expectEqual(@as(u32, 0), f.live);
    try std.testing.expectEqual(gc.Warmed.off, try c.warm(3, no_stream, 7, r.body(), true, 0));
}

test "graph cache: a cache that is off, or has a new fingerprint, warms as it runs" {
    var f: Fake = .{};
    var r: Ran = .{};
    var off = Cache(u32).init(std.testing.allocator, f.engine(), .{ .on = false });
    defer off.deinit();
    try std.testing.expectEqual(gc.Warmed.off, try off.warm(1, no_stream, 7, r.body(), true, 0));
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{});
    defer c.deinit();
    _ = try c.warm(1, no_stream, 7, r.body(), true, 0);
    try std.testing.expectEqual(gc.Warmed.captured, try c.warm(1, no_stream, 8, r.body(), true, 0)); // dropped, captured again
    try std.testing.expectEqual(@as(usize, 1), c.count());
    try std.testing.expectEqual(@as(u64, 1), c.stats.dropped);
}

test "graph cache: every miss is counted, each key's first request-path capture is logged once" {
    var f: Fake = .{};
    var r: Ran = .{};
    var c = Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 1 });
    defer c.deinit();
    _ = try c.run(1, no_stream, 7, r.body());
    _ = try c.run(1, no_stream, 7, r.body());
    _ = try c.run(2, no_stream, 7, r.body()); // evicts 1
    _ = try c.run(1, no_stream, 7, r.body()); // captured again: a miss, not a new log line
    try std.testing.expectEqual(@as(u64, 3), c.stats.misses);
    try std.testing.expectEqual(@as(u64, 1), c.stats.replayed);
    try std.testing.expectEqual(@as(u64, 3), c.stats.captured - c.stats.precaptured);
    try std.testing.expectEqual(@as(u64, 2), c.miss_log.logged);
    var buf: [256]u8 = undefined;
    const line = try std.fmt.bufPrint(&buf, "{f}", .{c.summary()});
    try std.testing.expect(std.mem.startsWith(u8, line, "1 hits, 3 misses (3 captured on the request path in "));
    try std.testing.expect(std.mem.indexOf(u8, line, "0 pre-captured") != null);
}
