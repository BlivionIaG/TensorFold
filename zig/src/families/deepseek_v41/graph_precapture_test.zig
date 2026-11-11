//! The boot's graph pre-capture over a fake engine: the plan covers the request keys, the floor stops it, no recapture.

const std = @import("std");
const cuda = @import("cuda");
const graphs = @import("graphs.zig");
const rowtab = @import("rowtab.zig");
const plan_mod = @import("graph_plan.zig");
const pre = @import("graph_precapture.zig");
const gc = graphs.gc;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

const no_stream: cuda.Stream = .{ .d = undefined, .handle = null };
const limit: u64 = 131072; // the served profiles' window

var avail_gib: f64 = 100;
var step_gib: f64 = 0; // what each capture takes from MemAvailable

fn meter() ?f64 {
    return avail_gib;
}

fn noMeter() ?f64 {
    return null;
}

const Fake = struct {
    captures: u32 = 0,
    live: u32 = 0,
    fail: bool = false,
    slots: [64]u8 = undefined,

    fn engine(f: *Fake) gc.Engine {
        return .{ .ctx = f, .capture = cap, .launch = lau, .free = fre };
    }
    fn cap(ctx: *anyopaque, _: cuda.Stream, _: gc.Body) anyerror!*anyopaque {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.fail) return error.CaptureFailed;
        f.captures += 1;
        f.live += 1;
        avail_gib -= step_gib;
        return &f.slots[f.captures % f.slots.len];
    }
    fn lau(_: *anyopaque, _: *anyopaque, _: cuda.Stream) anyerror!void {}
    fn fre(ctx: *anyopaque, _: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.live -= 1;
    }
};

fn go(_: *anyopaque, _: cuda.Stream) anyerror!void {}

/// A warmer over one cache the way the engine's are: a shape is one graph, or (`split`) a head's and a tail's.
const Harness = struct {
    cache: *graphs.Cache,
    floor: pre.Floor,
    split: bool = false,
    skip_rows: u32 = 0, // rows whose program has no capturable form

    fn parts(h: *const Harness) graphs.Parts {
        return if (h.split) .{ .at = 3, .spans = .{ .{ 0, 3 }, .{ 3, 9 } } } else .{ .at = null, .spans = .{ .{ 0, 9 }, .{ 9, 9 } } };
    }

    pub fn warm(h: *Harness, s: plan_mod.Shape) !pre.Step {
        if (s.rows == h.skip_rows) return .unavailable;
        var span: [2]usize = undefined;
        return pre.warmParts(h.cache, &h.floor, h.parts(), .{ .rows = s.rows, .ctx = s.ctx }, &span, no_stream, 7, .{ .ctx = h, .run = go });
    }

    pub fn facts(h: *Harness) pre.Facts {
        return .{ .rank = 0, .graphs = h.cache.stats.precaptured, .held = @intCast(h.cache.count()), .cap = h.cache.cap, .price_mib = h.floor.priceGib() * 1024, .avail_gib = avail_gib, .keep_gib = h.floor.room.keep_gib };
    }

    /// The request path's step (what `runRows` does with a key): the window's graphs through `cache.run`.
    fn request(h: *Harness, s: plan_mod.Shape) !gc.Outcome {
        const key: graphs.Key = .{ .rows = s.rows, .ctx = s.ctx };
        const ps = h.parts();
        var out: gc.Outcome = .replayed;
        for (ps.spans, 0..) |sp, pi| {
            if (sp[0] == sp[1]) continue;
            const o = try h.cache.run(ps.key(key, pi), no_stream, 7, .{ .ctx = h, .run = go });
            if (o == .captured or out != .captured) out = o;
        }
        return out;
    }
};

fn planOf(buf: *[plan_mod.rows_cap]u32, cap: u32, slots: u32, last: u64) plan_mod.Plan {
    return .{ .rows = plan_mod.rowBuckets(buf, plan_mod.roundRows(cap, slots), cap), .ctx = plan_mod.ctxBuckets(last, limit, .{}) };
}

fn newCache(f: *Fake, max: u32) graphs.Cache {
    avail_gib = 100;
    step_gib = 0;
    var c = graphs.Cache.init(std.testing.allocator, f.engine(), .{ .on = true, .max = max });
    c.meter = meter;
    return c;
}

test "graph plan: every shape once, context bucket by bucket, rows ascending" {
    var buf: [plan_mod.rows_cap]u32 = undefined;
    var seen = std.AutoHashMap(plan_mod.Shape, void).init(std.testing.allocator);
    defer seen.deinit();
    for ([_]u32{ 8, 16, 32, 48, 64 }) |cap| for ([_]u32{ 2, 4, 8 }) |slots| {
        seen.clearRetainingCapacity();
        const p = planOf(&buf, cap, slots, limit - 1);
        var prev: ?plan_mod.Shape = null;
        for (0..p.count()) |i| {
            const s = p.at(i);
            try expect(!(try seen.getOrPut(s)).found_existing);
            if (prev) |q| try expect(s.ctx > q.ctx or (s.ctx == q.ctx and s.rows > q.rows));
            prev = s;
        }
        try expectEqual(p.count(), seen.count());
    };
}

/// Last positions straddling every context bucket's end, and the first and the window's last.
fn edges(out: *[64]u64) []const u64 {
    var n: usize = 1;
    out[0] = 0;
    for (0..24) |b| {
        const e = graphs.endOf(@intCast(b), 2048, 1.25);
        for ([_]u64{ e - 1, e }) |p| if (p < limit) {
            out[n] = p;
            n += 1;
        };
    }
    out[n] = limit - 1;
    return out[0 .. n + 1];
}

test "graph plan: it holds the key every row window and every one-slot window asks for" {
    const s: graphs.Settings = .{};
    var buf: [plan_mod.rows_cap]u32 = undefined;
    var pos_buf: [64]u64 = undefined;
    const lasts = edges(&pos_buf);
    var set = std.AutoHashMap(plan_mod.Shape, void).init(std.testing.allocator);
    defer set.deinit();
    for ([_]u32{ 16, 48, 64 }) |cap| for ([_]u32{ 2, 4 }) |slots| {
        set.clearRetainingCapacity();
        const p = planOf(&buf, cap, slots, limit - 1);
        for (0..p.count()) |i| try set.put(p.at(i), {});
        for (lasts) |last| for (1..plan_mod.roundRows(cap, slots) + 1) |n| {
            const k = graphs.keyFor(rowtab.bucketRows(@intCast(n), cap).?, last, s);
            try expect(set.contains(.{ .rows = k.rows, .ctx = k.ctx }));
        };
    };
    set.clearRetainingCapacity();
    const most = plan_mod.slotRows(15, s.rows_max);
    const p: plan_mod.Plan = .{ .rows = plan_mod.windowRows(&buf, most), .ctx = plan_mod.ctxBuckets(limit - 1, limit, s) };
    for (0..p.count()) |i| try set.put(p.at(i), {});
    for (lasts) |last| for (1..most + 1) |n| if (last + 1 >= n) {
        const k = graphs.keyFor(@intCast(n), last, s);
        try expect(set.contains(.{ .rows = k.rows, .ctx = k.ctx }));
    };
}

test "graph plan: the served profiles' shape counts" {
    var buf: [plan_mod.rows_cap]u32 = undefined;
    try expectEqual(@as(usize, 14), planOf(&buf, 64, 4, limit - 1).rows.len);
    try expectEqual(@as(usize, 13), planOf(&buf, 48, 4, limit - 1).rows.len);
    try expectEqual(@as(usize, 16), plan_mod.windowRows(&buf, plan_mod.slotRows(15, 64)).len);
    try expectEqual(@as(u32, 23), plan_mod.ctxBuckets(limit - 1, limit, .{}));
    try expectEqual(@as(u32, 1), plan_mod.ctxBuckets(0, limit, .{}));
    try expectEqual(@as(u32, 2), plan_mod.ctxBuckets(4095, limit, .{}));
    try expectEqual(@as(u32, 23), plan_mod.ctxBuckets(limit * 4, limit, .{})); // a window ends inside the limit
}

test "pre-capture: every planned shape captured, and none captures again on the request path" {
    var f: Fake = .{};
    var c = newCache(&f, 48);
    defer c.deinit();
    var buf: [plan_mod.rows_cap]u32 = undefined;
    const plan = planOf(&buf, 64, 4, 4095); // 14 buckets x 2 context buckets
    var h: Harness = .{ .cache = &c, .floor = pre.Floor.init(.{}, meter) };
    const rep = try pre.run(&h, plan, "test");
    try expectEqual(@as(usize, 28), rep.planned);
    try expectEqual(@as(usize, 28), rep.captured);
    try expectEqual(pre.Stop.none, rep.stop);
    try expectEqual(@as(usize, 0), rep.left);
    try expectEqual(@as(u32, 28), f.captures);
    for (0..plan.count()) |i| try expectEqual(gc.Outcome.replayed, try h.request(plan.at(i)));
    try expectEqual(@as(u32, 28), f.captures);
    try expectEqual(@as(u64, 0), c.stats.misses);
    try expectEqual(c.stats.precaptured, c.stats.captured);
    try expectEqual(@as(u64, 0), c.stats.request_capture_ns);
    try expectEqual(@as(u64, 28), c.stats.replayed);
    // a second pre-capture finds everything held
    const again = try pre.run(&h, plan, "test");
    try expectEqual(@as(usize, 28), again.held);
    try expectEqual(@as(u32, 28), f.captures);
}

test "pre-capture: a window's head and tail are two graphs of a shape, both held and replayed" {
    var f: Fake = .{};
    var c = newCache(&f, 96); // `split` doubles the cap
    defer c.deinit();
    var buf: [plan_mod.rows_cap]u32 = undefined;
    const plan = planOf(&buf, 64, 4, 4095);
    var h: Harness = .{ .cache = &c, .floor = pre.Floor.init(.{}, meter), .split = true };
    const rep = try pre.run(&h, plan, "test");
    try expectEqual(@as(usize, 28), rep.captured);
    try expectEqual(@as(u32, 56), f.captures);
    try expectEqual(@as(u64, 56), c.stats.precaptured);
    for (0..plan.count()) |i| try expectEqual(gc.Outcome.replayed, try h.request(plan.at(i)));
    try expectEqual(@as(u32, 56), f.captures);
    try expectEqual(@as(u64, 56), c.stats.replayed);
}

test "pre-capture: the cache keeps a quarter free, so a shape outside the plan evicts nothing" {
    var f: Fake = .{};
    var c = newCache(&f, 48);
    defer c.deinit();
    var buf: [plan_mod.rows_cap]u32 = undefined;
    const plan = planOf(&buf, 64, 4, limit - 1); // 322 shapes
    try expectEqual(@as(usize, 14 * 23), plan.count());
    var h: Harness = .{ .cache = &c, .floor = pre.Floor.init(.{}, meter) };
    const rep = try pre.run(&h, plan, "test");
    try expectEqual(pre.reserveOf(48), 12);
    try expectEqual(@as(usize, 36), rep.captured); // 48 less the reserve of 12
    try expectEqual(pre.Stop.cache_full, rep.stop);
    try expectEqual(plan.at(36), rep.at.?);
    try expectEqual(@as(usize, 322 - 36), rep.left);
    for (0..36) |i| try expectEqual(gc.Outcome.replayed, try h.request(plan.at(i)));
    try expectEqual(gc.Outcome.captured, try h.request(plan.at(36))); // beyond the plan: one capture, no eviction
    try expectEqual(@as(u64, 0), c.stats.evicted);
    try expectEqual(@as(u64, 1), c.stats.misses);
    try expect(c.has(.{ .rows = plan.at(0).rows, .ctx = plan.at(0).ctx }));
}

test "pre-capture: it stops before a graph would cross the memory floor, priced at the configured cost first" {
    var f: Fake = .{};
    var c = newCache(&f, 48);
    defer c.deinit();
    avail_gib = 5.0;
    step_gib = 0.125; // 128 MiB a graph
    var buf: [plan_mod.rows_cap]u32 = undefined;
    const plan = planOf(&buf, 64, 4, limit - 1);
    var h: Harness = .{ .cache = &c, .floor = pre.Floor.init(.{ .keep_gib = 4.0, .cost_gib = 0.0625 }, meter) };
    const rep = try pre.run(&h, plan, "test");
    try expectEqual(@as(usize, 8), rep.captured); // 5.0 down to 4.0 GiB, the last graph leaving exactly the floor
    try expectEqual(pre.Stop.memory_floor, rep.stop);
    try expectEqual(plan.at(8), rep.at.?);
    try expectEqual(plan.count() - 8, rep.left);
    try expect(avail_gib >= 4.0);
    try expectEqual(@as(f64, 0.125), h.floor.priceGib());
    try expectEqual(@as(u64, 0), c.stats.misses);
}

test "pre-capture: a graph dearer than the configured cost teaches the price, and the floor still holds" {
    var f: Fake = .{};
    var c = newCache(&f, 48);
    defer c.deinit();
    avail_gib = 5.0;
    step_gib = 0.5;
    var buf: [plan_mod.rows_cap]u32 = undefined;
    const plan = planOf(&buf, 64, 4, limit - 1);
    var h: Harness = .{ .cache = &c, .floor = pre.Floor.init(.{ .keep_gib = 4.0, .cost_gib = 0.0625 }, meter) };
    const rep = try pre.run(&h, plan, "test");
    try expectEqual(@as(usize, 2), rep.captured);
    try expectEqual(pre.Stop.memory_floor, rep.stop);
    try expect(avail_gib >= 4.0);
}

test "pre-capture: the floor rule, a fixed floor, and no meter" {
    const keep: graphs.Room = .{ .keep_gib = 4.0 };
    try expect(keep.fitsPriced(4.25, 0.25) and !keep.fitsPriced(4.2, 0.25));
    try expect(keep.fits(4.0625) and !keep.fits(4.06));
    const fixed: graphs.Room = .{ .floor_gib = 3.0, .keep_gib = 4.0 };
    try expect(fixed.fitsPriced(3.0, 10) and !fixed.fitsPriced(2.99, 0));
    const blind = pre.Floor.init(keep, noMeter);
    try expect(blind.fits());
}

test "pre-capture: another rank without room stops this one, nothing is poisoned" {
    const Peer = struct {
        theirs: []const bool,
        at: usize = 0,
        fn agree(p: *@This()) gc.Agree {
            return .{ .ctx = p, .all = all };
        }
        fn all(ctx: *anyopaque, ok: bool) anyerror!bool {
            const p: *@This() = @ptrCast(@alignCast(ctx));
            defer p.at += 1;
            return ok and p.theirs[p.at];
        }
    };
    var f: Fake = .{};
    var c = newCache(&f, 48);
    defer c.deinit();
    var peer: Peer = .{ .theirs = &.{ true, true, true, true, false } }; // room and capture twice, then no room
    c.agree = peer.agree();
    var buf: [plan_mod.rows_cap]u32 = undefined;
    const plan = planOf(&buf, 64, 4, 4095);
    var h: Harness = .{ .cache = &c, .floor = pre.Floor.init(.{}, meter) };
    const rep = try pre.run(&h, plan, "test");
    try expectEqual(@as(usize, 2), rep.captured);
    try expectEqual(pre.Stop.memory_floor, rep.stop);
    try expect(!c.failed);
    try expectEqual(@as(u64, 0), c.stats.floor_hits);
}

test "pre-capture: shapes with no capturable program are counted and passed over, not a stop" {
    var f: Fake = .{};
    var c = newCache(&f, 48);
    defer c.deinit();
    var buf: [plan_mod.rows_cap]u32 = undefined;
    const plan = planOf(&buf, 64, 4, 4095);
    var h: Harness = .{ .cache = &c, .floor = pre.Floor.init(.{}, meter), .skip_rows = 3 };
    const rep = try pre.run(&h, plan, "test");
    try expectEqual(@as(usize, 2), rep.unavailable); // rows 3 at both context buckets
    try expectEqual(@as(usize, 26), rep.captured);
    try expectEqual(pre.Stop.none, rep.stop);
}

test "pre-capture: graphs off, or a failed capture, stop it and leave the cache as it is" {
    var f: Fake = .{};
    var off = graphs.Cache.init(std.testing.allocator, f.engine(), .{ .on = false });
    defer off.deinit();
    var buf: [plan_mod.rows_cap]u32 = undefined;
    const plan = planOf(&buf, 64, 4, 4095);
    var ho: Harness = .{ .cache = &off, .floor = pre.Floor.init(.{}, meter) };
    const r0 = try pre.run(&ho, plan, "test");
    try expectEqual(pre.Stop.cache_off, r0.stop);
    try expectEqual(@as(usize, 0), r0.captured);
    try expectEqual(plan.count(), r0.left);

    var bad = newCache(&f, 48);
    defer bad.deinit();
    f.fail = true;
    var hb: Harness = .{ .cache = &bad, .floor = pre.Floor.init(.{}, meter) };
    const r1 = try pre.run(&hb, plan, "test");
    try expectEqual(pre.Stop.cache_off, r1.stop);
    try expect(bad.failed);
    try expectEqual(@as(u32, 0), f.live);
}
