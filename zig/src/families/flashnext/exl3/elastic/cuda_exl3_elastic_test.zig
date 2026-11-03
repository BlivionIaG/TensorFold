//! The elastic memory controller on a model of two ranks: failures injected at any step, lockstep and rollback checked.
const std = @import("std");
const policy = @import("elastic_policy.zig");
const config = @import("elastic_config.zig");
const ctl = @import("cuda_exl3_elastic.zig");
const rates = @import("elastic_rates.zig");
const lines_mod = @import("cuda_exl3_lines.zig");
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

const mib: u64 = 1 << 20;
const line_bytes: u64 = 1_843_200;
const pools = 4;

/// One rank's device: its K/V pages and pools (lines) in a fixed amount of memory.
const Rank = struct {
    cap: u64,
    kv_pages: usize,
    stride: u64,
    pools: [pools]policy.Line,
    lines: lines_mod.Lines = undefined,

    fn kvBytes(r: *const Rank, pages: usize) u64 {
        const k: policy.Kv = .{ .mapped = 0, .count = 0, .min = 0, .floor = 0, .demand = 0, .stride = r.stride, .chunk = 2 * mib };
        return k.bytes(pages);
    }

    fn used(r: *const Rank) u64 {
        var sum = r.kvBytes(r.kv_pages);
        for (r.pools) |l| sum += l.mapped(l.active);
        return sum;
    }

    fn free(r: *const Rank) u64 {
        return r.cap -| r.used();
    }

    fn setKv(r: *Rank, pages: usize) !void {
        const need = r.kvBytes(pages);
        const have = r.kvBytes(r.kv_pages);
        if (need > have and need - have > r.free()) return error.OutOfDeviceMemory;
        r.kv_pages = pages;
    }

    fn shape(ctx: *anyopaque, i: usize) policy.Line {
        const r: *Rank = @ptrCast(@alignCast(ctx));
        return r.pools[i];
    }

    fn set(ctx: *anyopaque, i: usize, n: u32) anyerror!void {
        const r: *Rank = @ptrCast(@alignCast(ctx));
        const l = &r.pools[i];
        const need = l.mapped(n);
        const have = l.mapped(l.active);
        if (need > have and need - have > r.free()) return error.OutOfDeviceMemory;
        l.active = n;
    }

    fn freeOf(ctx: *anyopaque) u64 {
        const r: *Rank = @ptrCast(@alignCast(ctx));
        return r.free();
    }
};

/// Two ranks, the demand on the K/V pool and the failures to inject.
const World = struct {
    ranks: [2]Rank,
    demand: usize = 10,
    kv_count: usize = 400,
    kv_min: usize = 20,
    reserve: u64 = 100 * mib,
    clock: u64 = 0,
    /// Driver calls so far (observations excluded) and what to do at a given one.
    calls: usize = 0,
    stride1: u64 = 3 * mib,
    steal: ?struct { at: usize, rank: usize, bytes: u64 } = null,
    fatal_at: ?usize = null,
    observed: usize = 0,
    /// The decode counters each pool reports, and whether reading them fails.
    counters: [pools]rates.Pool = @splat(.{}),
    sample_fails: bool = false,

    fn init(w: *World) !void {
        for (&w.ranks, 0..) |*r, i| {
            r.* = .{ .cap = 0, .kv_pages = w.kv_count, .stride = if (i == 0) 3 * mib else w.stride1, .pools = undefined };
            for (&r.pools) |*l| l.* = .{ .active = 20, .floor = 10, .cap = 60, .bytes = 1_843_200, .chunk = 2 * mib };
            r.cap = r.used() + w.reserve; // nothing spare but the reserve
            r.lines = try lines_mod.Lines.init(.{ .ctx = r, .count = pools, .shape = Rank.shape, .set = Rank.set, .free = Rank.freeOf }, 10);
        }
    }

    fn driver(w: *World) ctl.Driver {
        return .{ .ctx = w, .observe = observe, .kv = kv, .lines = lines, .undo_lines = undo, .now_ns = now, .sample = sample };
    }

    fn sample(ctx: *anyopaque, out: *rates.Sample) anyerror!void {
        const w: *World = @ptrCast(@alignCast(ctx));
        if (w.sample_fails) return error.TelemetryCounterRegression;
        out.n = pools;
        out.pools[0..pools].* = w.counters;
    }

    fn observe(ctx: *anyopaque, m: *policy.Memory, peer: bool) anyerror!void {
        const w: *World = @ptrCast(@alignCast(ctx));
        w.observed += 1;
        const r = &w.ranks[0];
        m.* = .{ .kv = .{ .mapped = r.kv_pages, .count = w.kv_count, .min = w.kv_min, .floor = @min(r.kv_pages, @max(w.kv_min, w.demand)), .demand = w.demand, .stride = r.stride, .chunk = 2 * mib }, .free = r.free(), .reserve = w.reserve };
        if (peer) m.free = @min(m.free, w.ranks[1].free());
        m.n_pools = r.lines.shapes(&m.pools);
    }

    /// Counts a step, lets memory vanish under a rank when asked, and fails fatally when asked.
    fn step(w: *World) !void {
        w.calls += 1;
        if (w.steal) |s| {
            if (s.at == w.calls) w.ranks[s.rank].cap -= @min(s.bytes, w.ranks[s.rank].free()); // another process takes free memory
        }
        if (w.fatal_at) |n| if (n == w.calls) return error.RemoteRank;
    }

    fn kv(ctx: *anyopaque, pages: usize) anyerror!void {
        const w: *World = @ptrCast(@alignCast(ctx));
        try w.step();
        const before = w.ranks[0].kv_pages;
        w.ranks[0].setKv(pages) catch return error.ElasticRefused; // the peer has heard nothing
        w.ranks[1].setKv(pages) catch {
            try w.ranks[0].setKv(before); // the leader goes back; it cannot fail unless memory was taken meanwhile
            return error.ElasticRefused;
        };
    }

    fn lines(ctx: *anyopaque, delta: i32) anyerror!void {
        const w: *World = @ptrCast(@alignCast(ctx));
        try w.step();
        w.ranks[0].lines.apply(delta) catch |e| return e;
        w.ranks[1].lines.apply(delta) catch |e| {
            if (e != error.ElasticRefused) return e;
            try w.ranks[0].lines.undoLast();
            return error.ElasticRefused;
        };
    }

    fn undo(ctx: *anyopaque) anyerror!void {
        const w: *World = @ptrCast(@alignCast(ctx));
        try w.ranks[0].lines.undoLast();
        try w.ranks[1].lines.undoLast();
    }

    fn now(ctx: *anyopaque) u64 {
        const w: *World = @ptrCast(@alignCast(ctx));
        return w.clock;
    }

    const Snapshot = struct { kv: [2]usize, lines: [2][pools]u32, epoch: [2]u32 };

    fn snapshot(w: *const World) Snapshot {
        var s: Snapshot = undefined;
        for (w.ranks, 0..) |r, i| {
            s.kv[i] = r.kv_pages;
            for (r.pools, 0..) |l, j| s.lines[i][j] = l.active;
            s.epoch[i] = r.lines.epoch;
        }
        return s;
    }

    /// The ranks hold the same pages, every pool in its bounds, and nothing is mapped beyond the memory.
    fn consistent(w: *const World) !void {
        try equal(w.ranks[0].kv_pages, w.ranks[1].kv_pages);
        try equal(w.ranks[0].lines.epoch, w.ranks[1].lines.epoch);
        for (w.ranks) |r| {
            try expect(r.kv_pages >= w.kv_min and r.kv_pages <= w.kv_count);
            for (r.pools) |l| try expect(l.active >= l.floor and l.active <= l.cap);
            try expect(r.used() <= r.cap);
        }
    }
};

const settings: config.Settings = .{ .tick_ms = 100, .policy = .{ .idle_ticks = 3, .dwell_ticks = 5, .step_bytes = 64 * mib } };

fn started(w: *World) !ctl.Controller {
    try w.init();
    return ctl.Controller.init(settings, w.driver());
}

test "exl3: elastic controller: nothing is observed before a tick is due, and a calm server gives K/V memory to lines" {
    var w: World = .{ .ranks = undefined };
    var c = try started(&w);
    const lines_before = w.ranks[0].lines.total();
    for (0..50) |_| c.tick(.{}); // time has not moved: no tick is due
    try equal(@as(usize, 0), w.observed);
    try equal(@as(u64, 0), c.stats().ticks);
    var plans_at: usize = 0;
    for (1..40) |i| {
        w.clock += 100 * std.time.ns_per_ms;
        c.tick(.{});
        if (c.stats().plans > 0 and plans_at == 0) plans_at = i;
        try w.consistent();
    }
    try expect(plans_at >= 5); // the dwell: five ticks since the start
    try expect(c.stats().plans >= 1);
    try expect(w.ranks[0].lines.total() > lines_before);
    try expect(w.ranks[0].kv_pages < w.kv_count and w.ranks[0].kv_pages >= w.kv_min);
    try expect(c.stats().kv_pages_down > 0 and c.stats().lines_up > 0 and c.stats().bytes_moved > 0);
    for (w.ranks) |r| try expect(r.free() >= w.reserve); // the reserve is never spent
    try expect(!c.dead);
}

test "exl3: elastic controller: lines do not move while a prefill runs, and grow once decode or idle comes" {
    var w: World = .{ .ranks = undefined };
    var c = try started(&w);
    const before = w.snapshot();
    for (0..60) |_| { // a long prefill: calm for far longer than the dwell
        w.clock += 100 * std.time.ns_per_ms;
        c.tick(.{ .phase = .prefill });
    }
    try std.testing.expectEqualDeep(before, w.snapshot());
    try equal(@as(u64, 0), c.stats().plans);
    try expect(c.stats().ticks == 60);
    w.clock += 100 * std.time.ns_per_ms;
    c.tick(.{ .phase = .decode }); // the first decode tick finds it overdue and plans
    try equal(@as(u64, 1), c.stats().plans);
    try expect(w.ranks[0].lines.total() > before.lines[0][0] * pools);
    try w.consistent();
}

test "exl3: elastic controller: a request for pages shrinks lines to map them and the pair stays whole" {
    var w: World = .{ .ranks = undefined, .demand = 200 };
    var c = try started(&w);
    // K/V gave 200 pages back and something else took the memory: only the reserve is free
    for (&w.ranks) |*r| {
        try r.setKv(200);
        r.cap = r.used() + w.reserve;
    }
    try expect(c.couldFit(10));
    const lines_before = w.ranks[0].lines.total();
    try expect(c.ensure(20));
    try expect(w.ranks[0].kv_pages >= 220);
    try expect(w.ranks[0].lines.total() < lines_before); // the lines paid
    try expect(!c.dead);
    try equal(@as(u64, 1), c.stats().requests);
    try equal(@as(u64, 1), c.stats().requests_met);
    try w.consistent();
    for (w.ranks) |r| try expect(r.free() >= w.reserve);
    try expect(c.ensure(0));
    // more than the lines at their floors and the free bytes hold: refused without a move
    const snap = w.snapshot();
    try expect(!c.couldFit(10_000));
    try expect(!c.ensure(10_000));
    try std.testing.expectEqualDeep(snap, w.snapshot());
}

test "exl3: elastic controller: a plan reports the decode counters since the last one, and a failing sampler stops only the counting" {
    var w: World = .{ .ranks = undefined, .demand = 200 };
    var c = try started(&w);
    for (&w.ranks) |*r| {
        try r.setKv(200);
        r.cap = r.used() + w.reserve;
    }
    w.counters[0] = .{ .rounds = 100, .hit = 900, .miss = 100, .copies = 20 };
    w.counters[1] = .{ .rounds = 100, .hit = 400, .miss = 200, .copies = 5 };
    try expect(c.ensure(20)); // a plan: lines pay
    var d = c.stats().decode;
    try equal(@as(u64, 100), d.rounds);
    try equal(@as(u64, 1300), d.hit);
    try equal(@as(u64, 300), d.miss);
    try equal(@as(u64, 25), d.copies);
    try equal(@as(u64, 25 * line_bytes), d.copy_bytes);
    // later: more decode, and a plan from bytes that appeared; only what is new is added
    w.counters[0].rounds += 50;
    w.counters[0].hit += 450;
    w.counters[0].copies += 3;
    const plans = c.stats().plans;
    for (&w.ranks) |*r| {
        try r.setKv(200);
        r.cap = r.used() + w.reserve + 200 * mib;
    }
    w.demand = 200;
    try expect(c.ensure(20));
    try expect(c.stats().plans > plans);
    d = c.stats().decode;
    try equal(@as(u64, 150), d.rounds);
    try equal(@as(u64, 1750), d.hit);
    try equal(@as(u64, 28), d.copies);
    // a sampler that fails: the plan goes through, the counting stops, the controller lives
    w.sample_fails = true;
    const plans_2 = c.stats().plans;
    for (&w.ranks) |*r| {
        try r.setKv(200);
        r.cap = r.used() + w.reserve + 200 * mib;
    }
    w.demand = 200;
    try expect(c.ensure(20));
    try expect(c.stats().plans > plans_2);
    try equal(@as(u64, 150), c.stats().decode.rounds);
    try expect(!c.counting and !c.dead);
}

test "exl3: elastic controller: a forecast of the queue's pages is made at the tick, and the open then finds them" {
    var w: World = .{ .ranks = undefined, .demand = 200 };
    var c = try started(&w);
    for (&w.ranks) |*r| {
        try r.setKv(200);
        r.cap = r.used() + w.reserve;
    }
    const lines_before = w.ranks[0].lines.total();
    w.clock += 100 * std.time.ns_per_ms;
    c.tick(.{ .phase = .decode, .load = 3, .ahead = 20 }); // three streams are queued: 20 pages lacking
    try expect(w.ranks[0].kv_pages >= 220);
    try expect(w.ranks[0].lines.total() < lines_before); // the lines paid, in this one tick
    try equal(@as(u64, 1), c.stats().forecasts);
    try equal(@as(u64, 1), c.stats().forecasts_met);
    try equal(@as(u64, 0), c.stats().requests);
    try expect(c.stats().plans >= 1 and c.stats().plans <= 4);
    try w.consistent();
    for (w.ranks) |r| try expect(r.free() >= w.reserve);
    // the streams open on the pages the forecast mapped: a request for them moves nothing
    w.demand = 200 + 20;
    const calls = w.calls;
    try expect(c.ensure(0));
    try equal(calls, w.calls);
    // a forecast the lines and free bytes cannot meet moves nothing
    const snap = w.snapshot();
    w.clock += 100 * std.time.ns_per_ms;
    c.tick(.{ .load = 1, .ahead = 10_000 });
    try std.testing.expectEqualDeep(snap, w.snapshot());
    try equal(@as(u64, 1), c.stats().forecasts);
}

test "exl3: elastic controller: K/V keeps its pages for the hold after a burst of streams, then gives them back" {
    var held: u64 = 0;
    var free: u64 = 0;
    for ([_]u32{ 120, 0 }) |hold| {
        var w: World = .{ .ranks = undefined };
        try w.init();
        var s = settings;
        s.policy.hold_ticks = hold;
        var c = ctl.Controller.init(s, w.driver());
        for (0..60) |_| { // four streams ran, using little: the usual dwell has passed many times over
            w.clock += 100 * std.time.ns_per_ms;
            c.tick(.{ .phase = .decode, .load = 4 });
        }
        if (hold > 0) try equal(w.kv_count, w.ranks[0].kv_pages) else try expect(w.ranks[0].kv_pages < w.kv_count);
        var ticks: u64 = 0;
        while (c.stats().plans == 0 and ticks < 400) : (ticks += 1) {
            w.clock += 100 * std.time.ns_per_ms;
            c.tick(.{ .phase = .decode, .load = 1 });
        }
        if (hold > 0) held = ticks else free = ticks;
        try w.consistent();
    }
    try expect(held >= settings.policy.hold_ticks); // given back once the hold has run out, not before
    try expect(free < held);
}

test "exl3: elastic controller: a refusal in the middle of a plan takes the earlier steps back and backs off" {
    // the follower's pages are larger than the leader's: it refuses the K/V step after the lines were cut on both
    var w: World = .{ .ranks = undefined, .demand = 200, .stride1 = 12 * mib };
    var c = try started(&w);
    for (&w.ranks) |*r| {
        try r.setKv(200);
        r.cap = r.used() + w.reserve;
    }
    const before = w.snapshot();
    try expect(!c.ensure(20));
    try std.testing.expectEqualDeep(before, w.snapshot()); // both ranks exactly as they were
    try equal(@as(u64, 1), c.stats().refused);
    try equal(@as(u64, 0), c.stats().plans);
    try expect(w.calls >= 2); // the lines step was applied, then taken back
    try expect(c.st.blocked > 0);
    try expect(!c.dead);
    try w.consistent();
    // backed off: a request moves nothing until the ticks have passed
    const calls = w.calls;
    try expect(!c.ensure(5));
    try equal(calls, w.calls);
}

test "exl3: elastic controller: a step that cannot be taken back is counted, and a fatal step ends the controller" {
    var w: World = .{ .ranks = undefined, .demand = 200 };
    var c = try started(&w);
    for (&w.ranks) |*r| {
        try r.setKv(200);
        r.cap = r.used() + w.reserve;
    }
    w.steal = .{ .at = 2, .rank = 1, .bytes = 400 * mib };
    w.calls = 0;
    try expect(!c.ensure(20));
    try equal(@as(u64, 1), c.stats().rollback_failures);
    try expect(c.dead); // the put back found no memory for the lines: nothing more is attempted
    const calls = w.calls;
    w.clock += 1000 * std.time.ns_per_ms;
    c.tick(.{});
    try expect(!c.ensure(1));
    try expect(!c.couldFit(1));
    try equal(calls, w.calls);
    // a fatal error at the first step
    var w2: World = .{ .ranks = undefined, .demand = 200, .fatal_at = 1 };
    var c2 = try started(&w2);
    for (&w2.ranks) |*r| {
        try r.setKv(200);
        r.cap = r.used() + w2.reserve;
    }
    w2.calls = 0;
    try expect(!c2.ensure(20));
    try expect(c2.dead);
    try equal(@as(u64, 0), c2.stats().refused);
}

test "exl3: elastic controller: the same trace gives the same steps" {
    var traces: [2][64]usize = undefined;
    for (&traces) |*trace| {
        var w: World = .{ .ranks = undefined };
        var c = try started(&w);
        var prng = std.Random.DefaultPrng.init(7);
        for (trace) |*slot| {
            w.clock += 100 * std.time.ns_per_ms;
            w.demand = prng.random().uintAtMost(usize, w.ranks[0].kv_pages);
            c.tick(.{});
            slot.* = w.ranks[0].kv_pages * 1000 + w.ranks[0].lines.total();
        }
    }
    try std.testing.expectEqualSlices(usize, &traces[0], &traces[1]);
}

test "exl3: elastic controller: walks with random demand, requests, lost memory and refusals keep the pair whole" {
    const strides = [_]u64{ 3, 3, 4, 12 };
    var plans: u64 = 0;
    var refused: u64 = 0;
    var met: u64 = 0;
    for (0..120) |seed| {
        var prng = std.Random.DefaultPrng.init(0xc7a1_0000 + seed);
        const rnd = prng.random();
        var w: World = .{ .ranks = undefined, .stride1 = strides[seed % 4] * mib }; // pages that differ between ranks refuse
        var c = try started(&w);
        for (0..300) |_| {
            w.clock += (20 + rnd.uintLessThan(u64, 200)) * std.time.ns_per_ms;
            const r0 = &w.ranks[0];
            switch (rnd.uintLessThan(u8, 8)) {
                0 => w.demand = rnd.uintAtMost(usize, r0.kv_pages),
                1 => w.demand = @min(r0.kv_pages, w.demand * 2 + 3),
                2 => w.demand /= 4,
                else => {},
            }
            const before = w.snapshot();
            const refused_before = c.stats().refused;
            if (rnd.uintLessThan(u8, 10) == 0) { // a stream opens and wants pages
                const extra = 1 + rnd.uintLessThan(usize, 40);
                if (rnd.uintLessThan(u8, 4) == 0) w.steal = .{ .at = w.calls + 1 + rnd.uintLessThan(usize, 4), .rank = rnd.uintLessThan(usize, 2), .bytes = (1 + rnd.uintLessThan(u64, 60)) * mib };
                const got = c.ensure(extra);
                if (got) try expect(w.ranks[0].kv_pages >= @min(w.kv_count, before.kv[0] + extra));
                w.demand = @min(w.demand + (if (got) extra else 0), w.ranks[0].kv_pages);
            } else c.tick(.{});
            w.steal = null;
            try w.consistent();
            if (c.stats().refused > refused_before) try std.testing.expectEqualDeep(before, w.snapshot()); // a refused plan leaves nothing
            // free memory never falls below the reserve through a plan (a theft can, once: it is not the plan's doing)
            if (c.dead) break;
        }
        plans += c.stats().plans;
        refused += c.stats().refused;
        met += c.stats().requests_met;
    }
    try expect(plans > 0 and refused > 0 and met > 0);
}
