//! The queue-aware, length-aware route: concurrent long prompts use both machines' prefill, measured curves.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const remote = @import("lane_remote.zig");
const route = @import("lane_remote_route.zig");
const Route = remote.Route;
const Verdict = remote.Verdict;
const t = std.testing;

const s: u64 = std.time.ns_per_s;
const k: u64 = 1000;

/// Where each of `lens` prompts asked at 0 ends up: pulls run in turn, local prefills one after another.
const Plan = struct {
    done: [8]f64 = undefined,
    pulled: usize = 0,
    here: usize = 0,

    fn last(p: Plan, n: usize) f64 {
        return std.mem.max(f64, p.done[0..n]);
    }

    fn mean(p: Plan, n: usize) f64 {
        var sum: f64 = 0;
        for (p.done[0..n]) |d| sum += d;
        return sum / @as(f64, @floatFromInt(n));
    }
};

fn secs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

fn plan(r: *Route, lens: []const u64, b: route.Busy) !Plan {
    var out: Plan = .{};
    var ids: [8]u64 = undefined;
    var ests: [8]route.Estimate = undefined;
    for (lens, 0..) |len, i| {
        ests[i] = r.estimate(len - 1, 0, len);
        ids[i] = r.join(0, ests[i]).?;
    }
    r.busy(0, b);
    var remote_t: f64 = 0;
    var local_t: f64 = 0;
    for (lens, 0..) |_, i| {
        switch (r.poll(0, ids[i])) {
            .entered, .wait => {
                remote_t += secs(ests[i].remote_ns);
                out.done[i] = remote_t;
                out.pulled += 1;
            },
            .local => {
                local_t += secs(ests[i].local_ns);
                out.done[i] = local_t;
                out.here += 1;
            },
        }
    }
    return out;
}

test "queue: a 128k prompt alone goes to the pair, a 1M prompt alone is done sooner here, and two 1M prompts take one each" {
    var r: Route = .{};
    try t.expectEqual(@as(usize, 1), (try plan(&r, &.{128 * k}, .{})).pulled); // 66 s pulled against 69 s here
    var r1: Route = .{};
    const one = try plan(&r1, &.{1000 * k}, .{});
    try t.expectEqual(@as(usize, 1), one.here); // 620 s here against 695 s pulled (1.43k a second at 968k)
    var r2: Route = .{};
    const two = try plan(&r2, &.{ 1000 * k, 1000 * k }, .{});
    try t.expectEqual(@as(usize, 1), two.pulled); // the second would wait 1.24k s behind the first here
    try t.expect(two.done[1] > 650 and two.done[1] < 750);
}

test "queue: four 1M prompts at once split two and two, and the last one's first token comes in half the time" {
    var auto: Route = .{};
    const p = try plan(&auto, &.{ 1000 * k, 1000 * k, 1000 * k, 1000 * k }, .{});
    try t.expectEqual(@as(usize, 2), p.pulled);
    try t.expectEqual(@as(usize, 2), p.here);
    var always: Route = .{ .mode = .always };
    const a = try plan(&always, &.{ 1000 * k, 1000 * k, 1000 * k, 1000 * k }, .{});
    try t.expectEqual(@as(usize, 4), a.pulled);
    try t.expect(p.last(4) < a.last(4) * 0.6); // 1.39k s against 2.78k s
    try t.expect(p.mean(4) < a.mean(4));
}

test "queue: four 256k prompts: this machine takes some, and the last first token comes sooner than behind four pulls" {
    var r: Route = .{};
    const p = try plan(&r, &.{ 256 * k, 256 * k, 256 * k, 256 * k }, .{});
    try t.expect(p.here >= 1 and p.pulled >= 2); // a pull and a local prefill take about the same time at 256k
    var always: Route = .{ .mode = .always };
    const a = try plan(&always, &.{ 256 * k, 256 * k, 256 * k, 256 * k }, .{});
    try t.expect(p.last(4) < a.last(4) * 0.75);
    try t.expect(p.mean(4) < a.mean(4));
}

test "queue: a mixed burst keeps the short prompts on the pair or here in seconds, behind no long pull" {
    var r: Route = .{};
    // 16k, 1M, 24k, 1M: this machine takes the first 1M; the shorts and the other 1M go to the pair in turn
    const p = try plan(&r, &.{ 16 * k, 1000 * k, 24 * k, 1000 * k }, .{});
    try t.expectEqual(@as(usize, 3), p.pulled);
    try t.expectEqual(@as(usize, 1), p.here);
    try t.expect(p.done[0] < 20 and p.done[2] < 40); // the 16k and the 24k wait for nothing long
    try t.expect(p.last(4) < 760);
}

test "queue: this machine's own work ahead counts: a 1M prefill under way here keeps the next 1M on the pair" {
    var r: Route = .{};
    const p = try plan(&r, &.{1000 * k}, .{ .spans = &.{.{ .from = 0, .to = 1000 * k }} });
    try t.expectEqual(@as(usize, 1), p.pulled); // 1.24k s here behind it against 695 s pulled
    var idle: Route = .{};
    try t.expectEqual(@as(usize, 1), (try plan(&idle, &.{1000 * k}, .{})).here);
}

test "queue: decoding streams on this machine make a local prefill cost more, a smaller share makes it slower still" {
    const four = [_]u64{ 1000 * k, 1000 * k, 1000 * k, 1000 * k };
    var idle: Route = .{};
    try t.expectEqual(@as(usize, 2), (try plan(&idle, &four, .{})).here);
    var r: Route = .{};
    const p = try plan(&r, &four, .{ .decoding = 3 }); // each local prefill costs 4 x 620 s: only the fourth pull would be later
    try t.expectEqual(@as(usize, 1), p.here);
    var free: Route = .{ .queue = .{ .stall = 0 } };
    try t.expectEqual(@as(usize, 2), (try plan(&free, &four, .{ .decoding = 3 })).here); // no weight, no cost
    var half: Route = .{ .queue = .{ .stall = 0 } };
    const q = try plan(&half, &four, .{ .decoding = 3, .share = 0.5 }); // a prompt pass gets half the wall time beside them
    try t.expectEqual(@as(usize, 1), q.here); // 1.24k s here for each
}

test "queue: weigh counts the pair's queue: behind four forced 64k pulls a prompt is kept here, and says so" {
    var r: Route = .{};
    const ask: remote.Ask = .{ .len = 64 * k, .cached = 0, .decoding = 0 };
    try t.expectEqual(Verdict.remote, r.weigh(0, ask, 64 * k - 1)); // 30 s pulled against 33 s here
    for (0..4) |_| _ = r.join(0, r.estimateFor(64 * k - 1, 0, 64 * k, .force));
    try t.expectEqual(Verdict.busy, r.weigh(0, ask, 64 * k - 1)); // 150 s behind the pulls, or 33 s here
}

test "queue: a prompt sent here stays counted until the host's own count includes it, then no longer" {
    var r: Route = .{};
    const e = r.estimate(1000 * k - 1, 0, 1000 * k);
    var ids: [4]u64 = undefined;
    for (&ids) |*id| id.* = r.join(0, e).?;
    for (ids) |id| _ = r.poll(0, id);
    try t.expectEqual(@as(usize, 4), r.queue.len); // two waiting or pulling, two sent here and reserved
    for (ids) |id| r.leave(id);
    try t.expectEqual(@as(usize, 2), r.queue.len); // the reservations outlive the pull's leaving
    r.busy(100 * std.time.ns_per_ms, .{}); // a count taken inside the grace does not free them
    try t.expectEqual(@as(usize, 2), r.queue.len);
    r.busy(600 * std.time.ns_per_ms, .{});
    try t.expectEqual(@as(usize, 0), r.queue.len);
}

test "queue: a pull waiting longer than the pair's estimate is re-weighed against this machine, never on time already waited" {
    var r: Route = .{ .remote = .{ .base = .{ .v = 1000 } }, .local = .{ .base = .{ .v = 900 } } };
    const e = r.estimate(99_999, 0, 100_000); // 100.6 s pulled, 111.1 s here
    const a = r.join(0, e).?;
    const b = r.join(0, e).?;
    try t.expectEqual(.entered, r.poll(0, a));
    try t.expectEqual(.local, r.poll(0, b)); // 100.6 + 100.6 s behind the first against 111.1 s here
    r.leave(a);
    const c = r.join(50 * s, e).?;
    try t.expectEqual(.entered, r.poll(50 * s, c)); // the pair is free: nothing to wait for, whatever the first waited
}

test "queue: the table is bounded: a full one sends the prompt here" {
    var r: Route = .{ .mode = .always };
    const e = r.estimate(8191, 0, 8192);
    for (0..route.queue_capacity) |_| try t.expect(r.join(0, e) != null);
    try t.expectEqual(@as(?u64, null), r.join(0, e));
    var closing: std.atomic.Value(bool) = .init(false);
    var gone: std.atomic.Value(bool) = .init(false);
    try t.expectEqual(route.Admission.local, r.admit(t.io, e, &closing, &gone));
}

/// The pair as a source over the gate a family's pull keeps, with the host's own load said to it (`busy`).
const Pair = struct {
    route: Route,
    delay_ms: i64 = 600,
    fills: std.atomic.Value(u32) = .init(0),
    drops: std.atomic.Value(u32) = .init(0),
    running: std.atomic.Value(u32) = .init(0),
    closing: std.atomic.Value(bool) = .init(false),
    busies: std.atomic.Value(u32) = .init(0),

    fn wants(ptr: *anyopaque, ask: remote.Ask) Verdict {
        const p: *Pair = @ptrCast(@alignCast(ptr));
        return p.route.weigh(Route.nowNs(t.io), ask, 4);
    }

    fn fill(ptr: *anyopaque, prompt: []const u32, cached: u64, directive: remote.Directive, gone: *const std.atomic.Value(bool)) anyerror!?remote.Filled {
        const p: *Pair = @ptrCast(@alignCast(ptr));
        const entered = switch (p.route.admit(t.io, p.route.estimateFor(4, cached, prompt.len, directive), &p.closing, gone)) {
            .entered => |e| e,
            .local => return error.Shed,
            .stop => return null,
        };
        defer p.route.leave(entered.id);
        _ = p.fills.fetchAdd(1, .acq_rel);
        try std.Io.sleep(t.io, .fromMilliseconds(p.delay_ms), .awake);
        const kept = try t.allocator.create(std.ArrayList(u32));
        kept.* = .empty;
        errdefer t.allocator.destroy(kept);
        try kept.appendSlice(t.allocator, prompt[0..4]);
        return .{ .saved = kept, .at = 4 };
    }

    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const p: *Pair = @ptrCast(@alignCast(ptr));
        const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
        kept.deinit(t.allocator);
        t.allocator.destroy(kept);
        _ = p.drops.fetchAdd(1, .acq_rel);
    }

    fn busy(ptr: *anyopaque, b: route.Busy) void {
        const p: *Pair = @ptrCast(@alignCast(ptr));
        _ = p.busies.fetchAdd(1, .acq_rel);
        p.route.busy(Route.nowNs(t.io), b);
    }

    fn source(p: *Pair) remote.Source {
        return .{ .ptr = p, .wants = wants, .fill = fill, .drop = drop, .running = &p.running, .busy = busy };
    }
};

const Seen = struct {
    route: [16]u8 = undefined,
    n: usize = 0,
    done: std.atomic.Value(bool) = .init(false),

    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const x: *Seen = @ptrCast(@alignCast(ctx));
        switch (ev.*) {
            .finished => |f| {
                x.n = @min(f.stats.remote.route.len, x.route.len);
                @memcpy(x.route[0..x.n], f.stats.remote.route[0..x.n]);
                x.done.store(true, .release);
            },
            else => {},
        }
    }

    fn said(x: *const Seen) []const u8 {
        return x.route[0..x.n];
    }
};

/// A pass over a whole prompt (none of it pulled) outlasts a pull (600 ms), so the load the route counted still holds.
fn slowHere(_: *anyopaque, stream: *lanes.Stream, _: usize) void {
    if (stream.cached == 0) std.Io.sleep(t.io, .fromMilliseconds(1000), .awake) catch {};
}

fn waitFor(flag: *const std.atomic.Value(bool)) !void {
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 10 * std.time.ns_per_s;
    while (!flag.load(.acquire)) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() > until) return error.RemoteFixtureTimeout;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
}

test "queue: on the lane host three prompts at once, two pull in turn and the one the pair's queue would delay is prefilled here" {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var unused: u8 = 0;
    // a prompt prefilled here takes a while, as on the machine: the host's load stays what the route counted
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator, .prefill_hook = slowHere, .prefill_hook_ctx = &unused };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 4 });
    // 6 token prompts, a pull of 4: a pull takes 1.0 s on the books (1 s of tail), 1.2 s here at 5 a second
    var pair: Pair = .{ .route = .{ .local = .{ .base = .{ .v = 5 } }, .remote = .{ .base = .{ .v = 1000 } }, .tail_s = .{ .v = 1 } } };
    host.remote = pair.source();
    const prompts = [3][6]u32{ .{ 11, 12, 13, 14, 15, 16 }, .{ 21, 22, 23, 24, 25, 26 }, .{ 31, 32, 33, 34, 35, 36 } };
    var seen: [3]Seen = .{ .{}, .{}, .{} };
    var requests: [3]api.Request = undefined;
    // all three are queued before the engine looks, so it weighs them in one pass, as a burst of requests arrives
    for (&seen, &prompts, &requests, 1..) |*x, *p, *r, i| {
        r.* = .{ .prompt = p, .max_tokens = 2 };
        try host.engine().submit(@intCast(i), r, .{ .ctx = x, .event = Seen.event });
    }
    try host.start();
    defer host.stop();
    for (&seen) |*x| try waitFor(&x.done);
    while (pair.running.load(.acquire) > 0) try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    var remotes: usize = 0;
    var busy: usize = 0;
    for (seen) |x| {
        if (std.mem.eql(u8, x.said(), "remote")) remotes += 1;
        if (std.mem.eql(u8, x.said(), "busy")) busy += 1;
    }
    try t.expectEqual(@as(usize, 2), remotes);
    try t.expectEqual(@as(usize, 1), busy);
    try t.expectEqual(@as(u32, 2), pair.fills.load(.acquire)); // two pulls ran
    try t.expect(pair.busies.load(.acquire) > 0); // and the host said what it was doing
}
