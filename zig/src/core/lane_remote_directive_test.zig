//! A request's own `prefill` directive: force skips the weighing and waits for the pair, never keeps the prompt here.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const remote = @import("lane_remote.zig");
const Route = remote.Route;
const Verdict = remote.Verdict;
const Directive = remote.Directive;
const t = std.testing;

const s: u64 = std.time.ns_per_s;

/// The old seeds (950 a second here, 1000 a pull, 1 s of tail), for the tests whose numbers are written for them.
fn slowMac() Route {
    return .{ .local = .{ .base = .{ .v = 950 } }, .remote = .{ .base = .{ .v = 1000 } }, .tail_s = .{ .v = 1 } };
}

test "directive: the body's word names auto, force or never and nothing else" {
    try t.expectEqual(@as(?Directive, .force), Directive.parse("force"));
    try t.expectEqual(@as(?Directive, .never), Directive.parse("never"));
    try t.expectEqual(@as(?Directive, .auto), Directive.parse("auto"));
    try t.expectEqual(@as(?Directive, null), Directive.parse("Force"));
}

test "directive: force sends a prompt the weighing would keep here, busy pair or slower pull, and never is not weighed" {
    var r: Route = slowMac();
    const ask: remote.Ask = .{ .len = 8192, .cached = 0, .decoding = 0 };
    var forced = ask;
    forced.directive = .force;
    try t.expectEqual(Verdict.slower, r.weigh(0, ask, 8191)); // 9.2 s pulled against 8.6 s here
    try t.expectEqual(Verdict.remote, r.weigh(0, forced, 8191));
    const id = r.join(0, r.estimateFor(8191, 0, 8192, .force)).?;
    try t.expectEqual(.entered, r.poll(0, id));
    try t.expectEqual(Verdict.busy, r.weigh(1 * s, ask, 8191));
    try t.expectEqual(Verdict.remote, r.weigh(1 * s, forced, 8191));
}

test "directive: a forced prompt waits for the pull under way however long it runs, an auto one goes local" {
    var r: Route = slowMac();
    const e = r.estimate(8191, 0, 8192);
    const pinned = r.estimateFor(8191, 0, 8192, .force);
    const first = r.join(0, pinned).?; // the pair has it, as the weighing would not have said
    try t.expectEqual(.entered, r.poll(0, first));
    try t.expect(pinned.pinned and !r.estimateFor(8191, 0, 8192, .auto).pinned);
    const auto = r.join(1 * s, e).?;
    const hold = r.join(1 * s, pinned).?;
    try t.expectEqual(.local, r.poll(1 * s, auto));
    try t.expectEqual(.wait, r.poll(1 * s, hold));
    try t.expectEqual(.wait, r.poll(1000 * s, hold));
    r.leave(first);
    try t.expectEqual(.entered, r.poll(1001 * s, hold));
}

/// The pair as a source: the gate a family's pull keeps, `delay_ms` per pull, and what each fill was told.
const Pair = struct {
    /// 6 tokens, a pull of 4: an idle pair is faster, a busy one is not (the old 1000 a second and 1 s tail)
    route: Route = .{ .local = .{ .base = .{ .v = 5 } }, .remote = .{ .base = .{ .v = 1000 } }, .tail_s = .{ .v = 1 } },
    delay_ms: i64 = 600,
    fills: std.atomic.Value(u32) = .init(0),
    forced: std.atomic.Value(u32) = .init(0),
    looks: std.atomic.Value(u32) = .init(0),
    running: std.atomic.Value(u32) = .init(0),
    closing: std.atomic.Value(bool) = .init(false),

    fn wants(ptr: *anyopaque, ask: remote.Ask) Verdict {
        const p: *Pair = @ptrCast(@alignCast(ptr));
        _ = p.looks.fetchAdd(1, .acq_rel);
        return p.route.weigh(Route.nowNs(t.io), ask, 4);
    }

    fn fill(ptr: *anyopaque, prompt: []const u32, cached: u64, directive: Directive, gone: *const std.atomic.Value(bool)) anyerror!?remote.Filled {
        const p: *Pair = @ptrCast(@alignCast(ptr));
        const entered = switch (p.route.admit(t.io, p.route.estimateFor(4, cached, prompt.len, directive), &p.closing, gone)) {
            .entered => |e| e,
            .local => return error.Shed,
            .stop => return null,
        };
        defer p.route.leave(entered.id);
        _ = p.fills.fetchAdd(1, .acq_rel);
        if (directive == .force) _ = p.forced.fetchAdd(1, .acq_rel);
        try std.Io.sleep(t.io, .fromMilliseconds(p.delay_ms), .awake);
        const kept = try t.allocator.create(std.ArrayList(u32));
        kept.* = .empty;
        errdefer t.allocator.destroy(kept);
        try kept.appendSlice(t.allocator, prompt[0..4]);
        return .{ .saved = kept, .at = 4 };
    }

    fn drop(_: *anyopaque, saved: pc.Saved) void {
        const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
        kept.deinit(t.allocator);
        t.allocator.destroy(kept);
    }

    fn source(p: *Pair) remote.Source {
        return .{ .ptr = p, .wants = wants, .fill = fill, .drop = drop, .running = &p.running };
    }
};

/// A request's own clock: when its prompt was in, and whether it ended.
const Timed = struct {
    began: i96 = 0,
    prefilled_ms: std.atomic.Value(i64) = .init(-1),
    done: std.atomic.Value(bool) = .init(false),

    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const x: *Timed = @ptrCast(@alignCast(ctx));
        switch (ev.*) {
            .prefilled => x.prefilled_ms.store(@intCast(@divTrunc(std.Io.Clock.awake.now(t.io).toNanoseconds() - x.began, std.time.ns_per_ms)), .release),
            .finished => x.done.store(true, .release),
            else => {},
        }
    }
};

fn wait(flag: *const std.atomic.Value(bool)) !void {
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 10 * std.time.ns_per_s;
    while (!flag.load(.acquire)) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() > until) return error.RemoteFixtureTimeout;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
}

/// What the pair saw, and the second prompt's prefill time (ms): it is sent 50 ms into the first one's pull.
const Run = struct { prefilled_ms: i64, fills: u32, forced: u32, looks: u32 };

/// Two 6 token prompts on a two lane host, the first automatic, the second with `directive`.
fn twoPrompts(directive: Directive) !Run {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 2 });
    var pair: Pair = .{};
    host.remote = pair.source();
    try host.start();
    defer host.stop();
    const one = [_]u32{ 11, 12, 13, 14, 15, 16 };
    const two = [_]u32{ 21, 22, 23, 24, 25, 26 };
    var a: Timed = .{ .began = std.Io.Clock.awake.now(t.io).toNanoseconds() };
    try host.engine().submit(1, &.{ .prompt = &one, .max_tokens = 2, .prefill = .force }, .{ .ctx = &a, .event = Timed.event });
    while (pair.fills.load(.acquire) == 0) try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    try std.Io.sleep(t.io, .fromMilliseconds(50), .awake);
    var b: Timed = .{ .began = std.Io.Clock.awake.now(t.io).toNanoseconds() };
    try host.engine().submit(2, &.{ .prompt = &two, .max_tokens = 2, .prefill = directive }, .{ .ctx = &b, .event = Timed.event });
    try wait(&a.done);
    try wait(&b.done);
    while (pair.running.load(.acquire) > 0) try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    return .{ .prefilled_ms = b.prefilled_ms.load(.acquire), .fills = pair.fills.load(.acquire), .forced = pair.forced.load(.acquire), .looks = pair.looks.load(.acquire) };
}

test "directive: on the lane host a forced prompt waits for the pair, an automatic one goes local, a never prompt is not even weighed" {
    const auto = try twoPrompts(.auto);
    try t.expect(auto.prefilled_ms >= 0 and auto.prefilled_ms < 300); // the pull takes 600 ms
    try t.expectEqual(@as(u32, 1), auto.fills);
    try t.expectEqual(@as(u32, 1), auto.forced); // the first prompt's own force reached the fill
    const forced = try twoPrompts(.force);
    try t.expect(forced.prefilled_ms >= 500); // behind the first pull, then its own
    try t.expectEqual(@as(u32, 2), forced.fills);
    try t.expectEqual(@as(u32, 2), forced.forced);
    const never = try twoPrompts(.never);
    try t.expect(never.prefilled_ms >= 0 and never.prefilled_ms < 300);
    try t.expectEqual(@as(u32, 1), never.fills);
    try t.expectEqual(@as(u32, 1), never.looks); // only the first prompt was asked about
}
