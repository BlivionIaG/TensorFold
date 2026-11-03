//! The leader's elastic memory controller: observes, plans, and applies each plan step by step on both ranks.
const std = @import("std");
const builtin = @import("builtin");
const policy = @import("elastic_policy.zig");
const config = @import("elastic_config.zig");
const rates = @import("elastic_rates.zig");

const quiet = builtin.is_test;

/// What the controller drives: this rank pair's observations and the lockstep commands.
pub const Driver = struct {
    ctx: *anyopaque,
    /// Fills `m` from this rank (K/V arena, pools, free bytes, reserve); `peer` also takes the pair's least free bytes.
    observe: *const fn (ctx: *anyopaque, m: *policy.Memory, peer: bool) anyerror!void,
    /// Both ranks' K/V at `pages`, or ElasticRefused with both as they were.
    kv: *const fn (ctx: *anyopaque, pages: usize) anyerror!void,
    /// Both ranks' lines `delta` more, or ElasticRefused.
    lines: *const fn (ctx: *anyopaque, delta: i32) anyerror!void,
    /// The last lines step taken back on both ranks.
    undo_lines: *const fn (ctx: *anyopaque) anyerror!void,
    now_ns: *const fn (ctx: *anyopaque) u64,
    /// Fills `out` with this rank's cumulative decode counters per pool (they are never used to plan).
    sample: *const fn (ctx: *anyopaque, out: *rates.Sample) anyerror!void,
};

/// What the controller counts; read with `Controller.stats`.
pub const Stats = struct {
    ticks: u64 = 0,
    plans: u64 = 0,
    steps: u64 = 0,
    /// Admissions that asked for pages (`ensure` calls that found them missing) and those that got them.
    requests: u64 = 0,
    requests_met: u64 = 0,
    /// Ticks that saw the queue lack pages and made them ahead of the open, and those that got all of them.
    forecasts: u64 = 0,
    forecasts_met: u64 = 0,
    /// Plans that failed and were taken back, and put backs that failed too.
    refused: u64 = 0,
    rollback_failures: u64 = 0,
    kv_pages_up: u64 = 0,
    kv_pages_down: u64 = 0,
    lines_up: u64 = 0,
    lines_down: u64 = 0,
    bytes_moved: u64 = 0,
    stall_ns: u64 = 0,
    stall_max_ns: u64 = 0,
    /// Decode expert counters summed over the intervals between plans (up to the last plan): for the log only.
    decode: rates.Rate = .{},
};

pub const Controller = struct {
    set: config.Settings,
    st: policy.State = .{},
    drv: Driver,
    s: Stats = .{},
    next_tick: u64,
    /// A step poisoned the pair: nothing more is attempted.
    dead: bool = false,
    /// The decode counters at the last plan; `counting` is off once the driver could not read them.
    last: rates.Sample = .{},
    counting: bool = true,

    pub fn init(set: config.Settings, drv: Driver) Controller {
        return .{ .set = set, .drv = drv, .next_tick = drv.now_ns(drv.ctx) +| set.tick_ms * std.time.ns_per_ms };
    }

    pub fn stats(c: *const Controller) Stats {
        return c.s;
    }

    /// Whether a tick is due (a clock read): the caller reads the engine's phase only then.
    pub fn due(c: *Controller) bool {
        return !c.dead and c.drv.now_ns(c.drv.ctx) >= c.next_tick;
    }

    /// What the engine loop knows at a tick: its phase, the streams running or waiting, the pages the queue lacks.
    pub const Look = struct { phase: policy.Phase = .idle, load: u32 = 0, ahead: usize = 0 };

    fn arm(m: *policy.Memory, look: Look) void {
        m.phase = look.phase;
        m.load = look.load;
        m.ahead = look.ahead;
    }

    /// Called every pass of the engine loop when quiescent, once `due`: one look at the rank, maybe a plan.
    pub fn tick(c: *Controller, look: Look) void {
        if (!c.due()) return;
        c.next_tick = c.drv.now_ns(c.drv.ctx) +| c.set.tick_ms * std.time.ns_per_ms;
        c.s.ticks += 1;
        var m: policy.Memory = undefined;
        c.drv.observe(c.drv.ctx, &m, false) catch |e| return c.fail("observe", e);
        arm(&m, look);
        c.st.advance(c.set.policy, &m);
        if (policy.plan(c.set.policy, c.st, &m, 0).n == 0) return;
        c.drv.observe(c.drv.ctx, &m, true) catch |e| return c.fail("observe", e); // the pair's smaller free bytes
        arm(&m, look);
        if (look.ahead > 0) return c.forecast(&m, look);
        const p = policy.plan(c.set.policy, c.st, &m, 0);
        if (p.n > 0) _ = c.run(&m, p);
    }

    /// The pages the queue lacks, made before its streams open: a few bounded plans, so each open finds them mapped.
    fn forecast(c: *Controller, m: *policy.Memory, look: Look) void {
        c.s.forecasts += 1;
        const want = m.kv.mapped + look.ahead;
        var rounds: usize = 0;
        while (m.kv.mapped < want and rounds < 4) : (rounds += 1) {
            m.ahead = want - m.kv.mapped;
            const p = policy.plan(c.set.policy, c.st, m, 0);
            if (p.n == 0 or !c.run(m, p)) break;
            c.drv.observe(c.drv.ctx, m, true) catch |e| return c.fail("observe", e);
            arm(m, look);
        }
        c.s.forecasts_met += @intFromBool(m.kv.mapped >= want);
    }

    /// Whether `extra` more K/V pages could be had after a plan: the admission's question, answered without moving.
    pub fn couldFit(c: *Controller, extra: usize) bool {
        if (c.dead or c.st.blocked > 0) return false;
        var m: policy.Memory = undefined;
        c.drv.observe(c.drv.ctx, &m, false) catch return false;
        return policy.canGrow(&m, extra);
    }

    /// `extra` more K/V pages made, cutting lines where free bytes do not reach; true when mapped (quiescent only).
    pub fn ensure(c: *Controller, extra: usize) bool {
        if (extra == 0) return !c.dead;
        if (c.dead) return false;
        c.s.requests += 1;
        var m: policy.Memory = undefined;
        c.drv.observe(c.drv.ctx, &m, true) catch |e| {
            c.fail("observe", e);
            return false;
        };
        const want = m.kv.mapped + extra;
        var rounds: usize = 0;
        while (m.kv.mapped < want and rounds < 4) : (rounds += 1) {
            const p = policy.plan(c.set.policy, c.st, &m, want - m.kv.mapped);
            if (p.n == 0 or !c.run(&m, p)) break;
            c.drv.observe(c.drv.ctx, &m, true) catch |e| {
                c.fail("observe", e);
                return false;
            };
        }
        const met = m.kv.mapped >= want;
        c.s.requests_met += @intFromBool(met);
        return met;
    }

    fn fail(c: *Controller, what: []const u8, e: anyerror) void {
        c.dead = true;
        if (!quiet) std.log.warn("flashnext elastic memory: {s} failed ({t}): it takes no further steps", .{ what, e });
    }

    /// Applies `p`, which was planned from `before`; false when a step was refused (the earlier ones are taken back).
    fn run(c: *Controller, before: *const policy.Memory, p: policy.Plan) bool {
        const t0 = c.drv.now_ns(c.drv.ctx);
        var kv_was: [policy.max_steps]usize = undefined;
        var pages = before.kv.mapped;
        var done: usize = 0;
        for (p.slice(), 0..) |s, i| {
            const r = switch (s) {
                .kv => |to| c.drv.kv(c.drv.ctx, to),
                .lines => |k| c.drv.lines(c.drv.ctx, k),
            };
            r catch |e| {
                if (e != error.ElasticRefused) {
                    c.fail("a step", e);
                    return false;
                }
                c.takeBack(p, kv_was[0..done]);
                c.s.refused += 1;
                c.st.failed(c.set.policy);
                return false;
            };
            kv_was[i] = pages;
            if (s == .kv) pages = s.kv;
            done = i + 1;
        }
        const stall = c.drv.now_ns(c.drv.ctx) -| t0;
        var after: policy.Memory = undefined;
        c.drv.observe(c.drv.ctx, &after, false) catch |e| {
            c.fail("observe", e);
            return true;
        };
        c.account(before, &after, p, stall);
        c.st.applied();
        return true;
    }

    /// Steps `0 .. kv_was.len` of `p` undone in reverse; a put back that fails stops there.
    fn takeBack(c: *Controller, p: policy.Plan, kv_was: []const usize) void {
        var i = kv_was.len;
        while (i > 0) {
            i -= 1;
            const r = switch (p.steps[i]) {
                .kv => c.drv.kv(c.drv.ctx, kv_was[i]),
                .lines => c.drv.undo_lines(c.drv.ctx),
            };
            r catch |e| {
                c.s.rollback_failures += 1;
                if (e != error.ElasticRefused) c.dead = true;
                if (!quiet) std.log.warn("flashnext elastic memory: a step could not be taken back ({t}): the ranks keep the state they have", .{e});
                return;
            };
        }
    }

    fn account(c: *Controller, before: *const policy.Memory, after: *const policy.Memory, p: policy.Plan, stall_ns: u64) void {
        c.s.plans += 1;
        c.s.steps += p.n;
        c.s.bytes_moved += p.moved;
        c.s.stall_ns += stall_ns;
        c.s.stall_max_ns = @max(c.s.stall_max_ns, stall_ns);
        if (after.kv.mapped >= before.kv.mapped) c.s.kv_pages_up += after.kv.mapped - before.kv.mapped else c.s.kv_pages_down += before.kv.mapped - after.kv.mapped;
        const was = before.total();
        const now = after.total();
        if (now >= was) c.s.lines_up += now - was else c.s.lines_down += was - now;
        const seen = c.decodeSince(after);
        if (quiet) return;
        std.log.info("flashnext elastic memory: {t} plan, {d} steps: K/V pages {d} -> {d} of {d}, expert lines {d} -> {d}, free {d} MiB, moved {d} MiB, stall {d:.1} ms", .{ p.reason, p.n, before.kv.mapped, after.kv.mapped, after.kv.count, was, now, after.free >> 20, p.moved >> 20, @as(f64, @floatFromInt(stall_ns)) / 1e6 });
        if (seen) |r| std.log.info("flashnext elastic memory: decode since the last plan: {f}", .{r});
    }

    /// Decode counters since the last plan, added to the totals (null, and counting stops, when unreadable).
    fn decodeSince(c: *Controller, m: *const policy.Memory) ?rates.Rate {
        if (!c.counting) return null;
        var now: rates.Sample = .{};
        c.drv.sample(c.drv.ctx, &now) catch {
            c.counting = false;
            return null;
        };
        var bytes: [policy.max_pools]u64 = undefined;
        for (m.lines(), 0..) |l, i| bytes[i] = l.bytes;
        const r = rates.since(&c.last, &now, bytes[0..m.n_pools]);
        c.last = now;
        c.s.decode.add(r);
        return r;
    }

    /// What the controller did, for the log at close.
    pub fn report(c: *const Controller) void {
        if (quiet) return;
        const s = c.s;
        if (s.decode.rounds > 0) std.log.info("flashnext elastic memory: decode up to the last plan: {f}", .{s.decode});
        std.log.info("flashnext elastic memory: {d} ticks, {d} plans ({d} steps, {d} MiB moved, {d} refused, {d} put backs failed), {d}/{d} admissions got pages, {d}/{d} forecasts met, K/V pages +{d} -{d}, expert lines +{d} -{d}, stall {d:.1} ms total {d:.1} ms worst", .{ s.ticks, s.plans, s.steps, s.bytes_moved >> 20, s.refused, s.rollback_failures, s.requests_met, s.requests, s.forecasts_met, s.forecasts, s.kv_pages_up, s.kv_pages_down, s.lines_up, s.lines_down, @as(f64, @floatFromInt(s.stall_ns)) / 1e6, @as(f64, @floatFromInt(s.stall_max_ns)) / 1e6 });
    }
};

test {
    _ = @import("cuda_exl3_elastic_test.zig");
    _ = @import("elastic_rates.zig");
}
