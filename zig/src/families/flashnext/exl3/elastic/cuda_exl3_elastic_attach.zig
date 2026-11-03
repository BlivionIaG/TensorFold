//! Elastic memory on a loaded rank: its pools as `Lines`, its K/V arena, and on the leader the controller driving both.
const std = @import("std");
const cuda = @import("cuda");
const kvl = @import("../../cuda_kv_lanes.zig");
const wire = @import("../../cuda_rank_wire.zig");
const policy = @import("elastic_policy.zig");
const config = @import("elastic_config.zig");
const ctl = @import("cuda_exl3_elastic.zig");
const rates = @import("elastic_rates.zig");
const api = @import("engine_api");
const engine_lines = @import("cuda_exl3_lines_engine.zig");
const pool_ = @import("../offload/cuda_exl3_pool.zig");
const pass = @import("../model/cuda_exl3_pass.zig");

/// Queued requests a tick looks at: more than a few lanes' worth are not forecast.
const look_asks = 8;

pub const Elastic = struct {
    gpa: std.mem.Allocator,
    pools: *engine_lines.Engine,
    arena: *kvl.Pool,
    /// The rank pair's session (null with one rank).
    ranks: ?*@import("../../cuda_rank_engine.zig").Session,
    d: *const cuda.Driver,
    /// Bytes that stay free: the pools' reserve and the allocator margin.
    reserve: u64,
    /// The leader's (or the lone rank's) controller; null on a follower, which only applies commands.
    c: ?ctl.Controller = null,

    pub fn destroy(x: *Elastic) void {
        if (x.c) |*c| c.report();
        x.pools.destroy();
        x.gpa.destroy(x);
    }

    fn driver(x: *Elastic) ctl.Driver {
        return .{ .ctx = x, .observe = observe, .kv = kv, .lines = lines, .undo_lines = undoLines, .now_ns = now, .sample = sample };
    }

    fn observe(ctx: *anyopaque, m: *policy.Memory, peer: bool) anyerror!void {
        const x: *Elastic = @ptrCast(@alignCast(ctx));
        const v = x.arena.view() orelse return error.BadFlashNextElasticMemory;
        m.* = .{ .kv = .{ .mapped = v.mapped, .count = v.count, .min = v.min, .floor = v.floor, .demand = v.demand, .stride = v.stride, .chunk = v.chunk }, .free = try pass.freeBytes(x.d), .reserve = x.reserve };
        m.n_pools = x.pools.lines.shapes(&m.pools);
        if (peer) if (x.ranks) |r| {
            m.free = @min(m.free, try r.peerFree());
        };
    }

    fn kv(ctx: *anyopaque, pages: usize) anyerror!void {
        const x: *Elastic = @ptrCast(@alignCast(ctx));
        if (x.ranks) |r| return r.resizeKv(pages);
        x.arena.resizeMapped(pages) catch return error.ElasticRefused; // it changes nothing when it fails
    }

    fn lines(ctx: *anyopaque, delta: i32) anyerror!void {
        const x: *Elastic = @ptrCast(@alignCast(ctx));
        if (x.ranks) |r| return r.resizeLines(delta);
        return x.pools.lines.apply(delta);
    }

    fn undoLines(ctx: *anyopaque) anyerror!void {
        const x: *Elastic = @ptrCast(@alignCast(ctx));
        if (x.ranks) |r| return r.undoLines();
        return x.pools.lines.undoLast();
    }

    fn sample(ctx: *anyopaque, out: *rates.Sample) anyerror!void {
        const x: *Elastic = @ptrCast(@alignCast(ctx));
        return x.pools.sample(out);
    }

    fn now(_: *anyopaque) u64 {
        return wire.now();
    }
};

/// Bytes of K/V pages the arena keeps at least (kvel's default minimum, or TF_FLASHNEXT_KV_ELASTIC_MIN_PAGES).
pub fn kvFloorBytes(l: @import("core").kv_pool.Layout, lanes: u32) !u64 {
    const named = (try @import("../../cuda_kv_elastic.zig").fromEnv()).?.min_pages;
    const pages = @max(1, named orelse (lanes -| 1) * (l.runPages() + 1));
    return std.math.mul(u64, pages, l.pageBytes());
}

/// Makes the rank's elastic memory when ELASTIC_MEMORY=1: its lines go on the session, the leader gets a controller.
pub fn attach(own: anytype) !void {
    if (!try config.enabled()) return;
    const set = try config.fromEnv();
    const pool = own.kv.pool orelse return error.ElasticMemoryNeedsPagedKv;
    if (pool.view() == null) return error.ElasticMemoryNeedsBothElasticModes;
    const lines = (try engine_lines.Engine.create(own.gpa, own.engine, pool_.min_lines)) orelse return error.ElasticMemoryNoExpertPools;
    errdefer lines.destroy();
    const x = try own.gpa.create(Elastic);
    errdefer own.gpa.destroy(x);
    x.* = .{ .gpa = own.gpa, .pools = lines, .arena = pool, .ranks = own.ranks, .d = own.engine.driver, .reserve = own.elastic.reserve };
    if (own.ranks) |r| r.lines = &lines.lines;
    if (own.ranks == null or own.ranks.?.rank == 0) x.c = ctl.Controller.init(set, x.driver());
    own.elastic.x = x;
    std.log.info("flashnext elastic memory: {d} expert pools of {d}..{d} lines, K/V {d} of {d} pages; tick {d} ms, K/V use {d}% grows it and {d}% for {d} ticks lets lines grow, steps of {d} MiB", .{ lines.n, lines.lines.floor[0], lines.pools[0].opt.lines, pool.mappedPages(), pool.view().?.count, set.tick_ms, set.policy.high_ppm / 10_000, set.policy.low_ppm / 10_000, set.policy.idle_ticks, set.policy.step_bytes >> 20 });
}

/// What the lanes backend holds of elastic memory: the reserve the pools left (set at load) and the rank's `Elastic`.
pub const Slot = struct {
    reserve: u64 = 0,
    x: ?*Elastic = null,

    /// The engine loop's tick, on the leader only (a follower applies commands and never decides).
    pub fn tick(s: *Slot, own: anytype, q: api.Queue) void {
        const x = s.x orelse return;
        const c = if (x.c) |*c| c else return;
        if (!c.due()) return; // the lanes and the queue are read only when a tick is due
        var asks: [look_asks]api.Ask = undefined;
        const snap = q.snapshot(&asks);
        const next = asks[0..@min(snap.copied, snap.lanes -| snap.running)]; // the requests the free lanes take next
        c.tick(.{ .phase = phaseOf(own), .load = snap.running + snap.queued, .ahead = own.kv.shortAll(next) });
    }

    /// A stream still filling its prompt: prefill; else any stream: decode; else idle.
    fn phaseOf(own: anytype) policy.Phase {
        var any = false;
        var it = own.lanes.valueIterator();
        while (it.next()) |l| {
            if (l.filled != null) return .prefill;
            any = true;
        }
        return if (any) .decode else .idle;
    }

    /// Before a stream opens: the pages it lacks mapped by a plan when it can; else the open says LaneMemory as before.
    pub fn room(s: *Slot, kv: *const kvl.State, cap: usize, from: kvl.From) void {
        if (from != .pool) return;
        const x = s.x orelse return;
        const pool = kv.pool orelse return;
        const need = pool.short(cap);
        if (need > 0) if (x.c) |*c| {
            _ = c.ensure(need);
        };
    }

    pub fn destroy(s: *Slot) void {
        if (s.x) |x| x.destroy();
        s.x = null;
    }
};

/// Admission with elastic memory: a request the mapped pages do not hold is admitted when a plan could map them.
pub fn fits(own: anytype, rows: u32, max_tokens: u32) bool {
    if (own.kv.fits(rows, max_tokens)) return true;
    const x = own.elastic.x orelse return false;
    const c = if (x.c) |*c| c else return false;
    const need = own.kv.short(rows, max_tokens);
    return need > 0 and c.couldFit(need);
}
