//! The boot's graph pre-capture (TF_DSV41_GRAPH_PRECAPTURE): a plan's shapes, until the reserve or the floor stops it.

const std = @import("std");
const cuda = @import("cuda");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const ph = @import("phases.zig");
const graphs = @import("graphs.zig");
const plan_mod = @import("graph_plan.zig");

/// What one planned shape came to: its graphs captured or held, left out (not graphable), or the pre-capture must stop.
pub const Step = enum { captured, held, unavailable, full, no_room, off };

pub fn stepOf(w: graphs.gc.Warmed) Step {
    return switch (w) {
        .captured => .captured,
        .held => .held,
        .full => .full,
        .no_room => .no_room,
        .off => .off,
    };
}

/// TF_DSV41_GRAPH_PRECAPTURE: on unless `0` (every rank reads the one environment).
pub fn enabled() !bool {
    return @import("sys").environment.read(bool, "TF_DSV41_GRAPH_PRECAPTURE", true);
}

/// A quarter of the cache's cap stays free for the request path, so a shape outside the plan evicts nothing.
pub fn reserveOf(cap: u32) u32 {
    return cap / 4;
}

/// The memory rule each capture passes: MemAvailable after one more graph, priced by what captures have cost here.
pub const Floor = struct {
    room: graphs.Room,
    avail: *const fn () ?f64,
    cost_gib: f64,
    measured_mib: f64 = 0,
    measures: u32 = 0,

    pub fn init(room: graphs.Room, avail: *const fn () ?f64) Floor {
        return .{ .room = room, .avail = avail, .cost_gib = room.cost_gib };
    }

    /// The price of the next graph: the configured cost, or the mean a capture has cost so far when that is dearer.
    pub fn priceGib(f: *const Floor) f64 {
        if (f.measures == 0) return f.cost_gib;
        return @max(f.cost_gib, f.measured_mib / @as(f64, @floatFromInt(f.measures)) / 1024.0);
    }

    pub fn fits(f: *const Floor) bool {
        const a = f.avail() orelse return true; // where MemAvailable cannot be read there is no floor to keep
        return f.room.fitsPriced(a, f.priceGib());
    }

    /// A capture's measured cost (MiB; below 0 is noise, not a refund).
    pub fn learned(f: *Floor, capture_mib: f64) void {
        f.measured_mib += @max(0, capture_mib);
        f.measures += 1;
    }
};

/// One shape's graphs (`parts`, split as the request path does) through `cache.warm`; `span` is the body's call range.
pub fn warmParts(cache: *graphs.Cache, floor: *Floor, parts: graphs.Parts, key: graphs.Key, span: *[2]usize, stream: cuda.Stream, fp: u64, body: graphs.gc.Body) !Step {
    var out: Step = .held;
    for (parts.spans, 0..) |sp, pi| {
        if (sp[0] == sp[1]) continue;
        span.* = sp;
        const got = stepOf(try cache.warm(parts.key(key, pi), stream, fp, body, floor.fits(), reserveOf(cache.settings.max)));
        switch (got) {
            .captured => {
                floor.learned(cache.stats.capture_mib);
                out = .captured;
            },
            .held => {},
            else => return got,
        }
    }
    return out;
}

pub const Stop = enum { none, cache_full, memory_floor, cache_off };

pub const Report = struct {
    planned: usize = 0,
    captured: usize = 0, // shapes with a graph captured here
    held: usize = 0, // shapes whose graphs were already held
    unavailable: usize = 0, // shapes a program cannot be captured for (an AOT variant missing, a traced glue step)
    left: usize = 0, // planned shapes not reached: the request path captures them if they are asked for
    stop: Stop = .none,
    at: ?plan_mod.Shape = null, // where it stopped
    ns: u64 = 0,
};

/// What a warmer tells the log: the cache's graphs, cap and price, MemAvailable here, the floor it keeps.
pub const Facts = struct { rank: u32, graphs: u64, held: u32, cap: u32, price_mib: f64, avail_gib: ?f64, keep_gib: f64 };

/// Every shape of `plan` through `w.warm(shape) !Step` until it must stop; then one summary line (`w.facts()`).
pub fn run(w: anytype, plan: plan_mod.Plan, name: []const u8) !Report {
    var rep: Report = .{ .planned = plan.count() };
    const t0 = ph.nowNs();
    var i: usize = 0;
    while (i < plan.count()) : (i += 1) {
        const s = plan.at(i);
        const step = try w.warm(s);
        switch (step) {
            .captured => rep.captured += 1,
            .held => rep.held += 1,
            .unavailable => rep.unavailable += 1,
            .full, .no_room, .off => {
                rep.stop = switch (step) {
                    .full => .cache_full,
                    .no_room => .memory_floor,
                    else => .cache_off,
                };
                rep.at = s;
                rep.left = plan.count() - i;
                break;
            },
        }
    }
    rep.ns = ph.nowNs() - t0;
    if (!quiet) report(rep, plan, name, w.facts());
    return rep;
}

fn report(r: Report, plan: plan_mod.Plan, name: []const u8, f: Facts) void {
    const log = std.log.scoped(.dsv41);
    const ms = @as(f64, @floatFromInt(r.ns)) / 1e6;
    const rows_hi = if (plan.rows.len > 0) plan.rows[plan.rows.len - 1] else 0;
    log.info("graphs: {s}: rank {d} pre-captured {d} of {d} planned shapes ({d} graphs; rows up to {d}, context buckets 0-{d}) in {d:.0} ms; {d} held of {d}, a graph costs ~{d:.0} MiB, MemAvailable {?d:.2} GiB, floor {d:.2} GiB", .{ name, f.rank, r.captured + r.held, r.planned, f.graphs, rows_hi, plan.ctx -| 1, ms, f.held, f.cap, f.price_mib, f.avail_gib, f.keep_gib });
    if (r.unavailable > 0) log.warn("graphs: {s}: rank {d}: {d} shapes have no capturable program and stay eager", .{ name, f.rank, r.unavailable });
    const at = r.at orelse return;
    const why = switch (r.stop) {
        .cache_full => "the cache is at its cap less the room kept for the request path",
        .memory_floor => "one more graph would take MemAvailable below the floor on a rank",
        .cache_off => "graphs are off",
        .none => "",
    };
    log.warn("graphs: {s}: rank {d}: pre-capture stopped at rows {d}, context bucket {d}: {s}; {d} planned shapes are left to the request path", .{ name, f.rank, at.rows, at.ctx, why, r.left });
}
