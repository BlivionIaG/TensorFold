//! Keyed CUDA graphs for any family: one captured executable per key, launched again when the key comes back.

const std = @import("std");
const graph = @import("graph.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const registry = @import("graph_registry.zig");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const misses = @import("graph_cache_misses.zig");

/// One step's launches on `stream`; it must not synchronize, allocate device memory or read the device on the host.
pub const Body = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, stream: Stream) anyerror!void,
};

/// Every rank's verdict AND-ed on a miss, never a hit; an off cache never votes, so ranks match `Settings.digest`.
pub const Agree = struct {
    ctx: *anyopaque,
    all: *const fn (ctx: *anyopaque, ok: bool) anyerror!bool,
};

/// Capture a body into an executable graph, launch it, free it: a fake engine runs the policy in host tests.
pub const Engine = struct {
    ctx: *anyopaque,
    capture: *const fn (ctx: *anyopaque, stream: Stream, body: Body) anyerror!*anyopaque,
    launch: *const fn (ctx: *anyopaque, exec: *anyopaque, stream: Stream) anyerror!void,
    free: *const fn (ctx: *anyopaque, exec: *anyopaque) void,
};

pub const Settings = struct {
    on: bool = true, // off: every step runs eagerly
    max: u32 = 48, // graphs held at most
    min_keep: u32 = 8, // under the memory floor the cache evicts down to this many, then stops capturing
    hold: bool = false, // under the floor a miss runs eagerly with no eviction: held graphs cost next to no memory

    /// These settings under the operator's TF_GRAPH_CACHE value: `0` or `off` runs every step eagerly; else unchanged.
    pub fn operator(s: Settings, value: ?[]const u8) Settings {
        var out = s;
        const v = value orelse return out;
        if (std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off")) out.on = false;
        return out;
    }

    /// What a cache runs with: the operator's value, min_keep under max (0 with hold), and every off cache one value.
    pub fn effective(s: Settings, value: ?[]const u8) Settings {
        var out = s.operator(value);
        if (!out.on or out.max == 0) return .{ .on = false, .max = 0, .min_keep = 0, .hold = false };
        out.min_keep = if (out.hold) 0 else @min(out.min_keep, out.max);
        return out;
    }

    /// `effective` under this process's TF_GRAPH_CACHE.
    pub fn fromEnv(s: Settings) Settings {
        return s.effective(if (std.c.getenv("TF_GRAPH_CACHE")) |v| std.mem.span(v) else null);
    }

    /// The effective settings as bytes for ranks to compare at start (a family's bootstrap agreement), never per step.
    pub fn digest(s: Settings) [10]u8 {
        const e = s.effective(null);
        var out: [10]u8 = undefined;
        out[0] = @intFromBool(e.on);
        out[1] = @intFromBool(e.hold);
        std.mem.writeInt(u32, out[2..6], e.max, .little);
        std.mem.writeInt(u32, out[6..10], e.min_keep, .little);
        return out;
    }
};

pub const Outcome = enum { replayed, captured, eager };

/// Eager steps under the floor between two "still under" lines.
pub const report_every = 256;

/// One cache's name and counts as other threads read them (/health); `snapshots` copies the caches in use.
pub const Snapshot = registry.Snapshot;
pub const snapshots = registry.snapshots;

pub const Stats = struct {
    replayed: u64 = 0,
    captured: u64 = 0, // every graph made: on the request path, and at boot (`precaptured`)
    eager: u64 = 0,
    evicted: u64 = 0,
    dropped: u64 = 0,
    floor_hits: u64 = 0,
    capture_mib: f64 = 0,
    misses: u64 = 0, // steps that found no graph: captured on the request path, or run eagerly
    precaptured: u64 = 0,
    request_capture_ns: u64 = 0, // host time the request path spent capturing, instantiating and uploading
    precapture_ns: u64 = 0,
};

/// What `warm` did with a key (`full`: the cache is at its cap less the caller's reserve).
pub const Warmed = enum { captured, held, full, no_room, off };

/// `Cache.summary`'s counts as one log line fragment.
pub const Summary = struct {
    stats: Stats,
    held: u32,
    failed: bool,

    pub fn format(s: Summary, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const t = s.stats;
        const ms = struct {
            fn of(ns: u64) f64 {
                return @as(f64, @floatFromInt(ns)) / 1e6;
            }
        }.of;
        try w.print("{d} hits, {d} misses ({d} captured on the request path in {d:.1} ms, {d} eager), {d} pre-captured in {d:.1} ms, {d} keys held, {d} evicted, {d} dropped{s}", .{
            t.replayed, t.misses, t.captured - t.precaptured, ms(t.request_capture_ns), t.eager, t.precaptured, ms(t.precapture_ns), s.held, t.evicted, t.dropped, if (s.failed) " (off after a failed capture)" else "",
        });
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub fn Cache(comptime Key: type) type {
    return struct {
        const Self = @This();
        const Entry = struct { exec: *anyopaque, used: u64 };

        gpa: std.mem.Allocator,
        engine: Engine,
        settings: Settings,
        agree: ?Agree = null,
        room: ?*const fn () bool = null, // false when this rank is below its memory floor; null: always room
        meter: ?*const fn () ?f64 = null, // MemAvailable now in GiB, to log a capture's cost; null: not measured
        name: []const u8 = "graphs", // the cache's name in its log lines and snapshot
        entries: std.AutoArrayHashMapUnmanaged(Key, Entry) = .empty,
        cap: u32,
        tick: u64 = 0,
        fingerprint: ?u64 = null,
        failed: bool = false, // a capture failed on some rank: eager from here on
        under: bool = false, // the last miss met no room on some rank
        under_eager: u64 = 0, // eager steps since `under` began
        slot: ?usize = null, // this cache's registry slot, claimed at its first step; null: every slot taken
        claimed: bool = false,
        stats: Stats = .{},
        miss_log: misses.Log(Key) = .{},

        /// The cache with `settings.fromEnv()`: every rank sharing an `agree` must hold an equal `Settings.digest`.
        pub fn init(gpa: std.mem.Allocator, engine: Engine, settings: Settings) Self {
            const s = settings.fromEnv();
            if (!quiet and settings.on and settings.max > 0 and !s.on) std.log.info("graph cache: off (TF_GRAPH_CACHE), every step eager", .{});
            return .{ .gpa = gpa, .engine = engine, .settings = s, .cap = s.max };
        }

        pub fn deinit(c: *Self) void {
            if (c.slot) |i| registry.release(i);
            c.dropAll();
            c.entries.deinit(c.gpa);
            c.miss_log.deinit(c.gpa);
        }

        pub fn count(c: *const Self) usize {
            return c.entries.count();
        }

        pub fn has(c: *const Self, key: Key) bool {
            return c.entries.contains(key);
        }

        /// Every graph freed (a fingerprint change, a shutdown).
        pub fn dropAll(c: *Self) void {
            for (c.entries.values()) |e| c.engine.free(c.engine.ctx, e.exec);
            c.stats.dropped += c.entries.count();
            c.entries.clearRetainingCapacity();
        }

        fn agreed(c: *Self, ok: bool) !bool {
            const a = c.agree orelse return ok;
            return a.all(a.ctx, ok);
        }

        fn evictOldest(c: *Self) void {
            var at: ?usize = null;
            for (c.entries.values(), 0..) |e, i| {
                if (at == null or e.used < c.entries.values()[at.?].used) at = i;
            }
            const i = at orelse return;
            c.engine.free(c.engine.ctx, c.entries.values()[i].exec);
            c.entries.swapRemoveAt(i);
            c.stats.evicted += 1;
        }

        fn publish(c: *Self) void {
            if (!c.claimed) {
                c.claimed = true;
                c.slot = registry.claim(c.name);
            }
            const i = c.slot orelse return;
            registry.publish(i, .{ .held = @intCast(c.entries.count()), .captured = c.stats.captured, .replayed = c.stats.replayed, .eager = c.stats.eager, .floor_hits = c.stats.floor_hits, .under = c.under, .misses = c.stats.misses, .precaptured = c.stats.precaptured, .request_capture_ns = c.stats.request_capture_ns });
        }

        fn avail(c: *const Self) ?f64 {
            return if (c.meter) |m| m() else null;
        }

        /// A miss without room on some rank: the floor entered (logged), or a "still under" line every `report_every`.
        fn underFloor(c: *Self, room_here: bool) void {
            c.stats.floor_hits += 1;
            c.under_eager += 1;
            if (!c.under) {
                c.under = true;
                c.under_eager = 1;
                if (!quiet) std.log.warn("graph cache {s}: under the memory floor on {s} rank (MemAvailable {?d:.2} GiB here): new keys eager, {d} graphs held", .{ c.name, if (room_here) "another" else "this", c.avail(), c.entries.count() });
            } else if (c.under_eager % report_every == 0) {
                if (!quiet) std.log.warn("graph cache {s}: still under the memory floor: {d} eager steps since it began (MemAvailable {?d:.2} GiB here, {d} graphs held)", .{ c.name, c.under_eager, c.avail(), c.entries.count() });
            }
        }

        fn eager(c: *Self, stream: Stream, body: Body) !Outcome {
            try body.run(body.ctx, stream);
            c.stats.eager += 1;
            return .eager;
        }

        /// One step: the key's graph, captured first on a miss; a new `fingerprint` (same step on all ranks) drops all.
        pub fn run(c: *Self, key: Key, stream: Stream, fingerprint: u64, body: Body) !Outcome {
            defer c.publish();
            if (!c.settings.on or c.failed or c.settings.max == 0) return c.eager(stream, body);
            if (c.fingerprint != null and c.fingerprint.? != fingerprint) c.dropAll();
            c.fingerprint = fingerprint;
            c.tick += 1;
            if (c.entries.getPtr(key)) |e| {
                e.used = c.tick;
                try c.engine.launch(c.engine.ctx, e.exec, stream);
                c.stats.replayed += 1;
                return .replayed;
            }
            c.stats.misses += 1; // no graph from here: captured on the request path, or the step runs eagerly
            // a miss: room on every rank (a host map that cannot grow is none), else shed a quarter and lower the cap
            const reserved = if (c.entries.ensureUnusedCapacity(c.gpa, 1)) |_| true else |_| false;
            const room_here = reserved and (if (c.room) |f| f() else true);
            if (!try c.agreed(room_here)) {
                c.underFloor(room_here);
                if (c.settings.hold) return c.eager(stream, body);
                if (c.entries.count() > c.settings.min_keep) {
                    const drop = @max(1, c.entries.count() / 4);
                    for (0..drop) |_| if (c.entries.count() > c.settings.min_keep) c.evictOldest();
                }
                const was = c.cap;
                c.cap = @intCast(@max(c.entries.count(), c.settings.min_keep)); // min_keep <= max (init)
                if (!quiet and c.cap != was) std.log.warn("graph cache: under the memory floor on {s} rank: cap {d} -> {d} ({d} held, {d} evicted, {d} eager so far)", .{ if (room_here) "another" else "this", was, c.cap, c.entries.count(), c.stats.evicted, c.stats.eager + 1 });
                return c.eager(stream, body);
            }
            if (c.under) {
                if (!quiet) std.log.info("graph cache {s}: room again on every rank (MemAvailable {?d:.2} GiB here) after {d} eager steps under the floor, {d} graphs held", .{ c.name, c.avail(), c.under_eager, c.entries.count() });
                c.under = false;
            }
            if (c.cap < c.settings.max) {
                if (!quiet) std.log.info("graph cache: room again on every rank: cap {d} -> {d}", .{ c.cap, c.settings.max });
                c.cap = c.settings.max;
            }
            while (c.entries.count() >= c.cap) c.evictOldest();
            const before = c.avail();
            const t0 = nowNs();
            const exec = (try c.captureAgreed(key, stream, body)) orelse return c.eager(stream, body);
            const took = nowNs() - t0;
            c.entries.putAssumeCapacityNoClobber(key, .{ .exec = exec, .used = c.tick });
            try c.engine.launch(c.engine.ctx, exec, stream);
            c.stats.captured += 1;
            c.stats.request_capture_ns += took;
            c.noteCapture(key, took, before);
            return .captured;
        }

        /// `key`'s body captured and every rank agreeing it was; null: some rank failed, so the cache is off and empty.
        fn captureAgreed(c: *Self, key: Key, stream: Stream, body: Body) !?*anyopaque {
            const exec = c.engine.capture(c.engine.ctx, stream, body) catch |err| blk: {
                // the only trace of why graphs went off (the step then runs eagerly, which may fail the same way)
                if (!quiet) std.log.warn("graph capture failed: {s} (key {any}, {d} graphs held)", .{ @errorName(err), key, c.entries.count() });
                break :blk null;
            };
            errdefer if (exec) |x| c.engine.free(c.engine.ctx, x); // a failed agreement cannot leak it
            if (!try c.agreed(exec != null)) {
                if (exec) |x| c.engine.free(c.engine.ctx, x);
                c.failed = true;
                c.dropAll();
                return null;
            }
            return exec;
        }

        /// MemAvailable after a capture that began at `before`; `capture_mib` is what it cost this rank.
        fn priced(c: *Self, before: ?f64) ?f64 {
            const b = before orelse return null;
            const a = c.avail() orelse return null;
            c.stats.capture_mib = (b - a) * 1024;
            return a;
        }

        /// A request-path capture's price, and its log line (the host ms it stalled the step) at each key's first.
        fn noteCapture(c: *Self, key: Key, ns: u64, before: ?f64) void {
            const after = c.priced(before);
            if (!c.miss_log.first(c.gpa, key, c.stats.captured - c.stats.precaptured) or quiet) return;
            const ms = @as(f64, @floatFromInt(ns)) / 1e6;
            if (before != null and after != null) {
                std.log.info("graph cache {s}: capture on the request path: key {any}: {d:.1} ms, {d} held, MemAvailable {d:.2} -> {d:.2} GiB ({d:.0} MiB)", .{ c.name, key, ms, c.entries.count(), before.?, after.?, c.stats.capture_mib });
            } else std.log.info("graph cache {s}: capture on the request path: key {any}: {d:.1} ms, {d} held", .{ c.name, key, ms, c.entries.count() });
        }

        /// `key`'s graph captured, not run or launched (boot pre-capture): `full` within `reserve` of the cap.
        pub fn warm(c: *Self, key: Key, stream: Stream, fingerprint: u64, body: Body, fits: bool, reserve: u32) !Warmed {
            defer c.publish();
            if (!c.settings.on or c.failed or c.settings.max == 0) return .off;
            if (c.fingerprint != null and c.fingerprint.? != fingerprint) c.dropAll();
            c.fingerprint = fingerprint;
            if (c.entries.contains(key)) return .held;
            if (c.entries.count() + reserve >= c.cap) return .full;
            const reserved = if (c.entries.ensureUnusedCapacity(c.gpa, 1)) |_| true else |_| false;
            if (!try c.agreed(reserved and fits and (if (c.room) |f| f() else true))) return .no_room;
            const before = c.avail();
            const t0 = nowNs();
            const exec = (try c.captureAgreed(key, stream, body)) orelse return .off;
            c.tick += 1;
            c.entries.putAssumeCapacityNoClobber(key, .{ .exec = exec, .used = c.tick });
            c.stats.captured += 1;
            c.stats.precaptured += 1;
            c.stats.precapture_ns += nowNs() - t0;
            _ = c.priced(before);
            return .captured;
        }

        /// The counts a cache logs at its end (`{f}`): hits and misses, what was captured where, what it holds.
        pub fn summary(c: *const Self) Summary {
            return .{ .stats = c.stats, .held = @intCast(c.entries.count()), .failed = c.failed };
        }
    };
}

/// The driver's engine: thread-local stream capture (other threads' CUDA calls stay out), instantiate, upload, launch.
pub const CudaEngine = struct {
    d: *const Driver,

    pub fn engine(e: *CudaEngine) Engine {
        return .{ .ctx = e, .capture = capture, .launch = launch, .free = free };
    }

    const Held = struct { exec: graph.Exec };

    fn capture(ctx: *anyopaque, stream: Stream, body: Body) anyerror!*anyopaque {
        const e: *CudaEngine = @ptrCast(@alignCast(ctx));
        try graph.beginCapture(stream, .thread_local);
        body.run(body.ctx, stream) catch |err| {
            // end the capture so the stream is usable again; the partial graph is dropped
            if (graph.endCapture(stream)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return err;
        };
        var g = try graph.endCapture(stream);
        defer g.deinit();
        const h = try std.heap.c_allocator.create(Held);
        errdefer std.heap.c_allocator.destroy(h);
        h.exec = try g.instantiate();
        errdefer h.exec.deinit();
        try h.exec.upload(stream);
        _ = e;
        return h;
    }

    fn launch(_: *anyopaque, exec: *anyopaque, stream: Stream) anyerror!void {
        const h: *Held = @ptrCast(@alignCast(exec));
        try h.exec.launchOn(stream);
    }

    fn free(_: *anyopaque, exec: *anyopaque) void {
        const h: *Held = @ptrCast(@alignCast(exec));
        h.exec.deinit();
        std.heap.c_allocator.destroy(h);
    }
};

/// FNV-1a over the addresses a body captures (buffer bases, sizes): the fingerprint `run` compares, on one rank only.
pub fn fingerprintOf(words: []const u64) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (words) |w| {
        var x = w;
        for (0..8) |_| {
            h = (h ^ (x & 0xff)) *% 0x100000001b3;
            x >>= 8;
        }
    }
    return h;
}

test {
    _ = @import("graph_cache_policy_test.zig");
    _ = @import("graph_cache_regression_test.zig");
    _ = @import("graph_cache_ranks_test.zig");
    _ = @import("graph_cache_shapes_test.zig");
}
