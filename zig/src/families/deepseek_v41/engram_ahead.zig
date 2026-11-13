//! A prompt's Engram rows gathered by a worker while the GPU runs the segment before: same function, same bytes.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const eh = @import("engram_host.zig");

const log = std.log.scoped(.dsv41);

/// Windows this small are decode's (the forward's `hs.chunk`): never read ahead, never counted.
pub const min_rows: usize = 128;

/// Windows kept at once (queued, running, or gathered and not yet taken): the one the GPU runs and the one after.
const kept_windows = 2;

/// The layer of a job that only starts the window's reads on every layer; nobody takes it.
const warm_layer = std.math.maxInt(u32);

const State = enum { queued, running, done, failed };

const Job = struct {
    layer: u32,
    ids: []u32,
    lookback: []u32,
    rows: []u16 = &.{},
    state: State = .queued,
};

pub const Stats = struct {
    /// windows' layers whose rows came from the worker, were not there (gathered inline) or failed there
    took: u64 = 0,
    missed: u64 = 0,
    failed: u64 = 0,
    /// ns the launch thread waited for the worker, and the worker spent gathering
    wait_ns: u64 = 0,
    gather_ns: u64 = 0,
};

/// The table under one lock: the worker's reads and every other read of the forward while the worker lives.
pub const Locked = struct {
    inner: eh.Source,
    lock: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,

    pub fn source(l: *Locked) eh.Source {
        return .{ .ptr = l, .vtable = &.{ .issue = issueFn, .rows = rowsFn } };
    }

    fn issueFn(p: *anyopaque, li: usize, idx: []const i64) anyerror!void {
        const l: *Locked = @ptrCast(@alignCast(p));
        _ = std.c.pthread_mutex_lock(&l.lock);
        defer _ = std.c.pthread_mutex_unlock(&l.lock);
        return l.inner.vtable.issue(l.inner.ptr, li, idx);
    }

    fn rowsFn(p: *anyopaque, li: usize, idx: []const i64, dim: usize, out: []u16) anyerror!void {
        const l: *Locked = @ptrCast(@alignCast(p));
        _ = std.c.pthread_mutex_lock(&l.lock);
        defer _ = std.c.pthread_mutex_unlock(&l.lock);
        return l.inner.vtable.rows(l.inner.ptr, li, idx, dim, out);
    }
};

/// The forward's handle: opened by the first prompt that reads ahead; `off` once it may not be (knob, gate) or cannot.
pub const Handle = struct { ahead: ?*Ahead = null, off: bool = false };

pub const Ahead = struct {
    a: std.mem.Allocator,
    host: *const eh.Host,
    table: Locked,
    rank: u32,
    world: u32,
    dim: usize,
    /// this rank's columns of a row (its heads x dim)
    cols: usize,
    mu: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    work: std.c.pthread_cond_t = .{},
    done: std.c.pthread_cond_t = .{},
    /// oldest first: the launch thread alone adds, takes and drops; the worker runs the first queued one
    jobs: std.ArrayList(*Job) = .empty,
    /// gathered buffers kept for the next window (the same size every segment)
    spare: std.ArrayList([]u16) = .empty,
    stop: bool = false,
    thread: ?std.Thread = null,
    stats: Stats = .{},

    /// The worker over `inner` (the rank's table), `a` safe to use from two threads.
    pub fn open(a: std.mem.Allocator, host: *const eh.Host, inner: eh.Source, rank: u32, world: u32, dim: usize) !*Ahead {
        if (world == 0 or rank >= world or host.layer_ids.len == 0) return error.BadEngramHost;
        const g = try a.create(Ahead);
        errdefer a.destroy(g);
        g.* = .{ .a = a, .host = host, .table = .{ .inner = inner }, .rank = rank, .world = world, .dim = dim, .cols = host.headShard(rank, world)[1] * dim };
        g.thread = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, loop, .{g});
        return g;
    }

    /// The worker stopped after the job it runs; everything held freed.
    pub fn close(g: *Ahead) void {
        _ = std.c.pthread_mutex_lock(&g.mu);
        g.stop = true;
        _ = std.c.pthread_cond_broadcast(&g.work);
        _ = std.c.pthread_mutex_unlock(&g.mu);
        if (g.thread) |t| t.join();
        for (g.jobs.items) |j| g.free(j);
        g.jobs.deinit(g.a);
        for (g.spare.items) |b| g.a.free(b);
        g.spare.deinit(g.a);
        g.a.destroy(g);
    }

    /// The table every other Engram read of the forward goes through while the worker lives.
    pub fn source(g: *Ahead) eh.Source {
        return g.table.source();
    }

    /// Jobs held (queued, running, or gathered and not yet taken).
    pub fn held(g: *Ahead) usize {
        _ = std.c.pthread_mutex_lock(&g.mu);
        defer _ = std.c.pthread_mutex_unlock(&g.mu);
        return g.jobs.items.len;
    }

    fn free(g: *Ahead, j: *Job) void {
        g.a.free(j.ids);
        g.a.free(j.lookback);
        if (j.rows.len > 0) g.a.free(j.rows);
        g.a.destroy(j);
    }

    fn same(j: *const Job, layer: u32, ids: []const u32, lookback: []const u32) bool {
        return j.layer == layer and std.mem.eql(u32, j.ids, ids) and std.mem.eql(u32, j.lookback, lookback);
    }

    fn find(g: *Ahead, layer: u32, ids: []const u32, lookback: []const u32) ?*Job {
        for (g.jobs.items) |j| if (same(j, layer, ids, lookback)) return j;
        return null;
    }

    fn push(g: *Ahead, layer: u32, ids: []const u32, lookback: []const u32) !void {
        const j = try g.a.create(Job);
        errdefer g.a.destroy(j);
        const own = try g.a.dupe(u32, ids);
        errdefer g.a.free(own);
        const lb = try g.a.dupe(u32, lookback);
        errdefer g.a.free(lb);
        j.* = .{ .layer = layer, .ids = own, .lookback = lb };
        try g.jobs.append(g.a, j);
    }

    /// The oldest job that is not running, dropped; false when there is none.
    fn evictOldest(g: *Ahead) bool {
        for (g.jobs.items, 0..) |j, i| if (j.state != .running) {
            _ = g.jobs.orderedRemove(i);
            g.free(j);
            return true;
        };
        return false;
    }

    /// Queues the window's rows on every Engram layer, all their reads first; a held window is not queued again.
    pub fn submit(g: *Ahead, ids: []const u32, lookback: []const u32) void {
        if (ids.len <= min_rows) return;
        _ = std.c.pthread_mutex_lock(&g.mu);
        defer _ = std.c.pthread_mutex_unlock(&g.mu);
        const layers = g.host.layer_ids;
        if (g.stop or g.find(layers[0], ids, lookback) != null) return;
        while (g.jobs.items.len + layers.len + 1 > kept_windows * (layers.len + 1)) if (!g.evictOldest()) return;
        defer _ = std.c.pthread_cond_broadcast(&g.work); // a window queued in part still wakes the worker
        g.push(warm_layer, ids, lookback) catch return;
        for (layers) |L| g.push(L, ids, lookback) catch return;
    }

    /// Layer `layer`'s rows of the window into `out` once the worker has them (waits); false: gather them inline.
    pub fn take(g: *Ahead, layer: u32, ids: []const u32, lookback: []const u32, out: []u16) bool {
        if (ids.len <= min_rows) return false;
        _ = std.c.pthread_mutex_lock(&g.mu);
        const j = g.find(layer, ids, lookback) orelse {
            g.stats.missed += 1;
            _ = std.c.pthread_mutex_unlock(&g.mu);
            return false;
        };
        if (j.state == .queued or j.state == .running) {
            const t0 = nowNs();
            while (j.state == .queued or j.state == .running) _ = std.c.pthread_cond_wait(&g.done, &g.mu);
            g.stats.wait_ns += nowNs() - t0;
        }
        const ok = j.state == .done and j.rows.len == out.len;
        _ = std.c.pthread_mutex_unlock(&g.mu); // only this thread drops a job: the copy needs no lock
        if (ok) @memcpy(out, j.rows);
        _ = std.c.pthread_mutex_lock(&g.mu);
        defer _ = std.c.pthread_mutex_unlock(&g.mu);
        if (ok) g.stats.took += 1 else g.stats.failed += 1;
        for (g.jobs.items, 0..) |x, i| if (x == j) {
            _ = g.jobs.orderedRemove(i);
            break;
        };
        g.recycle(j);
        return ok;
    }

    /// A taken job's buffer back to the spares (the layers' worth), the rest freed; the lock held.
    fn recycle(g: *Ahead, j: *Job) void {
        if (j.rows.len > 0 and g.spare.items.len < g.host.layer_ids.len) {
            g.spare.append(g.a, j.rows) catch g.a.free(j.rows);
            j.rows = &.{};
        }
        g.free(j);
    }

    /// A prompt's end: queued windows dropped, the running job awaited, all held freed; logs what the prompt saw.
    pub fn release(g: *Ahead) void {
        _ = std.c.pthread_mutex_lock(&g.mu);
        var i: usize = 0;
        while (i < g.jobs.items.len) {
            if (g.jobs.items[i].state == .queued) g.free(g.jobs.orderedRemove(i)) else i += 1;
        }
        while (g.running()) _ = std.c.pthread_cond_wait(&g.done, &g.mu);
        for (g.jobs.items) |j| g.free(j);
        g.jobs.clearRetainingCapacity();
        for (g.spare.items) |b| g.a.free(b);
        g.spare.clearRetainingCapacity();
        const s = g.stats;
        g.stats = .{};
        _ = std.c.pthread_mutex_unlock(&g.mu);
        if (s.took + s.missed + s.failed == 0 or quiet) return;
        log.info("engram ahead (rank {d}): {d} segment layers' rows from the worker, {d} inline, {d} failed there; the launch thread waited {d:.1} ms, the worker gathered {d:.1} ms", .{ g.rank, s.took, s.missed, s.failed, ms(s.wait_ns), ms(s.gather_ns) });
    }

    fn ms(ns: u64) f64 {
        return @as(f64, @floatFromInt(ns)) / 1e6;
    }

    // -- the worker ------------------------------------------------------------------------------------------------

    fn nextQueued(g: *Ahead) ?*Job {
        for (g.jobs.items) |j| if (j.state == .queued) return j;
        return null;
    }

    fn running(g: *Ahead) bool {
        for (g.jobs.items) |j| if (j.state == .running) return true;
        return false;
    }

    fn loop(g: *Ahead) void {
        while (true) {
            _ = std.c.pthread_mutex_lock(&g.mu);
            const j = while (true) {
                if (g.stop) {
                    _ = std.c.pthread_mutex_unlock(&g.mu);
                    return;
                }
                if (g.nextQueued()) |q| break q;
                _ = std.c.pthread_cond_wait(&g.work, &g.mu);
            };
            j.state = .running;
            _ = std.c.pthread_mutex_unlock(&g.mu);
            const t0 = nowNs();
            const result = g.gather(j);
            const spent = nowNs() - t0;
            _ = std.c.pthread_mutex_lock(&g.mu);
            g.stats.gather_ns += spent;
            if (result) |_| j.state = .done else |e| {
                j.state = .failed;
                if (!quiet) log.warn("engram ahead: layer {d}'s rows failed on the worker ({t}); the launch thread reads them", .{ j.layer, e });
            }
            if (j.layer == warm_layer) {
                for (g.jobs.items, 0..) |x, i| if (x == j) {
                    _ = g.jobs.orderedRemove(i);
                    break;
                };
                g.free(j);
            }
            _ = std.c.pthread_cond_broadcast(&g.done);
            _ = std.c.pthread_mutex_unlock(&g.mu);
        }
    }

    fn spareOf(g: *Ahead, n: usize) ?[]u16 {
        _ = std.c.pthread_mutex_lock(&g.mu);
        defer _ = std.c.pthread_mutex_unlock(&g.mu);
        for (g.spare.items, 0..) |b, i| if (b.len == n) return g.spare.swapRemove(i);
        return null;
    }

    fn gather(g: *Ahead, j: *Job) !void {
        const src = g.table.source();
        if (j.layer == warm_layer) return g.host.prefetch(g.a, src, j.ids, j.lookback, g.rank, g.world);
        const n = j.ids.len * g.cols;
        const buf = g.spareOf(n) orelse try g.a.alloc(u16, n);
        errdefer g.a.free(buf);
        try g.host.windowRows(g.a, src, j.ids, j.lookback, j.layer, g.rank, g.world, g.dim, buf);
        j.rows = buf;
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
