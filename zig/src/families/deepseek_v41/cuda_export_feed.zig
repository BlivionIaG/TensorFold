//! The CUDA page feed of a streamed hand-off: finished pool pages read back on the worker thread, behind an event.
const std = @import("std");
const cuda = @import("cuda");
const kvs = @import("cuda_kv.zig");
const cx = @import("cuda_export.zig");

const Allocator = std.mem.Allocator;

/// One queued read: logical pages [first, upto) of the slot, behind `ev`.
const Job = struct { ev: cuda.Event, first: u32, upto: u32, ids: []u32 };

pub const Feed = struct {
    gpa: Allocator,
    kx: *kvs.Kv,
    stream: cuda.Stream,
    pages: []cx.Pages,
    bufs: [][]u8,
    /// the logical pages queued so far
    cursor: u32 = 0,
    total: u32,
    lock: std.atomic.Mutex = .unlocked,
    jobs: std.ArrayList(*Job) = .empty,

    /// A feed for a prompt of `at` tokens on the active slot; full-size host buffers for every KV source's pages.
    pub fn open(gpa: Allocator, kx: *kvs.Kv, at: u64) !*Feed {
        const page = kx.pool.page;
        const total: u32 = @intCast((at + page - 1) / page);
        const f = try gpa.create(Feed);
        errdefer gpa.destroy(f);
        const sources = kx.layout.sources[0..kx.layout.nsources];
        f.* = .{ .gpa = gpa, .kx = kx, .stream = try cuda.Stream.init(kx.dev.d, true), .pages = try gpa.alloc(cx.Pages, sources.len), .bufs = try gpa.alloc([]u8, 2 * sources.len), .total = total };
        errdefer f.stream.deinit();
        errdefer gpa.free(f.pages);
        errdefer gpa.free(f.bufs);
        var made: usize = 0;
        errdefer for (f.bufs[0..made]) |b| gpa.free(b);
        const fams = kx.layout.families();
        for (sources, f.pages, 0..) |layer, *pg, i| {
            const comp = kx.layout.compOf(layer).?;
            const index = kx.layout.indexOf(layer).?;
            f.bufs[2 * i] = try gpa.alloc(u8, @as(usize, total) * @as(usize, @intCast(fams[comp].pageBytes(page))));
            made += 1;
            f.bufs[2 * i + 1] = try gpa.alloc(u8, @as(usize, total) * @as(usize, @intCast(fams[index].pageBytes(page))));
            made += 1;
            pg.* = .{ .layer = layer, .comp = f.bufs[2 * i], .index = f.bufs[2 * i + 1] };
        }
        return f;
    }

    pub fn feed(f: *Feed) cx.Feed {
        return .{ .ptr = f, .pages = f.pages, .queue = queue, .land = land, .close = close };
    }

    fn self(ptr: *anyopaque) *Feed {
        return @ptrCast(@alignCast(ptr));
    }

    /// Engine thread: the pages whole at `at` (all when `final`) behind an event after the work queued so far.
    fn queue(ptr: *anyopaque, at: u64, final: bool) anyerror!?*anyopaque {
        const f = self(ptr);
        const kx = f.kx;
        const page = kx.pool.page;
        const upto: u32 = @intCast(if (final) (at + page - 1) / page else at / page);
        if (upto <= f.cursor) return null;
        if (upto > f.total or kx.slot.len < upto) return error.BadWindow;
        const job = try f.gpa.create(Job);
        errdefer f.gpa.destroy(job);
        job.ids = try f.gpa.dupe(u32, kx.slot.pages[f.cursor..upto]);
        errdefer f.gpa.free(job.ids);
        job.ev = try cuda.Event.init(kx.dev.d, false);
        errdefer job.ev.deinit();
        try job.ev.record(kx.runner.stream);
        job.first = f.cursor;
        job.upto = upto;
        while (!f.lock.tryLock()) std.atomic.spinLoopHint();
        defer f.lock.unlock();
        try f.jobs.append(f.gpa, job);
        f.cursor = upto;
        return job;
    }

    /// Worker thread: waits for the event, copies the pages into the buffers, and returns the pages valid now.
    fn land(ptr: *anyopaque, raw: *anyopaque) anyerror!u64 {
        const f = self(ptr);
        const job: *Job = @ptrCast(@alignCast(raw));
        const kx = f.kx;
        try kx.dev.ctx.makeCurrent();
        try job.ev.synchronize();
        const page = kx.pool.page;
        const fams = kx.layout.families();
        const sources = kx.layout.sources[0..kx.layout.nsources];
        for (sources, 0..) |layer, i| {
            inline for (.{ kx.layout.compOf(layer).?, kx.layout.indexOf(layer).? }, 0..) |fam, k| {
                const pb: usize = @intCast(fams[fam].pageBytes(page));
                var at: usize = 0;
                while (at < job.ids.len) {
                    var end = at + 1;
                    while (end < job.ids.len and job.ids[end] == job.ids[end - 1] + 1) end += 1;
                    const dst = f.bufs[2 * i + k][(job.first + at) * pb ..][0 .. (end - at) * pb];
                    try kx.dev.tensors[fam].downloadAsync(job.ids[at] * pb, dst, f.stream.handle);
                    at = end;
                }
            }
        }
        try f.stream.synchronize();
        const upto = job.upto;
        f.retire(job);
        return upto;
    }

    fn retire(f: *Feed, job: *Job) void {
        while (!f.lock.tryLock()) std.atomic.spinLoopHint();
        for (f.jobs.items, 0..) |j, i| if (j == job) {
            _ = f.jobs.swapRemove(i);
            break;
        };
        f.lock.unlock();
        job.ev.deinit();
        f.gpa.free(job.ids);
        f.gpa.destroy(job);
    }

    /// Frees the buffers, the stream and the reads that never landed; any thread, after the last `land`.
    fn close(ptr: *anyopaque) void {
        const f = self(ptr);
        f.kx.dev.ctx.makeCurrent() catch {};
        f.stream.synchronize() catch {};
        for (f.jobs.items) |job| {
            job.ev.deinit();
            f.gpa.free(job.ids);
            f.gpa.destroy(job);
        }
        f.jobs.deinit(f.gpa);
        f.stream.deinit();
        for (f.bufs) |b| f.gpa.free(b);
        f.gpa.free(f.bufs);
        f.gpa.free(f.pages);
        f.gpa.destroy(f);
    }
};
