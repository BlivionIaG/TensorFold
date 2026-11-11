//! A hand-off export converted while its prompt prefills: the parts of finished pool pages are built as the pages land.
const std = @import("std");
const fabric = @import("fabric");
const cx = @import("cuda_export.zig");
const ex = @import("handoff/export.zig");
const plan = @import("handoff/plan.zig");
const producer = fabric.producer;
const words = fabric.words;
const Allocator = std.mem.Allocator;

pub const Error = error{ NoStream, StreamFailed } || cx.Error || Allocator.Error;

/// What a stream tells its owner: `touch` after each batch of parts, `ended` last (the stream is freed after it).
pub const Hooks = struct {
    ptr: *anyopaque,
    touch: *const fn (ptr: *anyopaque, id: [16]u8) void,
    ended: *const fn (ptr: *anyopaque, s: *Stream, end: End) void,
};

/// How a stream finished.
pub const End = struct {
    ok: bool,
    parts: u32,
    /// the parts that were ready before the cut
    streamed: u32,
    bytes: u64,
    /// the time the cut's own conversion took, after the last page landed
    cut_ns: u64,
};

/// The export's readiness, owned by the export's arena so a responder may read it after the stream is gone.
const Shared = struct {
    ready: []std.atomic.Value(bool),
    first: []usize,
    failed: std.atomic.Value(bool) = .init(false),
    reason: [96]u8 = undefined,
    reason_len: usize = 0,

    fn state(ptr: *anyopaque, position: u32, part: u64) producer.PartState {
        const s: *Shared = @ptrCast(@alignCast(ptr));
        if (s.failed.load(.acquire)) return .failed;
        return if (s.ready[s.first[position] + part].load(.acquire)) .ready else .pending;
    }

    fn why(ptr: *anyopaque) []const u8 {
        const s: *Shared = @ptrCast(@alignCast(ptr));
        return s.reason[0..s.reason_len];
    }
};

const Cut = struct { state: cx.State, ctx: *anyopaque, done: *const fn (ctx: *anyopaque) void };
const Job = union(enum) { pages: *anyopaque, cut: Cut, abort };

/// The part's last page needs the cut (its rows come from the snapshot, or the page is not whole before it).
const at_cut = std.math.maxInt(u64);

pub const Stream = struct {
    gpa: Allocator,
    built: *cx.Built,
    shape: plan.Shape,
    feed: cx.Feed,
    /// the pages alone: what comp and ixk parts convert from
    pages_only: ex.Cuda,
    /// pages (from 0) each part needs landed, or `at_cut`
    need: []u64,
    converted: []bool,
    shared: *Shared,
    hooks: Hooks,
    lock: std.atomic.Mutex = .unlocked,
    jobs: std.ArrayList(Job) = .empty,
    /// the engine's side and the worker's: the last to finish frees the stream
    refs: std.atomic.Value(u32) = .init(2),
    /// the engine thread already cut or aborted
    engine_done: bool = false,
    /// the worker is over (under `lock`): a job pushed after it is not run
    over: bool = false,
    /// the owner's record of this export (`Held`), for `ended`
    user: ?*anyopaque = null,

    /// A stream over `feed` for `tokens`: the export's shell allocated and gated; `start` it once registered.
    pub fn open(gpa: Allocator, id: [16]u8, tokens: []const u32, shape: plan.Shape, feed: cx.Feed, index_format: ex.IndexFormat, hooks: Hooks) Error!*Stream {
        const built = try cx.shell(gpa, id, tokens, shape);
        errdefer built.deinit(gpa);
        const a = built.arena.allocator();
        const s = try gpa.create(Stream);
        errdefer gpa.destroy(s);
        const parts = built.parts;
        const shared = try a.create(Shared);
        shared.* = .{ .ready = try a.alloc(std.atomic.Value(bool), parts.len), .first = try a.alloc(usize, built.exp.layers.len) };
        for (shared.ready) |*r| r.* = .init(false);
        var at: usize = 0;
        for (built.exp.layers, shared.first) |l, *f| {
            f.* = at;
            at += l.parts.len;
        }
        const need = try gpa.alloc(u64, parts.len);
        errdefer gpa.free(need);
        for (parts, need) |p, *n| n.* = pagesNeeded(shape, p.spec);
        const converted = try gpa.alloc(bool, parts.len);
        errdefer gpa.free(converted);
        @memset(converted, false);
        const state: cx.State = .{ .pos = shape.upto, .ds_rows = 0, .index_format = index_format, .pieces = &.{}, .pages = feed.pages, .shard = shape.shard, .shards = shape.shards };
        const pages_only: ex.Cuda = .{ .index_format = index_format, .sources = try cx.sourcesOf(a, state), .swa = &.{}, .carries = &.{} };
        built.exp.gate = .{ .ptr = shared, .state_fn = Shared.state, .why_fn = Shared.why };
        s.* = .{ .gpa = gpa, .built = built, .shape = shape, .feed = feed, .pages_only = pages_only, .need = need, .converted = converted, .shared = shared, .hooks = hooks };
        return s;
    }

    /// The pages (counted from 0) a part's rows are in, all of them whole: `at_cut` for the parts that are not pages.
    fn pagesNeeded(shape: plan.Shape, spec: plan.Spec) u64 {
        if (spec.kind != .comp and spec.kind != .ixk) return at_cut;
        const si = shape.geo.sourceIndex(spec.layer).?;
        var it = plan.partPages(shape, si, spec.kind, spec.first_page).?;
        var last: u64 = 0;
        while (it.next()) |r| last = r.page;
        return last + 1;
    }

    /// Starts the worker; without a thread the stream is over at once (failed).
    pub fn start(s: *Stream) void {
        const t = std.Thread.spawn(.{}, run, .{s}) catch {
            s.fail("a thread could not start");
            return s.finishWorker(s.summary(false, 0, 0));
        };
        t.detach();
    }

    /// A stream that was never started (its export could not be registered): the shell, feed and stream are freed.
    pub fn discard(s: *Stream) void {
        s.built.deinit(s.gpa);
        s.refs.store(1, .release);
        s.unref();
    }

    /// Why the stream failed (empty while it has not).
    pub fn reason(s: *const Stream) []const u8 {
        return if (s.shared.failed.load(.acquire)) s.shared.reason[0..s.shared.reason_len] else "";
    }

    /// False when the job will never run (the worker is over, or there is no memory to queue it).
    fn push(s: *Stream, job: Job) bool {
        while (!s.lock.tryLock()) std.atomic.spinLoopHint();
        defer s.lock.unlock();
        if (s.over) return false;
        s.jobs.append(s.gpa, job) catch {
            s.fail("out of memory");
            return false;
        };
        return true;
    }

    fn pop(s: *Stream) ?Job {
        while (!s.lock.tryLock()) std.atomic.spinLoopHint();
        defer s.lock.unlock();
        if (s.jobs.items.len == 0) return null;
        return s.jobs.orderedRemove(0);
    }

    /// Engine thread, the slot standing at `at` tokens: the pages that are whole by now are read and converted.
    pub fn progress(s: *Stream, at: u64) void {
        if (s.engine_done or s.shared.failed.load(.acquire)) return;
        const job = s.feed.queue(s.feed.ptr, at, false) catch return s.fail("the page read could not be queued");
        if (job) |j| _ = s.push(.{ .pages = j });
    }

    /// Engine thread, at the cut: the rest of the pages are read and `state` converts the others; then `done(ctx)`.
    pub fn cut(s: *Stream, state: cx.State, ctx: *anyopaque, done: *const fn (ctx: *anyopaque) void) void {
        defer s.finishEngine();
        const job = s.feed.queue(s.feed.ptr, state.pos, true) catch {
            s.fail("the page read could not be queued");
            _ = s.push(.abort);
            return done(ctx);
        };
        if (job) |j| _ = s.push(.{ .pages = j });
        if (!s.push(.{ .cut = .{ .state = state, .ctx = ctx, .done = done } })) done(ctx);
    }

    /// Engine thread: no state will come; the stream fails with `why` and ends.
    pub fn abort(s: *Stream) void {
        defer s.finishEngine();
        _ = s.push(.abort);
    }

    /// Whether the worker has failed (the engine's `cut` would be wasted).
    pub fn failedAlready(s: *const Stream) bool {
        return s.shared.failed.load(.acquire);
    }

    /// The engine's side is over: the stream is freed when the worker's is too.
    pub fn finishEngine(s: *Stream) void {
        s.engine_done = true;
        s.unref();
    }

    fn unref(s: *Stream) void {
        if (s.refs.fetchSub(1, .acq_rel) != 1) return;
        s.feed.close(s.feed.ptr);
        s.jobs.deinit(s.gpa);
        s.gpa.free(s.need);
        s.gpa.free(s.converted);
        s.gpa.destroy(s);
    }

    fn fail(s: *Stream, why: []const u8) void {
        if (s.shared.failed.load(.acquire)) return;
        const n = @min(why.len, s.shared.reason.len);
        @memcpy(s.shared.reason[0..n], why[0..n]);
        s.shared.reason_len = n;
        s.shared.failed.store(true, .release);
    }

    fn run(s: *Stream) void {
        s.finishWorker(s.work());
    }

    fn finishWorker(s: *Stream, end: End) void {
        while (!s.lock.tryLock()) std.atomic.spinLoopHint();
        s.over = true;
        for (s.jobs.items) |job| switch (job) {
            .cut => |c| c.done(c.ctx), // a cut the worker never ran still hands its state back
            else => {},
        };
        s.jobs.clearRetainingCapacity();
        s.lock.unlock();
        s.hooks.ended(s.hooks.ptr, s, end);
        s.unref();
    }

    fn summary(s: *const Stream, ok: bool, streamed: u32, cut_ns: u64) End {
        return .{ .ok = ok, .parts = @intCast(s.built.parts.len), .streamed = streamed, .bytes = s.built.bytes, .cut_ns = cut_ns };
    }

    fn work(s: *Stream) End {
        var landed: u64 = 0;
        var streamed: u32 = 0;
        while (true) {
            const job = s.pop() orelse {
                if (s.shared.failed.load(.acquire)) return s.summary(false, streamed, 0);
                words.pause(std.time.ns_per_ms);
                continue;
            };
            switch (job) {
                .abort => {
                    s.fail("the prompt pass ended before its state was saved");
                    return s.summary(false, streamed, 0);
                },
                .pages => |j| {
                    landed = s.feed.land(s.feed.ptr, j) catch |err| {
                        s.fail(@errorName(err));
                        return s.summary(false, streamed, 0);
                    };
                    streamed += s.convertPages(landed) catch |err| {
                        s.fail(@errorName(err));
                        return s.summary(false, streamed, 0);
                    };
                    s.hooks.touch(s.hooks.ptr, s.built.exp.handoff);
                },
                .cut => |c| {
                    defer c.done(c.ctx);
                    const t0 = words.nowNs();
                    s.convertCut(c.state) catch |err| {
                        s.fail(@errorName(err));
                        return s.summary(false, streamed, 0);
                    };
                    return s.summary(true, streamed, words.nowNs() - t0);
                },
            }
        }
    }

    /// The comp and ixk parts whose pages have all landed; how many became ready.
    fn convertPages(s: *Stream, landed: u64) Error!u32 {
        var n: u32 = 0;
        for (s.built.parts, s.need, s.converted, 0..) |*p, need, *done, i| {
            if (done.* or need > landed) continue;
            try ex.fill(p, s.shape, s.pages_only);
            done.* = true;
            s.shared.ready[i].store(true, .release);
            n += 1;
        }
        return n;
    }

    /// Every part still waiting, from the cut's state (its pages have all landed).
    fn convertCut(s: *Stream, st: cx.State) Error!void {
        if (st.pos != s.shape.upto) return error.BadSnapshot;
        var arena: std.heap.ArenaAllocator = .init(s.gpa);
        defer arena.deinit();
        var ds: u32 = 0;
        const cuda = try cx.cudaOf(arena.allocator(), st, &ds);
        if (ds != s.shape.ds_rows) return error.BadSnapshot; // the drafter rows the plan counted on are not the state's
        try ex.checkRings(s.shape, cuda);
        for (s.built.parts, s.converted, 0..) |*p, *done, i| {
            if (done.*) continue;
            try ex.fill(p, s.shape, cuda);
            done.* = true;
            s.shared.ready[i].store(true, .release);
        }
    }
};
