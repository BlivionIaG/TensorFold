//! A streamed export over a synthetic engine equals the export built at the cut; failures reach the gate.
const std = @import("std");
const fabric = @import("fabric");
const g = @import("handoff/geometry.zig");
const plan = @import("handoff/plan.zig");
const ex = @import("handoff/export.zig");
const s = @import("handoff/synth.zig");
const cx = @import("cuda_export.zig");
const cxt = @import("cuda_export_test.zig");
const stream = @import("cuda_export_stream.zig");
const serve = @import("cuda_export_serve.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;
const page: u64 = g.release.page;

/// A page feed over a synthetic world: `land` copies the queued pages out of the truth, as the device read would.
const FakeFeed = struct {
    truth: []const cx.Pages,
    pages: []cx.Pages,
    cursor: u64 = 0,
    queued: u32 = 0,
    landed: u32 = 0,
    fail_land: bool = false,
    jobs: std.ArrayList(*Job) = .empty,
    closed: *std.atomic.Value(bool),

    const Job = struct { first: u32, upto: u32 };

    fn open(truth: []const cx.Pages, closed: *std.atomic.Value(bool)) !*FakeFeed {
        const f = try gpa.create(FakeFeed);
        f.* = .{ .truth = truth, .pages = try gpa.alloc(cx.Pages, truth.len), .closed = closed };
        for (truth, f.pages) |t, *p| {
            const comp = try gpa.alloc(u8, t.comp.len);
            @memset(comp, 0xEE); // a part read before its pages landed would show
            const index = try gpa.alloc(u8, t.index.len);
            @memset(index, 0xEE);
            p.* = .{ .layer = t.layer, .comp = comp, .index = index };
        }
        return f;
    }

    fn feed(f: *FakeFeed) cx.Feed {
        return .{ .ptr = f, .pages = f.pages, .queue = queue, .land = land, .close = close };
    }

    fn queue(ptr: *anyopaque, at: u64, final: bool) anyerror!?*anyopaque {
        const f: *FakeFeed = @ptrCast(@alignCast(ptr));
        const upto = if (final) (at + page - 1) / page else at / page;
        if (upto <= f.cursor) return null;
        const job = try gpa.create(Job);
        job.* = .{ .first = @intCast(f.cursor), .upto = @intCast(upto) };
        try f.jobs.append(gpa, job);
        f.cursor = upto;
        f.queued += 1;
        return job;
    }

    fn land(ptr: *anyopaque, raw: *anyopaque) anyerror!u64 {
        const f: *FakeFeed = @ptrCast(@alignCast(ptr));
        const job: *Job = @ptrCast(@alignCast(raw));
        if (f.fail_land) return error.DeviceLost;
        for (g.release.sources, f.truth, f.pages) |src, t, *p| {
            const rows: u64 = g.release.pageRows(@intCast(g.release.sourceIndex(src.layer).?));
            inline for (.{ .{ t.comp, p.comp, g.cuda_record_bytes }, .{ t.index, p.index, g.cuda_index_fp8_bytes } }) |c| {
                const pb = rows * c[2];
                const lo = @min(job.first * pb, c[0].len);
                const hi = @min(job.upto * pb, c[0].len);
                @memcpy(@constCast(c[1][lo..hi]), c[0][lo..hi]);
            }
        }
        f.landed = job.upto;
        return job.upto;
    }

    fn close(ptr: *anyopaque) void {
        const f: *FakeFeed = @ptrCast(@alignCast(ptr));
        for (f.jobs.items) |j| gpa.destroy(j);
        f.jobs.deinit(gpa);
        for (f.pages) |p| {
            gpa.free(@constCast(p.comp));
            gpa.free(@constCast(p.index));
        }
        gpa.free(f.pages);
        f.closed.store(true, .release);
        gpa.destroy(f);
    }
};

/// The feed is closed when the stream is freed: after the worker's last word, on its thread.
fn waitClosed(closed: *const std.atomic.Value(bool)) !void {
    for (0..5000) |_| {
        if (closed.load(.acquire)) return;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.Timeout;
}

/// Records what a stream tells its owner.
const Watch = struct {
    touches: std.atomic.Value(u32) = .init(0),
    ended: std.atomic.Value(bool) = .init(false),
    end: stream.End = undefined,
    done: std.atomic.Value(u32) = .init(0),

    fn touch(ptr: *anyopaque, _: [16]u8) void {
        const w: *Watch = @ptrCast(@alignCast(ptr));
        _ = w.touches.fetchAdd(1, .acq_rel);
    }

    fn ended_(ptr: *anyopaque, _: *stream.Stream, end: stream.End) void {
        const w: *Watch = @ptrCast(@alignCast(ptr));
        w.end = end;
        w.ended.store(true, .release);
    }

    fn done_(ptr: *anyopaque) void {
        const w: *Watch = @ptrCast(@alignCast(ptr));
        _ = w.done.fetchAdd(1, .acq_rel);
    }

    fn hooks(w: *Watch) stream.Hooks {
        return .{ .ptr = w, .touch = touch, .ended = ended_ };
    }

    fn await(w: *Watch) !void {
        for (0..5000) |_| {
            if (w.ended.load(.acquire)) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

/// A synthetic world, its snapshot and the export the library builds from it at the cut.
const Rig = struct {
    arena: std.heap.ArenaAllocator,
    snap: cxt.Snap,
    tokens: []u32,
    shape: plan.Shape,
    want: []ex.Part,

    fn init(upto: u64, ds_rows: u32) !*Rig {
        const r = try gpa.create(Rig);
        r.arena = .init(gpa);
        const a = r.arena.allocator();
        const w = try s.make(a, .{ .upto = upto, .ds_rows = ds_rows, .seed = upto });
        r.snap = try cxt.snapshotOf(a, w);
        r.tokens = try a.alloc(u32, upto);
        for (r.tokens, 0..) |*t, i| t.* = @intCast(i * 5 + 3);
        r.shape = .{ .upto = upto, .ds_rows = ds_rows, .chunk = 70 << 10 };
        r.want = try ex.build(a, r.shape, w.cuda);
        return r;
    }

    fn deinit(r: *Rig) void {
        r.arena.deinit();
        gpa.destroy(r);
    }

    fn state(r: *const Rig, ds_rows: u64, feed: cx.Feed) cx.State {
        return .{ .pos = r.shape.upto, .ds_rows = ds_rows, .pieces = r.snap.pieces, .pages = feed.pages };
    }
};

fn readyCount(built: *cx.Built) usize {
    const gate = built.exp.gate.?;
    var n: usize = 0;
    for (built.exp.layers, 0..) |l, i| for (0..l.parts.len) |k| {
        if (gate.state_fn(gate.ptr, @intCast(i), k) == .ready) n += 1;
    };
    return n;
}

test "pages that landed before the cut convert at once, the rest at the cut, and the export equals the cut-time build" {
    const rig = try Rig.init(1400, 128);
    defer rig.deinit();
    var closed: std.atomic.Value(bool) = .init(false);
    const ff = try FakeFeed.open(rig.snap.pages, &closed);
    var watch: Watch = .{};
    const st = try stream.Stream.open(gpa, @splat(3), rig.tokens, rig.shape, ff.feed(), .fp8, watch.hooks());
    const built = st.built; // the stream frees itself once it and the pass are done with it
    defer built.deinit(gpa);
    st.start();

    st.progress(520); // two whole pages
    for (0..2000) |_| {
        if (watch.touches.load(.acquire) >= 1) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(watch.touches.load(.acquire) >= 1);
    const early = readyCount(built);
    try std.testing.expect(early > 0 and early < built.parts.len);
    // the rings, carries and the pages of the last, partial page are not ready before the cut
    try std.testing.expect(built.exp.gate.?.state_fn(built.exp.gate.?.ptr, 0, 0) == .pending);

    st.cut(rig.state(128, ff.feed()), &watch, Watch.done_);
    try watch.await();
    try std.testing.expect(watch.end.ok);
    try std.testing.expectEqual(@as(u32, 1), watch.done.load(.acquire));
    try std.testing.expect(watch.end.streamed >= early and watch.end.streamed < built.parts.len);
    try waitClosed(&closed);
    for (rig.want, built.parts) |a, b| try std.testing.expectEqualSlices(u8, a.bytes, b.bytes);
    try std.testing.expectEqual(built.parts.len, readyCount(built));
}

test "a page read that fails fails the gate with its reason, and the cut's state is still handed back" {
    const rig = try Rig.init(700, 0);
    defer rig.deinit();
    var closed: std.atomic.Value(bool) = .init(false);
    const ff = try FakeFeed.open(rig.snap.pages, &closed);
    ff.fail_land = true;
    var watch: Watch = .{};
    const st = try stream.Stream.open(gpa, @splat(4), rig.tokens, rig.shape, ff.feed(), .fp8, watch.hooks());
    const built = st.built;
    defer built.deinit(gpa);
    st.start();
    st.progress(600);
    try watch.await();
    try std.testing.expect(!watch.end.ok);
    const gate = built.exp.gate.?;
    try std.testing.expect(gate.state_fn(gate.ptr, 0, 0) == .failed);
    try std.testing.expectEqualStrings("DeviceLost", gate.why_fn(gate.ptr));
    st.cut(rig.state(0, ff.feed()), &watch, Watch.done_); // the worker is over: the state comes straight back
    try std.testing.expectEqual(@as(u32, 1), watch.done.load(.acquire));
    try waitClosed(&closed);
}

test "a stream the pass abandons fails and frees; a drafter row count that is not the plan's fails at the cut" {
    const rig = try Rig.init(700, 0);
    defer rig.deinit();
    var closed: std.atomic.Value(bool) = .init(false);
    const ff = try FakeFeed.open(rig.snap.pages, &closed);
    var watch: Watch = .{};
    const st = try stream.Stream.open(gpa, @splat(5), rig.tokens, rig.shape, ff.feed(), .fp8, watch.hooks());
    const built = st.built;
    defer built.deinit(gpa);
    st.start();
    st.abort();
    try watch.await();
    try std.testing.expect(!watch.end.ok);
    try std.testing.expectEqualStrings("the prompt pass ended before its state was saved", built.exp.gate.?.why_fn(built.exp.gate.?.ptr));
    try waitClosed(&closed);

    // the plan counted on no drafter rows; the state holds the rings (valid rows 128)
    const rig2 = try Rig.init(700, 128);
    defer rig2.deinit();
    var closed2: std.atomic.Value(bool) = .init(false);
    const ff2 = try FakeFeed.open(rig2.snap.pages, &closed2);
    var watch2: Watch = .{};
    var shape = rig2.shape;
    shape.ds_rows = 0;
    const st2 = try stream.Stream.open(gpa, @splat(6), rig2.tokens, shape, ff2.feed(), .fp8, watch2.hooks());
    const built2 = st2.built;
    defer built2.deinit(gpa);
    st2.start();
    st2.cut(rig2.state(128, ff2.feed()), &watch2, Watch.done_);
    try watch2.await();
    try std.testing.expect(!watch2.end.ok);
    try std.testing.expectEqual(@as(u32, 1), watch2.done.load(.acquire));
    try waitClosed(&closed2);
}

/// A source over the synthetic world with the page feed; counts the states it hands out and gets back.
const Fake = struct {
    rig: *Rig,
    feeds: u32 = 0,
    released: u32 = 0,
    closed: std.atomic.Value(bool) = .init(false),
    refuse: bool = false,

    fn source(f: *Fake) cx.Source {
        return .{ .ptr = f, .take = take, .release = release, .open_feed = openFeed, .take_state = takeState };
    }

    fn take(_: *anyopaque, _: std.mem.Allocator, _: u64) anyerror!cx.Captured {
        return error.NotUsed;
    }

    fn release(ctx: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.released += 1;
    }

    fn openFeed(ptr: *anyopaque, _: std.mem.Allocator, _: u64, out: *cx.Plan) anyerror!cx.Feed {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        if (f.refuse) return error.NoStream;
        out.* = .{ .ds_rows = f.rig.shape.ds_rows, .index_format = .fp8 };
        f.feeds += 1;
        return (try FakeFeed.open(f.rig.snap.pages, &f.closed)).feed();
    }

    fn takeState(ptr: *anyopaque, _: std.mem.Allocator, _: u64, feed: cx.Feed) anyerror!cx.Captured {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        return .{ .ctx = f, .state = f.rig.state(f.rig.shape.ds_rows, feed) };
    }
};

fn producer(f: *Fake) !*serve.Produce {
    const p = try gpa.create(serve.Produce);
    p.* = .{ .gpa = gpa, .io = io, .source = f.source(), .opts = .{ .link = "test", .model = "dsv41-test", .stream = true }, .table = .{ .gpa = gpa, .ttl_ns = 60 * std.time.ns_per_s, .clock = fabric.words.nowNs } };
    return p;
}

fn waitBuilt(p: *serve.Produce) !void {
    for (0..5000) |_| {
        if (p.building.load(.acquire) == 0) return;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.Timeout;
}

test "the producer streams a hand-off: begin, progress, the cut's saveFor, exported; the table holds the finished export" {
    const rig = try Rig.init(1100, 128);
    defer rig.deinit();
    var fake: Fake = .{ .rig = rig };
    const p = try producer(&fake);
    defer p.stop();
    const id: [16]u8 = @splat(0x61);

    p.begin(id, rig.tokens);
    try std.testing.expectEqual(@as(u32, 1), fake.feeds);
    try std.testing.expect(p.table.get(id) != null); // registered at admission: the decoder's OPEN is answered
    p.progress(id, 700);
    const saved = try p.saveFor(id, 1100);
    try std.testing.expect(saved.stream != null);
    p.exportedBg(id, rig.tokens, saved);
    try waitBuilt(p);
    try std.testing.expectEqual(@as(u32, 1), fake.released);
    try waitClosed(&fake.closed);
    const exp = p.table.get(id).?.ready;
    try std.testing.expect(exp.gate != null);
    const gate = exp.gate.?;
    var parts: usize = 0;
    for (exp.layers, 0..) |l, i| for (0..l.parts.len) |k| {
        parts += 1;
        try std.testing.expect(gate.state_fn(gate.ptr, @intCast(i), k) == .ready);
    };
    try std.testing.expect(parts > 0);
    // a stream for an id nobody began is the plain save: the engine's capture
    try std.testing.expectError(error.NotUsed, p.saveFor(@splat(0x62), 1100));
}

test "a hand-off the pass never saves is failed: its stream ends, its export is released, nothing leaks" {
    const rig = try Rig.init(700, 0);
    defer rig.deinit();
    var fake: Fake = .{ .rig = rig };
    const p = try producer(&fake);
    defer p.stop();
    const id: [16]u8 = @splat(0x63);
    p.begin(id, rig.tokens);
    p.progress(id, 600);
    p.failed(id, "the prompt pass ended before its state was saved");
    try waitBuilt(p);
    try waitClosed(&fake.closed);
    // a source that cannot stream, and a reused id, leave the cut-time build in charge
    var no: Fake = .{ .rig = rig, .refuse = true };
    const q = try producer(&no);
    defer q.stop();
    q.begin(id, rig.tokens);
    try std.testing.expect(q.table.get(id) == null);
    try std.testing.expectError(error.NotUsed, q.saveFor(id, 700));
}
