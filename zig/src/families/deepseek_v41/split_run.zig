//! The M5's layer-split prompt pass over a `Feed`: when to run which streamed rows, the state's restore at the end, and the
//! fall back to a local pass. Model calls come in as `Ops`, so the machine is host-testable (backend_prefill.zig wires it).
const std = @import("std");
const fabric = @import("fabric");
const feed_mod = @import("split_feed.zig");
const split = @import("handoff/split.zig");
const geometry = @import("handoff/geometry.zig");

pub const Feed = feed_mod.Feed;
pub const Stats = feed_mod.Stats;
pub const Pull = feed_mod.Pull;

pub const Limits = struct {
    /// the engine thread waits at most this for rows or the state in one step
    slice_ns: u64 = 2 * std.time.ns_per_ms,
    /// a pass waits for this many rows (the pull delivers them in frames) unless the stream is complete or stood still
    /// for a slice
    min_pass: u32 = 256,
    /// no new row, or no state once the rows are in, for this long: the split is given up
    stall_ns: u64 = 120 * std.time.ns_per_s,
};
pub var limits: Limits = .{};

/// What the machine asks of the model; `ptr` is the caller's context for this step.
pub const Ops = struct {
    ptr: *anyopaque,
    /// the slot back to an empty prompt
    reset: *const fn (ptr: *anyopaque) anyerror!void,
    /// the slot's position
    position: *const fn (ptr: *anyopaque) u32,
    /// layers from the cut over prompt rows [at, at + n), entered from `rows` (n boundary rows)
    enter: *const fn (ptr: *anyopaque, at: u32, n: u32, rows: []const u8) anyerror!void,
    /// layers below the cut from the landed state (the feed's snapshot)
    restore: *const fn (ptr: *anyopaque, snap: *anyopaque) anyerror!void,
};

pub const Next = enum {
    /// rows entered or still waiting: the call did what it could
    more,
    /// every row ran and the state is restored: the slot stands at `upto`, the tail rows are the normal path's
    restored,
    /// the split was given up and the slot reset: the prompt is a normal local pass from row 0
    fell_back,
};

/// Whether the engine can run this feed for a prompt of `prompt_len` tokens: the cut is one the M5 enters at, and the
/// rows end before the prompt does (the tail runs locally).
pub fn admits(feed: Feed, prompt_len: usize) bool {
    return split.cutOk(geometry.release, feed.cut) and feed.upto >= 1 and feed.upto < prompt_len and feed.upto <= std.math.maxInt(u32);
}

pub const Run = struct {
    feed: Feed,
    upto: u32,
    /// rows [0, at) ran
    at: u32 = 0,
    stats: Stats = .{},
    /// the pull's timings, read just before the feed was finished
    pull: Pull = .{},
    finished: bool = false,
    /// the feed was finished as a success
    ok: bool = false,
    fell_back: bool = false,
    err: ?anyerror = null,
    /// the stream stood still for a slice: run what has arrived
    idle: bool = false,
    progress_ns: u64,
    /// when the last row ran (the state's wait starts); 0 until then
    state_since_ns: u64 = 0,

    pub fn init(feed: Feed) Run {
        return .{ .feed = feed, .upto = @intCast(feed.upto), .progress_ns = fabric.words.nowNs() };
    }

    /// The feed is over, once: `ok` for a restored state, else the puller may stop. Its timings are read first.
    pub fn finish(r: *Run, ok: bool) void {
        if (r.finished) return;
        r.finished = true;
        r.ok = ok;
        r.pull = r.feed.pull();
        r.feed.finish(ok, r.stats);
    }

    /// The stream ended or was cancelled before the state landed.
    pub fn abort(r: *Run) void {
        r.finish(false);
    }

    /// A feed the engine does not run: finished as failed, the prompt is a local pass.
    pub fn refuse(r: *Run) void {
        r.fell_back = true;
        r.finish(false);
    }

    /// One slice of work: a pass over the rows that have arrived, a short wait for more, or the restore. Any failure
    /// (the feed's, a model call's, a stall) resets the slot and answers `.fell_back`.
    pub fn advance(r: *Run, ops: Ops, max_rows: u32) !Next {
        std.debug.assert(!r.finished);
        return r.step(ops, max_rows) catch |err| {
            std.log.warn("layer split: falling back to a local pass: {s}", .{@errorName(err)});
            r.err = err;
            r.fell_back = true;
            r.finish(false);
            try ops.reset(ops.ptr);
            r.at = 0;
            return .fell_back;
        };
    }

    fn step(r: *Run, ops: Ops, max_rows: u32) !Next {
        if (r.feed.state() == .failed) return error.StateFailed;
        if (r.at < r.upto) {
            try r.rows(ops, max_rows);
            return .more;
        }
        const now = fabric.words.nowNs();
        if (r.state_since_ns == 0) r.state_since_ns = now;
        switch (r.feed.state()) {
            .failed => return error.StateFailed,
            .pulling => {
                if (now - r.state_since_ns > limits.stall_ns) return error.StateStall;
                fabric.words.pause(limits.slice_ns);
                return .more;
            },
            .landed => {
                const snap = r.feed.snapshot() orelse return error.NoSnapshot;
                r.stats.state_wait_ns = now - r.state_since_ns;
                try ops.restore(ops.ptr, snap);
                r.stats.restore_ns = fabric.words.nowNs() - now;
                if (ops.position(ops.ptr) != r.upto) return error.PositionMismatch;
                r.finish(true);
                return .restored;
            },
        }
    }

    fn rows(r: *Run, ops: Ops, max_rows: u32) !void {
        const ready = r.feed.ready();
        if (split.passRows(r.at, ready, r.upto, max_rows)) |n| {
            if (n >= limits.min_pass or ready >= r.upto or r.idle) {
                const t0 = fabric.words.nowNs();
                try ops.enter(ops.ptr, r.at, n, r.feed.rows(r.at, n));
                const t1 = fabric.words.nowNs();
                r.stats.compute_ns += t1 - t0;
                r.stats.rows += n;
                r.stats.passes += 1;
                r.at += n;
                r.idle = false;
                r.progress_ns = t1;
                return;
            }
        }
        // wait for one row more than there are, a slice at most: the engine thread stays responsive
        const t0 = fabric.words.nowNs();
        const w = r.feed.wait(@min(ready + 1, r.upto), limits.slice_ns);
        const t1 = fabric.words.nowNs();
        r.stats.wait_ns += t1 - t0;
        switch (w) {
            .failed => return error.RowsFailed,
            .ready => {
                r.idle = false;
                r.progress_ns = t1;
            },
            .timeout => {
                r.idle = true;
                if (t1 - r.progress_ns > limits.stall_ns) return error.RowStall;
            },
        }
    }
};

// ---- tests ----

const Probe = struct {
    bytes: []const u8,
    pos: u32 = 0,
    resets: u32 = 0,
    restores: u32 = 0,
    restored: ?*anyopaque = null,
    passes: std.ArrayList(struct { at: u32, n: u32 }) = .empty,
    fail_enter_at: ?u32 = null,

    fn ops(p: *Probe) Ops {
        return .{ .ptr = p, .reset = reset, .position = position, .enter = enter, .restore = restore };
    }
    fn self(ptr: *anyopaque) *Probe {
        return @ptrCast(@alignCast(ptr));
    }
    fn reset(ptr: *anyopaque) anyerror!void {
        self(ptr).resets += 1;
        self(ptr).pos = 0;
    }
    fn position(ptr: *anyopaque) u32 {
        return self(ptr).pos;
    }
    fn enter(ptr: *anyopaque, at: u32, n: u32, rows: []const u8) anyerror!void {
        const p = self(ptr);
        if (at != p.pos) return error.PositionMismatch;
        if (p.fail_enter_at) |f| if (f == at) return error.DeviceLost;
        // exactly the rows the feed holds at [at, at + n)
        try std.testing.expectEqualSlices(u8, p.bytes[@as(usize, at) * feed_mod.row_bytes ..][0 .. @as(usize, n) * feed_mod.row_bytes], rows);
        try p.passes.append(std.testing.allocator, .{ .at = at, .n = n });
        p.pos += n;
    }
    fn restore(ptr: *anyopaque, snap: *anyopaque) anyerror!void {
        self(ptr).restores += 1;
        self(ptr).restored = snap;
    }
};

fn rowsOf(n: usize) ![]u8 {
    const bytes = try std.testing.allocator.alloc(u8, n * feed_mod.row_bytes);
    for (bytes, 0..) |*b, i| b.* = @truncate(i / 7 + i / feed_mod.row_bytes * 13);
    return bytes;
}

test "rows streamed in odd sizes are run in order, no pass leaves a stub, and the state restores once it lands" {
    limits = .{ .min_pass = 1, .slice_ns = 0 };
    defer limits = .{};
    const upto: u32 = 1000;
    const bytes = try rowsOf(upto);
    defer std.testing.allocator.free(bytes);
    var fake: feed_mod.Fake = .{ .bytes = bytes, .cut = 14, .upto = upto };
    var snap: u8 = 0;
    fake.snap = &snap;
    var probe: Probe = .{ .bytes = bytes };
    defer probe.passes.deinit(std.testing.allocator);
    var run: Run = .init(fake.feed());
    try std.testing.expect(admits(fake.feed(), upto + 1));
    const arrivals = [_]u64{ 7, 7, 300, 301, 355, 640, 641, 999, 1000 };
    var next: Next = .more;
    for (arrivals) |ready| {
        fake.ready_rows.store(ready, .release);
        // run what has arrived, then one more call to take the timeout as an idle stream
        for (0..8) |_| {
            next = try run.advance(probe.ops(), 256);
            try std.testing.expectEqual(Next.more, next);
        }
        try std.testing.expect(run.at <= ready);
    }
    try std.testing.expectEqual(upto, run.at);
    for (probe.passes.items) |ps| try std.testing.expect(ps.n <= 256);
    var covered: u32 = 0;
    for (probe.passes.items) |ps| {
        try std.testing.expectEqual(covered, ps.at);
        covered += ps.n;
        const left = upto - covered;
        try std.testing.expect(left == 0 or left >= split.min_tail_rows);
    }
    try std.testing.expectEqual(upto, covered);
    try std.testing.expectEqual(@as(u32, @intCast(probe.passes.items.len)), run.stats.passes);
    try std.testing.expectEqual(@as(u64, upto), run.stats.rows);
    // rows are in, the state is not: waiting, nothing restored, the feed not finished
    for (0..3) |_| try std.testing.expectEqual(Next.more, try run.advance(probe.ops(), 256));
    try std.testing.expectEqual(@as(?bool, null), fake.finished);
    try std.testing.expectEqual(@as(u32, 0), probe.restores);
    // a late state
    fake.times = .{ .first_row_ns = 5, .rows_ns = 6, .state_ns = 7, .bytes = 8 };
    fake.landed.store(@backingInt(feed_mod.State.landed), .release);
    try std.testing.expectEqual(Next.restored, try run.advance(probe.ops(), 256));
    try std.testing.expectEqual(@as(u32, 1), probe.restores);
    try std.testing.expectEqual(@as(?*anyopaque, &snap), probe.restored);
    try std.testing.expectEqual(@as(?bool, true), fake.finished);
    try std.testing.expectEqual(@as(u64, upto), fake.stats.rows);
    try std.testing.expectEqual(@as(u64, 8), run.pull.bytes);
    try std.testing.expect(run.ok and !run.fell_back and run.finished);
    run.abort(); // a finished feed is never finished again
    fake.finished = null;
    run.abort();
    try std.testing.expectEqual(@as(?bool, null), fake.finished);
}

test "a pass waits for a worthwhile batch, then runs what is there once the stream stands still" {
    limits = .{ .min_pass = 256, .slice_ns = 0 };
    defer limits = .{};
    const upto: u32 = 600;
    const bytes = try rowsOf(upto);
    defer std.testing.allocator.free(bytes);
    var fake: feed_mod.Fake = .{ .bytes = bytes, .cut = 8, .upto = upto };
    var probe: Probe = .{ .bytes = bytes };
    defer probe.passes.deinit(std.testing.allocator);
    var run: Run = .init(fake.feed());
    fake.ready_rows.store(51, .release);
    // the first call finds 51 rows, under the batch: it waits and takes the timeout as a still stream
    try std.testing.expectEqual(Next.more, try run.advance(probe.ops(), 4096));
    try std.testing.expectEqual(@as(usize, 0), probe.passes.items.len);
    try std.testing.expect(run.idle);
    try std.testing.expectEqual(Next.more, try run.advance(probe.ops(), 4096));
    try std.testing.expectEqual(@as(usize, 1), probe.passes.items.len);
    try std.testing.expectEqual(@as(u32, 51), probe.passes.items[0].n);
    // a full batch runs at once
    fake.ready_rows.store(600, .release);
    try std.testing.expectEqual(Next.more, try run.advance(probe.ops(), 4096));
    try std.testing.expectEqual(@as(u32, 549), probe.passes.items[1].n);
    try std.testing.expectEqual(upto, run.at);
}

test "a failure mid-stream resets the slot, finishes the feed failed once and leaves a local pass" {
    limits = .{ .min_pass = 1, .slice_ns = 0 };
    defer limits = .{};
    const upto: u32 = 400;
    const bytes = try rowsOf(upto);
    defer std.testing.allocator.free(bytes);
    var fake: feed_mod.Fake = .{ .bytes = bytes, .cut = 14, .upto = upto };
    var probe: Probe = .{ .bytes = bytes };
    defer probe.passes.deinit(std.testing.allocator);
    var run: Run = .init(fake.feed());
    fake.ready_rows.store(200, .release);
    try std.testing.expectEqual(Next.more, try run.advance(probe.ops(), 128));
    try std.testing.expectEqual(@as(u32, 128), probe.pos);
    fake.landed.store(@backingInt(feed_mod.State.failed), .release);
    try std.testing.expectEqual(Next.fell_back, try run.advance(probe.ops(), 128));
    try std.testing.expectEqual(@as(u32, 1), probe.resets);
    try std.testing.expectEqual(@as(u32, 0), probe.pos);
    try std.testing.expectEqual(@as(?bool, false), fake.finished);
    try std.testing.expect(run.fell_back and run.finished and !run.ok);
    try std.testing.expectEqual(@as(u64, 128), fake.stats.rows);
    try std.testing.expectEqual(@as(?anyerror, error.StateFailed), run.err);
    // a model failure inside a pass is the same fall back
    var fake2: feed_mod.Fake = .{ .bytes = bytes, .cut = 14, .upto = upto };
    var probe2: Probe = .{ .bytes = bytes, .fail_enter_at = 0 };
    defer probe2.passes.deinit(std.testing.allocator);
    var run2: Run = .init(fake2.feed());
    fake2.ready_rows.store(400, .release);
    try std.testing.expectEqual(Next.fell_back, try run2.advance(probe2.ops(), 128));
    try std.testing.expectEqual(@as(?bool, false), fake2.finished);
    try std.testing.expectEqual(@as(u32, 1), probe2.resets);
}

test "a stream that stands still too long, rows or state, is given up; a refused feed and a cancel finish once" {
    limits = .{ .min_pass = 1, .slice_ns = 0, .stall_ns = 1 };
    defer limits = .{};
    const upto: u32 = 100;
    const bytes = try rowsOf(upto);
    defer std.testing.allocator.free(bytes);
    var fake: feed_mod.Fake = .{ .bytes = bytes, .cut = 14, .upto = upto };
    var probe: Probe = .{ .bytes = bytes };
    defer probe.passes.deinit(std.testing.allocator);
    var run: Run = .init(fake.feed());
    fabric.words.pause(100_000);
    try std.testing.expectEqual(Next.fell_back, try run.advance(probe.ops(), 128));
    try std.testing.expectEqual(@as(?anyerror, error.RowStall), run.err);
    // rows in, state never lands
    var fake2: feed_mod.Fake = .{ .bytes = bytes, .cut = 14, .upto = upto };
    var probe2: Probe = .{ .bytes = bytes };
    defer probe2.passes.deinit(std.testing.allocator);
    var run2: Run = .init(fake2.feed());
    fake2.ready_rows.store(100, .release);
    try std.testing.expectEqual(Next.more, try run2.advance(probe2.ops(), 128));
    try std.testing.expectEqual(Next.more, try run2.advance(probe2.ops(), 128)); // the clock starts here
    fabric.words.pause(100_000);
    try std.testing.expectEqual(Next.fell_back, try run2.advance(probe2.ops(), 128));
    try std.testing.expectEqual(@as(?anyerror, error.StateStall), run2.err);
    // cancelled while rows are still coming
    var fake3: feed_mod.Fake = .{ .bytes = bytes, .cut = 14, .upto = upto };
    var run3: Run = .init(fake3.feed());
    run3.abort();
    try std.testing.expectEqual(@as(?bool, false), fake3.finished);
    fake3.finished = null;
    run3.abort();
    try std.testing.expectEqual(@as(?bool, null), fake3.finished);
    // cuts and lengths the engine does not run
    try std.testing.expect(!admits(.{ .ptr = &fake3, .cut = 11, .upto = 50, .vtable = fake3.feed().vtable }, 51));
    try std.testing.expect(!admits(fake3.feed(), 100)); // no tail
    try std.testing.expect(admits(fake3.feed(), 101));
}
