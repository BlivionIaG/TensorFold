//! The layer-split producer's boundary-row ring: pinned staging slots filled by async copies, published in order.
const std = @import("std");
const split = @import("handoff/split.zig");

/// Staging slots: chunk N's rows copy out while chunk N + 1 computes and N + 2 may queue behind it.
pub const slots = 3;

/// Where the rows go: `rows` is the stage's next region (a ring may end it before the chunk), `publish` frees readers.
pub const Sink = struct {
    ptr: *anyopaque,
    rows: *const fn (ptr: *anyopaque, row0: u64, n: u64) anyerror![]u8,
    publish: *const fn (ptr: *anyopaque, end: u64) anyerror!void,
};

/// The copies' completion: `done` polls a slot's copy event, `wait` blocks on it (CUDA events; a fake in tests).
pub const Backend = struct {
    ptr: *anyopaque,
    done: *const fn (ptr: *anyopaque, slot: usize) anyerror!bool,
    wait: *const fn (ptr: *anyopaque, slot: usize) anyerror!void,
};

const Chunk = struct { row0: u64, n: u64 };

pub const Ring = struct {
    sink: Sink,
    be: Backend,
    /// per slot: the chunk's stream rows (`n` x 40,960 bytes) and pre rows (`n` x 16), as the copies leave them
    x: [slots][]const u8,
    pre: [slots][]const u8,
    chunk: [slots]Chunk = undefined,
    /// the oldest slot in flight, and how many are
    first: usize = 0,
    live: usize = 0,
    /// rows published so far (chunks go out in row order)
    end: u64 = 0,

    pub const Error = error{ OutOfOrder, TooManyRows };

    /// The slot the next chunk's copy fills: the one after the newest in flight
    pub fn acquire(r: *Ring) !usize {
        if (r.live == slots) try r.publishOldest(true);
        return (r.first + r.live) % slots;
    }

    /// Rows [row0, row0 + n) are being copied into `slot` (the one `acquire` named); the copy's event is recorded.
    pub fn submit(r: *Ring, slot: usize, row0: u64, n: u64) Error!void {
        if (slot != (r.first + r.live) % slots or r.live == slots) return error.OutOfOrder;
        if (n * split.stream_bytes > r.x[slot].len or n * split.pre_bytes > r.pre[slot].len) return error.TooManyRows;
        r.chunk[slot] = .{ .row0 = row0, .n = n };
        r.live += 1;
    }

    /// Publishes the oldest chunks whose copies are done, in order, and stops at the first that is not.
    pub fn poll(r: *Ring) !void {
        while (r.live > 0 and try r.be.done(r.be.ptr, r.first)) try r.publishOldest(false);
    }

    /// Awaits and publishes every chunk in flight.
    pub fn drain(r: *Ring) !void {
        while (r.live > 0) try r.publishOldest(true);
    }

    fn publishOldest(r: *Ring, block: bool) !void {
        const s = r.first;
        if (block) try r.be.wait(r.be.ptr, s);
        const c = r.chunk[s];
        if (c.row0 != r.end) return error.OutOfOrder;
        // a row on the wire is the stream's 40,960 bytes then the pre's 16; a stage region may hold part of a chunk
        const stride: usize = @intCast(split.row_bytes);
        const xrow: usize = @intCast(split.stream_bytes);
        const prow: usize = @intCast(split.pre_bytes);
        var done: u64 = 0;
        while (done < c.n) {
            const dst = try r.sink.rows(r.sink.ptr, c.row0 + done, c.n - done);
            const n: usize = @min(dst.len / stride, @as(usize, @intCast(c.n - done)));
            if (n == 0) return error.OutOfOrder;
            const from: usize = @intCast(done);
            for (0..n) |i| {
                @memcpy(dst[i * stride ..][0..xrow], r.x[s][(from + i) * xrow ..][0..xrow]);
                @memcpy(dst[i * stride + xrow ..][0..prow], r.pre[s][(from + i) * prow ..][0..prow]);
            }
            done += n;
            try r.sink.publish(r.sink.ptr, c.row0 + done);
        }
        r.end = c.row0 + c.n;
        r.first = (r.first + 1) % slots;
        r.live -= 1;
    }
};

const Fake = struct {
    finished: [slots]bool = @splat(false),
    waits: u32 = 0,
    stage: std.ArrayList(u8) = .empty,
    published: std.ArrayList(u64) = .empty,
    /// the most rows a region of the stage holds (a ring that wraps hands out less than asked)
    region: u64 = std.math.maxInt(u64),

    fn done(ptr: *anyopaque, slot: usize) anyerror!bool {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        return f.finished[slot];
    }

    fn wait(ptr: *anyopaque, slot: usize) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        f.waits += 1;
        f.finished[slot] = true;
    }

    fn rows(ptr: *anyopaque, row0: u64, n0: u64) anyerror![]u8 {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        const row: usize = @intCast(split.row_bytes);
        const n = @min(n0, f.region);
        f.stage.resize(std.testing.allocator, @intCast((row0 + n) * row)) catch unreachable;
        return f.stage.items[@intCast(row0 * row)..][0..@intCast(n * row)];
    }

    fn publish(ptr: *anyopaque, end: u64) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        try f.published.append(std.testing.allocator, end);
    }

    fn deinit(f: *Fake) void {
        f.stage.deinit(std.testing.allocator);
        f.published.deinit(std.testing.allocator);
    }
};

/// A ring over fake events and `rows_max`-row slots.
fn testRing(f: *Fake, bufs: *[slots][2][]u8, rows_max: usize) !Ring {
    var x: [slots][]const u8 = undefined;
    var pre: [slots][]const u8 = undefined;
    for (0..slots) |s| {
        bufs[s][0] = try std.testing.allocator.alloc(u8, rows_max * split.stream_bytes);
        bufs[s][1] = try std.testing.allocator.alloc(u8, rows_max * split.pre_bytes);
        x[s] = bufs[s][0];
        pre[s] = bufs[s][1];
    }
    return .{
        .sink = .{ .ptr = f, .rows = Fake.rows, .publish = Fake.publish },
        .be = .{ .ptr = f, .done = Fake.done, .wait = Fake.wait },
        .x = x,
        .pre = pre,
    };
}

fn freeBufs(bufs: *[slots][2][]u8) void {
    for (bufs) |b| for (b) |p| std.testing.allocator.free(p);
}

/// Submits a chunk of `n` rows at `row0` into the slot `acquire` names, its bytes `tag` and `tag + 1`.
fn put(r: *Ring, bufs: *[slots][2][]u8, row0: u64, n: u64, tag: u8) !usize {
    const s = try r.acquire();
    @memset(bufs[s][0], tag);
    @memset(bufs[s][1], tag +% 1);
    try r.submit(s, row0, n);
    return s;
}

test "slots rotate in order and a chunk publishes only after its copy is done" {
    var f: Fake = .{};
    defer f.deinit();
    var bufs: [slots][2][]u8 = undefined;
    defer freeBufs(&bufs);
    var r = try testRing(&f, &bufs, 4);
    try std.testing.expectEqual(@as(usize, 0), try put(&r, &bufs, 0, 4, 10));
    try std.testing.expectEqual(@as(usize, 1), try put(&r, &bufs, 4, 3, 20));
    try r.poll();
    try std.testing.expectEqual(@as(usize, 0), f.published.items.len); // no copy finished: nothing out
    f.finished[0] = true;
    try r.poll();
    try std.testing.expectEqualSlices(u64, &.{4}, f.published.items);
    try std.testing.expectEqual(@as(usize, 2), try put(&r, &bufs, 7, 4, 30));
    try std.testing.expectEqual(@as(usize, 0), try put(&r, &bufs, 11, 2, 40)); // slot 0 again, its rows long out
    try std.testing.expectEqual(@as(u32, 0), f.waits);
}

test "chunks publish in row order even when a later copy finishes first" {
    var f: Fake = .{};
    defer f.deinit();
    var bufs: [slots][2][]u8 = undefined;
    defer freeBufs(&bufs);
    var r = try testRing(&f, &bufs, 4);
    _ = try put(&r, &bufs, 0, 4, 1);
    _ = try put(&r, &bufs, 4, 4, 2);
    f.finished[1] = true; // the newer copy is done, the older is not
    try r.poll();
    try std.testing.expectEqual(@as(usize, 0), f.published.items.len);
    f.finished[0] = true;
    try r.poll();
    try std.testing.expectEqualSlices(u64, &.{ 4, 8 }, f.published.items);
    try std.testing.expectEqual(@as(usize, 0), r.live);
}

test "a full ring awaits and publishes its oldest chunk before a slot is refilled" {
    var f: Fake = .{};
    defer f.deinit();
    var bufs: [slots][2][]u8 = undefined;
    defer freeBufs(&bufs);
    var r = try testRing(&f, &bufs, 2);
    for (0..slots) |i| _ = try put(&r, &bufs, i * 2, 2, @intCast(i * 10));
    try std.testing.expectEqual(@as(usize, 0), f.published.items.len);
    // the fourth chunk needs slot 0 back: its chunk is waited for and goes out first
    try std.testing.expectEqual(@as(usize, 0), try put(&r, &bufs, 6, 2, 99));
    try std.testing.expectEqual(@as(u32, 1), f.waits);
    try std.testing.expectEqualSlices(u64, &.{2}, f.published.items);
    try r.drain();
    try std.testing.expectEqualSlices(u64, &.{ 2, 4, 6, 8 }, f.published.items);
    try std.testing.expectEqual(@as(u32, 4), f.waits);
}

test "a published row is the stream's bytes then its pre, and an out-of-order or oversized chunk is refused" {
    var f: Fake = .{};
    defer f.deinit();
    var bufs: [slots][2][]u8 = undefined;
    defer freeBufs(&bufs);
    var r = try testRing(&f, &bufs, 3);
    _ = try put(&r, &bufs, 0, 3, 0x41);
    try r.drain();
    const row: usize = @intCast(split.row_bytes);
    for (0..3) |i| {
        const got = f.stage.items[i * row ..][0..row];
        for (got[0..40960]) |b| try std.testing.expectEqual(@as(u8, 0x41), b);
        for (got[40960..]) |b| try std.testing.expectEqual(@as(u8, 0x42), b);
    }
    const s = try r.acquire();
    try std.testing.expectError(error.TooManyRows, r.submit(s, 3, 4));
    try std.testing.expectError(error.OutOfOrder, r.submit((s + 1) % slots, 3, 1));
    try r.submit(s, 5, 1); // a gap in the rows: refused when it comes to publish
    try std.testing.expectError(error.OutOfOrder, r.drain());
}

test "a chunk the stage takes in parts is copied and published part by part, in order" {
    var f: Fake = .{ .region = 3 };
    defer f.deinit();
    var bufs: [slots][2][]u8 = undefined;
    defer freeBufs(&bufs);
    var r = try testRing(&f, &bufs, 8);
    const s = try r.acquire();
    for (bufs[s][0], 0..) |*b, i| b.* = @truncate(i / split.stream_bytes);
    @memset(bufs[s][1], 9);
    try r.submit(s, 0, 8);
    try r.drain();
    try std.testing.expectEqualSlices(u64, &.{ 3, 6, 8 }, f.published.items);
    const row: usize = @intCast(split.row_bytes);
    for (0..8) |i| try std.testing.expectEqual(@as(u8, @intCast(i)), f.stage.items[i * row]);
    try std.testing.expectEqual(@as(u64, 8), r.end);
}
