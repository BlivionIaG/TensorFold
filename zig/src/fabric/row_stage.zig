//! A host row buffer one side fills in order and another reads in order while it fills: a stream's rows in flight.
const std = @import("std");
const words = @import("words.zig");
const Allocator = std.mem.Allocator;

/// Rows [0, `available`) are complete; the writer publishes, readers wait. A whole stage never changes a published
/// row. A ring stage keeps `cap` rows at once: the writer waits for the readers to free the oldest (`release`, `served`).
pub const Stage = struct {
    gpa: Allocator,
    bytes: []align(16) u8,
    rows: u64,
    row_bytes: u64,
    ready: std.atomic.Value(u64) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),
    /// rows held at once; `rows` for a whole stage
    cap: u64,
    /// ring: rows after the `cap` ones that copy the first rows, so a read across the wrap stays one slice
    mirror: u64 = 0,
    /// ring: rows [0, released) are the readers' no more, the writer may overwrite them
    released: std.atomic.Value(u64) = .init(0),
    /// ring: frames served out of order wait here until the rows below them are served
    lock: std.atomic.Mutex = .unlocked,
    contig: u64 = 0,
    pending: [pending_max]Range = undefined,
    pending_n: usize = 0,
    /// ring: ns the writer spent waiting for room (the readers are the bottleneck)
    blocked_ns: std.atomic.Value(u64) = .init(0),

    const pending_max = 16;
    const Range = struct { start: u64, end: u64 };

    pub const Error = error{ BadStage, OutOfOrder, OutOfRows, OutOfMemory, WindowFull, Released, TooManyPending };

    /// What a wait ended on.
    pub const Wait = enum { ready, failed, timeout, aborted };

    /// `rows` rows of `row_bytes` each, zero rows or bytes allowed (a latch is one row of no bytes).
    pub fn init(gpa: Allocator, rows: u64, row_bytes: u64) Error!Stage {
        return initRing(gpa, rows, row_bytes, rows, 0);
    }

    /// A stage of `rows` rows that holds `cap` of them at once (a whole stage when `cap` >= `rows`) and `mirror` more
    /// rows after them, for readers that take a slice across the wrap.
    pub fn initRing(gpa: Allocator, rows: u64, row_bytes: u64, cap: u64, mirror: u64) Error!Stage {
        if (cap == 0 and rows != 0) return error.BadStage;
        const held = @min(cap, rows);
        const extra = if (held < rows) @min(mirror, held) else 0;
        const n = std.math.mul(u64, held + extra, row_bytes) catch return error.BadStage;
        const bytes = try gpa.alignedAlloc(u8, .@"16", std.math.cast(usize, n) orelse return error.BadStage);
        return .{ .gpa = gpa, .bytes = bytes, .rows = rows, .row_bytes = row_bytes, .cap = held, .mirror = extra };
    }

    pub fn deinit(s: *Stage) void {
        s.gpa.free(s.bytes);
        s.* = undefined;
    }

    pub fn isRing(s: *const Stage) bool {
        return s.cap < s.rows;
    }

    fn at(s: *const Stage, row: u64) usize {
        return @intCast(row % s.cap * s.row_bytes);
    }

    /// Rows [row0, row0 + n) as bytes, for the writer to fill or a reader to take; a ring's rows must still be held, and
    /// either lie before the wrap or end within the mirror.
    pub fn slice(s: *const Stage, row0: u64, n: u64) []u8 {
        std.debug.assert(row0 + n <= s.rows);
        if (s.isRing()) {
            std.debug.assert(row0 >= s.released.load(.acquire) and row0 % s.cap + n <= s.cap + s.mirror);
        }
        const from = if (s.isRing()) s.at(row0) else @as(usize, @intCast(row0 * s.row_bytes));
        return s.bytes[from..][0..@intCast(n * s.row_bytes)];
    }

    pub fn available(s: *const Stage) u64 {
        return s.ready.load(.acquire);
    }

    /// The writer's rows [0, end) are complete (never backwards, never past the end).
    pub fn publish(s: *Stage, end: u64) Error!void {
        if (end > s.rows) return error.OutOfRows;
        if (end < s.ready.load(.monotonic)) return error.OutOfOrder;
        s.ready.store(end, .release);
    }

    /// The writer's next region of a ring, rows from `row0`: as many as lie before the wrap (at most `n`), once the readers
    /// have freed them; fill it, then `publish`. A whole stage's region is rows [row0, row0 + n).
    pub fn reserve(s: *Stage, row0: u64, n: u64, timeout_ns: u64, abort: ?*const std.atomic.Value(bool)) Error![]u8 {
        if (row0 + n > s.rows) return error.OutOfRows;
        if (!s.isRing()) return s.slice(row0, n);
        const seg = @min(n, s.cap - row0 % s.cap);
        switch (s.waitSpace(row0 + seg, timeout_ns, abort)) {
            .ready => {},
            .failed => return error.Released,
            .timeout, .aborted => return error.WindowFull,
        }
        return s.bytes[s.at(row0)..][0..@intCast(seg * s.row_bytes)];
    }

    /// Wait until rows below `end` fit in the ring: the readers freed rows [0, end - cap).
    fn waitSpace(s: *Stage, end: u64, timeout_ns: u64, abort: ?*const std.atomic.Value(bool)) Wait {
        if (end <= s.released.load(.acquire) + s.cap) return .ready;
        const t0 = words.nowNs();
        const deadline = t0 +| timeout_ns;
        var nap: u64 = 20_000;
        defer _ = s.blocked_ns.fetchAdd(words.nowNs() - t0, .monotonic);
        while (true) {
            if (end <= s.released.load(.acquire) + s.cap) return .ready;
            if (s.failed.load(.acquire)) return .failed;
            if (abort) |a| if (a.load(.acquire)) return .aborted;
            if (words.nowNs() >= deadline) return .timeout;
            words.pause(nap);
            nap = @min(nap * 2, 1_000_000);
        }
    }

    /// Copy `src` (whole rows) in as rows from `row0`, which must follow the last published one, and publish them. A ring
    /// waits up to `timeout_ns` (or until `abort`) for room: error.WindowFull.
    pub fn putWait(s: *Stage, row0: u64, src: []const u8, timeout_ns: u64, abort: ?*const std.atomic.Value(bool)) Error!void {
        if (s.row_bytes == 0 or src.len % s.row_bytes != 0) return error.BadStage;
        if (row0 != s.ready.load(.monotonic)) return error.OutOfOrder;
        const n = src.len / s.row_bytes;
        if (row0 + n > s.rows) return error.OutOfRows;
        if (!s.isRing()) {
            @memcpy(s.slice(row0, n), src);
            return s.publish(row0 + n);
        }
        var done: u64 = 0;
        while (done < n) {
            const dst = try s.reserve(row0 + done, n - done, timeout_ns, abort);
            const from: usize = @intCast(done * s.row_bytes);
            @memcpy(dst, src[from..][0..dst.len]);
            // the first rows of the ring are copied after the last, for a reader's slice across the wrap
            const phys = (row0 + done) % s.cap;
            const seg = dst.len / @as(usize, @intCast(s.row_bytes));
            if (phys < s.mirror) {
                const m = @min(s.mirror - phys, seg);
                const o = s.at(row0 + done) + @as(usize, @intCast(s.cap * s.row_bytes));
                @memcpy(s.bytes[o..][0 .. m * @as(usize, @intCast(s.row_bytes))], dst[0 .. m * @as(usize, @intCast(s.row_bytes))]);
            }
            done += seg;
        }
        try s.publish(row0 + n);
    }

    /// `putWait` without waiting.
    pub fn put(s: *Stage, row0: u64, src: []const u8) Error!void {
        return s.putWait(row0, src, 0, null);
    }

    /// Rows [start, start + count) copied into `out` (a ring may wrap); a ring's rows the readers already freed are gone.
    pub fn copyOut(s: *const Stage, start: u64, count: u64, out: []u8) Error!void {
        if (start + count > s.rows) return error.OutOfRows;
        if (!s.isRing()) {
            @memcpy(out[0..@intCast(count * s.row_bytes)], s.slice(start, count));
            return;
        }
        if (start < s.released.load(.acquire)) return error.Released;
        var done: u64 = 0;
        while (done < count) {
            const seg = @min(count - done, s.cap - (start + done) % s.cap);
            const n: usize = @intCast(seg * s.row_bytes);
            const o: usize = @intCast(done * s.row_bytes);
            @memcpy(out[o..][0..n], s.bytes[s.at(start + done)..][0..n]);
            done += seg;
        }
    }

    /// A ring's reader is done with rows [0, end): the writer may overwrite them (never backwards).
    pub fn release(s: *Stage, end: u64) void {
        if (!s.isRing()) return;
        var cur = s.released.load(.monotonic);
        while (end > cur) cur = s.released.cmpxchgWeak(cur, end, .release, .monotonic) orelse return;
    }

    /// A ring's frame [start, start + count) went out: rows are freed as the served ones reach back to row 0, whatever the
    /// order the frames were taken in (error.TooManyPending: more frames out of order than a link set has in flight).
    pub fn served(s: *Stage, start: u64, count: u64) Error!void {
        if (!s.isRing()) return;
        while (!s.lock.tryLock()) std.atomic.spinLoopHint();
        defer s.lock.unlock();
        const end = start + count;
        if (start > s.contig) {
            if (s.pending_n == pending_max) return error.TooManyPending;
            s.pending[s.pending_n] = .{ .start = start, .end = end };
            s.pending_n += 1;
            return;
        }
        s.contig = @max(s.contig, end);
        var moved = true;
        while (moved) {
            moved = false;
            var i: usize = 0;
            while (i < s.pending_n) {
                if (s.pending[i].start <= s.contig) {
                    s.contig = @max(s.contig, s.pending[i].end);
                    s.pending_n -= 1;
                    s.pending[i] = s.pending[s.pending_n];
                    moved = true;
                } else i += 1;
            }
        }
        s.release(s.contig);
    }

    /// The writer gave up: every waiter is released with `.failed`.
    pub fn fail(s: *Stage) void {
        s.failed.store(true, .release);
    }

    /// Wait for rows [0, end) up to `timeout_ns`, or until the stage fails or `abort` is set.
    pub fn wait(s: *const Stage, end: u64, timeout_ns: u64, abort: ?*const std.atomic.Value(bool)) Wait {
        const deadline = words.nowNs() +| timeout_ns;
        var nap: u64 = 20_000;
        while (true) {
            if (s.ready.load(.acquire) >= end) return .ready;
            if (s.failed.load(.acquire)) return .failed;
            if (abort) |a| if (a.load(.acquire)) return .aborted;
            if (words.nowNs() >= deadline) return .timeout;
            words.pause(nap);
            nap = @min(nap * 2, 1_000_000);
        }
    }
};

test "rows published in order are readable, and a reader waits for them" {
    var s = try Stage.init(std.testing.allocator, 4, 3);
    defer s.deinit();
    try std.testing.expectEqual(Stage.Wait.timeout, s.wait(1, 1_000_000, null));
    try s.put(0, &.{ 1, 2, 3, 4, 5, 6 });
    try std.testing.expectEqual(@as(u64, 2), s.available());
    try std.testing.expectEqual(Stage.Wait.ready, s.wait(2, 0, null));
    try std.testing.expectEqualSlices(u8, &.{ 4, 5, 6 }, s.slice(1, 1));
    try std.testing.expectError(error.OutOfOrder, s.put(3, &.{ 7, 8, 9 }));
    try std.testing.expectError(error.OutOfRows, s.put(2, &@as([9]u8, @splat(0))));
    try std.testing.expectError(error.BadStage, s.put(2, &.{ 1, 2 }));
    try s.put(2, &.{ 7, 8, 9 });
    try std.testing.expectError(error.OutOfOrder, s.publish(1));
    try std.testing.expectError(error.OutOfRows, s.publish(5));
}

test "a failed or aborted stage releases its waiters, and another thread's rows arrive" {
    var s = try Stage.init(std.testing.allocator, 8, 4);
    defer s.deinit();
    var abort: std.atomic.Value(bool) = .init(false);
    abort.store(true, .release);
    try std.testing.expectEqual(Stage.Wait.aborted, s.wait(1, 1 << 30, &abort));
    abort.store(false, .release);
    const Writer = struct {
        fn run(st: *Stage) void {
            for (0..4) |i| {
                words.pause(500_000);
                st.put(i * 2, &.{ 1, 1, 1, 1, 2, 2, 2, 2 }) catch return st.fail();
            }
        }
    };
    const t = try std.Thread.spawn(.{}, Writer.run, .{&s});
    try std.testing.expectEqual(Stage.Wait.ready, s.wait(8, 5 * std.time.ns_per_s, null));
    t.join();
    var f = try Stage.init(std.testing.allocator, 1, 0);
    defer f.deinit();
    f.fail();
    try std.testing.expectEqual(Stage.Wait.failed, f.wait(1, 1 << 30, null));
}

test "a ring holds its window, frees rows as readers release them, and a read across the wrap is one slice" {
    var s = try Stage.initRing(std.testing.allocator, 12, 2, 4, 2);
    defer s.deinit();
    try std.testing.expect(s.isRing());
    var rows: [24]u8 = undefined;
    for (&rows, 0..) |*b, i| b.* = @intCast(i);
    try s.put(0, rows[0..6]); // rows 0..2
    try s.put(3, rows[6..8]); // row 3: the ring is full
    try std.testing.expectError(error.WindowFull, s.put(4, rows[8..10]));
    try std.testing.expectEqual(@as(u64, 4), s.available());
    s.release(3);
    try s.put(4, rows[8..14]); // rows 4..6 wrap onto physical 0..2
    try std.testing.expectEqual(@as(u64, 7), s.available());
    // rows 3..5 cross the wrap: physical 3, then the mirror of physical 0 and 1
    try std.testing.expectEqualSlices(u8, rows[6..12], s.slice(3, 3));
    try std.testing.expectEqualSlices(u8, rows[8..14], s.slice(4, 3));
    var out: [6]u8 = undefined;
    try s.copyOut(4, 3, &out);
    try std.testing.expectEqualSlices(u8, rows[8..14], &out);
    try std.testing.expectError(error.Released, s.copyOut(2, 1, &out));
    s.release(2); // never backwards
    try std.testing.expectEqual(@as(u64, 3), s.released.load(.acquire));
}

test "a writer past the ring waits for the reader, and a failed or aborted ring releases it" {
    var s = try Stage.initRing(std.testing.allocator, 40, 4, 8, 4);
    defer s.deinit();
    const Reader = struct {
        fn run(st: *Stage) void {
            var next: u64 = 0;
            var buf: [20]u8 = undefined;
            while (next < 40) {
                if (st.wait(next + 5, 5 * std.time.ns_per_s, null) != .ready) return;
                const row = st.slice(next, 5);
                for (row, 0..) |b, i| if (b != @as(u8, @truncate(next * 4 + i))) return st.fail();
                @memcpy(buf[0..20], row);
                next += 5;
                st.release(next);
            }
        }
    };
    const t = try std.Thread.spawn(.{}, Reader.run, .{&s});
    var src: [20]u8 = undefined;
    var at: u64 = 0;
    while (at < 40) : (at += 5) {
        for (&src, 0..) |*b, i| b.* = @truncate(at * 4 + i);
        try s.putWait(at, &src, 5 * std.time.ns_per_s, null);
    }
    t.join();
    try std.testing.expect(!s.failed.load(.acquire));
    try std.testing.expectEqual(@as(u64, 40), s.available());
    var f = try Stage.initRing(std.testing.allocator, 20, 1, 4, 0);
    defer f.deinit();
    try f.put(0, &.{ 1, 2, 3, 4 });
    var abort: std.atomic.Value(bool) = .init(true);
    try std.testing.expectError(error.WindowFull, f.putWait(4, &.{5}, 1 << 30, &abort));
    f.fail();
    try std.testing.expectError(error.Released, f.putWait(4, &.{5}, 1 << 30, null));
}

test "frames served in any order free the ring from the front only" {
    var s = try Stage.initRing(std.testing.allocator, 64, 1, 16, 0);
    defer s.deinit();
    try s.served(8, 4);
    try s.served(4, 4);
    try std.testing.expectEqual(@as(u64, 0), s.released.load(.acquire));
    try s.served(0, 4);
    try std.testing.expectEqual(@as(u64, 12), s.released.load(.acquire));
    try s.served(12, 4);
    try std.testing.expectEqual(@as(u64, 16), s.released.load(.acquire));
    for (0..16) |i| try s.served(100 + i * 2, 1); // none reaches the front
    try std.testing.expectError(error.TooManyPending, s.served(200, 1));
    var whole = try Stage.init(std.testing.allocator, 4, 1);
    defer whole.deinit();
    try whole.served(0, 4); // a whole stage ignores it
    whole.release(3);
    try std.testing.expect(!whole.isRing());
}
