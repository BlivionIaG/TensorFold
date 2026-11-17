//! A host row buffer one side fills in order and another reads in order while it fills: a stream's rows in flight.
const std = @import("std");
const words = @import("words.zig");
const Allocator = std.mem.Allocator;

/// Rows [0, `available`) are complete and never change again; the writer publishes, readers wait.
pub const Stage = struct {
    gpa: Allocator,
    bytes: []align(16) u8,
    rows: u64,
    row_bytes: u64,
    ready: std.atomic.Value(u64) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    pub const Error = error{ BadStage, OutOfOrder, OutOfRows, OutOfMemory };

    /// What a wait ended on.
    pub const Wait = enum { ready, failed, timeout, aborted };

    /// `rows` rows of `row_bytes` each, zero rows or bytes allowed (a latch is one row of no bytes).
    pub fn init(gpa: Allocator, rows: u64, row_bytes: u64) Error!Stage {
        const n = std.math.mul(u64, rows, row_bytes) catch return error.BadStage;
        const bytes = try gpa.alignedAlloc(u8, .@"16", std.math.cast(usize, n) orelse return error.BadStage);
        return .{ .gpa = gpa, .bytes = bytes, .rows = rows, .row_bytes = row_bytes };
    }

    pub fn deinit(s: *Stage) void {
        s.gpa.free(s.bytes);
        s.* = undefined;
    }

    /// Rows [row0, row0 + n) as bytes, for the writer to fill or a reader to take.
    pub fn slice(s: *const Stage, row0: u64, n: u64) []u8 {
        std.debug.assert(row0 + n <= s.rows);
        const at: usize = @intCast(row0 * s.row_bytes);
        return s.bytes[at..][0..@intCast(n * s.row_bytes)];
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

    /// Copy `src` (whole rows) in as rows from `row0`, which must follow the last published one, and publish them.
    pub fn put(s: *Stage, row0: u64, src: []const u8) Error!void {
        if (s.row_bytes == 0 or src.len % s.row_bytes != 0) return error.BadStage;
        if (row0 != s.ready.load(.monotonic)) return error.OutOfOrder;
        const n = src.len / s.row_bytes;
        if (row0 + n > s.rows) return error.OutOfRows;
        @memcpy(s.slice(row0, n), src);
        try s.publish(row0 + n);
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
