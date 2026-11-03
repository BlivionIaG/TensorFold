//! One rank's expert pools as one elastic quantity: a uniform step clamped per pool, all or nothing, undone exactly.
const std = @import("std");
const policy = @import("elastic_policy.zig");

const max_pools = policy.max_pools;

/// What `Lines` needs of the pools: how many, each one's shape and mover, and the device's free bytes.
pub const Backend = struct {
    ctx: *anyopaque,
    count: usize,
    /// Pool `i`: lines in use, the most it can map, the bytes a line holds and the chunk it maps in.
    shape: *const fn (ctx: *anyopaque, i: usize) policy.Line,
    /// Pool `i` to `n` lines, streams quiescent. A failure leaves it at its old count or at `n`.
    set: *const fn (ctx: *anyopaque, i: usize, n: u32) anyerror!void,
    free: *const fn (ctx: *anyopaque) u64,
};

pub const Lines = struct {
    b: Backend,
    /// The fewest lines of each pool.
    floor: [max_pools]u32,
    /// Lines of every pool before each of the last steps, oldest first.
    undo: [policy.max_steps][max_pools]u32 = undefined,
    undo_n: usize = 0,
    /// Steps applied and not taken back: ranks compare it to learn that they did the same.
    epoch: u32 = 0,

    /// `floor` lines at least in each pool (but never more than it holds now); null: no pool at all.
    pub fn init(b: Backend, floor: u32) !Lines {
        if (b.count == 0 or b.count > max_pools) return error.Invalid;
        var l: Lines = .{ .b = b, .floor = undefined };
        for (0..b.count) |i| l.floor[i] = @min(floor, b.shape(b.ctx, i).active);
        return l;
    }

    /// The policy's view: each pool with its floor.
    pub fn shapes(l: *const Lines, out: []policy.Line) usize {
        for (0..l.b.count) |i| {
            out[i] = l.b.shape(l.b.ctx, i);
            out[i].floor = l.floor[i];
        }
        return l.b.count;
    }

    pub fn total(l: *const Lines) u64 {
        var sum: u64 = 0;
        for (0..l.b.count) |i| sum += l.b.shape(l.b.ctx, i).active;
        return sum;
    }

    fn target(active: u32, floor: u32, cap: u32, delta: i32) u32 {
        if (delta >= 0) return @min(cap, active +| @as(u32, @intCast(delta)));
        return @max(floor, active -| @as(u32, @intCast(-@as(i64, delta))));
    }

    /// Every pool `delta` lines more, clamped to its bounds; one that cannot move puts all back (ElasticRefused).
    pub fn apply(l: *Lines, delta: i32) !void {
        var prev: [max_pools]u32 = undefined;
        for (0..l.b.count) |i| prev[i] = l.b.shape(l.b.ctx, i).active;
        for (0..l.b.count) |i| {
            const sh = l.b.shape(l.b.ctx, i);
            const want = target(sh.active, l.floor[i], sh.cap, delta);
            if (want == sh.active) continue;
            l.b.set(l.b.ctx, i, want) catch {
                try l.restore(prev[0 .. i + 1]);
                return error.ElasticRefused;
            };
        }
        if (l.undo_n == l.undo.len) {
            std.mem.copyForwards([max_pools]u32, l.undo[0 .. l.undo.len - 1], l.undo[1..]);
            l.undo_n -= 1;
        }
        l.undo[l.undo_n] = prev;
        l.undo_n += 1;
        l.epoch +%= 1;
    }

    /// The pools of `prev` back at their counts (what a failed step touched).
    fn restore(l: *Lines, prev: []const u32) !void {
        for (prev, 0..) |n, i| {
            if (l.b.shape(l.b.ctx, i).active == n) continue;
            l.b.set(l.b.ctx, i, n) catch return error.ElasticRollbackFailed;
        }
    }

    /// The last applied step taken back exactly.
    pub fn undoLast(l: *Lines) !void {
        if (l.undo_n == 0) return error.ElasticNothingToUndo;
        // growing back may need memory the step released: fatal if it is gone, as the ranks would differ
        try l.restore(l.undo[l.undo_n - 1][0..l.b.count]);
        l.undo_n -= 1;
        l.epoch -%= 1;
    }

    /// What a rank answers a lines command with: its epoch and its free MiB.
    pub fn ack(l: *const Lines) u64 {
        return (@as(u64, l.epoch) << 32) | @as(u32, @intCast(@min(l.b.free(l.b.ctx) >> 20, std.math.maxInt(u32))));
    }
};

/// Pools of fixed shape and a log of what was asked, for the tests.
pub const Fake = struct {
    pools: [max_pools]policy.Line = undefined,
    n: usize,
    free_bytes: u64 = 1 << 40,
    /// Attempts at `set` numbered from `fail_after` up to (not including) `fail_until` fail (null: none do).
    fail_after: ?usize = null,
    fail_until: usize = std.math.maxInt(usize),
    calls: usize = 0,

    pub fn backend(f: *Fake) Backend {
        return .{ .ctx = f, .count = f.n, .shape = shape, .set = set, .free = free };
    }

    fn shape(ctx: *anyopaque, i: usize) policy.Line {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        return f.pools[i];
    }

    fn set(ctx: *anyopaque, i: usize, n: u32) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        const at = f.calls;
        f.calls += 1;
        if (f.fail_after) |left| if (at >= left and at < f.fail_until) return error.OutOfDeviceMemory;
        f.pools[i].active = n;
    }

    fn free(ctx: *anyopaque) u64 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        return f.free_bytes;
    }
};

fn fakePools(n: usize) Fake {
    var f: Fake = .{ .n = n };
    for (f.pools[0..n], 0..) |*p, i| p.* = .{ .active = 20 + @as(u32, @intCast(i)), .floor = 0, .cap = 60, .bytes = 1_843_200, .chunk = 2 << 20 };
    return f;
}

test "exl3: elastic lines: a uniform step clamps each pool to its floor and cap and is taken back exactly" {
    var f = fakePools(4);
    f.pools[1].active = 11;
    f.pools[2].active = 58;
    var l = try Lines.init(f.backend(), 10);
    try std.testing.expectEqualSlices(u32, &.{ 10, 10, 10, 10 }, l.floor[0..4]);
    try l.apply(-3);
    try std.testing.expectEqual([4]u32{ 17, 10, 55, 20 }, [4]u32{ f.pools[0].active, f.pools[1].active, f.pools[2].active, f.pools[3].active });
    try l.apply(9);
    try std.testing.expectEqual([4]u32{ 26, 19, 60, 29 }, [4]u32{ f.pools[0].active, f.pools[1].active, f.pools[2].active, f.pools[3].active });
    try std.testing.expectEqual(@as(u32, 2), l.epoch);
    try l.undoLast(); // the clamped pool goes to where it was, not by -9
    try std.testing.expectEqual([4]u32{ 17, 10, 55, 20 }, [4]u32{ f.pools[0].active, f.pools[1].active, f.pools[2].active, f.pools[3].active });
    try l.undoLast();
    try std.testing.expectEqual([4]u32{ 20, 11, 58, 23 }, [4]u32{ f.pools[0].active, f.pools[1].active, f.pools[2].active, f.pools[3].active });
    try std.testing.expectEqual(@as(u32, 0), l.epoch);
    try std.testing.expectError(error.ElasticNothingToUndo, l.undoLast());
}

test "exl3: elastic lines: a pool that fails puts the earlier ones back and nothing changes" {
    var f = fakePools(5);
    var l = try Lines.init(f.backend(), 10);
    f.fail_after = 3; // the fourth pool's map fails, the puts back go through
    f.fail_until = 4;
    try std.testing.expectError(error.ElasticRefused, l.apply(5));
    for (f.pools[0..5], 0..) |p, i| try std.testing.expectEqual(20 + @as(u32, @intCast(i)), p.active);
    try std.testing.expectEqual(@as(u32, 0), l.epoch);
    try std.testing.expectEqual(@as(usize, 0), l.undo_n);
}

test "exl3: elastic lines: a put back that fails too is fatal, and says so" {
    var f = fakePools(5);
    var l = try Lines.init(f.backend(), 10);
    f.fail_after = 2; // the third pool fails and so does every put back after it
    try std.testing.expectError(error.ElasticRollbackFailed, l.apply(5));
    try std.testing.expectEqual(@as(u32, 0), l.epoch);
}

test "exl3: elastic lines: more steps than the log holds drop the oldest, and the newest undo is still exact" {
    var f = fakePools(2);
    var l = try Lines.init(f.backend(), 10);
    for (0..policy.max_steps + 5) |i| try l.apply(if (i % 2 == 0) 1 else -1);
    try std.testing.expectEqual(policy.max_steps, l.undo_n);
    const before = f.pools[0].active;
    try l.apply(7);
    try l.undoLast();
    try std.testing.expectEqual(before, f.pools[0].active);
}

test "exl3: elastic lines: the ack carries the epoch and the free MiB" {
    var f = fakePools(2);
    f.free_bytes = 123 << 20;
    var l = try Lines.init(f.backend(), 10);
    try l.apply(1);
    try std.testing.expectEqual((@as(u64, 1) << 32) | 123, l.ack());
}
