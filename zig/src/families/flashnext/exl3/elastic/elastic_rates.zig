//! Decode expert-residency counters per pool, sampled when a plan is applied: for the log, never for a decision.
const std = @import("std");
const policy = @import("elastic_policy.zig");

/// One pool's cumulative decode counters: planner rounds, picks that found their expert resident or not, line copies.
pub const Pool = struct { rounds: u64 = 0, hit: u64 = 0, miss: u64 = 0, copies: u64 = 0 };

pub const Sample = struct {
    n: usize = 0,
    pools: [policy.max_pools]Pool = @splat(.{}),

    pub fn slice(s: *const Sample) []const Pool {
        return s.pools[0..s.n];
    }
};

/// What happened between two samples, summed over the pools.
pub const Rate = struct {
    /// Decode rounds: the most any pool planned (each pool plans once a round).
    rounds: u64 = 0,
    hit: u64 = 0,
    miss: u64 = 0,
    copies: u64 = 0,
    copy_bytes: u64 = 0,
    /// The pools with the lowest and the highest resident share of their picks (null: none had a pick).
    low: ?Extreme = null,
    high: ?Extreme = null,

    pub const Extreme = struct { pool: usize, ppm: u32 };

    /// The resident share of all picks, in parts per million (a million when there were none).
    pub fn hitPpm(r: Rate) u32 {
        return share(r.hit, r.miss);
    }

    /// Adds `o`'s counts (the extremes are of one interval only).
    pub fn add(r: *Rate, o: Rate) void {
        r.rounds +|= o.rounds;
        r.hit +|= o.hit;
        r.miss +|= o.miss;
        r.copies +|= o.copies;
        r.copy_bytes +|= o.copy_bytes;
    }

    pub fn format(r: Rate, w: *std.Io.Writer) std.Io.Writer.Error!void {
        if (r.rounds == 0) return w.writeAll("no decode round");
        const mib = @as(f64, @floatFromInt(r.copy_bytes)) / (1 << 20);
        try w.print("{d} decode rounds, {d} picks {d:.1}% resident, {d} line copies {d:.0} MiB ({d:.2} MiB a round)", .{ r.rounds, r.hit + r.miss, pct(r.hitPpm()), r.copies, mib, mib / @as(f64, @floatFromInt(r.rounds)) });
        if (r.low) |lo| if (r.high) |hi| try w.print(", pools {d:.0}% (#{d}) to {d:.0}% (#{d})", .{ pct(lo.ppm), lo.pool, pct(hi.ppm), hi.pool });
    }
};

fn pct(ppm: u32) f64 {
    return @as(f64, @floatFromInt(ppm)) / 10_000;
}

fn share(hit: u64, miss: u64) u32 {
    const all = hit +| miss;
    if (all == 0) return 1_000_000;
    return @intCast(@as(u128, hit) * 1_000_000 / all);
}

/// `now` less `prev`. A counter that went back (a reset) counts as no news. A copy moves `line_bytes[i]` of pool `i`.
pub fn since(prev: *const Sample, now: *const Sample, line_bytes: []const u64) Rate {
    var r: Rate = .{};
    for (now.slice(), 0..) |p, i| {
        const was: Pool = if (i < prev.n) prev.pools[i] else .{};
        const rounds = p.rounds -| was.rounds;
        const hit = p.hit -| was.hit;
        const miss = p.miss -| was.miss;
        const copies = p.copies -| was.copies;
        r.rounds = @max(r.rounds, rounds);
        r.hit +|= hit;
        r.miss +|= miss;
        r.copies +|= copies;
        r.copy_bytes +|= copies *| (if (i < line_bytes.len) line_bytes[i] else 0);
        if (hit + miss == 0) continue;
        const ppm = share(hit, miss);
        if (r.low == null or ppm < r.low.?.ppm) r.low = .{ .pool = i, .ppm = ppm };
        if (r.high == null or ppm > r.high.?.ppm) r.high = .{ .pool = i, .ppm = ppm };
    }
    return r;
}

fn sample(rows: []const Pool) Sample {
    var s: Sample = .{ .n = rows.len };
    @memcpy(s.pools[0..rows.len], rows);
    return s;
}

test "exl3: elastic rates: the interval is the difference, bytes follow each pool's line size, extremes name the pools" {
    const a = sample(&.{ .{ .rounds = 10, .hit = 80, .miss = 20, .copies = 5 }, .{ .rounds = 10, .hit = 50, .miss = 50, .copies = 10 } });
    const b = sample(&.{ .{ .rounds = 30, .hit = 180, .miss = 60, .copies = 15 }, .{ .rounds = 30, .hit = 100, .miss = 140, .copies = 30 } });
    const r = since(&a, &b, &.{ 1 << 20, 2 << 20 });
    try std.testing.expectEqual(@as(u64, 20), r.rounds);
    try std.testing.expectEqual(@as(u64, 150), r.hit);
    try std.testing.expectEqual(@as(u64, 130), r.miss);
    try std.testing.expectEqual(@as(u64, 30), r.copies);
    try std.testing.expectEqual(@as(u64, 10 * (1 << 20) + 20 * (2 << 20)), r.copy_bytes);
    try std.testing.expectEqual(@as(u32, 535_714), r.hitPpm()); // 150 / 280
    try std.testing.expectEqual(@as(usize, 1), r.low.?.pool); // 50 of 140 resident
    try std.testing.expectEqual(@as(usize, 0), r.high.?.pool); // 100 of 140 resident
    var buf: [256]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf, "{f}", .{r});
    try std.testing.expect(std.mem.startsWith(u8, text, "20 decode rounds, 280 picks 53.6% resident, 30 line copies 50 MiB (2.50 MiB a round)"));
}

test "exl3: elastic rates: a first sample counts from zero, a counter that went back counts for nothing, no round says so" {
    const b = sample(&.{.{ .rounds = 4, .hit = 3, .miss = 1, .copies = 1 }});
    const first = since(&.{}, &b, &.{1000});
    try std.testing.expectEqual(@as(u64, 4), first.rounds);
    try std.testing.expectEqual(@as(u64, 1000), first.copy_bytes);
    const reset = since(&b, &sample(&.{.{ .rounds = 1 }}), &.{1000});
    try std.testing.expectEqual(@as(u64, 0), reset.rounds + reset.hit + reset.miss + reset.copies + reset.copy_bytes);
    try std.testing.expect(reset.low == null);
    try std.testing.expectEqual(@as(u32, 1_000_000), reset.hitPpm());
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("no decode round", try std.fmt.bufPrint(&buf, "{f}", .{reset}));
    var sum: Rate = .{};
    sum.add(first);
    sum.add(first);
    try std.testing.expectEqual(@as(u64, 8), sum.rounds);
}
