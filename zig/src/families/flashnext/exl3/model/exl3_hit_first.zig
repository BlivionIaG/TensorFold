//! Hits first: a planned one-pass round with held and missed experts becomes two passes over the same plan.

const std = @import("std");
const round = @import("exl3_round.zig");

/// Splits `r` when it is one pass that holds at least `min_held` of its experts beside a miss; true when it did.
pub fn split(r: *round.Round, min_held: usize) bool {
    if (min_held == 0 or r.n_passes != 1) return false;
    const p = r.passes[0];
    const n_miss = p.copy_hi - p.copy_lo;
    const n_held = p.id_hi - p.id_lo - n_miss;
    if (n_miss == 0 or n_held < min_held) return false;
    std.debug.assert(r.passes.len >= 2);
    // the plan lists held experts first and a miss costs one copy, in order: ids and copies cut at one index
    const at = p.id_lo + n_held;
    r.passes[0] = .{ .id_lo = p.id_lo, .id_hi = at, .copy_lo = p.copy_lo, .copy_hi = p.copy_lo };
    r.passes[1] = .{ .id_lo = at, .id_hi = p.id_hi, .copy_lo = p.copy_lo, .copy_hi = p.copy_hi };
    r.n_passes = 2;
    return true;
}

// ------------------------------------------------------------------------------------------------ tests

const testing = std.testing;
const offload = @import("../offload/exl3_offload.zig");
const shape = offload.shape;

const Fix = struct {
    slots: [16]u16 = undefined,
    line: [16]offload.Line = undefined,
    stamp: [16]u32 = undefined,
    lay: [16]u8 = undefined,
    v: offload.Vram = undefined,
    counts: [shape.layers][shape.experts]u32 = @splat(@splat(0)),
    r: round.Round = undefined,

    fn init(f: *Fix, lines: usize, max_picks: usize) !void {
        f.v = offload.Vram.init(f.slots[0..lines], f.line[0..lines], f.stamp[0..lines], f.lay[0..lines]);
        f.r = try round.Round.init(testing.allocator, lines, max_picks);
    }
    fn deinit(f: *Fix) void {
        f.r.deinit(testing.allocator);
    }
    fn go(f: *Fix, picks: []const i32, now: u32, opt: round.Options) !void {
        return round.plan(&f.v, 3, picks, &f.counts, now, now, opt, &f.r);
    }
};

fn contains(list: []const u32, x: u32) bool {
    return std.mem.indexOfScalar(u32, list, x) != null;
}

test "exl3: hit first splits a one-pass round at its first miss, the held experts without a copy" {
    var f: Fix = .{};
    try f.init(8, 32);
    defer f.deinit();
    try f.go(&[_]i32{ 1, 2, 3, 4 }, 1, .{});
    // nothing held yet: the round stays one pass
    try testing.expect(!split(&f.r, 1));
    try f.go(&[_]i32{ 1, 9, 2, 512, 8, 9, -1, 3 }, 2, .{});
    try testing.expect(!split(&f.r, 0));
    try testing.expect(!split(&f.r, 4));
    try testing.expect(split(&f.r, 3));
    try testing.expect(f.r.masked());
    try testing.expectEqualSlices(u16, &.{ 1, 2, 3 }, f.r.passIds(0));
    try testing.expectEqualSlices(u16, &.{ 9, 8 }, f.r.passIds(1)); // 9 is picked twice: the hotter miss
    try testing.expectEqual(@as(usize, 0), f.r.passCopies(0).len);
    try testing.expectEqual(@as(usize, 2), f.r.passCopies(1).len);
    try testing.expectEqual(@as(usize, 2), f.r.misses_());
    for (f.r.passIds(1), f.r.passLines(1), f.r.passCopies(1)) |id, line, c| {
        try testing.expectEqual(id, c.id);
        try testing.expectEqual(line, c.line);
        try testing.expect(!contains(f.r.passLines(0), c.line));
    }
    // a split round is two passes: it is not split again
    try testing.expect(!split(&f.r, 1));
    // every pick held: one pass
    try f.go(&[_]i32{ 1, 8, 9 }, 3, .{});
    try testing.expect(f.r.allHits() and !split(&f.r, 1));
}

test "exl3: hit first leaves a round that needs more passes anyway as planned" {
    var f: Fix = .{};
    try f.init(4, 64);
    defer f.deinit();
    try f.go(&[_]i32{ 20, 21 }, 1, .{});
    var picks: [30]i32 = undefined;
    for (&picks, 0..) |*p, i| p.* = @intCast(10 + (i % 11)); // 10..20, 11 distinct; 20 held
    try f.go(&picks, 2, .{});
    const passes = f.r.n_passes;
    try testing.expect(passes > 2);
    try testing.expect(!split(&f.r, 1));
    try testing.expectEqual(passes, f.r.n_passes);
}

test "exl3: hit first never moves a copy into a line pass 0 reads, double buffering on, over random rounds" {
    var f: Fix = .{};
    try f.init(6, 64);
    defer f.deinit();
    var prng = std.Random.DefaultPrng.init(0x4817);
    const rng = prng.random();
    var prev: [shape.experts]u16 = undefined;
    var prev_n: usize = 0;
    var splits: usize = 0;
    for (1..400) |i| {
        var picks: [5]i32 = undefined;
        for (&picks) |*p| p.* = rng.uintLessThan(u8, 12);
        try f.go(&picks, @intCast(i), .{ .protect = prev[0..prev_n] });
        prev_n = f.r.n_ids;
        @memcpy(prev[0..prev_n], f.r.ids[0..prev_n]);
        const one = f.r.n_passes == 1;
        const held = f.r.n_ids - f.r.misses_();
        const did = split(&f.r, 1);
        try testing.expectEqual(one and held > 0 and f.r.misses_() > 0, did);
        if (!did) continue;
        splits += 1;
        try testing.expectEqual(held, f.r.passIds(0).len);
        try testing.expectEqual(f.r.misses_(), f.r.passCopies(1).len);
        for (f.r.passCopies(1), f.r.passIds(1), f.r.passLines(1)) |c, id, line| {
            try testing.expect(c.id == id and c.line == line);
            try testing.expect(!contains(f.r.passLines(0), c.line));
        }
    }
    try testing.expect(splits > 50);
}
