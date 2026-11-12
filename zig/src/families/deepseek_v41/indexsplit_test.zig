//! Host test of TF_DSV41_INDEX_SPLIT: the reader, and the long-prefill program with half of the rows a rank.
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const check = @import("calls_check.zig");
const pk = @import("serve_knobs.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "TF_DSV41_INDEX_SPLIT: 0 / 1" {
    defer Fake.pairs = &.{};
    Fake.pairs = &.{};
    try testing.expect(!try pk.indexSplit(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_INDEX_SPLIT", "1" }};
    try testing.expect(try pk.indexSplit(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_INDEX_SPLIT", "2" }};
    try testing.expectError(error.BadKnob, pk.indexSplit(&Fake.get));
}

fn arg(c: calls.Call, name: []const u8) ?calls.Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    return null;
}

const Tally = struct { rows: i64 = 0, streams: usize = 0, merges: usize = 0, gathers: usize = 0, first: i64 = -1 };

fn tally(cs: []const calls.Call, n: i64) !Tally {
    var t: Tally = .{};
    for (cs) |c| {
        if (std.mem.eql(u8, c.name, "_stream")) {
            t.rows += c.grid[0];
            t.streams += 1;
            if (t.first < 0) t.first = arg(c, "QI").?.t.offset;
        }
        if (std.mem.eql(u8, c.name, "glue.stream_merge")) t.merges += 1;
        if (std.mem.eql(u8, c.name, "glue.sel_gather")) {
            // sel [n, K], this rank's rows, the gathered rows, the half: one shape each
            const half = @divExact(n, 2);
            try testing.expectEqual(half, c.args[3].arg.i);
            try testing.expectEqualSlices(i64, &.{ n, 512 }, c.args[0].arg.t.shape);
            try testing.expectEqualSlices(i64, &.{ half, 512 }, c.args[1].arg.t.shape);
            try testing.expectEqualSlices(i64, &.{ n, 512 }, c.args[2].arg.t.shape);
            t.gathers += 1;
        }
    }
    return t;
}

test "TF_DSV41_INDEX_SPLIT on the long-context prefill: each rank's half of the rows, then one gather a streamed layer" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    var w: block.Widths = .{};
    try check.parseDense(aa, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(aa, &w, @embedFile("fixtures/expert-k2.txt"));
    var enc: [21]u32 = undefined;
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    const base: block.Options = .{ .gm_v2 = .one, .limit = 1 << 20, .rope_rows = (1 << 20) + 2048, .index_budget = 64 << 20 };
    for ([_]i64{ 2048, 1024, 300 }) |n| for ([_]i64{ 2 * 4096, 131072 - 2048 }) |start| {
        const off = try tally(try bp.emitReplay(aa, &cfg, &w, base, &enc, n, start, .encoder), n);
        try testing.expect(off.streams > 0);
        try testing.expectEqual(@as(usize, 0), off.gathers);
        var firsts: [2]i64 = undefined;
        for ([_]u32{ 0, 1 }) |rank| {
            var o = base;
            o.index_split = true;
            o.rank = rank;
            const cs = try bp.emitReplay(aa, &cfg, &w, o, &enc, n, start, .encoder);
            const on = try tally(cs, n);
            // every streamed layer: half the rows here, the merge a block, one gather
            try testing.expectEqual(@divExact(off.rows, 2), on.rows);
            try testing.expectEqual(on.streams, on.merges);
            try testing.expect(on.gathers > 0 and on.gathers <= on.streams);
            firsts[rank] = on.first;
            var p: buffers.Plan = .{ .a = aa };
            try p.add(cs);
            try testing.expect(p.sizes.get("L.ix.selall").? >= @as(u64, @intCast(n * 512 * 4)));
            try testing.expect(p.sizes.get("L.ix.selh").? >= @as(u64, @intCast(@divExact(n, 2) * 512 * 4)));
        }
        // rank 1 starts its first block half a segment's rows later (whole rows of QI)
        try testing.expect(firsts[1] > firsts[0]);
        try testing.expectEqual(@as(i64, 0), @rem(firsts[1] - firsts[0], @divExact(n, 2)));
    };
}

test "TF_DSV41_INDEX_SPLIT leaves odd, short and single-rank segments as they were" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    var w: block.Widths = .{};
    try check.parseDense(aa, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(aa, &w, @embedFile("fixtures/expert-k2.txt"));
    var enc: [21]u32 = undefined;
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    const base: block.Options = .{ .gm_v2 = .one, .limit = 1 << 20, .rope_rows = (1 << 20) + 2048, .index_budget = 64 << 20 };
    const Case = struct { n: i64, world: u32 };
    for ([_]Case{ .{ .n = 2047, .world = 2 }, .{ .n = 200, .world = 2 } }) |c| {
        var o = base;
        o.world = c.world;
        const off = try bp.emitReplay(aa, &cfg, &w, o, &enc, c.n, 2 * 4096, .encoder);
        o.index_split = true;
        const on = try bp.emitReplay(aa, &cfg, &w, o, &enc, c.n, 2 * 4096, .encoder);
        try testing.expectEqual(off.len, on.len);
        for (off, on) |x, y| try testing.expectEqualStrings(x.name, y.name);
    }
}
