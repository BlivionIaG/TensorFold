const std = @import("std");
const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
const Tree = @import("lanes.zig").Tree;
const A = mx.Array;
const ti = mx.ti;

pub const max_streams = 8;
pub const max_rows = 128;

// Indices are local to this launch; parent values remain local to each stream.
pub const Layout = struct {
    streams: usize = 0,
    rows: usize = 0,
    firsts: [8]i32 = undefined,
    widths: [8]i32 = undefined,
    parents: [128]i32 = undefined,
    windows: [128 * 4]i32 = undefined,
    row_stream: [128]i32 = undefined,
    tree_meta: [16]i32 = undefined,
    chain: bool = true,
    slots: i32 = 1,
    base: [104]i32 = @splat(0),
    tile_stream: [56]i32 = undefined,
    q_rows: [56 * 16]i32 = undefined,
    tiles: i32 = 0,
    ca: i32 = 0,
    ncb: i32 = 0,
    nodes: [128 * 2]i32 = undefined,
    paths: [128 * 128]i32 = @splat(0),
    arrays: struct {
        windows: A = mx.empty,
        row_stream: A = mx.empty,
        parents: A = mx.empty,
        tree_meta: A = mx.empty,
        tile_stream: A = mx.empty,
        q_rows: A = mx.empty,
        nodes: A = mx.empty,
        paths: A = mx.empty,
        scale: A = mx.empty,
        padding: A = mx.empty,
        zero: A = mx.empty,
    } = .{},
    attention_meta: A = mx.empty,
    attention_strides: [16]i32 = @splat(0),

    pub fn init(parents: []const []const i32, starts: []const i32) !Layout {
        if (parents.len == 0 or parents.len > 8 or starts.len != parents.len) return error.InvalidStreams;
        var p = Layout{ .streams = parents.len };
        var widest: usize = 1;
        for (parents, starts, 0..) |rp, start, st| {
            if (start < 0 or rp.len > 128 - p.rows) return error.InvalidStreams;
            const t = try Tree.init(rp);
            _ = std.math.add(i32, start, @intCast(rp.len)) catch return error.InvalidStreams;
            const first: i32 = @intCast(p.rows);
            const width: i32 = @intCast(rp.len);
            p.firsts[st] = first;
            p.widths[st] = width;
            p.tree_meta[2 * st] = first;
            p.tree_meta[2 * st + 1] = width;
            p.chain = p.chain and t.chain;
            widest = @max(widest, rp.len);
            @memcpy(p.parents[p.rows..][0..rp.len], rp);
            const pt = @divTrunc(start, 64) * 64;
            const ca = @divTrunc(pt + 511, 512);
            const ncb = @divTrunc(start + t.max_depth, 512) - @divTrunc(pt, 512) + 1;
            const tiles = @divTrunc(6 * width + 15, 16);
            @memcpy(p.base[8 + 12 * st ..][0..7], &[_]i32{ pt, ca, width, p.tiles, start, ncb, first });
            for (0..@intCast(tiles)) |j| p.tile_stream[@as(usize, @intCast(p.tiles)) + j] = @intCast(st);
            for (0..@intCast(16 * tiles)) |j| p.q_rows[@as(usize, @intCast(16 * p.tiles)) + j] = 6 * first + if (j < 6 * rp.len) @as(i32, @intCast(j)) else 0;
            for (0..rp.len) |j| {
                const row = p.rows + j;
                p.row_stream[row] = @intCast(st);
                p.nodes[2 * row] = t.depths[j];
                p.nodes[2 * row + 1] = @intCast(st);
                for (0..4) |tap| {
                    const local = t.windows[4 * j + tap];
                    p.windows[4 * row + tap] = local + if (local >= 3) first else 0;
                }
                @memcpy(p.paths[128 * row ..][0..@intCast(t.depths[j] + 1)], t.paths[128 * j ..][0..@intCast(t.depths[j] + 1)]);
            }
            p.rows += rp.len;
            p.tiles += tiles;
            p.ca = @max(p.ca, ca);
            p.ncb = @max(p.ncb, ncb);
        }
        if (!p.chain and widest > 32) return error.InvalidStreams;
        p.slots = if (p.chain) 1 else if (widest <= 16) 16 else 32;
        @memcpy(p.base[0..7], &[_]i32{ @intCast(p.streams), p.ca, p.tiles, p.ncb, @intCast(p.rows), 16 * p.tiles, p.ca });
        return p;
    }

    pub fn prepare(p: *Layout, s: *mx.Scope) !void {
        p.arrays.windows = try s.ints(p.windows[0 .. p.rows * 4]);
        p.arrays.row_stream = try s.ints(p.row_stream[0..p.rows]);
        p.arrays.parents = try s.ints(p.parents[0..p.rows]);
        p.arrays.tree_meta = try s.ints(p.tree_meta[0 .. p.streams * 2]);
        if (mx.tensor_units) {
            p.arrays.tile_stream = try s.ints(p.tile_stream[0..@intCast(p.tiles)]);
            p.arrays.q_rows = try s.ints(p.q_rows[0..@intCast(16 * p.tiles)]);
            p.arrays.nodes = try s.ints(p.nodes[0 .. 2 * p.rows]);
            p.arrays.paths = try s.ints(p.paths[0 .. 128 * p.rows]);
            p.arrays.scale = try s.scalar(0.0625);
            p.arrays.padding = try s.zeros(&.{ 4, @intCast(p.rows), 10, 256 }, mx.bf16);
            if (p.ca == 0) p.arrays.zero = try s.zeros(&.{16}, mx.f32t);
        }
    }

    fn attentionMeta(p: *Layout, s: *mx.Scope, keys: []const A, values: []const A) !A {
        var strides: [16]i32 = @splat(0);
        for (keys, values, 0..) |key, value, i| {
            strides[2 * i] = mx.dim(key, 2) * 256;
            strides[2 * i + 1] = mx.dim(value, 2) * 256;
        }
        if (p.attention_meta.ctx != null and std.mem.eql(i32, &strides, &p.attention_strides)) return p.attention_meta;
        var base = p.base;
        for (0..p.streams) |i| @memcpy(base[8 + i * 12 + 7 ..][0..2], strides[2 * i ..][0..2]);
        const meta = try s.ints(&base);
        p.attention_meta = meta;
        p.attention_strides = strides;
        return meta;
    }
};

pub const Commit = struct {
    rows: [128]i32 = undefined,
    count: usize = 0,
    meta: [16]i32 = undefined,
    tails: [24]i32 = undefined,
    streams: usize = 0,
    arrays: struct { rows: A = mx.empty, meta: A = mx.empty, tails: A = mx.empty } = .{},

    pub fn init(p: *const Layout, paths: []const []const i32) !Commit {
        if (paths.len != p.streams) return error.InvalidCommit;
        var c = Commit{ .streams = p.streams };
        for (paths, 0..) |path, st| {
            if (path.len > p.widths[st]) return error.InvalidCommit;
            const first: usize = @intCast(p.firsts[st]);
            c.meta[2 * st] = @intCast(c.count);
            c.meta[2 * st + 1] = @intCast(path.len);
            for (path, 0..) |row, j| {
                if (row < 0 or row >= p.widths[st] or p.parents[first + @as(usize, @intCast(row))] != (if (j == 0) @as(i32, -1) else path[j - 1])) return error.InvalidCommit;
                c.rows[c.count] = p.firsts[st] + row;
                c.count += 1;
            }
            for (0..3) |j| {
                const n = path.len + j;
                c.tails[3 * st + j] = if (n < 3) @intCast(n) else 3 + p.firsts[st] + path[n - 3];
            }
        }
        return c;
    }

    pub fn prepare(c: *Commit, s: *mx.Scope) !void {
        c.arrays.rows = try s.ints(if (c.count == 0) &.{0} else c.rows[0..c.count]);
        c.arrays.meta = try s.ints(c.meta[0 .. c.streams * 2]);
        c.arrays.tails = try s.ints(c.tails[0 .. c.streams * 3]);
    }
};

pub fn pre(k: *mx.Kernels, s: *mx.Scope, p: *const Layout, qkv: A, states: []const A, cw: A, a: A, b: A, alog: A, dt: A) ![5]A {
    return preImpl(k, s, p, qkv, states, cw, a, b, alog, dt, 48, 0, 0);
}

pub fn preStack(k: *mx.Kernels, s: *mx.Scope, p: *const Layout, qkv: A, states: []const A, cw: A, zba: A, alog: A, dt: A) ![5]A {
    return preImpl(k, s, p, qkv, states, cw, zba, zba, alog, dt, 6240, 6192, 6144);
}

fn preImpl(k: *mx.Kernels, s: *mx.Scope, p: *const Layout, qkv: A, states: []const A, cw: A, a: A, b: A, alog: A, dt: A, stride: i32, a_offset: i32, b_offset: i32) ![5]A {
    if (states.len != p.streams) return error.InvalidStreams;
    var inputs: [16]A = undefined;
    inputs[0] = qkv;
    for (0..8) |i| inputs[1 + i] = states[if (i < states.len) i else 0];
    @memcpy(inputs[9..], &[_]A{ cw, p.arrays.windows, a, b, alog, dt, p.arrays.row_stream });
    const r: i32 = @intCast(p.rows);
    return k.run(s, src.stream_gdn_pre, &inputs, &.{ ti("NK", 16), ti("NV", 48), ti("DK", 128), ti("DV", 128), ti("TAPS", 4), ti("ZS", stride), ti("AO", a_offset), ti("BO", b_offset) }, .{ 32, 80, r }, .{ 32, 1, 1 }, &.{ .{ .shape = &.{ 1, r, 16, 128 } }, .{ .shape = &.{ 1, r, 16, 128 } }, .{ .shape = &.{ 1, r, 48, 128 } }, .{ .shape = &.{ 1, r, 48 }, .dtype = mx.f32t }, .{ .shape = &.{ 1, r, 48 } } });
}

pub fn recurrence(k: *mx.Kernels, s: *mx.Scope, p: *const Layout, vals: [5]A, states: []const A) !A {
    if (states.len != p.streams) return error.InvalidStreams;
    var inputs: [15]A = undefined;
    @memcpy(inputs[0..5], &vals);
    for (0..8) |i| inputs[5 + i] = states[if (i < states.len) i else 0];
    inputs[13] = p.arrays.parents;
    inputs[14] = p.arrays.tree_meta;
    return (try k.run(s, src.stream_gdn_tree, &inputs, &.{ mx.td("InT", mx.bf16), ti("Dk", 128), ti("Dv", 128), ti("Hk", 16), ti("Hv", 48), ti("MAXW", p.slots), mx.tb("CHAIN", p.chain) }, .{ 32, 128, @intCast(48 * p.streams) }, .{ 32, 4, 1 }, &.{.{ .shape = &.{ 1, @intCast(p.rows), 48, 128 } }}))[0];
}

pub const Step = struct { y: A, states: [8]A };

pub fn step(k: *mx.Kernels, s: *mx.Scope, p: *const Layout, vals: [5]A, states: []const A) !Step {
    if (states.len != p.streams or p.rows != p.streams) return error.InvalidStreams;
    var inputs: [15]A = undefined;
    @memcpy(inputs[0..5], &vals);
    var outputs: [9]mx.Output = undefined;
    outputs[0] = .{ .shape = &.{ 1, @intCast(p.rows), 48, 128 } };
    for (0..8) |i| {
        inputs[5 + i] = states[if (i < states.len) i else 0];
        outputs[1 + i] = .{ .shape = if (i < states.len) &.{ 1, 48, 128, 128 } else &.{16}, .dtype = mx.f32t };
    }
    inputs[13] = p.arrays.parents;
    inputs[14] = p.arrays.tree_meta;
    var out: [9]A = undefined;
    try k.runInto(s, src.stream_gdn_step, &inputs, &.{ mx.td("InT", mx.bf16), ti("Dk", 128), ti("Dv", 128), ti("Hk", 16), ti("Hv", 48), ti("MAXW", 1), mx.tb("CHAIN", true) }, .{ 32, 128, @intCast(48 * p.streams) }, .{ 32, 4, 1 }, &outputs, &out, null);
    return .{ .y = out[0], .states = out[1..9].* };
}

pub fn replay(k: *mx.Kernels, s: *mx.Scope, c: *const Commit, vals: [5]A, states: []const A) ![8]A {
    if (states.len == 0 or states.len != c.streams) return error.InvalidStreams;
    var inputs: [15]A = undefined;
    @memcpy(inputs[0..5], &vals);
    var outputs: [8]mx.Output = undefined;
    for (0..8) |i| {
        inputs[5 + i] = states[if (i < states.len) i else 0];
        outputs[i] = .{ .shape = if (i < states.len) &.{ 1, 48, 128, 128 } else &.{16}, .dtype = mx.f32t };
    }
    inputs[13] = c.arrays.rows;
    inputs[14] = c.arrays.meta;
    var out: [8]A = undefined;
    try k.runInto(s, src.stream_gdn_replay, &inputs, &.{ ti("Dk", 128), ti("Dv", 128), ti("Hk", 16), ti("Hv", 48) }, .{ 32, 128, @intCast(48 * states.len) }, .{ 32, 4, 1 }, &outputs, &out, null);
    return out;
}

pub fn tails(k: *mx.Kernels, s: *mx.Scope, c: *const Commit, qkv: A, states: []const A) ![8]A {
    if (states.len == 0 or states.len != c.streams) return error.InvalidStreams;
    var inputs: [10]A = undefined;
    var outputs: [8]mx.Output = undefined;
    for (0..8) |i| {
        inputs[i] = states[if (i < states.len) i else 0];
        outputs[i] = .{ .shape = if (i < states.len) &.{ 1, 3, 10240 } else &.{16} };
    }
    inputs[8] = qkv;
    inputs[9] = c.arrays.tails;
    var out: [8]A = undefined;
    try k.runInto(s, src.stream_gdn_tails, &inputs, &.{ ti("C", 10240), ti("NKEEP", 3) }, .{ 10240, 3, @intCast(states.len) }, .{ 256, 1, 1 }, &outputs, &out, null);
    return out;
}

pub fn attention(k: *mx.Kernels, s: *mx.Scope, p: *Layout, q: A, keys: []const A, values: []const A) !A {
    if (keys.len != p.streams or values.len != p.streams) return error.InvalidStreams;
    const meta = try p.attentionMeta(s, keys, values);
    const scale = p.arrays.scale;
    const nodes = p.arrays.nodes;
    const r: i32 = @intCast(p.rows);
    const per_head = try s.transpose(try s.reshape(q, &.{ 4, 6, r, 256 }), &.{ 0, 2, 1, 3 });
    var inputs: [24]A = undefined;
    for (0..8) |i| {
        inputs[1 + i] = keys[if (i < keys.len) i else 0];
        inputs[9 + i] = values[if (i < values.len) i else 0];
    }
    inputs[17] = scale;
    inputs[18] = meta;
    var a: [5]A = @splat(mx.empty);
    if (p.ca > 0) {
        inputs[0] = try s.contiguous(try s.take(try s.reshape(per_head, &.{ 4, r * 6, 256 }), p.arrays.q_rows, 1));
        inputs[19] = p.arrays.tile_stream;
        const sg = @min(p.tiles, 16);
        const n = 4 * p.ca * 16 * p.tiles;
        a = try k.run(s, src.stream_attention_partial, inputs[0..20], &.{ ti("G", 6), ti("D", 256), ti("SG", sg), ti("CK", 512), ti("TK", 64), ti("MG", 8), ti("MS", 12) }, .{ 4 * 32 * sg, p.ca, @divTrunc(p.tiles + sg - 1, sg) }, .{ 32 * sg, 1, 1 }, &.{ .{ .shape = &.{n * 256}, .dtype = mx.f32t }, .{ .shape = &.{n}, .dtype = mx.f32t }, .{ .shape = &.{n}, .dtype = mx.f32t } });
    } else {
        @memset(a[0..3], p.arrays.zero);
    }
    inputs[0] = try s.contiguous(try s.cat(&.{ per_head, p.arrays.padding }, 2));
    inputs[19] = p.arrays.paths;
    inputs[20] = nodes;
    @memcpy(inputs[21..24], a[0..3]);
    // Tail has 24 inputs, including all eight independent K/V buffers.
    const n = 4 * p.ncb * r * 16;
    const b = try k.run(s, src.stream_attention_tail, inputs[0..24], &.{ ti("G", 6), ti("D", 256), ti("CK", 512), ti("TK", 64), ti("MAXD", 128), ti("MG", 8), ti("MS", 12) }, .{ 4 * 32, p.ncb, r }, .{ 32, 1, 1 }, &.{ .{ .shape = &.{n * 256}, .dtype = mx.f32t }, .{ .shape = &.{n}, .dtype = mx.f32t }, .{ .shape = &.{n}, .dtype = mx.f32t } });
    return (try k.run(s, src.stream_attention_merge, &.{ a[0], a[1], a[2], b[0], b[1], b[2], meta, nodes }, &.{ ti("G", 6), ti("D", 256), ti("CK", 512), ti("MG", 8), ti("MS", 12) }, .{ 4 * 32, r * 6, 1 }, .{ 32, 1, 1 }, &.{.{ .shape = &.{ 1, 24, r, 256 } }}))[0];
}

test "shared stream layouts preserve local trees and chunk boundaries" {
    const parents = [_][]const i32{ &.{ -1, 0, 0, 2 }, &.{-1} };
    const p = try Layout.init(&parents, &.{ 511, 64 });
    try std.testing.expectEqualSlices(i32, &.{ 0, 4 }, p.firsts[0..2]);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0, 2, -1 }, p.parents[0..5]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 7 }, p.windows[16..20]);
    try std.testing.expectEqualSlices(i32, &.{ 448, 1, 4, 0, 511, 2, 0 }, p.base[8..15]);
    try std.testing.expectEqualSlices(i32, &.{ 64, 1, 1, 2, 64, 1, 4 }, p.base[20..27]);
    const c = try Commit.init(&p, &.{ &.{ 0, 2, 3 }, &.{} });
    try std.testing.expectEqualSlices(i32, &.{ 0, 2, 3 }, c.rows[0..c.count]);
    try std.testing.expectEqualSlices(i32, &.{ 3, 5, 6, 0, 1, 2 }, c.tails[0..6]);
    try std.testing.expectError(error.InvalidCommit, Commit.init(&p, &.{ &.{ 0, 1, 3 }, &.{} }));
    try std.testing.expectError(error.InvalidStreams, Layout.init(&parents, &.{ -1, 0 }));
    try std.testing.expectError(error.InvalidStreams, Layout.init(&.{}, &.{}));
}
