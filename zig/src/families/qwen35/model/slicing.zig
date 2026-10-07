//! Tensor-parallel slicing on the host: each rank's heads, columns, groups, experts and vocabulary rows, byte for byte.

const std = @import("std");
const config = @import("config.zig");
const host = @import("host.zig");
const table = @import("table.zig");

const Tensor = table.Tensor;
const Allocator = std.mem.Allocator;

pub const Error = error{ UnevenSplit, DenseProjection, KvHeads } || Allocator.Error;

pub const Rank = struct { rank: usize, world: usize };

/// Rows [from, to) of a tensor's first dimension.
const Span = struct { from: usize, to: usize };

fn even(size: usize, world: usize, what: []const u8) Error!usize {
    if (size % world != 0) {
        std.log.err("{s}: {d} does not split into {d} equal parts", .{ what, size, world });
        return error.UnevenSplit;
    }
    return size / world;
}

/// The spec as a rank sees it: its heads and value heads (the vocabulary stays whole; the head's rows are split).
pub fn localSpec(s: config.Spec, r: Rank) Error!config.Spec {
    var out = s;
    out.heads = try even(s.heads, r.world, "heads");
    out.key_heads = try even(s.key_heads, r.world, "key_heads");
    out.value_heads = try even(s.value_heads, r.world, "value_heads");
    _ = try even(s.vocab, r.world, "vocab");
    out.kv_heads = @max(1, s.kv_heads / r.world);
    return out;
}

fn rowBytes(t: Tensor) usize {
    return t.numel() / t.shape[0] * t.dtype.size();
}

/// `spans` of `t`'s rows joined in order; one span shares the source's bytes.
fn takeRows(a: Allocator, t: Tensor, spans: []const Span) Error!Tensor {
    const row = rowBytes(t);
    var rows: usize = 0;
    for (spans) |s| rows += s.to - s.from;
    var out = t;
    out.shape[0] = rows;
    if (spans.len == 1) {
        out.bytes = t.bytes[spans[0].from * row .. spans[0].to * row];
        return out;
    }
    const bytes = try a.alloc(u8, rows * row);
    var at: usize = 0;
    for (spans) |s| {
        const n = (s.to - s.from) * row;
        @memcpy(bytes[at..][0..n], t.bytes[s.from * row ..][0..n]);
        at += n;
    }
    out.bytes = bytes;
    return out;
}

/// Columns [from, to) of a rank-2 tensor, copied.
fn takeCols(a: Allocator, t: Tensor, from: usize, to: usize) Error!Tensor {
    const size = t.dtype.size();
    const width = (to - from) * size;
    const bytes = try a.alloc(u8, t.shape[0] * width);
    for (0..t.shape[0]) |r| @memcpy(bytes[r * width ..][0..width], t.bytes[(r * t.shape[1] + from) * size ..][0..width]);
    var out = t;
    out.shape[1] = to - from;
    out.bytes = bytes;
    return out;
}

fn affine(p: host.Projection) Error!host.Affine {
    return switch (p) {
        .affine => |x| x,
        .dense => error.DenseProjection,
    };
}

/// `_rows`: output rows `spans` of a projection, the share of a column-split one.
fn sliceRows(a: Allocator, p: host.Projection, spans: []const Span) Error!host.Projection {
    const x = try affine(p);
    return .{ .affine = .{ .words = try takeRows(a, x.words, spans), .scales = try takeRows(a, x.scales, spans), .biases = try takeRows(a, x.biases, spans), .bits = x.bits, .group = x.group } };
}

/// `_cols`: one rank's whole input groups of a row-split projection, whose fp32 outputs the ranks sum.
fn cols(a: Allocator, p: host.Projection, r: Rank, what: []const u8) Error!host.Projection {
    const x = try affine(p);
    const groups = try even(x.scales.shape[1], r.world, what);
    const words = groups * x.group * x.bits / 32;
    return .{ .affine = .{
        .words = try takeCols(a, x.words, r.rank * words, (r.rank + 1) * words),
        .scales = try takeCols(a, x.scales, r.rank * groups, (r.rank + 1) * groups),
        .biases = try takeCols(a, x.biases, r.rank * groups, (r.rank + 1) * groups),
        .bits = x.bits,
        .group = x.group,
    } };
}

/// This rank's rows of each segment of a concatenated output (q | k | v), each segment split evenly.
fn segments(widths: []const usize, r: Rank, out: []Span, what: []const u8) Error![]const Span {
    var base: usize = 0;
    for (widths, out[0..widths.len]) |w, *span| {
        const part = try even(w, r.world, what);
        span.* = .{ .from = base + r.rank * part, .to = base + (r.rank + 1) * part };
        base += w;
    }
    return out[0..widths.len];
}

/// This rank's k or v rows; with fewer KV heads than ranks each head is kept by its query ranks.
fn kvSpan(width: usize, kv_heads: usize, r: Rank, out: []Span) Error![]const Span {
    if (kv_heads >= r.world) return segments(&.{width}, r, out, "kv rows");
    if (r.world % kv_heads != 0) return error.KvHeads;
    const head = r.rank / (r.world / kv_heads);
    const part = width / kv_heads;
    out[0] = .{ .from = head * part, .to = (head + 1) * part };
    return out[0..1];
}

/// One layer's tensors for rank `r`, cut in place (`l.arena` holds the copies). `s` is the whole model's spec.
pub fn layer(l: *host.Layer, s: config.Spec, r: Rank) Error!void {
    const a = l.arena.allocator();
    var buf: [3]Span = undefined;
    switch (l.body) {
        .full => |*f| {
            f.q = try sliceRows(a, f.q, try segments(&.{s.heads * s.head_dim * 2}, r, &buf, "q_proj"));
            const kv = s.kv_heads * s.head_dim;
            f.k = try sliceRows(a, f.k, try kvSpan(kv, s.kv_heads, r, &buf));
            f.v = try sliceRows(a, f.v, try kvSpan(kv, s.kv_heads, r, &buf));
            f.o = try cols(a, f.o, r, "o_proj groups");
            try mlp(a, &f.mlp, r);
        },
        .linear => |*x| {
            const qkv = try segments(&.{ s.keyWidth(), s.keyWidth(), s.valueWidth() }, r, &buf, "in_proj_qkv");
            x.qkv = try sliceRows(a, x.qkv, qkv);
            x.conv = try takeRows(a, x.conv, qkv);
            x.z = try sliceRows(a, x.z, try segments(&.{s.valueWidth()}, r, &buf, "in_proj_z"));
            const heads = try segments(&.{s.value_heads}, r, &buf, "value heads");
            x.a = try sliceRows(a, x.a, heads);
            x.b = try sliceRows(a, x.b, heads);
            x.a_log = try takeRows(a, x.a_log, heads);
            x.dt_bias = try takeRows(a, x.dt_bias, heads);
            x.out = try cols(a, x.out, r, "out_proj groups");
            try mlp(a, &x.mlp, r);
        },
    }
}

fn mlp(a: Allocator, m: *host.Mlp, r: Rank) Error!void {
    switch (m.*) {
        .dense => |*d| {
            var buf: [1]Span = undefined;
            const spans = try segments(&.{d.gate.rows()}, r, &buf, "mlp");
            d.gate = try sliceRows(a, d.gate, spans);
            d.up = try sliceRows(a, d.up, spans);
            d.down = try cols(a, d.down, r, "mlp.down_proj groups");
        },
        .routed => |*x| try routed(a, x, r),
    }
}

/// Experts [E / world * rank, ...) of a stacked tensor (E + 1 experts, the shared one last), and the shared one on rank 0.
fn takeExperts(a: Allocator, t: Tensor, r: Rank, part: usize) Error!Tensor {
    const total = t.shape[0] - 1;
    const row = rowBytes(t);
    const mine = t.bytes[r.rank * part * row ..][0 .. part * row];
    if (r.rank != 0) {
        var out = t;
        out.shape[0] = part;
        out.bytes = mine;
        return out;
    }
    const bytes = try a.alloc(u8, (part + 1) * row);
    @memcpy(bytes[0 .. part * row], mine);
    @memcpy(bytes[part * row ..], t.bytes[total * row ..][0..row]);
    var out = t;
    out.shape[0] = part + 1;
    out.bytes = bytes;
    return out;
}

fn side(a: Allocator, x: host.Side, r: Rank, part: usize) Error!host.Side {
    return .{ .words = try takeExperts(a, x.words, r, part), .scales = try takeExperts(a, x.scales, r, part), .biases = try takeExperts(a, x.biases, r, part) };
}

/// A rank's contiguous routed experts, the shared one on rank 0, and the remap to its own ids (-1 where another holds it).
fn routed(a: Allocator, x: *host.Routed, r: Rank) Error!void {
    const total = x.experts.count - 1;
    const part = try even(total, r.world, "experts");
    x.experts.fused = try side(a, x.experts.fused, r, part);
    x.experts.down = try side(a, x.experts.down, r, part);
    x.experts.count = part + @intFromBool(r.rank == 0);
    const remap = try a.alloc(i32, total + 1);
    @memset(remap, -1);
    for (0..part) |i| remap[r.rank * part + i] = @intCast(i);
    if (r.rank == 0) remap[total] = @intCast(part);
    x.remap = .{ .dtype = .i32, .rank = 1, .shape = .{ total + 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(remap) };
}

/// The output head's rows for rank `r`: its share of the vocabulary.
pub fn vocabRows(a: Allocator, p: host.Projection, vocab: usize, r: Rank) Error!host.Projection {
    var buf: [1]Span = undefined;
    return sliceRows(a, p, try segments(&.{vocab}, r, &buf, "vocab"));
}

test "a column split keeps whole groups and a row split its own rows" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var words: [4 * 4]u32 = undefined;
    for (&words, 0..) |*w, i| w.* = @intCast(i);
    var scales: [4 * 2]u16 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const p: host.Projection = .{ .affine = .{
        .words = .{ .dtype = .i32, .rank = 2, .shape = .{ 4, 4, 1, 1 }, .bytes = std.mem.sliceAsBytes(&words) },
        .scales = .{ .dtype = .bf16, .rank = 2, .shape = .{ 4, 2, 1, 1 }, .bytes = std.mem.sliceAsBytes(&scales) },
        .biases = .{ .dtype = .bf16, .rank = 2, .shape = .{ 4, 2, 1, 1 }, .bytes = std.mem.sliceAsBytes(&scales) },
        .bits = 8, // four words a row: sixteen codes, two groups of eight
        .group = 8,
    } };
    const c = try cols(a, p, .{ .rank = 1, .world = 2 }, "test");
    try std.testing.expectEqual(@as(usize, 2), c.affine.words.shape[1]);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 6, 7, 10, 11, 14, 15 }, @alignCast(std.mem.bytesAsSlice(u32, c.affine.words.bytes)));
    try std.testing.expectEqualSlices(u16, &.{ 1, 3, 5, 7 }, @alignCast(std.mem.bytesAsSlice(u16, c.affine.scales.bytes)));
    const q = try sliceRows(a, p, &.{ .{ .from = 1, .to = 2 }, .{ .from = 3, .to = 4 } });
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 6, 7, 12, 13, 14, 15 }, @alignCast(std.mem.bytesAsSlice(u32, q.affine.words.bytes)));
}

test "a kv head is kept by its query ranks" {
    var buf: [1]Span = undefined;
    const spans = try kvSpan(256, 2, .{ .rank = 3, .world = 4 }, &buf);
    try std.testing.expectEqual(Span{ .from = 128, .to = 256 }, spans[0]);
    try std.testing.expectError(error.KvHeads, kvSpan(256, 3, .{ .rank = 0, .world = 4 }, &buf));
}
