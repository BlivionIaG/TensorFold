//! One rank's full-attention head, the o_proj exchange, and the indexer.
//! `attend` does not add the other rank's head: that would be an all-reduce
//! inside the head loop. `exchangeO` is the split_k partial, outside `attend`.

const std = @import("std");
const shape = @import("exl3_shape.zig");
const kv = @import("exl3_kv.zig");

/// Scores for `head` only. This table does not hold the other rank's head.
pub fn attend(table: *const kv.Table, head: usize, layer: usize, query: []const f32, scores: []f32) error{ Head, Slot, Length }!void {
    if (scores.len != table.tokens) return error.Length;
    // int8 and int4 slots are not an f32 dot. Refuse before any store.
    if (table.dtype != .bf16) return error.Length;
    if (table.heads == 0 or head < table.head or head >= table.head + table.heads) return error.Head;

    // Same split `locate` uses: `contextKvRank` bytes over full-attention layers, tokens, and this rank's heads.
    const layers = shape.layers / shape.full_attention_interval;
    if (table.tokens == 0 or layers == 0) return error.Slot;
    const denom = layers * table.tokens * table.heads;
    if (table.bytes.len % denom != 0) return error.Slot;
    const one = table.bytes.len / denom;
    // bf16 K and V. Query length is that slot in f32 lanes, not `head_dim` alone.
    var raw: [2 * shape.head_dim * 2]u8 = undefined;
    if (one != raw.len or one % 2 != 0 or query.len != one / 2) return error.Length;

    for (scores, 0..) |*score, token| {
        try table.read(head, layer, token, raw[0..one]);
        var sum: f32 = 0;
        for (query, 0..) |q, i| {
            const bits = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
            const lane: f32 = @as(f32, @bitCast(@as(u32, bits) << 16));
            sum += q * lane;
        }
        score.* = sum;
    }
}

/// fp32 sum of one `o_proj` split_k partial, then the other. Not called from `attend`.
pub fn exchangeO(partial0: []const f32, partial1: []const f32, out: []f32) error{Length}!void {
    if (partial0.len != partial1.len or partial0.len != out.len) return error.Length;
    for (out, partial0, partial1) |*slot, a, b| slot.* = a + b;
}

/// exl3_bounce_bytes for the o_proj partial: D2H of the partial plus H2D of the summed hidden, one full-attention layer.
pub fn bounceBytes(tokens: usize) u64 {
    return @as(u64, tokens) * 5120 * 2;
}

/// One indexer query against each key row. Does not read a KV table and does not allocate.
pub fn indexScores(index_q: []const f32, index_k: []const f32, out: []f32) error{Length}!void {
    if (index_k.len != index_q.len * out.len) return error.Length;
    for (out, 0..) |*score, t| {
        const row = index_k[t * index_q.len ..][0..index_q.len];
        var sum: f32 = 0;
        for (index_q, row) |q, k| sum += q * k;
        score.* = sum;
    }
}

test "attend reads only this rank's head" {
    const fill = struct {
        fn bf16(dst: []u8, v: f32) void {
            const u: u32 = @bitCast(v);
            const h: u16 = @truncate(u >> 16);
            for (0..dst.len / 2) |i| std.mem.writeInt(u16, dst[i * 2 ..][0..2], h, .little);
        }
    };
    const gpa = std.testing.allocator;
    const tokens: usize = 4;
    const ranks: usize = 2;
    const layer: usize = 0;
    // Slot `read` expects: bf16 K and V, from the same width `contextKvRank` stores.
    const slot_len = 2 * shape.head_dim * 2;
    var r0 = try kv.Table.init(gpa, tokens, .bf16, 0, ranks);
    defer r0.deinit(gpa);
    var r1 = try kv.Table.init(gpa, tokens, .bf16, 1, ranks);
    defer r1.deinit(gpa);

    var query: [slot_len / 2]f32 = @splat(1);
    var slot: [slot_len]u8 = undefined;
    var expect: [tokens]f32 = undefined;
    for (0..tokens) |t| {
        const v: f32 = @floatFromInt(t + 1);
        fill.bf16(&slot, v);
        try r0.write(0, layer, t, &slot);
        expect[t] = v * @as(f32, @floatFromInt(query.len));
    }
    for (0..tokens) |t| {
        fill.bf16(&slot, 7);
        try r1.write(1, layer, t, &slot);
    }

    var scores: [tokens]f32 = @splat(9);
    try attend(&r0, 0, layer, &query, &scores);
    try std.testing.expectEqualSlices(f32, &expect, &scores);

    // Head 1 lives on rank 1. Rank 0 must not write scores for it.
    scores = @splat(3);
    try std.testing.expectError(error.Head, attend(&r0, 1, layer, &query, &scores));
    try std.testing.expect(std.mem.allEqual(f32, &scores, 3));

    const head1 = 7 * @as(f32, @floatFromInt(query.len));
    try attend(&r1, 1, layer, &query, &scores);
    try std.testing.expect(std.mem.allEqual(f32, &scores, head1));
    scores = @splat(3);
    try std.testing.expectError(error.Head, attend(&r1, 0, layer, &query, &scores));
    try std.testing.expect(std.mem.allEqual(f32, &scores, 3));

    var short_scores: [tokens - 1]f32 = @splat(4);
    try std.testing.expectError(error.Length, attend(&r0, 0, layer, &query, &short_scores));
    try std.testing.expect(std.mem.allEqual(f32, &short_scores, 4));

    var short_query: [1]f32 = .{1};
    scores = @splat(6);
    try std.testing.expectError(error.Length, attend(&r0, 0, layer, &short_query, &scores));
    try std.testing.expect(std.mem.allEqual(f32, &scores, 6));

    var quant = try kv.Table.init(gpa, tokens, .int8, 0, ranks);
    defer quant.deinit(gpa);
    scores = @splat(2);
    try std.testing.expectError(error.Length, attend(&quant, 0, layer, &query, &scores));
    try std.testing.expect(std.mem.allEqual(f32, &scores, 2));
}

test "exchangeO sums split_k partials outside attend" {
    const partial0 = [_]f32{ 1, 2, 3 };
    const partial1 = [_]f32{ 4, 5, 6 };
    var out: [3]f32 = @splat(99);
    try exchangeO(&partial0, &partial1, &out);
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9 }, &out);

    var held: [2]f32 = @splat(8);
    try std.testing.expectError(error.Length, exchangeO(&partial0, &partial1, &held));
    try std.testing.expect(std.mem.allEqual(f32, &held, 8));

    const ragged = [_]f32{ 4, 5 };
    var held3: [3]f32 = @splat(5);
    try std.testing.expectError(error.Length, exchangeO(&partial0, &ragged, &held3));
    try std.testing.expect(std.mem.allEqual(f32, &held3, 5));

    try std.testing.expectEqual(@as(u64, 10240), bounceBytes(1));
}

test "indexScores dots one query against each key row" {
    const index_q = [_]f32{ 1, 2 };
    const index_k = [_]f32{ 3, 4, 5, 6 };
    var out: [2]f32 = @splat(0);
    try indexScores(&index_q, &index_k, &out);
    try std.testing.expectEqualSlices(f32, &.{ 11, 17 }, &out);

    var held: [2]f32 = @splat(7);
    const ragged = [_]f32{ 3, 4, 5 };
    try std.testing.expectError(error.Length, indexScores(&index_q, &ragged, &held));
    try std.testing.expect(std.mem.allEqual(f32, &held, 7));
}
