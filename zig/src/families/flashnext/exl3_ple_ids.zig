//! Host n-gram row ids for the EXL3 pack (qwen4_exp `NGram`, tensorfold/families/qwen4_exp/cuda/ngram.py).
//! A token's 16 heads hash the token and its 1 or 2 predecessors: heads 0..7 use the bigram, heads 8..15 the trigram.
//! The predecessors reset at an EOS: a position `k` back that lies before the segment start reads EOS instead.
//! Every constant here is derived from the config, then compared with the four tensors the pack ships
//! (`layer_multipliers`, `head_vocab_sizes`, `head_offsets`, shard row counts). The pack's own table wins on a mismatch.

const std = @import("std");
const shape = @import("exl3_shape.zig");

pub const heads = 16;
pub const context = 2;

pub const Params = struct {
    vocab: u64 = shape.vocab,
    seed: u64 = 1234,
    eos: i64 = 248044,
    vocab_base: u64 = shape.ngram_vocab_base,
    divisor: u64 = 128,
    shards: u64 = shape.ngram_shards,
    heads_per: usize = shape.ngram_heads_per,
};

pub const Layout = struct {
    eos: i64,
    heads_per: usize,
    multipliers: [3]i64,
    sizes: [heads]i64,
    offsets: [heads]i64,
    /// Sum of the head sizes.
    total: u64,
    /// `total` rounded up to the divisor: the rows the shards hold in all.
    rows: u64,
    shards: u64,

    /// Rows in shard `i`: the first `rows % shards` shards carry one extra.
    pub fn shardRows(self: Layout, i: u64) u64 {
        return self.rows / self.shards + @intFromBool(i < self.rows % self.shards);
    }
};

const golden: u64 = 0x9E3779B97F4A7C15;
const mix1: u64 = 0xBF58476D1CE4E5B9;
const mix2: u64 = 0x94D049BB133111EB;
const prime_gap: u64 = 10007;

fn splitmix64(value: u64) u64 {
    var v = value +% golden;
    v = (v ^ (v >> 30)) *% mix1;
    v = (v ^ (v >> 27)) *% mix2;
    return v ^ (v >> 31);
}

fn isPrime(value: u64) bool {
    if (value < 2) return false;
    if (value % 2 == 0) return value == 2;
    var d: u64 = 3;
    while (d * d <= value) : (d += 2) if (value % d == 0) return false;
    return true;
}

/// The `count`-th prime strictly after `start`.
fn nthPrimeAfter(start: u64, count: u64) u64 {
    var prime = start;
    for (0..count) |_| {
        prime += 1;
        while (!isPrime(prime)) prime += 1;
    }
    return prime;
}

/// Layout of the `ple_index`-th PLE layer (0 for the first id in `ple_layer_ids`).
pub fn derive(p: Params, ple_index: u64) Layout {
    var l: Layout = .{ .eos = p.eos, .heads_per = p.heads_per, .multipliers = undefined, .sizes = undefined, .offsets = undefined, .total = 0, .rows = 0, .shards = p.shards };
    for (0..heads) |h| {
        const size = nthPrimeAfter(p.vocab_base - 1, ple_index * heads + h + 1);
        l.sizes[h] = @intCast(size);
        l.offsets[h] = @intCast(l.total);
        l.total += size;
    }
    const half = @max(1, ((@as(u64, 1) << 63) - 1) / @max(p.vocab, 1) / 2);
    const base = p.seed +% prime_gap *% ple_index;
    for (0..3) |i| l.multipliers[i] = @intCast(2 * (splitmix64(base +% golden *% (i + 1)) % half) + 1);
    l.rows = (l.total + p.divisor - 1) / p.divisor * p.divisor;
    return l;
}

/// Row ids [tokens.len][16] for `tokens` after `history` (the 2 tokens before them, oldest first).
/// The initial history is [eos, eos]. A token equal to EOS starts a new segment after itself.
pub fn ids(l: Layout, history: [context]i64, tokens: []const u32, out: [][heads]u64) void {
    std.debug.assert(out.len >= tokens.len);
    var seq: [context + 1]i64 = undefined; // context tokens then the row's token, oldest first
    @memcpy(seq[0..context], &history);
    var run: usize = 0; // tokens of the current segment before the row, capped at `context`
    for (history) |t| run = if (t == l.eos) 0 else @min(run + 1, context);
    for (tokens, out[0..tokens.len]) |tok, *row| {
        seq[context] = tok;
        for (row, 0..) |*id, head| {
            var mixed: u64 = 0;
            for (0..2 + head / l.heads_per) |q| {
                const t = if (run >= q) seq[context - q] else l.eos;
                const term = @as(u64, @bitCast(t)) *% @as(u64, @bitCast(l.multipliers[q]));
                mixed = if (q == 0) term else mixed ^ term;
            }
            id.* = @as(u64, @intCast(l.offsets[head])) + mixed % @as(u64, @intCast(l.sizes[head]));
        }
        run = if (tok == l.eos) 0 else @min(run + 1, context);
        seq[0] = seq[1];
        seq[1] = seq[2];
    }
}

/// The history after `tokens`: their last two tokens, topped up from the older history.
pub fn after(history: [context]i64, tokens: []const u32) [context]i64 {
    var out = history;
    for (tokens) |t| {
        out[0] = out[1];
        out[1] = t;
    }
    return out;
}

/// The same ids, written as the numpy reference reads: the whole sequence at once, segment starts from
/// the last EOS strictly before each position, and a shifted copy per n-gram term.
pub fn idsSpec(gpa: std.mem.Allocator, l: Layout, history: [context]i64, tokens: []const u32, out: [][heads]u64) !void {
    const n = context + tokens.len;
    const seq = try gpa.alloc(i64, n);
    defer gpa.free(seq);
    @memcpy(seq[0..context], &history);
    for (tokens, seq[context..]) |t, *s| s.* = t;
    for (0..tokens.len) |i| {
        const p = context + i;
        var last: isize = -1; // last EOS strictly before p
        for (0..p) |q| if (seq[q] == l.eos) {
            last = @intCast(q);
        };
        const in_segment: isize = @as(isize, @intCast(p)) - (last + 1);
        var shifted: [3]i64 = undefined;
        for (0..3) |k| {
            const src: isize = @as(isize, @intCast(p)) - @as(isize, @intCast(k));
            shifted[k] = if (in_segment >= @as(isize, @intCast(k)) and src >= 0) seq[@intCast(src)] else l.eos;
        }
        for (0..heads) |head| {
            const terms: usize = 2 + head / l.heads_per;
            var mixed: i64 = shifted[0] *% l.multipliers[0];
            for (1..terms) |q| mixed ^= shifted[q] *% l.multipliers[q];
            out[i][head] = @intCast(@mod(mixed, l.sizes[head]) + l.offsets[head]);
        }
    }
}

test "layout of the first PLE layer matches the pack's shipped constants (qwen3.8-flash-next-exl3, ple index 0)" {
    // Printed from tensors in ngram_embedding.safetensors on par1-cs13 (head_vocab_sizes / head_offsets / layer_multipliers).
    const l = derive(.{}, 0);
    const sizes = [_]i64{ 20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171 };
    const offsets = [_]i64{ 0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275 };
    try std.testing.expectEqualSlices(i64, &sizes, &l.sizes);
    try std.testing.expectEqualSlices(i64, &offsets, &l.offsets);
    try std.testing.expectEqualSlices(i64, &.{ 23703573157769, 20109073645365, 8052911324071 }, &l.multipliers);
    for (1..heads) |h| {
        try std.testing.expect(l.sizes[h] > l.sizes[h - 1]);
        try std.testing.expectEqual(l.offsets[h - 1] + l.sizes[h - 1], l.offsets[h]);
    }
    try std.testing.expectEqual(@as(u64, 320001536), l.rows);
    try std.testing.expectEqual(@as(u64, 2500012), l.shardRows(0));
    try std.testing.expectEqual(@as(u64, 2500012), l.shardRows(127));
    var sum: u64 = 0;
    for (0..128) |i| sum += l.shardRows(i);
    try std.testing.expectEqual(l.rows, sum);
}

test "the incremental hash equals the whole-sequence spec, with EOS inside and across a split window" {
    const l = derive(.{}, 0);
    const eos: u32 = 248044;
    const tokens = [_]u32{ 5, eos, 77, 90000, eos, eos, 3, 4, 248319, 0, 12 };
    var a: [tokens.len][heads]u64 = undefined;
    var b: [tokens.len][heads]u64 = undefined;
    ids(l, .{ 11, eos }, &tokens, &a);
    try idsSpec(std.testing.allocator, l, .{ 11, eos }, &tokens, &b);
    try std.testing.expectEqualSlices([heads]u64, &b, &a);
    var split: [tokens.len][heads]u64 = undefined;
    ids(l, .{ 11, eos }, tokens[0..4], split[0..4]);
    ids(l, after(.{ 11, eos }, tokens[0..4]), tokens[4..], split[4..]);
    try std.testing.expectEqualSlices([heads]u64, &a, &split);
    // every id sits inside its own head's range
    for (a) |row| for (row, 0..) |id, h| {
        try std.testing.expect(id >= @as(u64, @intCast(l.offsets[h])));
        try std.testing.expect(id < @as(u64, @intCast(l.offsets[h] + l.sizes[h])));
    };
}

test "a bigram head ignores the second predecessor; the trigram head does not" {
    const l = derive(.{}, 0);
    var a: [1][heads]u64 = undefined;
    var b: [1][heads]u64 = undefined;
    ids(l, .{ 1, 2 }, &.{3}, &a);
    ids(l, .{ 9, 2 }, &.{3}, &b);
    for (0..8) |h| try std.testing.expectEqual(a[0][h], b[0][h]);
    for (8..16) |h| try std.testing.expect(a[0][h] != b[0][h]);
}
