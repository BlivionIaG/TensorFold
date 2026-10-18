//! Host reference for the Qwen3.8-Flash-Next PLE block (n-gram embedding + gated, dilated-conv add-back to the residual
//! streams), the one that sits at the start of decoder layer `ple_layer_ids[0] - 1` (layer 1 in this pack: the tensors are
//! `model.language_model.layers.1.ple.*` and the config lists `ple_layer_ids [2]`, one-indexed).
//!
//! Written from the TensorFold Python family (`qwen4_exp/model.py` `NGramEmbedding.ids` / `PLELayer.__call__` and
//! `cuda/ngram.py` `NGram.ids`, read on par1-cs13), not from memory:
//!   ids      row ids of the 2- and 3-gram hashes, 8 heads each (16 heads), EOS resets the n-gram window
//!   emb      the 16 rows (160 dims each, `exl3_ngram.decodeRow`, 5 bit) concatenated -> 2560
//!   keys     norm_key(key_proj emb)  [4 streams][2560], per-stream centred RMSNorm
//!   values   value_proj emb [2560]
//!   queries  norm_query(h)           [4][2560]
//!   gate     sign(g) sqrt(max(|g|, 1e-6)), g = sum_d keys*queries / sqrt(2560) per stream; gated = sigmoid(gate) * values
//!   out      gated + silu(conv1d(norm_conv(gated)))  (depthwise, kernel 4, dilation 3, 9-row carried tail)
//! The caller adds `out` to the streams. The hash constants (head offsets/sizes, multipliers) are read from the n-gram file.
//! f64 throughout. This is a host path; the row path on the device is another stage's work.

const std = @import("std");
const ngram = @import("exl3_ngram.zig");
const fwd = @import("exl3_fwd_cpu.zig");

pub const hidden = fwd.hidden;
pub const streams = fwd.streams;
pub const wide = fwd.wide;
pub const heads = 16;
pub const dh = 160;
pub const bits = 5;
pub const context = 2;
pub const kernel = 4;
pub const dilation = 3;
pub const tail_rows = (kernel - 1) * dilation;
pub const words_per_row = 51;

pub const Hash = struct {
    head_offsets: [heads]i64,
    head_sizes: [heads]i64,
    multipliers: [3]i64,
    eos: i64,

    /// Row ids [heads] of the token at index `p` of `seq` (history first), `p >= 0`.
    pub fn idsAt(h: Hash, seq: []const i64, p: usize, out: *[heads]u64) void {
        var before: isize = -1;
        for (seq[0..p], 0..) |v, i| {
            if (v == h.eos) before = @intCast(i);
        }
        const in_segment: isize = @as(isize, @intCast(p)) - (before + 1);
        var shifted: [3]i64 = undefined;
        for (0..3) |s| {
            const source: isize = @as(isize, @intCast(p)) - @as(isize, @intCast(s));
            const taken = seq[@intCast(@max(source, 0))];
            shifted[s] = if (in_segment >= @as(isize, @intCast(s)) and source >= 0) taken else h.eos;
        }
        for (2..4) |n| {
            const first = (n - 2) * 8;
            var mixed: i64 = shifted[0] * h.multipliers[0];
            for (1..n) |q| mixed ^= shifted[q] * h.multipliers[q];
            for (0..8) |i| {
                const m: u64 = @intCast(mixed);
                out[first + i] = m % @as(u64, @intCast(h.head_sizes[first + i])) + @as(u64, @intCast(h.head_offsets[first + i]));
            }
        }
    }
};

pub const Ple = struct {
    gpa: std.mem.Allocator,
    key_proj: []f32, // [wide][hidden]
    value_proj: []f32, // [hidden][hidden]
    scale_key: []f32, // 1 + w, [wide]
    scale_query: []f32,
    scale_conv: []f32,
    conv_w: []f32, // [wide][4]
    head_bias: []f16, // [16][160]
    hash: Hash,
    shard_rows: usize,
    shards: []const []const u8, // trellis bytes of each shard, [shard_rows][51] i16
    history: [context]i64,
    tail: []f64, // [tail_rows][wide]

    pub fn deinit(p: *Ple) void {
        p.gpa.free(p.key_proj);
        p.gpa.free(p.value_proj);
        p.gpa.free(p.scale_key);
        p.gpa.free(p.scale_query);
        p.gpa.free(p.scale_conv);
        p.gpa.free(p.conv_w);
        p.gpa.free(p.head_bias);
        p.gpa.free(p.tail);
    }

    pub fn reset(p: *Ple) void {
        p.history = @splat(p.hash.eos);
        @memset(p.tail, 0);
    }

    fn row(p: *const Ple, global: u64, head: usize, out: []f64) void {
        const shard = global / p.shard_rows;
        const local = global % p.shard_rows;
        const bytes = p.shards[shard][local * words_per_row * 2 ..][0 .. words_per_row * 2];
        var words: [words_per_row]i16 = undefined;
        @memcpy(std.mem.sliceAsBytes(&words), bytes);
        var r: [dh]f16 = undefined;
        ngram.decodeRow(&words, bits, p.head_bias[head * dh ..][0..dh], &r) catch unreachable;
        for (out, r) |*d, s| d.* = @floatCast(s);
    }

    const MatJob = struct {
        w: []const f32,
        n_in: usize,
        x: []const f64,
        y: []f64,
        fn run(j: *MatJob, o: usize) void {
            const row_ = j.w[o * j.n_in ..][0..j.n_in];
            var s: f64 = 0;
            for (row_, j.x) |a, b| s += @as(f64, a) * b;
            j.y[o] = s;
        }
    };

    fn matvec(w: []const f32, n_in: usize, x: []const f64, y: []f64) void {
        var j: MatJob = .{ .w = w, .n_in = n_in, .x = x, .y = y };
        fwd.parallelFor(y.len, 48, &j, MatJob.run);
    }

    fn normGroups(x: []const f64, scale: []const f32, out: []f64) void {
        for (0..streams) |s| {
            const seg = x[s * hidden ..][0..hidden];
            var ss: f64 = 0;
            for (seg) |v| ss += v * v;
            const inv = 1.0 / @sqrt(ss / @as(f64, hidden) + 1e-6);
            for (0..hidden) |d| out[s * hidden + d] = seg[d] * inv * @as(f64, scale[s * hidden + d]);
        }
    }

    /// `out[t]` = the add-back for row t of `tokens`, given the streams `h[t]` before it. Advances history and conv tail.
    pub fn rows(p: *Ple, tokens: []const u32, h: []const f64, out: []f64) !void {
        const t_n = tokens.len;
        if (h.len != t_n * wide or out.len != t_n * wide) return error.Length;
        const gpa = p.gpa;
        const seq = try gpa.alloc(i64, context + t_n);
        defer gpa.free(seq);
        seq[0..context].* = p.history;
        for (tokens, 0..) |tk, i| seq[context + i] = tk;

        const conv_in = try gpa.alloc(f64, (tail_rows + t_n) * wide);
        defer gpa.free(conv_in);
        @memcpy(conv_in[0 .. tail_rows * wide], p.tail);
        const gated = try gpa.alloc(f64, t_n * wide);
        defer gpa.free(gated);

        const emb = try gpa.alloc(f64, heads * dh);
        defer gpa.free(emb);
        const keys_raw = try gpa.alloc(f64, wide);
        defer gpa.free(keys_raw);
        const keys = try gpa.alloc(f64, wide);
        defer gpa.free(keys);
        const queries = try gpa.alloc(f64, wide);
        defer gpa.free(queries);
        const values = try gpa.alloc(f64, hidden);
        defer gpa.free(values);

        for (0..t_n) |t| {
            var ids: [heads]u64 = undefined;
            p.hash.idsAt(seq, context + t, &ids);
            for (0..heads) |hd| p.row(ids[hd], hd, emb[hd * dh ..][0..dh]);
            matvec(p.key_proj, hidden, emb, keys_raw);
            matvec(p.value_proj, hidden, emb, values);
            normGroups(keys_raw, p.scale_key, keys);
            normGroups(h[t * wide ..][0..wide], p.scale_query, queries);
            const g = gated[t * wide ..][0..wide];
            for (0..streams) |s| {
                var dot: f64 = 0;
                for (0..hidden) |d| dot += keys[s * hidden + d] * queries[s * hidden + d];
                var gate = dot / @sqrt(@as(f64, hidden));
                gate = std.math.sign(gate) * @sqrt(@max(@abs(gate), 1e-6));
                const sg = fwd.sigmoid(gate);
                for (0..hidden) |d| g[s * hidden + d] = sg * values[d];
            }
            normGroups(g, p.scale_conv, conv_in[(tail_rows + t) * wide ..][0..wide]);
        }
        for (0..t_n) |t| {
            for (0..wide) |c| {
                var acc: f64 = 0;
                for (0..kernel) |j| acc += @as(f64, p.conv_w[c * kernel + j]) * conv_in[(t + j * dilation) * wide + c];
                out[t * wide + c] = gated[t * wide + c] + fwd.silu(acc);
            }
        }
        @memcpy(p.tail, conv_in[t_n * wide ..][0 .. tail_rows * wide]);
        for (0..context) |i| p.history[i] = seq[seq.len - context + i];
    }
};

test "idsAt: first token after the EOS history has only its own id, later tokens see their predecessors" {
    const h: Hash = .{
        .head_offsets = .{ 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500 },
        .head_sizes = @splat(97),
        .multipliers = .{ 3, 5, 7 },
        .eos = 99,
    };
    var ids: [heads]u64 = undefined;
    // [eos, eos, 10, 11]: position 2 sees only itself (shifted 1 and 2 are eos)
    const seq = [_]i64{ 99, 99, 10, 11 };
    h.idsAt(&seq, 2, &ids);
    // 2-gram: 10*3 ^ 99*5 ; 3-gram: 10*3 ^ 99*5 ^ 99*7
    const m2: u64 = @intCast((10 * 3) ^ (99 * 5));
    const m3: u64 = @intCast(((10 * 3) ^ (99 * 5)) ^ (99 * 7));
    try std.testing.expectEqual(m2 % 97, ids[0]);
    try std.testing.expectEqual(m3 % 97 + 1000, ids[10]);
    h.idsAt(&seq, 3, &ids);
    const n2: u64 = @intCast((11 * 3) ^ (10 * 5));
    const n3: u64 = @intCast(((11 * 3) ^ (10 * 5)) ^ (99 * 7));
    try std.testing.expectEqual(n2 % 97 + 100, ids[1]);
    try std.testing.expectEqual(n3 % 97 + 900, ids[9]);
}
