//! CPU side of the EXL3 PLE (n-gram embedding) layer of Qwen3.8-Flash-Next.
//!
//! Verified against the tensorfold reference on par1-cs13 (tf-final/src/tensorfold/families/qwen4_exp):
//!   model.py `PLELayer.__call__` and `DecoderLayer.__call__` (the model definition),
//!   cuda/glue.py `_ple_gate` / `_ple_conv`, cuda/forward.py `ple_block` / `_pre_moe` (the engine's roundings),
//!   cuda/exl3_mm.py `_ple_rows` (the row codec).
//!
//! Where it sits: decoder layer index 1 (one-indexed id 2 in `ple_layer_ids`), the first thing the layer does. It adds
//! the vector below to the 4-stream residual h [4 * 2560] BEFORE `attn_hyper_connection`: h' = bf16(h + ple).
//!
//!   emb    = 16 head rows (160 values each, fp16) of the n-gram table, concatenated: [2560]
//!   keys   = key_proj(emb) [10240], values = value_proj(emb) [2560]            (dense F16 weights, no bias)
//!   per stream s of 4 (D = 2560):
//!     k = norm_key(keys_s), q = norm_query(h_s)                                (centred RMSNorm, scale 1 + w)
//!     gate_s = sum(k * q) / sqrt(D); gate_s = sign(gate_s) * sqrt(max(|gate_s|, 1e-6))
//!     gated_s = sigmoid(gate_s) * values                                        (values shared by the 4 streams)
//!   normed = norm_conv(gated) per stream; conv = depthwise causal conv1d over [tail; normed],
//!   kernel 4, dilation 3 (= ngram_size), tail = the last 9 normed rows of earlier tokens
//!   ple = gated + silu(conv)                                                    [10240]

const std = @import("std");
const exl3_format = @import("exl3_format.zig");

pub const streams = 4;
pub const hidden = 2560;
pub const wide = streams * hidden;
pub const embed_dim = 2560;
pub const taps = 4;
pub const dilation = 3;
pub const tail_rows = (taps - 1) * dilation;
pub const eps: f32 = 1e-6;

/// Round to bf16 and back (round to nearest even), as the device stores its intermediates.
pub fn bf(x: f32) f32 {
    if (x != x) return x;
    const u: u32 = @bitCast(x);
    return @bitCast((u +% 0x7FFF +% ((u >> 16) & 1)) & 0xFFFF0000);
}

// ---------------------------------------------------------------------------------------------------------------
// Row codec, second implementation: the Triton `_ple_rows` formulation (hash of the state, byte sum, f32 multiply-add).
// It shares nothing with `exl3_ngram.decodeRow` / `exl3_format.codebookValue` except the three fp16 bit patterns.

/// Row = [fp16 scale][160 values of `bits` bits, tail-biting]. out[i] = f16(f16(mul1(state_i)) * scale + bias[i]).
pub fn decodeRowRef(words: []const i16, bits: usize, bias: []const f16, out: []f16) void {
    const dh = out.len;
    const k_inv: f32 = @floatCast(@as(f16, @bitCast(exl3_format.Mul1ScaleBits)));
    const k_bias: f32 = @floatCast(@as(f16, @bitCast(exl3_format.Mul1BiasBits)));
    const scale: f32 = @floatCast(@as(f16, @bitCast(words[0])));
    for (0..dh) |i| {
        var state: u64 = 0;
        for (0..16) |m| {
            var pos: isize = @as(isize, @intCast(i)) - @as(isize, @intCast(m / bits));
            if (pos < 0) pos += @as(isize, @intCast(dh));
            const sb: usize = @as(usize, @intCast(pos)) * bits + (m % bits);
            const word: u64 = @as(u16, @bitCast(words[1 + (sb >> 4)]));
            state |= ((word >> @intCast(sb & 15)) & 1) << @intCast(m);
        }
        const prod: u64 = (state * 0x83DCD12D) & 0xFFFFFFFF;
        const hs: u64 = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255);
        const cb: f16 = @floatCast(@as(f32, @floatFromInt(1024 + hs)) * k_inv + k_bias);
        const b: f32 = @floatCast(bias[i]);
        out[i] = @floatCast(@as(f32, @floatCast(cb)) * scale + b);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The layer.

/// Dense weights as f32 (F16 on disk widens exactly). Norm scales already carry the centring: 1 + stored (gamma - 1).
pub const Weights = struct {
    key: []const f32, // [wide][embed_dim]
    value: []const f32, // [hidden][embed_dim]
    norm_key: []const f32, // [wide]
    norm_query: []const f32, // [wide]
    norm_conv: []const f32, // [wide]
    conv: []const f32, // [wide][taps], as stored (F16)
};

fn dot32(w: []const f32, x: []const f32) f32 {
    var acc: f32 = 0;
    for (w, x) |a, b| acc += a * b;
    return acc;
}

fn sigmoid32(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// The engine's PLE branch for one token at a time, rounding where glue.py does (bf16 stores, fp32 reductions).
/// The engine converts the conv weights to bf16 at load (exl3.py `ple_layer`), so this does too.
pub const Branch = struct {
    gpa: std.mem.Allocator,
    w: Weights,
    conv_bf: []f32,
    keys: []f32,
    vals: []f32,
    gated: []f32,
    nrow: []f32,
    /// The last `tail_rows` normed rows, oldest first. Zero at the start of a sequence.
    tail: []f32,

    pub fn init(gpa: std.mem.Allocator, w: Weights) !Branch {
        const conv_bf = try gpa.alloc(f32, w.conv.len);
        errdefer gpa.free(conv_bf);
        for (conv_bf, w.conv) |*d, s| d.* = bf(s);
        const bufs = try gpa.alloc(f32, 4 * wide + tail_rows * wide);
        @memset(bufs, 0);
        return .{
            .gpa = gpa,
            .w = w,
            .conv_bf = conv_bf,
            .keys = bufs[0..wide],
            .vals = bufs[wide..][0..hidden],
            .gated = bufs[2 * wide ..][0..wide],
            .nrow = bufs[3 * wide ..][0..wide],
            .tail = bufs[4 * wide ..][0 .. tail_rows * wide],
        };
    }

    pub fn deinit(self: *Branch) void {
        self.gpa.free(self.conv_bf);
        self.gpa.free(self.keys.ptr[0 .. 4 * wide + tail_rows * wide]);
        self.* = undefined;
    }

    pub fn reset(self: *Branch) void {
        @memset(self.tail, 0);
    }

    /// `emb` [2560] the token's 16 gathered rows (fp16 values), `h` [10240] the layer input (bf16 values),
    /// `out` [10240] the vector to add to h (bf16 values). Advances the conv tail by this token.
    pub fn step(self: *Branch, emb: []const f32, h: []const f32, out: []f32) void {
        std.debug.assert(emb.len == embed_dim and h.len == wide and out.len == wide);
        const w = self.w;
        for (0..wide) |o| self.keys[o] = bf(dot32(w.key[o * embed_dim ..][0..embed_dim], emb));
        for (0..hidden) |o| self.vals[o] = bf(dot32(w.value[o * embed_dim ..][0..embed_dim], emb));
        const d: f32 = hidden;
        for (0..streams) |s| {
            const k = self.keys[s * hidden ..][0..hidden];
            const q = h[s * hidden ..][0..hidden];
            var kss: f32 = 0;
            var qss: f32 = 0;
            for (k, q) |a, b| {
                kss += a * a;
                qss += b * b;
            }
            const krinv = 1.0 / @sqrt(kss / d + eps);
            const qrinv = 1.0 / @sqrt(qss / d + eps);
            var dot: f32 = 0;
            for (0..hidden) |i| {
                const kn = bf(k[i] * krinv * w.norm_key[s * hidden + i]);
                const qn = bf(q[i] * qrinv * w.norm_query[s * hidden + i]);
                dot += bf(kn * qn);
            }
            var gate = bf(dot / @sqrt(d));
            const mag = @sqrt(@max(@abs(gate), 1e-6));
            gate = bf(if (gate > 0) mag else if (gate < 0) -mag else 0);
            const sg = bf(sigmoid32(gate));
            var gss: f32 = 0;
            for (0..hidden) |i| {
                const g = bf(sg * self.vals[i]);
                self.gated[s * hidden + i] = g;
                gss += g * g;
            }
            const rinv = 1.0 / @sqrt(gss / d + eps);
            for (0..hidden) |i| {
                const e = s * hidden + i;
                self.nrow[e] = bf(self.gated[e] * rinv * w.norm_conv[e]);
            }
        }
        for (0..wide) |e| {
            var acc: f32 = 0;
            for (0..taps) |t| {
                const at = t * dilation; // index into [tail; this row]
                const x = if (at < tail_rows) self.tail[at * wide + e] else self.nrow[e];
                acc += self.conv_bf[e * taps + t] * x;
            }
            const c = bf(acc);
            const act = bf(c / (1.0 + @exp(-c)));
            out[e] = bf(self.gated[e] + act);
        }
        std.mem.copyForwards(f32, self.tail[0 .. (tail_rows - 1) * wide], self.tail[wide..]);
        @memcpy(self.tail[(tail_rows - 1) * wide ..], self.nrow);
    }
};

/// model.py maths in f64 with no intermediate rounding, state kept the same way (a tail of normed rows).
pub const Oracle = struct {
    gpa: std.mem.Allocator,
    w: Weights,
    tail: []f64,

    pub fn init(gpa: std.mem.Allocator, w: Weights) !Oracle {
        const tail = try gpa.alloc(f64, tail_rows * wide);
        @memset(tail, 0);
        return .{ .gpa = gpa, .w = w, .tail = tail };
    }

    pub fn deinit(self: *Oracle) void {
        self.gpa.free(self.tail);
        self.* = undefined;
    }

    fn proj(w: []const f32, rows: usize, emb: []const f64, out: []f64) void {
        for (0..rows) |o| {
            var acc: f64 = 0;
            for (w[o * embed_dim ..][0..embed_dim], emb) |a, b| acc += @as(f64, a) * b;
            out[o] = acc;
        }
    }

    fn centred(x: []const f64, scale: []const f32, out: []f64) void {
        var ss: f64 = 0;
        for (x) |v| ss += v * v;
        const r = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + @as(f64, eps));
        for (out, x, scale) |*o, v, s| o.* = v * r * s;
    }

    pub fn step(self: *Oracle, emb: []const f64, h: []const f64, out: []f64) !void {
        const w = self.w;
        const keys = try self.gpa.alloc(f64, wide);
        defer self.gpa.free(keys);
        const vals = try self.gpa.alloc(f64, hidden);
        defer self.gpa.free(vals);
        const gated = try self.gpa.alloc(f64, wide);
        defer self.gpa.free(gated);
        const normed = try self.gpa.alloc(f64, wide);
        defer self.gpa.free(normed);
        const kn = try self.gpa.alloc(f64, hidden);
        defer self.gpa.free(kn);
        const qn = try self.gpa.alloc(f64, hidden);
        defer self.gpa.free(qn);
        proj(w.key, wide, emb, keys);
        proj(w.value, hidden, emb, vals);
        for (0..streams) |s| {
            centred(keys[s * hidden ..][0..hidden], w.norm_key[s * hidden ..][0..hidden], kn);
            centred(h[s * hidden ..][0..hidden], w.norm_query[s * hidden ..][0..hidden], qn);
            var dot: f64 = 0;
            for (kn, qn) |a, b| dot += a * b;
            var gate = dot / @sqrt(@as(f64, hidden));
            const sgn: f64 = if (gate > 0) 1 else if (gate < 0) -1 else 0;
            gate = sgn * @sqrt(@max(@abs(gate), 1e-6));
            const sg = 1.0 / (1.0 + @exp(-gate));
            for (0..hidden) |i| gated[s * hidden + i] = sg * vals[i];
            centred(gated[s * hidden ..][0..hidden], w.norm_conv[s * hidden ..][0..hidden], normed[s * hidden ..][0..hidden]);
        }
        for (0..wide) |e| {
            var acc: f64 = 0;
            for (0..taps) |t| {
                const at = t * dilation;
                const x = if (at < tail_rows) self.tail[at * wide + e] else normed[e];
                acc += @as(f64, w.conv[e * taps + t]) * x;
            }
            out[e] = gated[e] + acc / (1.0 + @exp(-acc));
        }
        std.mem.copyForwards(f64, self.tail[0 .. (tail_rows - 1) * wide], self.tail[wide..]);
        @memcpy(self.tail[(tail_rows - 1) * wide ..], normed);
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Tests: synthetic weights (the real-file test is tf-cuda-test exl3-ple).

test "second row decoder equals decodeRow for every width and all 65536 mul1 states" {
    const ngram = @import("exl3_ngram.zig");
    // exhaustive over the codebook: the row whose value 0 carries state s (5 bits: only bits 0..4 plus its neighbours
    // matter), so instead drive full rows from a counter and compare whole rows bit for bit.
    var rng = std.Random.DefaultPrng.init(0x91e);
    var words: [81]i16 = undefined;
    var bias: [160]f16 = undefined;
    var a: [160]f16 = undefined;
    var b: [160]f16 = undefined;
    for ([_]usize{ 2, 3, 4, 5, 6, 8 }) |bits| {
        const n = try ngram.wordsPerRow(bits);
        for (0..400) |_| {
            for (words[0..n]) |*x| x.* = @bitCast(rng.random().int(u16));
            words[0] = @bitCast(@as(f16, @floatCast((rng.random().float(f32) - 0.5) * 4)));
            for (&bias) |*x| x.* = @floatCast((rng.random().float(f32) - 0.5) * 0.25);
            try ngram.decodeRow(words[0..n], bits, &bias, &a);
            decodeRowRef(words[0..n], bits, &bias, &b);
            for (a, b) |x, y| try std.testing.expectEqual(@as(u16, @bitCast(x)), @as(u16, @bitCast(y)));
        }
    }
    // the codebook itself, all states: f32 form against the format's f64 form
    const k_inv: f32 = @floatCast(@as(f16, @bitCast(exl3_format.Mul1ScaleBits)));
    const k_bias: f32 = @floatCast(@as(f16, @bitCast(exl3_format.Mul1BiasBits)));
    for (0..65536) |s| {
        const prod: u64 = (@as(u64, s) * 0x83DCD12D) & 0xFFFFFFFF;
        const hs: u64 = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255);
        const cb: f16 = @floatCast(@as(f32, @floatFromInt(1024 + hs)) * k_inv + k_bias);
        try std.testing.expectEqual(@as(u16, @bitCast(exl3_format.codebookValue(.mul1, @intCast(s)))), @as(u16, @bitCast(cb)));
    }
}

test "bf16 rounding is round to nearest even" {
    try std.testing.expectEqual(@as(f32, 1.0), bf(1.0));
    try std.testing.expectEqual(@as(f32, 1.0), bf(1.0 + 1.0 / 512.0)); // below the half step: down
    try std.testing.expectEqual(@as(f32, 1.0 + 1.0 / 128.0), bf(1.0 + 1.0 / 128.0 + 1.0 / 512.0)); // above: up
    try std.testing.expectEqual(@as(f32, 1.0), bf(1.0 + 1.0 / 256.0)); // tie to even (down)
    try std.testing.expectEqual(@as(f32, 1.0 + 2.0 / 128.0), bf(1.0 + 3.0 / 256.0)); // tie to even (up)
}

test "engine branch tracks the f64 oracle across a sequence, tails included" {
    const gpa = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x91e2);
    const r = rng.random();
    const key = try gpa.alloc(f32, wide * embed_dim);
    defer gpa.free(key);
    const value = try gpa.alloc(f32, hidden * embed_dim);
    defer gpa.free(value);
    const norms = try gpa.alloc(f32, 3 * wide);
    defer gpa.free(norms);
    const conv = try gpa.alloc(f32, wide * taps);
    defer gpa.free(conv);
    for (key) |*x| x.* = @floatCast(@as(f16, @floatCast((r.float(f32) - 0.5) * 0.06)));
    for (value) |*x| x.* = @floatCast(@as(f16, @floatCast((r.float(f32) - 0.5) * 0.06)));
    for (norms) |*x| x.* = 1.0 + (r.float(f32) - 0.5) * 0.4;
    for (conv) |*x| x.* = @floatCast(@as(f16, @floatCast((r.float(f32) - 0.5) * 1.0)));
    const w: Weights = .{ .key = key, .value = value, .norm_key = norms[0..wide], .norm_query = norms[wide..][0..wide], .norm_conv = norms[2 * wide ..], .conv = conv };
    var branch = try Branch.init(gpa, w);
    defer branch.deinit();
    var oracle = try Oracle.init(gpa, w);
    defer oracle.deinit();
    const emb = try gpa.alloc(f32, embed_dim);
    defer gpa.free(emb);
    const emb64 = try gpa.alloc(f64, embed_dim);
    defer gpa.free(emb64);
    const h = try gpa.alloc(f32, wide);
    defer gpa.free(h);
    const h64 = try gpa.alloc(f64, wide);
    defer gpa.free(h64);
    const got = try gpa.alloc(f32, wide);
    defer gpa.free(got);
    const want = try gpa.alloc(f64, wide);
    defer gpa.free(want);
    var worst: f64 = 0;
    var peak: f64 = 0;
    for (0..14) |_| { // more than the 9-row tail, so every tap reads a real earlier token
        for (emb, emb64) |*x, *y| {
            x.* = @floatCast(@as(f16, @floatCast((r.float(f32) - 0.5) * 6)));
            y.* = x.*;
        }
        for (h, h64) |*x, *y| {
            x.* = bf((r.float(f32) - 0.5) * 8);
            y.* = x.*;
        }
        branch.step(emb, h, got);
        try oracle.step(emb64, h64, want);
        for (got, want) |g, o| {
            worst = @max(worst, @abs(@as(f64, g) - o));
            peak = @max(peak, @abs(o));
        }
    }
    try std.testing.expect(peak > 0.05);
    // bf16 keeps 8 bits: each stored intermediate is good to 2^-9 relative; a few of them in a chain
    try std.testing.expect(worst <= 0.03 * peak);
}
