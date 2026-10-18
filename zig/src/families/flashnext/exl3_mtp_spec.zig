//! Single-draft MTP speculation for Qwen3.8-Flash-Next EXL3: the host maths of the head's input and the round driver.
//!
//! What the head computes (reference: families/qwen4_exp/mtp.py `FlashMTP.__call__`, verified on par1-cs13):
//!   e   = fc_embedding(CenteredRMSNorm_2560(embed(token t+1)))
//!   hs  = fc_hidden(CenteredRMSNorm_10240(streams_t) reshaped to [4][2560])      (one RMS over all 10240, NOT per stream)
//!   x   = e + hs[s] per stream s                                                  (the 4 streams of one decoder layer's input)
//!   x   = one sparse_attention decoder layer (attention + indexer + MoE, own KV, no PLE)
//!   out = hyper_connection_mixer(x) (collapse, no inject), then the main lm_head (no cut draft head here)
//! Row t of the head's KV pairs main position t with the committed token t+1, so its cache needs no rollback.
//!
//! The round driver is generic over a model `M` so the accept / rollback logic is unit-tested on the host:
//!   M.forward(tokens, next)   run rows from the current position, next[i] = greedy token after row i; keeps the rows' streams
//!   M.snapshot() / M.restore()  GDN state, position, PLE history and tail (attention KV is position-addressed: nothing to save)
//!   M.absorb(next_tokens)     MTP rows for the streams of the last `forward` paired with `next_tokens`; returns the draft after the last
//! Verification (greedy): rows [pending, draft] run in one call; row 0's greedy token must equal the draft
//! (`exl3_window.accept`: match advances 2, mismatch drops the draft). On a mismatch the state is restored and `pending` is
//! re-run alone, so the rejected path is bit-for-bit the serial step.

const std = @import("std");
const window = @import("exl3_window.zig");

pub const eps: f32 = 1e-6;

/// `out = x * rsqrt(mean(x^2) + eps) * scale` over the whole row (scale already holds 1 + w). In f32 like the reference.
pub fn centredNorm(x: []const f32, scale: []const f32, out: []f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    for (out, x, scale) |*d, v, s| d.* = v * inv * s;
}

pub fn centredNormF64(x: []const f64, scale: []const f32, out: []f64) void {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + 1e-6);
    for (out, x, scale) |*d, v, s| d.* = v * inv * @as(f64, s);
}

pub const Stats = struct {
    rounds: usize = 0,
    accepted: usize = 0,
    /// Rows run by the model's main forward (prefill rows excluded), serial equivalent = tokens produced.
    decode_rows: usize = 0,
    /// Single-row re-runs after a rejected draft.
    reruns: usize = 0,

    pub fn rate(s: Stats) f64 {
        if (s.rounds == 0) return 0;
        return @as(f64, @floatFromInt(s.accepted)) / @as(f64, @floatFromInt(s.rounds));
    }
};

/// Greedy decode, one row per step. `out[0]` is the token after the prompt.
pub fn serial(m: anytype, prompt: []const u32, out: []u32) !void {
    var nx: [64]u32 = undefined;
    if (prompt.len > nx.len) return error.Length;
    try m.forward(prompt, nx[0..prompt.len]);
    var cur = nx[prompt.len - 1];
    out[0] = cur;
    for (1..out.len) |i| {
        var one = [_]u32{cur};
        try m.forward(&one, nx[0..1]);
        cur = nx[0];
        out[i] = cur;
    }
}

/// Greedy decode with one MTP draft per round. Produces exactly `out.len` tokens.
pub fn speculative(m: anytype, prompt: []const u32, out: []u32, stats: *Stats) !void {
    var nx: [64]u32 = undefined;
    if (prompt.len > nx.len or prompt.len == 0 or out.len == 0) return error.Length;
    try m.forward(prompt, nx[0..prompt.len]);
    var pending = nx[prompt.len - 1];
    out[0] = pending;
    var n: usize = 1;
    // the head sees every prompt row paired with its successor, the last one with the first generated token
    var absorb_tokens: [64]u32 = undefined;
    @memcpy(absorb_tokens[0 .. prompt.len - 1], prompt[1..]);
    absorb_tokens[prompt.len - 1] = pending;
    var draft = try m.absorb(absorb_tokens[0..prompt.len]);
    while (n < out.len) {
        try m.snapshot();
        var rows = [_]u32{ pending, draft };
        try m.forward(&rows, nx[0..2]);
        stats.rounds += 1;
        stats.decode_rows += 2;
        const verdict = window.accept(draft, nx[0]);
        if (verdict.accepted) {
            stats.accepted += 1;
            out[n] = nx[0];
            n += 1;
            const v2 = nx[1];
            if (n < out.len) {
                out[n] = v2;
                n += 1;
            }
            pending = v2;
            const pair = [_]u32{ nx[0], v2 };
            draft = try m.absorb(&pair);
        } else {
            try m.restore();
            var one = [_]u32{pending};
            try m.forward(&one, nx[0..1]);
            stats.reruns += 1;
            stats.decode_rows += 1;
            pending = nx[0];
            out[n] = pending;
            n += 1;
            const pair = [_]u32{pending};
            draft = try m.absorb(&pair);
        }
    }
}

// -- tests -------------------------------------------------------------------------------------------------

/// A deterministic fake: the token after a sequence is a hash of its last two tokens. The "draft" is right unless
/// `wrong_every` says otherwise. State = the token history (snapshot / restore truncate it).
const Fake = struct {
    hist: [256]u32 = undefined,
    len: usize = 0,
    saved: usize = 0,
    last_rows: [2]u32 = undefined, // tokens whose streams were last produced
    last_n: usize = 0,
    calls: usize = 0,
    wrong_every: usize,

    fn succ(a: u32, b: u32) u32 {
        return (a *% 31 +% b *% 17 +% 7) % 1000;
    }

    pub fn forward(m: *Fake, tokens: []const u32, next: []u32) !void {
        for (tokens, 0..) |t, i| {
            m.hist[m.len] = t;
            m.len += 1;
            const prev: u32 = if (m.len >= 2) m.hist[m.len - 2] else 0;
            next[i] = succ(prev, t);
        }
        m.last_n = @min(tokens.len, 2);
    }

    pub fn snapshot(m: *Fake) !void {
        m.saved = m.len;
    }

    pub fn restore(m: *Fake) !void {
        m.len = m.saved;
    }

    pub fn absorb(m: *Fake, next_tokens: []const u32) !u32 {
        m.calls += 1;
        // the token after `last` (the committed token whose forward has not run) is succ(last forwarded token, last)
        const truth = succ(m.hist[m.len - 1], next_tokens[next_tokens.len - 1]);
        if (m.wrong_every != 0 and m.calls % m.wrong_every == 0) return truth +% 1;
        return truth;
    }
};

test "speculation yields the serial stream for every accept pattern" {
    const prompt = [_]u32{ 5, 9, 2, 40 };
    var want: [24]u32 = undefined;
    var serial_model: Fake = .{ .wrong_every = 0 };
    try serial(&serial_model, &prompt, &want);
    for ([_]usize{ 0, 1, 2, 3, 5 }) |wrong| {
        var m: Fake = .{ .wrong_every = wrong };
        var got: [24]u32 = undefined;
        var stats: Stats = .{};
        try speculative(&m, &prompt, &got, &stats);
        try std.testing.expectEqualSlices(u32, &want, &got);
        if (wrong == 0) try std.testing.expect(stats.accepted == stats.rounds and stats.rounds > 0);
        if (wrong == 1) try std.testing.expectEqual(@as(usize, 0), stats.accepted);
        if (wrong == 1) try std.testing.expectEqual(stats.rounds, stats.reruns);
    }
}

test "centredNorm normalises the whole row and applies 1 + w" {
    const x = [_]f32{ 3, 4 };
    const s = [_]f32{ 1, 2 };
    var out: [2]f32 = undefined;
    centredNorm(&x, &s, &out);
    const rms = @sqrt((9.0 + 16.0) / 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0 / rms), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0 / rms), out[1], 1e-5);
    var o64: [2]f64 = undefined;
    centredNormF64(&.{ 3, 4 }, &s, &o64);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0 / rms), o64[0], 1e-6);
}
