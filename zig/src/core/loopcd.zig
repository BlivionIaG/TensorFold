//! LoopCD (arXiv:2610.02185): the first recurrent pass guides the last before the coda layers.
const std = @import("std");
const bf16 = @import("bf16_round");

/// The most strength accepted: beyond it the paper's generation curves collapse (Appendix B.4).
pub const max_strength: f32 = 2.0;
/// The least strength accepted: -1 reproduces the first pass's own state, the paper's weak reference.
pub const min_strength: f32 = -1.0;
/// The strength the paper selects for a fixed rule in generation (Appendix B.4: 0.2 to 0.3).
pub const default_strength: f32 = 0.3;

pub const Error = error{InvalidStrength};

/// A strength this slice runs: finite, from -1 to 2, and not 0 (0 is the unguided state, the flag's own off).
pub fn accepted(w: f32) bool {
    return std.math.isFinite(w) and w >= min_strength and w <= max_strength and w != 0;
}

/// h' = h_R + w (h_R - h_1) for one element in float64: the serial reference a combine is checked against.
pub fn combine(w: f64, hr: f64, h1: f64) f64 {
    return hr + w * (hr - h1);
}

/// The order the kernel rounds in: BF16 inputs promoted to FP32, one fused multiply-add, one BF16 rounding.
pub fn rounded(w: f32, hr: f32, h1: f32) f32 {
    return bf16.value(stored(w, hr, h1));
}

/// The BF16 bits of one element's guided state, as the kernel stores it.
pub fn stored(w: f32, hr: f32, h1: f32) u16 {
    return bf16.bits(@mulAdd(f32, w, hr - h1, hr));
}

/// The prompt cache's extra key word for a strength: a kept state of another mode never resumes this one.
pub fn key(w: f32) u64 {
    var h = std.hash.Wyhash.init(0x4c6f6f704344);
    h.update(std.mem.asBytes(&w));
    return h.final();
}

pub const enabled_name = "TF_LOOPCD";
pub const strength_name = "TF_LOOPCD_W";

/// --loopcd and --loopcd-w, else TF_LOOPCD and TF_LOOPCD_W; null: off.
pub fn fromEnvironment(on: bool, w: ?f32) !?f32 {
    const env = @import("sys").environment;
    const off = !on and !try env.read(bool, enabled_name, false);
    if (off) return null;
    const strength = w orelse try env.read(f32, strength_name, default_strength);
    if (!accepted(strength)) return error.InvalidStrength;
    return strength;
}

const t = std.testing;

test "loopcd: the f32 rounding order meets the float64 reference at a BF16 output" {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rnd = prng.random();
    var worst: f64 = 0;
    for (0..4000) |_| {
        const w: f32 = rnd.float(f32) * 2;
        const hr: f32 = rnd.float(f32) * 8 - 4;
        const h1: f32 = rnd.float(f32) * 8 - 4;
        const want = combine(w, hr, h1);
        const got: f64 = rounded(w, hr, h1);
        const half_ulp = 0x1p-8 * @abs(want);
        const slack = @as(f64, 0x1p-23) * (@abs(hr) + @abs(@as(f64, w) * (@as(f64, hr) - h1)));
        worst = @max(worst, @abs(got - want) - half_ulp - slack);
    }
    try t.expect(worst <= 0);
}

test "loopcd: the accepted strengths are the slice's, and 0 is the unguided path" {
    try t.expect(accepted(default_strength) and accepted(-1) and accepted(2));
    try t.expect(!accepted(0) and !accepted(-1.5) and !accepted(2.5));
    try t.expect(!accepted(std.math.nan(f32)) and !accepted(std.math.inf(f32)));
    // -1 gives the first pass's own state back, whatever the strength's rounding
    try t.expectEqual(@as(f32, 7), rounded(-1, -2.5, 7)); // h' = h_R - (h_R - h_1) = h_1
    try t.expectEqual(@as(f32, 1), rounded(1, 1, 1)); // the same state at both passes: no contrast
}

test "loopcd: a strength's key separates the modes and repeats for the same strength" {
    try t.expectEqual(key(0.3), key(0.3));
    try t.expect(key(0.3) != key(0.31));
    try t.expect(key(-1) != key(1));
}
