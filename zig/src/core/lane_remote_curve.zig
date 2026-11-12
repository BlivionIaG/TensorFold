//! A machine's prompt-prefill speed by prompt length: the average rate falls as the context grows, on both machines.
const std = @import("std");

/// Prompt rows up to which a rate is flat; beyond it the average rate over n rows is `base * (ref/n)^decay`.
pub const ref_rows: f64 = 32768;

/// A rate or time, seeded and then blended with what is measured.
pub const Rate = struct {
    v: f64,
    seen: bool = false,

    /// A sample within a third to three times the standing value: the first replaces the seed, then 70% / 30%.
    pub fn blend(r: *Rate, sample: f64) void {
        if (!std.math.isFinite(sample) or sample <= 0) return;
        const s = std.math.clamp(sample, r.v / 3, r.v * 3);
        r.v = if (r.seen) 0.7 * r.v + 0.3 * s else s;
        r.seen = true;
    }
};

/// Seconds a prompt takes at `base` tokens a second when it is `rows` long and the average rate decays by `decay`.
fn unit(rows: u64, decay: f64) f64 {
    const n: f64 = @floatFromInt(rows);
    return n * std.math.pow(f64, @max(n, ref_rows) / ref_rows, decay);
}

/// Prefill from an empty context: `base` tokens a second up to `ref_rows`, the average over n rows lower by `decay`.
pub const Curve = struct {
    base: Rate,
    decay: f64 = 0,

    /// Seconds to prefill the rows from `from` to `to` (a position in the prompt): the marginal cost of that stretch.
    pub fn span(c: Curve, from: u64, to: u64) f64 {
        if (to <= from) return 0;
        return @max(0, (unit(to, c.decay) - unit(from, c.decay)) / c.base.v);
    }

    /// A pass over rows `from` to `to` took `ns`: the base rate it implies, blended in.
    pub fn observe(c: *Curve, from: u64, to: u64, ns: u64) void {
        if (ns == 0 or to <= from) return;
        const s = @as(f64, @floatFromInt(ns)) / 1e9;
        c.base.blend((unit(to, c.decay) - unit(from, c.decay)) / s);
    }

    /// Nanoseconds for `span`.
    pub fn spanNs(c: Curve, from: u64, to: u64) u64 {
        return @intFromFloat(c.span(from, to) * 1e9);
    }
};

test "a curve is flat up to 32k rows and slows beyond: the M5's measured 480k pass and the pair's 968k pass" {
    const local: Curve = .{ .base = .{ .v = 2000 }, .decay = 0.44 };
    try std.testing.expectApproxEqAbs(@as(f64, 16.384), local.span(0, 32768), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 8.192), local.span(0, 16384), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 784), local.span(0, 480_000), 10); // measured 784 s
    const pair: Curve = .{ .base = .{ .v = 2400 }, .decay = 0.15 };
    try std.testing.expectApproxEqAbs(@as(f64, 677), pair.span(0, 968_000), 8); // measured 677 s
    // the same rows cost more at the far end of a prompt than at its start
    try std.testing.expect(local.span(400_000, 480_000) > local.span(0, 80_000));
    try std.testing.expectEqual(@as(f64, 0), local.span(5, 5));
    try std.testing.expectEqual(@as(f64, 0), local.span(9, 5));
}

test "observe: a pass is normalised to the base rate, so a long pass and a short one teach the same number" {
    var c: Curve = .{ .base = .{ .v = 1000 }, .decay = 0.44 };
    c.observe(0, 480_000, 784 * std.time.ns_per_s);
    try std.testing.expectApproxEqAbs(@as(f64, 2000), c.base.v, 40);
    var d: Curve = .{ .base = .{ .v = 1000 }, .decay = 0.44 };
    d.observe(0, 16384, 8192 * std.time.ns_per_ms);
    try std.testing.expectApproxEqAbs(@as(f64, 2000), d.base.v, 1e-6);
    d.observe(0, 0, 100); // nothing prefilled, and no time, say nothing
    d.observe(0, 100, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 2000), d.base.v, 1e-6);
}
