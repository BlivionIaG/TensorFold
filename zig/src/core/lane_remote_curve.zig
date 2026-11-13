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

/// One calibration point: a pass over `rows` prompt rows from an empty context took `s` seconds.
pub const Point = struct { rows: u64, s: f64 };

pub const max_points = 8;

/// `rows:seconds,rows:seconds,...` (at most `max_points`, at least one) as points; null when anything is malformed.
pub fn parsePoints(text: []const u8, buf: *[max_points]Point) ?[]const Point {
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |item| {
        if (n == max_points) return null;
        const colon = std.mem.indexOfScalar(u8, item, ':') orelse return null;
        const rows = std.fmt.parseInt(u64, std.mem.trim(u8, item[0..colon], " "), 10) catch return null;
        const s = std.fmt.parseFloat(f64, std.mem.trim(u8, item[colon + 1 ..], " ")) catch return null;
        if (rows == 0 or !std.math.isFinite(s) or s <= 0) return null;
        buf[n] = .{ .rows = rows, .s = s };
        n += 1;
    }
    return if (n == 0) null else buf[0..n];
}

/// The curve through points: least squares of ln rate on ln(rows / ref), slope = decay (0 to 2); no spread keeps seed.
pub fn fit(points: []const Point, seed: Curve) Curve {
    var sx: f64 = 0;
    var sy: f64 = 0;
    var sxx: f64 = 0;
    var sxy: f64 = 0;
    for (points) |p| {
        const rows: f64 = @floatFromInt(p.rows);
        const x = @log(@max(rows, ref_rows) / ref_rows);
        const y = @log(rows / p.s);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const n: f64 = @floatFromInt(points.len);
    const spread = sxx - sx * sx / n;
    const decay = if (spread > 1e-9) std.math.clamp(-(sxy - sx * sy / n) / spread, 0, 2) else seed.decay;
    return .{ .base = .{ .v = @exp((sy + decay * sx) / n), .seen = true }, .decay = decay };
}

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

test "fit: points from a curve give the curve back, one point keeps the seeded decay, a flat set fits the base only" {
    const local: Curve = .{ .base = .{ .v = 2000 }, .decay = 0.44 };
    var pts: [3]Point = undefined;
    for (&pts, [_]u64{ 32768, 131072, 524288 }) |*p, rows| p.* = .{ .rows = rows, .s = local.span(0, rows) };
    const got = fit(&pts, .{ .base = .{ .v = 1 }, .decay = 0 });
    try std.testing.expectApproxEqAbs(@as(f64, 2000), got.base.v, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 0.44), got.decay, 1e-6);
    try std.testing.expect(got.base.seen);
    const one = fit(pts[2..], .{ .base = .{ .v = 1 }, .decay = 0.44 });
    try std.testing.expectApproxEqAbs(@as(f64, 2000), one.base.v, 1);
    const flat = fit(&.{ .{ .rows = 8192, .s = 4 }, .{ .rows = 32768, .s = 16 } }, .{ .base = .{ .v = 1 }, .decay = 0.3 });
    try std.testing.expectApproxEqAbs(@as(f64, 2048), flat.base.v, 1e-6);
    try std.testing.expectEqual(@as(f64, 0.3), flat.decay);
    // a pull that gets faster with length is flat, not a negative decay
    const rising = fit(&.{ .{ .rows = 32768, .s = 20 }, .{ .rows = 131072, .s = 60 } }, .{ .base = .{ .v = 1 }, .decay = 0.3 });
    try std.testing.expectEqual(@as(f64, 0), rising.decay);
}

test "parsePoints: rows:seconds pairs, nothing else" {
    var buf: [max_points]Point = undefined;
    const ok = parsePoints(" 32768:14.5, 131072:70 ", &buf).?;
    try std.testing.expectEqual(@as(usize, 2), ok.len);
    try std.testing.expectEqual(@as(u64, 131072), ok[1].rows);
    try std.testing.expectEqual(@as(f64, 14.5), ok[0].s);
    for ([_][]const u8{ "", ",", "32768", "32768:", ":5", "0:5", "32768:0", "32768:-1", "x:5", "32768:nan", "1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9" }) |bad|
        try std.testing.expectEqual(@as(?[]const Point, null), parsePoints(bad, &buf));
}
