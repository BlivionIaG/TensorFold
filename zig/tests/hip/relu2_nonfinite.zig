const std = @import("std");

/// A fused epilogue must propagate every NaN; only a plain tile's infinity-to-NaN finding is inherited.
pub fn matches(raw: f64, sum: f64, out: f64) bool {
    const r = if (std.math.isNan(raw)) raw else @max(raw, 0);
    const want = r * r;
    if (std.math.isNan(want)) return std.math.isNan(out);
    if (std.math.isFinite(want) and std.math.isFinite(out)) return true;
    if (std.math.isInf(want) and out == want) return true;
    return std.math.isInf(raw) and std.math.isNan(sum) and std.math.isNan(out);
}

test "relu2 rejects dropped NaNs even beside the shipped infinity finding" {
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    try std.testing.expect(matches(nan, nan, nan));
    try std.testing.expect(!matches(nan, nan, 0));
    try std.testing.expect(matches(-inf, nan, nan));
    try std.testing.expect(!matches(1, nan, nan));
    try std.testing.expect(!matches(-inf, -inf, nan));
    try std.testing.expect(!matches(inf, nan, 0));
}
