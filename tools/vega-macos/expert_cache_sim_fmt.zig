//! What expert_cache_sim.py's output needs from Python and NumPy: `{x:.Nf}` / `{x:+.Nf}` / `%+.2f` (the exact binary value rounded half to even,
//! "nan" and "inf" spelled as Python does) and NumPy's pairwise float64 sum with the mean and sample standard deviation built on it.
const std = @import("std");

/// `format(x, ".<prec>f")`, or `"+.<prec>f"` with `plus`; the same text as CPython for finite values of ordinary size.
pub fn fixed(w: *std.Io.Writer, x: f64, prec: u8, plus: bool) std.Io.Writer.Error!void {
    if (std.math.isNan(x)) return w.writeAll(if (plus) "+nan" else "nan");
    const neg = std.math.signbit(x);
    if (neg) try w.writeByte('-') else if (plus) try w.writeByte('+');
    if (std.math.isInf(x)) return w.writeAll("inf");
    const bits: u64 = @bitCast(@abs(x));
    const biased: i32 = @intCast(bits >> 52);
    const frac = bits & ((1 << 52) - 1);
    // |x| = m * 2^e
    const m: u128 = if (biased == 0) frac else frac | (1 << 52);
    const e: i32 = (if (biased == 0) 1 else biased) - 1075;
    var scale: u128 = 1;
    for (0..prec) |_| scale *= 10;
    const n = m * scale;
    var q: u128 = undefined;
    if (e >= 0) {
        if (e > 60) return w.print("{d}", .{@abs(x)}); // not a number this tool prints
        q = n << @intCast(e);
    } else if (-e >= 128) {
        q = 0; // below half a unit in the last place: n < 2**127 <= half
    } else {
        const s: u7 = @intCast(-e);
        q = n >> s;
        const rem = n & ((@as(u128, 1) << s) - 1);
        const half = @as(u128, 1) << (s - 1);
        if (rem > half or (rem == half and q & 1 == 1)) q += 1;
    }
    try w.print("{d}", .{q / scale});
    if (prec > 0) {
        var digits: [40]u8 = undefined;
        const text = std.fmt.bufPrint(&digits, "{d}", .{q % scale}) catch unreachable;
        try w.writeByte('.');
        try w.splatByteAll('0', prec - text.len);
        try w.writeAll(text);
    }
}

/// An allocation-free `fixed` into a small buffer, for table cells.
pub fn fixedBuf(buf: []u8, x: f64, prec: u8, plus: bool) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    fixed(&w, x, prec, plus) catch unreachable;
    return w.buffered();
}

/// NumPy's pairwise summation of a contiguous float64 array (`np.add.reduce`): eight interleaved partial sums for blocks of up to 128.
pub fn pairwiseSum(a: []const f64) f64 {
    const n = a.len;
    if (n < 8) {
        var res: f64 = 0;
        for (a) |v| res += v;
        return res;
    }
    if (n <= 128) {
        var r: [8]f64 = undefined;
        for (0..8) |j| r[j] = a[j];
        var i: usize = 8;
        while (i < n - n % 8) : (i += 8) for (0..8) |j| {
            r[j] += a[i + j];
        };
        var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while (i < n) : (i += 1) res += a[i];
        return res;
    }
    var n2 = n / 2;
    n2 -= n2 % 8;
    return pairwiseSum(a[0..n2]) + pairwiseSum(a[n2..]);
}

/// `a.mean()`.
pub fn mean(a: []const f64) f64 {
    return pairwiseSum(a) / @as(f64, @floatFromInt(a.len));
}

/// `a.std(ddof=1)`; `scratch` (same length as `a`) holds the squared deviations.
pub fn stdDdof1(a: []const f64, scratch: []f64) f64 {
    const mu = mean(a);
    for (a, scratch[0..a.len]) |v, *s| {
        const d = v - mu;
        s.* = d * d;
    }
    const denom: f64 = @floatFromInt(@max(a.len, 1) - 1);
    return @sqrt(pairwiseSum(scratch[0..a.len]) / denom);
}
