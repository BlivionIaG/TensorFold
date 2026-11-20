//! Independent FP64 and documented-order FP32 GLPv3 models shared by device cases and registered contracts.

const std = @import("std");

pub const nt: usize = 320; // threads a CTA
pub const gpt: usize = 4; // groups of eight columns a thread holds
pub const chain: usize = 128; // columns a block of the final's square chain
pub const max_width: usize = 5120; // the served width: the launch checks refuse more (the kernel's rows hold nt * 8 * gpt)
pub const max_world: usize = 8;

pub const Mode = enum(c_int) { add = 0, project = 1 };
pub const Acc = enum { f32, f16, bf16 };

/// The arithmetic a planted precision fault changes: everything else stays the documented fp32.
pub const Arith = struct { dir_bf16: bool = false, mean_acc: Acc = .f32, dot_acc: Acc = .f32 };

// ---- bf16 -------------------------------------------------------------------------------------------------------

pub fn widen(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn isNan16(h: u16) bool {
    return (h & 0x7f80) == 0x7f80 and (h & 0x7f) != 0;
}

/// fp32 to bf16, round to nearest even; any NaN stays a NaN (payloads are not compared).
pub fn rne32(f: f32) u16 {
    const u: u32 = @bitCast(f);
    if (f != f) return @intCast((u >> 16) | 0x40);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// FP64 to bf16 in one rounding (never through fp32): the ulp of the value's binade, ties to the even significand.
pub fn rne64(v: f64) u16 {
    if (std.math.isNan(v)) return 0x7fc0;
    const neg: u16 = if (std.math.signbit(v)) 0x8000 else 0;
    if (std.math.isInf(v)) return neg | 0x7f80;
    if (v == 0) return neg;
    const a = @abs(v);
    const e = std.math.frexp(a).exponent; // a = m 2^e, m in [0.5, 1)
    const q = std.math.ldexp(@as(f64, 1), @max(e - 8, -133)); // bf16: 8 significant bits, subnormal ulp 2^-133
    const t = a / q;
    var n = @floor(t);
    const frac = t - n;
    if (frac > 0.5 or (frac == 0.5 and @mod(n, 2) == 1)) n += 1;
    const r = n * q;
    if (r >= 0x1p128) return neg | 0x7f80;
    const f: f32 = @floatCast(r); // r has at most 8 significant bits: exact in fp32
    return neg | @as(u16, @intCast(@as(u32, @bitCast(f)) >> 16));
}

pub fn same(got: u16, want: u16) bool {
    return got == want or (isNan16(got) and isNan16(want));
}

fn key(h: u16) i32 {
    return if (h & 0x8000 != 0) -@as(i32, h & 0x7fff) else @as(i32, h);
}

/// Distance in bf16 steps between two finite values (a non-finite difference counts as huge).
pub fn ulps(a: u16, b: u16) u32 {
    if ((a & 0x7f80) == 0x7f80 or (b & 0x7f80) == 0x7f80) return if (same(a, b)) 0 else 0xffff;
    return @abs(key(a) - key(b));
}

fn accRound(a: Acc, v: f32) f32 {
    return switch (a) {
        .f32 => v,
        .f16 => @as(f32, @floatCast(@as(f16, @floatCast(v)))),
        .bf16 => widen(rne32(v)),
    };
}

// ---- the host fp32 model of the documented order -----------------------------------------------------------------

/// Each copy times the exact 0.25, then added in copy order (no overflow for finite copies).
fn mean4(v: [4]f32, acc: Acc) f32 {
    var s = accRound(acc, v[0] * 0.25);
    for (1..4) |c| s = accRound(acc, s + accRound(acc, v[c] * 0.25));
    return s;
}

/// Model the per-thread FMA chain, 32-lane XOR butterfly and ordered warp-partial sum.
pub fn dot(m: []const f32, d: []const f32, acc: Acc) f32 {
    const ng = m.len / 8;
    var total: f32 = 0;
    for (0..nt / 32) |w| {
        var lane: [32]f32 = undefined;
        for (&lane, 0..) |*slot, l| {
            const t = w * 32 + l;
            var s: f32 = 0;
            for (0..gpt) |i| {
                const g = t + i * nt;
                if (g >= ng) continue;
                for (0..8) |k| s = accRound(acc, @mulAdd(f32, m[g * 8 + k], d[g * 8 + k], s));
            }
            slot.* = s;
        }
        var off: usize = 16;
        while (off > 0) : (off >>= 1) {
            var next: [32]f32 = undefined;
            for (&next, 0..) |*slot, l| slot.* = lane[l] + lane[l ^ off];
            lane = next;
        }
        total += lane[0];
    }
    return total;
}

fn dirOf(d: []const f32, k: usize, a: Arith) f32 {
    return if (a.dir_bf16) widen(rne32(d[k])) else d[k];
}

/// One fold row: `x` is the four copies (4 width), `y` their steered values, `tap` the 0.25 mean of the steered copies.
pub fn foldRow(x: []const u16, d: []const f32, alpha: f32, mode: Mode, a: Arith, y: []u16, tap: ?[]u16) void {
    const w = d.len;
    var s: f32 = 0;
    if (mode == .project) {
        var mb: [max_width]f32 = undefined;
        var db: [max_width]f32 = undefined;
        for (0..w) |k| {
            mb[k] = mean4(.{ widen(x[k]), widen(x[w + k]), widen(x[2 * w + k]), widen(x[3 * w + k]) }, a.mean_acc);
            db[k] = dirOf(d, k, a);
        }
        s = alpha * dot(mb[0..w], db[0..w], a.dot_acc);
    }
    for (0..w) |k| {
        const dk = dirOf(d, k, a);
        for (0..4) |c| {
            const xc = widen(x[c * w + k]);
            y[c * w + k] = rne32(if (mode == .project) xc - s * dk else xc + alpha * dk);
        }
        if (tap) |t| t[k] = rne32(mean4(.{ widen(y[k]), widen(y[w + k]), widen(y[2 * w + k]), widen(y[3 * w + k]) }, a.mean_acc));
    }
}

/// One writer row: `ranks[r]` is rank r's partial row; the result is rank 0's slot (the others read -0.0).
pub fn writerRow(world: usize, ranks: []const []const u16, d: []const f32, alpha: f32, mode: Mode, a: Arith, y: []u16) void {
    const w = d.len;
    var ob: [max_width]f32 = undefined;
    for (0..w) |k| {
        var s = widen(ranks[0][k]);
        for (1..world) |r| s += widen(ranks[r][k]);
        ob[k] = widen(rne32(s)); // the post reads the rounded partial sum
    }
    var s: f32 = 0;
    if (mode == .project) {
        var db: [max_width]f32 = undefined;
        for (0..w) |k| db[k] = dirOf(d, k, a);
        s = alpha * dot(ob[0..w], db[0..w], a.dot_acc);
    }
    for (0..w) |k| {
        const dk = dirOf(d, k, a);
        y[k] = rne32(if (mode == .project) ob[k] - s * dk else ob[k] + alpha * dk);
    }
}

/// Reproduce mhc_cuda's BF16 collapse, ordered 128-column square chains and RMS/weight output.
pub fn finalRow(x: []const u16, q: [4]f32, nw: []const u8, nw16: bool, eps: f32, out: []u16) void {
    const w = out.len;
    var cb: [max_width]f32 = undefined;
    for (0..w) |k| {
        cb[k] = widen(rne32(q[0] * widen(x[k]) + q[1] * widen(x[w + k]) + q[2] * widen(x[2 * w + k]) + q[3] * widen(x[3 * w + k])));
    }
    var total: f32 = 0;
    for (0..w / chain) |b| {
        var acc: f32 = 0;
        for (0..chain) |e| acc = @mulAdd(f32, cb[b * chain + e], cb[b * chain + e], acc);
        total += acc;
    }
    const rn: f32 = 1.0 / @sqrt(total / @as(f32, @floatFromInt(w)) + eps);
    for (0..w) |k| {
        const wv: f32 = if (nw16) widen(std.mem.readInt(u16, nw[2 * k ..][0..2], .little)) else @bitCast(std.mem.readInt(u32, nw[4 * k ..][0..4], .little));
        out[k] = rne32((cb[k] * rn) * wv);
    }
}

// ---- the FP64 formula -------------------------------------------------------------------------------------------

/// Include copy magnitudes in the dot error scale so cancellation cannot erase mean-rounding error.
pub const Row = struct { dot: f64 = 0, mag: f64 = 0 };

/// Fold: m = 0.25 (x0 + x1 + x2 + x3), s = alpha dot(m, d), project y = x - s d, add y = x + alpha d, in FP64.
pub fn foldRef(x: []const u16, d: []const f32, alpha: f32, mode: Mode, y: []f64) Row {
    const w = d.len;
    var row: Row = .{};
    if (mode == .project) for (0..w) |k| {
        const m = 0.25 * (@as(f64, widen(x[k])) + widen(x[w + k]) + widen(x[2 * w + k]) + widen(x[3 * w + k]));
        row.dot += m * d[k];
        row.mag += @abs(d[k]) * 0.25 * (@abs(@as(f64, widen(x[k]))) + @abs(widen(x[w + k])) + @abs(widen(x[2 * w + k])) + @abs(widen(x[3 * w + k])));
    };
    const s = @as(f64, alpha) * row.dot;
    for (0..w) |k| for (0..4) |c| {
        const xc: f64 = widen(x[c * w + k]);
        y[c * w + k] = if (mode == .project) xc - s * d[k] else xc + @as(f64, alpha) * d[k];
    };
    return row;
}

/// Writer: o = bf16(sum of the ranks) once, then the same edit on the one vector, in FP64.
pub fn writerRef(world: usize, ranks: []const []const u16, d: []const f32, alpha: f32, mode: Mode, o16: []u16, y: []f64) Row {
    const w = d.len;
    var row: Row = .{};
    for (0..w) |k| {
        var s: f64 = 0;
        for (0..world) |r| s += widen(ranks[r][k]);
        o16[k] = rne64(s);
        const o: f64 = widen(o16[k]);
        if (mode == .project) {
            row.dot += o * d[k];
            row.mag += @abs(o * d[k]);
        }
    }
    const s = @as(f64, alpha) * row.dot;
    for (0..w) |k| {
        const o: f64 = widen(o16[k]);
        y[k] = if (mode == .project) o - s * d[k] else o + @as(f64, alpha) * d[k];
    }
    return row;
}

/// Final in FP64: c = bf16(sum q x) once, rn from the FP64 squares of the bf16 c, out = c rn w (unrounded).
pub fn finalRef(x: []const u16, q: [4]f32, nw: []const u8, nw16: bool, eps: f32, out: []f64) void {
    const w = out.len;
    var cb: [max_width]f64 = undefined;
    var ss: f64 = 0;
    for (0..w) |k| {
        cb[k] = widen(rne64(@as(f64, q[0]) * widen(x[k]) + @as(f64, q[1]) * widen(x[w + k]) + @as(f64, q[2]) * widen(x[2 * w + k]) + @as(f64, q[3]) * widen(x[3 * w + k])));
        ss += cb[k] * cb[k];
    }
    const rn = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(w)) + eps);
    for (0..w) |k| {
        const wv: f64 = if (nw16) widen(std.mem.readInt(u16, nw[2 * k ..][0..2], .little)) else @as(f32, @bitCast(std.mem.readInt(u32, nw[4 * k ..][0..4], .little)));
        out[k] = cb[k] * rn * wv;
    }
}

/// Bound FP32 dot by gamma(50)*copy-magnitude, post-dot by 3u, then BF16 half-ulp plus writer rounding flips.
pub fn allow(want: f64, x: f64, dk: f64, alpha: f32, mode: Mode, row: Row, extra: f64) f64 {
    const u = 0x1p-24;
    const a: f64 = alpha;
    var work: f64 = 2 * u * (@abs(x) + @abs(a * dk));
    if (mode == .project) {
        const g = 50 * u / (1 - 50 * u);
        work = g * @abs(a * dk) * row.mag + 3 * u * (@abs(x) + @abs(a * row.dot * dk)) + @abs(a * dk) * extra;
    }
    return work + 0x1p-8 * (@abs(want) + work) + 0x1p-133;
}

// ---- seeded generators ---------------------------------------------------------------------------------------------

pub const Rng = struct {
    s: u64,

    pub fn init(seed: u64) Rng {
        return .{ .s = seed *% 0x9e3779b97f4a7c15 | 1 };
    }

    pub fn next(r: *Rng) u64 {
        var x = r.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        r.s = x;
        return x *% 0x2545f4914f6cdd1d;
    }

    pub fn below(r: *Rng, n: u64) u64 {
        return (r.next() >> 11) % n;
    }

    pub fn unit(r: *Rng) f64 {
        return @as(f64, @floatFromInt(r.next() >> 11)) * 0x1p-53;
    }
};

/// A bf16 with magnitude spread over 2^-6 .. 2^6 and seven random mantissa bits: sums of these sit on bf16 ties often.
pub fn spread(r: *Rng) u16 {
    const sign: u16 = @intCast(r.below(2));
    const exp: u16 = @intCast(121 + r.below(13));
    const man: u16 = @intCast(r.below(128));
    return (sign << 15) | (exp << 7) | man;
}

/// An fp32 direction entry: a full 24-bit mantissa, magnitude 2^-4 .. 1.
pub fn dirEntry(r: *Rng) f32 {
    const m: f32 = @floatCast(r.unit() * 2 - 1);
    return m * std.math.ldexp(@as(f32, 1), -@as(i32, @intCast(r.below(5))));
}

/// Generate ordinary, tie, cancellation, exact-sum, BF16-max and nonfinite fixtures deterministically.
pub const Fill = enum { random, ties, cancel, stress, extreme, nonfinite };

pub const nan16: u16 = 0x7fc0;
pub const inf16: u16 = 0x7f80;
pub const ninf16: u16 = 0xff80;

pub fn fillDir(kind: Fill, r: *Rng, d: []f32) void {
    const h = d.len / 2;
    for (d, 0..) |*v, k| v.* = switch (kind) {
        .ties => if (k % 2 == 0) 0x1p-8 else -0x1p-8,
        .stress => @floatFromInt(1 + r.below(15)),
        .extreme => if (k < 4) 0x1p-12 else 0,
        else => dirEntry(r),
    };
    switch (kind) {
        .cancel => for (0..h) |i| {
            d[h + i] = -d[i] * (1 + @as(f32, @floatFromInt(r.below(64))) * 0x1p-18); // the mirror, perturbed
        },
        .stress => for (0..h) |i| {
            d[h + i] = -d[i];
        },
        else => {},
    }
}

/// One row of four copies `x` (4 width) for a fill; `row` picks the nonfinite placements.
pub fn fillRow(kind: Fill, r: *Rng, row: usize, x: []u16, d: []const f32) void {
    const w = d.len;
    const h = w / 2;
    switch (kind) {
        .random, .nonfinite => for (x) |*v| {
            v.* = spread(r);
        },
        .ties => for (0..4) |c| for (0..w) |k| {
            x[c * w + k] = 0x3f80 | @as(u16, @intCast((k + 5 * c) & 127));
        },
        .cancel => for (0..h) |k| {
            const v = spread(r);
            for (0..4) |c| {
                x[c * w + k] = v;
                x[c * w + h + k] = v;
            }
        },
        .stress => for (0..h) |k| {
            const v: f32 = @floatFromInt(1 + r.below(127));
            const lo: u16 = @intCast(@as(u32, @bitCast(v)) >> 16);
            const hi: u16 = @intCast(@as(u32, @bitCast(v + if (k < 8) @as(f32, 1) else 0)) >> 16);
            for (0..4) |c| {
                x[c * w + k] = lo;
                x[c * w + h + k] = hi;
            }
        },
        .extreme => @memset(x, if (row % 2 == 0) 0x7f7f else 0xff7f),
    }
    if (kind != .nonfinite) return;
    switch (row) {
        0 => x[w + w / 3] = nan16,
        1 => x[w / 2] = inf16,
        2 => x[3 * w + w - 1] = ninf16,
        3 => {
            x[0] = inf16;
            x[w - 1] = ninf16;
        },
        else => {},
    }
}

/// bf16 word of a pad column: a finite value no kernel computes, distinct per position.
pub fn padWord(i: usize) u16 {
    return 0x3f00 | @as(u16, @truncate(i *% 37 +% 11)) & 0x7f;
}

// ---- the comparison tally ------------------------------------------------------------------------------------------

pub const Tally = struct {
    n: usize = 0,
    model_diff: usize = 0, // bits that differ from the fp32 order model
    class: usize = 0, // NaN / infinity class differs from FP64
    flips: usize = 0, // bits that differ from the FP64 result rounded once
    max_ulp: u32 = 0,
    over: usize = 0, // beyond the stated bound
    bad_pad: usize = 0, // words outside the edited region that changed
    first: ?usize = null,

    pub fn add(t: *Tally, i: usize, got: u16, model: u16, want: f64, bound: f64) void {
        t.n += 1;
        var bad = false;
        if (!same(got, model)) {
            t.model_diff += 1;
            bad = true;
        }
        if (!std.math.isFinite(want)) {
            const w16 = rne64(want);
            if (!same(got, w16)) {
                t.class += 1;
                bad = true;
            }
        } else {
            const w16 = rne64(want);
            if (!same(got, w16)) {
                t.flips += 1;
                t.max_ulp = @max(t.max_ulp, ulps(got, w16));
            }
            if (!(@abs(@as(f64, widen(got)) - want) <= bound)) {
                t.over += 1;
                bad = true;
            }
        }
        if (bad and t.first == null) t.first = i;
    }

    pub fn clean(t: Tally) bool {
        return t.model_diff == 0 and t.class == 0 and t.over == 0 and t.bad_pad == 0;
    }

    pub fn bits(t: Tally) usize {
        return t.model_diff + t.class + t.over + t.bad_pad;
    }
};

test "rne64 rounds once, ties to even, and matches rne32 where fp32 is exact" {
    try std.testing.expectEqual(@as(u16, 0x3f80), rne64(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), rne64(1.0 + 0x1p-8)); // tie: even is 1.0
    try std.testing.expectEqual(@as(u16, 0x3f82), rne64(1.0 + 0x1p-7 + 0x1p-8)); // tie: even is 1.015625
    try std.testing.expectEqual(@as(u16, 0x3f81), rne64(1.0 + 0x1p-8 + 0x1p-40)); // past the tie: up
    try std.testing.expectEqual(@as(u16, 0xbf80), rne64(-1.0));
    try std.testing.expectEqual(@as(u16, 0x7f80), rne64(0x1p128));
    try std.testing.expectEqual(@as(u16, 0x7f7f), rne64(0x1.fep127));
    var r = Rng.init(7);
    for (0..1000) |_| {
        const h = spread(&r);
        try std.testing.expectEqual(h, rne64(widen(h)));
        const f: f32 = @floatCast((r.unit() - 0.5) * 100);
        try std.testing.expectEqual(rne32(f), rne64(f));
    }
}

test "the model's fold of equal copies projects the mean" {
    var x: [4 * 16]u16 = undefined;
    var d: [16]f32 = undefined;
    var y: [4 * 16]u16 = undefined;
    for (&d) |*v| v.* = 1;
    for (0..4) |c| for (0..16) |k| {
        x[c * 16 + k] = rne32(@floatFromInt(k));
    };
    foldRow(&x, &d, 1, .project, .{}, &y, null);
    // dot(m, d) = 120 over 16 columns, so every column loses 120
    try std.testing.expectEqual(rne32(@as(f32, 5) - 120), y[5]);
}
