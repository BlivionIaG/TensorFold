//! FP64 GLPv3 fold/writer reference; error derivation and deterministic stress fixtures are in the GLP recipe.
const std = @import("std");
const contract = @import("contract");

const c = contract.spec;
const rows = contract.rows;

pub const Fill = rows.Fill;
pub const Order = rows.Order;

/// The strength random fills use; the stress fill uses 1 so every product and sum is an exact integer.
pub const strength: f32 = 0.75;

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

fn plain(t: []const f64) f64 {
    var s: f64 = 0;
    for (t) |v| s += v;
    return s;
}

/// One call's host inputs: `rows` rows of `copies` bf16 copies of `dim` values, and the fp32 direction.
pub const Case = struct {
    rows: usize,
    dim: usize,
    copies: usize,
    project: bool,
    alpha: f32 = strength,
    x: []u16,
    d: []f32,

    pub fn init(a: std.mem.Allocator, copies: usize, project: bool, n: usize, dim: usize) !Case {
        const x = try a.alloc(u16, n * copies * dim);
        return .{ .rows = n, .dim = dim, .copies = copies, .project = project, .x = x, .d = try a.alloc(f32, dim) };
    }

    pub fn count(s: Case) usize {
        return s.rows * s.copies * s.dim;
    }

    /// Ordinary values, NaN and infinities at known places, gate 4's integers, or (`one_hot`) the rank-one probe's.
    pub fn fill(s: *Case, mode: Fill, seed: u64, one_hot: bool) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        s.alpha = if (mode == .stress) 1 else strength;
        for (s.d) |*v| v.* = rng.float(f32) * 2 - 1;
        for (s.x, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 4 - 2) * (if (i % 64 == 7) @as(f32, 10) else 1));
        if (one_hot) return s.oneHot();
        switch (mode) {
            .random => {},
            .nonfinite => s.plant(),
            .stress => s.stress(rng),
        }
    }

    /// A NaN and both infinities in some rows (the rest stay finite); a zero direction entry under one infinity.
    fn plant(s: *Case) void {
        const k = s.dim;
        if (k < 4) return;
        s.d[k / 2] = 0;
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        const at = [_][3]usize{ .{ 0, 1, k / 3 }, .{ 1, 0, k / 2 }, .{ 2, 3, k - 1 }, .{ 3, 0, 0 }, .{ 3, 2, k - 1 } };
        const put = [_]u16{ nan, inf, ninf, inf, ninf };
        for (at, put) |q, v| if (q[0] < s.rows) {
            s.x[(q[0] * s.copies + q[1] % s.copies) * k + q[2]] = v;
        };
    }

    /// Mirrored integer halves keep FP32 sums exact; one copy's nudge leaves a small dot amid large partial sums.
    fn stress(s: *Case, rng: std.Random) void {
        const k = s.dim;
        const h = k / 2;
        for (0..h) |i| {
            const dv: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 12));
            s.d[i] = dv;
            s.d[h + i] = dv;
        }
        if (k % 2 == 1) s.d[k - 1] = 0;
        for (0..s.rows) |r| {
            for (0..s.copies) |cc| {
                const x = s.x[(r * s.copies + cc) * k ..][0..k];
                for (0..h) |i| {
                    const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 32));
                    x[i] = c.toBf16(v);
                    x[h + i] = c.toBf16(-v);
                }
                if (k % 2 == 1) x[k - 1] = 0;
            }
            if (h > 0) {
                const x = s.x[r * s.copies * k ..][0..k];
                x[h] = c.toBf16(@floatCast(-bf(x[0]) + @as(f64, @floatFromInt(s.copies))));
            }
        }
    }

    /// A one-hot direction removes only its selected mean component and leaves every other BF16 value unchanged.
    fn oneHot(s: *Case) void {
        const at = s.dim / 3;
        @memset(s.d, 0);
        s.d[at] = 1;
        s.alpha = 1;
        for (0..s.rows) |r| for (0..s.copies) |cc| {
            const dr: f32 = @floatFromInt(r);
            const dc: f32 = @floatFromInt(cc);
            s.x[(r * s.copies + cc) * s.dim + at] = c.toBf16(2 + 0.5 * dc + 0.125 * dr);
        };
    }

    /// One row's float64 outputs and magnitudes; `acc` other than none runs the dot's chain in that accumulator.
    fn row(s: Case, r: usize, want: []f64, mag: []f64, t: []f64, acc: c.Accum, order: Order) void {
        const k = s.dim;
        const cp = s.copies;
        const a: f64 = s.alpha;
        const xr = s.x[r * cp * k ..][0 .. cp * k];
        if (!s.project) {
            for (xr, 0..) |h, n| {
                const dv: f64 = s.d[n % k];
                want[n] = bf(h) + a * dv;
                mag[n] = @abs(bf(h)) + a * @abs(dv);
            }
            return;
        }
        const inv: f64 = 1.0 / @as(f64, @floatFromInt(cp));
        var big: f64 = 0;
        for (0..k) |j| {
            var m: f64 = 0;
            var am: f64 = 0;
            for (0..cp) |cc| {
                const v = bf(xr[cc * k + j]);
                m += v;
                am += @abs(v);
            }
            t[j] = m * inv * s.d[j];
            big += @abs(@as(f64, s.d[j])) * am * inv;
        }
        const dot = if (acc == .none) plain(t[0..k]) else rows.sumIn(acc, order, t[0..k]);
        const step = a * dot;
        for (xr, 0..) |h, n| {
            const dv: f64 = s.d[n % k];
            want[n] = bf(h) - step * dv;
            mag[n] = @abs(bf(h)) + a * @abs(dv) * big;
        }
    }

    /// Every output in float64 and its term magnitude; with `acc`, the outputs if the dot's sum ran in it (gate 4).
    pub fn reference(s: Case, a: std.mem.Allocator, want: []f64, mag: []f64, acc: c.Accum, order: Order) !void {
        const t = try a.alloc(f64, s.dim);
        defer a.free(t);
        const per = s.copies * s.dim;
        for (0..s.rows) |r| s.row(r, want[r * per ..][0..per], mag[r * per ..][0..per], t, acc, order);
    }
};

/// Rank one: the worst |dot(mean(out), d)| over its bound, and the least |dot(mean(x), d)| over the same bound.
pub const Rank = struct { worst: f64, least: f64 };

/// Bound residual projection by ideal dot*(1-alpha*|d|^2) plus direction-weighted output error.
pub fn rankOne(s: Case, a: std.mem.Allocator, out: []const u8, sp: c.Spec) !Rank {
    const k = s.dim;
    const cp = s.copies;
    const buf = try a.alloc(f64, 2 * cp * k + k);
    defer a.free(buf);
    const want = buf[0 .. cp * k];
    const mag = buf[cp * k .. 2 * cp * k];
    const t = buf[2 * cp * k ..];
    var n2: f64 = 0;
    for (s.d) |v| n2 += @as(f64, v) * v;
    const shape = c.Shape{ .rows = s.rows, .n = 1, .k = k };
    const inv: f64 = 1.0 / @as(f64, @floatFromInt(cp));
    var rank = Rank{ .worst = 0, .least = std.math.inf(f64) };
    for (0..s.rows) |r| {
        s.row(r, want, mag, t, .none, .sequential);
        var got: f64 = 0;
        var before: f64 = 0;
        var slack: f64 = 0;
        for (0..k) |j| {
            var m: f64 = 0;
            var mx: f64 = 0;
            var al: f64 = 0;
            for (0..cp) |cc| {
                m += c.read(.bf16, out, (r * cp + cc) * k + j);
                mx += bf(s.x[(r * cp + cc) * k + j]);
                al += c.bound(sp, shape, want[cc * k + j], mag[cc * k + j]);
            }
            got += m * inv * s.d[j];
            before += mx * inv * s.d[j];
            slack += @abs(@as(f64, s.d[j])) * al * inv;
        }
        slack += @abs(before * (1 - @as(f64, s.alpha) * n2));
        rank.worst = @max(rank.worst, @abs(got) / slack);
        rank.least = @min(rank.least, @abs(before) / slack);
    }
    return rank;
}
