//! The affine 4-bit product on the host: the kernel's fp32 recipe, a float64 reference and the bound between them.

const std = @import("std");

/// One seeded product: bf16 x, packed 4-bit codes, bf16 scales and biases.
pub const Problem = struct {
    m: usize,
    n: usize,
    k: usize,
    group: usize,
    x: []u16,
    words: []u32,
    scale: []u16,
    bias: []u16,

    pub fn init(gpa: std.mem.Allocator, seed: u64, m: usize, n: usize, k: usize, group: usize) !Problem {
        var r: Rng = .{ .state = seed | 1 };
        const p: Problem = .{
            .m = m,
            .n = n,
            .k = k,
            .group = group,
            .x = try gpa.alloc(u16, m * k),
            .words = try gpa.alloc(u32, n * k / 8),
            .scale = try gpa.alloc(u16, n * (k / group)),
            .bias = try gpa.alloc(u16, n * (k / group)),
        };
        for (p.x) |*v| v.* = bf16Bits(r.fine());
        for (p.words) |*v| v.* = @truncate(r.next() >> 16);
        for (p.scale) |*v| v.* = bf16Bits((r.unit() + 1.0) / 16.0);
        for (p.bias) |*v| v.* = bf16Bits(r.unit() / 2.0);
        return p;
    }

    pub fn deinit(p: Problem, gpa: std.mem.Allocator) void {
        gpa.free(p.x);
        gpa.free(p.words);
        gpa.free(p.scale);
        gpa.free(p.bias);
    }

    fn code(p: Problem, col: usize, k: usize) u32 {
        return p.words[col * (p.k / 8) + k / 8] >> @intCast(4 * (k % 8)) & 15;
    }

    /// Output (row, col) as the kernel computes it: per group an ordered fma and a plain sum, then two fmas.
    pub fn serial(p: Problem, row: usize, col: usize) f32 {
        const groups = p.k / p.group;
        var acc: f32 = 0;
        for (0..groups) |g| {
            var dot: f32 = 0;
            var sum: f32 = 0;
            for (g * p.group..(g + 1) * p.group) |t| {
                const xv = bf16(p.x[row * p.k + t]);
                dot = @mulAdd(f32, xv, @floatFromInt(p.code(col, t)), dot);
                sum += xv;
            }
            acc = @mulAdd(f32, dot, bf16(p.scale[col * groups + g]), acc);
            acc = @mulAdd(f32, sum, bf16(p.bias[col * groups + g]), acc);
        }
        return acc;
    }

    /// Output (row, col) in float64, and the sum of its terms' magnitudes.
    pub fn reference(p: Problem, row: usize, col: usize) Value {
        const groups = p.k / p.group;
        var v: Value = .{ .y = 0, .norm = 0 };
        for (0..p.k) |t| {
            const at = col * groups + t / p.group;
            const w = @as(f64, @floatFromInt(p.code(col, t))) * bf16(p.scale[at]) + bf16(p.bias[at]);
            const term = @as(f64, bf16(p.x[row * p.k + t])) * w;
            v.y += term;
            v.norm += @abs(term);
        }
        return v;
    }

    /// Every output of the host recipe, row major, for a byte comparison with the GPU's.
    pub fn serialAll(p: Problem, gpa: std.mem.Allocator) ![]f32 {
        const out = try gpa.alloc(f32, p.m * p.n);
        for (0..p.m) |row| for (0..p.n) |col| {
            out[row * p.n + col] = p.serial(row, col);
        };
        return out;
    }
};

pub const Value = struct { y: f64, norm: f64 };

/// The largest |y - ref| over the sum of the terms' magnitudes an fp32 sum may have: a few units of rounding a term.
pub fn bound(k: usize) f64 {
    return 0x1p-24 * (8 + 2 * @sqrt(@as(f64, @floatFromInt(k))));
}

/// |y - ref| of an output in units of its bound; above 1 fails.
pub fn excess(y: f32, v: Value, k: usize) f64 {
    return @abs(@as(f64, y) - v.y) / (bound(k) * @max(v.norm, 1e-30));
}

fn bf16(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

fn bf16Bits(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16);
}

const Rng = struct {
    state: u64,

    fn next(r: *Rng) u64 {
        r.state ^= r.state >> 12;
        r.state ^= r.state << 25;
        r.state ^= r.state >> 27;
        return r.state *% 0x2545F4914F6CDD1D;
    }

    /// A value in [-1, 1) with a few mantissa bits.
    fn unit(r: *Rng) f32 {
        const v: i32 = @intCast(r.next() >> 40 & 0x7ff);
        return @as(f32, @floatFromInt(v - 1024)) / 1024.0;
    }

    /// A full-mantissa value in [-1, 1) at one of six scales: sums of products then round, in any order.
    fn fine(r: *Rng) f32 {
        const v = r.next();
        const m: i32 = @intCast(v >> 41 & 0x3fffff);
        return @as(f32, @floatFromInt(m - 0x200000)) / 2097152.0 * std.math.ldexp(@as(f32, 1), -@as(i32, @intCast(v % 6)));
    }
};

/// The pinned problem whose output digest the host and the GPU tests both check.
pub const pinned = .{ .seed = 0x7f4a7c15, .m = 4, .n = 96, .k = 1024, .group = 64 };

/// Wyhash of the pinned problem's fp32 outputs as the recipe computes them; a moved bit changes it.
pub const pinned_digest: u64 = 0x550e1ce1da51621d;

pub fn digest(out: []const f32) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(out));
}

test "the fp32 recipe stays within the fp32 bound of float64 for every group size" {
    const gpa = std.testing.allocator;
    for ([_]usize{ 32, 64, 128 }) |group| {
        const p = try Problem.init(gpa, 0x51ed + group, 3, 40, 4096, group);
        defer p.deinit(gpa);
        for (0..p.m) |row| for (0..p.n) |col| {
            try std.testing.expect(excess(p.serial(row, col), p.reference(row, col), p.k) <= 1);
        };
    }
}

test "the pinned problem's outputs keep their bits" {
    const gpa = std.testing.allocator;
    const p = try Problem.init(gpa, pinned.seed, pinned.m, pinned.n, pinned.k, pinned.group);
    defer p.deinit(gpa);
    const out = try p.serialAll(gpa);
    defer gpa.free(out);
    try std.testing.expectEqual(pinned_digest, digest(out));
}
