//! The split offload's boundary rows between a host window (fp32 rows) and the activation rows (split/rows.hip).

const std = @import("std");
const t = @import("types.zig");
const Ops = @import("ops.zig").Ops;
const launches = @import("../launches/split_rows.zig");
const Error = t.Error;
const Tensor = t.Tensor;

/// Stamped groups of a row of `n` values, and bytes of the stamped row: wire.stampedLen(4 * n).
pub const groups = launches.groups;

pub fn stampedBytes(n: usize) usize {
    return groups(n) * 16;
}

/// dst (n values of dst.kind) = the fp32 row at the window address `src_f32`, rounded to nearest even.
pub fn importRow(o: Ops, src_f32: u64, dst: Tensor, n: usize) Error!void {
    try launches.importRow(o.l, src_f32, dst.ptr, @backingInt(dst.kind), n, o.stream);
}

/// The fp32 row at the window address `dst_f32` = src (n values of src.kind) widened, exactly.
pub fn exportRow(o: Ops, src: Tensor, dst_f32: u64, n: usize) Error!void {
    try launches.exportRow(o.l, src.ptr, dst_f32, @backingInt(src.kind), n, o.stream);
}

/// dst = the stamped row at `src` once every stamp is `seq`; `status` (a window word) is 0, or 1 if one stayed stale.
pub fn consumeStamped(o: Ops, src: u64, seq: u32, dst: Tensor, n: usize, status: u64, retries: u32) Error!void {
    try launches.consumeStamped(o.l, src, seq, dst.ptr, @backingInt(dst.kind), n, status, retries, o.stream);
}

// The semantics the kernels implement, on the host: what the tests below and the contract entry check them against.

/// One value in the activation kind's bits, rounded to nearest even; NaN stays NaN (bf16: the quiet NaN 0x7fc0).
pub fn roundBits(kind: t.Kind, v: f32) u32 {
    return switch (kind) {
        .f32 => @bitCast(v),
        .f16 => @as(u16, @bitCast(@as(f16, @floatCast(v)))),
        .bf16 => bf16Bits(v),
    };
}

fn bf16Bits(v: f32) u16 {
    if (std.math.isNan(v)) return 0x7fc0;
    const u: u32 = @bitCast(v);
    return @truncate((u +% ((u >> 16) & 1) +% 0x7fff) >> 16);
}

/// What `consumeStamped` leaves: the status word and the reads in all; group g is stale for `stale_reads[g]` reads.
pub const Consumed = struct { status: u32, reads: usize };

pub fn consumeReference(out: []u32, kind: t.Kind, region: []const [4]u32, seq: u32, stale_reads: []const u32, retries: u32) Consumed {
    var stale = false;
    var reads: usize = 0;
    for (0..groups(out.len)) |g| {
        const need: u64 = if (region[g][3] != seq) std.math.maxInt(u64) else stale_reads[g];
        reads += @intCast(@min(need, retries) + 1);
        if (need > retries) stale = true;
        for (0..3) |j| {
            if (g * 3 + j < out.len) out[g * 3 + j] = roundBits(kind, @bitCast(region[g][j]));
        }
    }
    return .{ .status = @intFromBool(stale), .reads = reads };
}

test "import rounds to nearest even in fp16 and bf16, keeps fp32 and NaN" {
    // a bf16 tie rounds to the even mantissa: 1 + 2^-8 is halfway between 1 and 1 + 2^-7
    try std.testing.expectEqual(@as(u32, 0x3f80), roundBits(.bf16, 1.0 + 0x1p-8));
    try std.testing.expectEqual(@as(u32, 0x3f82), roundBits(.bf16, 1.0 + 3 * 0x1p-8));
    try std.testing.expectEqual(@as(u32, 0x3f81), roundBits(.bf16, 1.0 + 0x1p-8 + 0x1p-20));
    try std.testing.expectEqual(@as(u32, 0x7fc0), roundBits(.bf16, std.math.nan(f32)));
    try std.testing.expectEqual(@as(u32, 0x7f80), roundBits(.bf16, std.math.inf(f32)));
    // fp16: 1 + 2^-11 is the tie between 1 and 1 + 2^-10
    try std.testing.expectEqual(@as(u32, 0x3c00), roundBits(.f16, 1.0 + 0x1p-11));
    try std.testing.expectEqual(@as(u32, 0x3c02), roundBits(.f16, 1.0 + 3 * 0x1p-11));
    try std.testing.expectEqual(@as(u32, 0x7c00), roundBits(.f16, 70000.0));
    try std.testing.expectEqual(@as(u32, 0x3f800000), roundBits(.f32, 1.0));
}

test "widening is exact: a bf16 or fp16 value survives the round trip through fp32" {
    var bits: u32 = 0;
    while (bits < 0x10000) : (bits += 7) {
        const b: f32 = @bitCast(bits << 16);
        if (!std.math.isNan(b)) try std.testing.expectEqual(bits, roundBits(.bf16, b));
        const h: f16 = @bitCast(@as(u16, @intCast(bits)));
        if (!std.math.isNan(h)) try std.testing.expectEqual(bits, roundBits(.f16, h));
    }
}

fn group(a: f32, b: f32, c: f32, stamp: u32) [4]u32 {
    return .{ @bitCast(a), @bitCast(b), @bitCast(c), stamp };
}

test "a current row is accepted whole, padding ignored" {
    // n = 7: three groups, the last holds one value and two padding slots that must not be read as data
    const region = [_][4]u32{ group(1, 2, 3, 9), group(4, 5, 6, 9), group(7, 123, 456, 9) };
    var out: [7]u32 = @splat(0xdead);
    const r = consumeReference(&out, .f16, &region, 9, &.{ 0, 0, 0 }, 0);
    try std.testing.expectEqual(@as(u32, 0), r.status);
    try std.testing.expectEqual(@as(usize, 3), r.reads);
    for (out, 1..) |v, i| try std.testing.expectEqual(roundBits(.f16, @floatFromInt(i)), v);
}

test "a stale group is re-read until fresh, within the retries" {
    const region = [_][4]u32{ group(1, 2, 3, 9), group(4, 5, 6, 9) };
    var out: [6]u32 = undefined;
    // group 1 is stale for three reads: with 3 retries its fourth read is current
    const r = consumeReference(&out, .bf16, &region, 9, &.{ 0, 3 }, 3);
    try std.testing.expectEqual(@as(u32, 0), r.status);
    try std.testing.expectEqual(@as(usize, 1 + 4), r.reads);
}

test "a group that stays stale makes status 1 after retries + 1 reads, and its values are the last read" {
    const region = [_][4]u32{ group(1, 2, 3, 9), group(4, 5, 6, 8) };
    var out: [6]u32 = undefined;
    const r = consumeReference(&out, .f16, &region, 9, &.{ 0, 0 }, 5);
    try std.testing.expectEqual(@as(u32, 1), r.status);
    try std.testing.expectEqual(@as(usize, 1 + 6), r.reads);
    try std.testing.expectEqual(roundBits(.f16, 4), out[3]);
    // retries 0 is one read: a stale group times out at once
    const z = consumeReference(&out, .f16, &region, 9, &.{ 0, 0 }, 0);
    try std.testing.expectEqual(@as(u32, 1), z.status);
    try std.testing.expectEqual(@as(usize, 2), z.reads);
}
