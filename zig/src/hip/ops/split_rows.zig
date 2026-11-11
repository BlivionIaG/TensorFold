//! The split offload's boundary rows between a host window and the activation rows (split/rows.hip).
//!
//! A hidden row crosses the window as `n` little-endian fp32 values whatever the activation kind is: import rounds
//! to the activation kind to nearest even, export widens exactly. A level-B reply is a stamped row (wire.zig): groups of
//! three fp32 values and a u32 stamp, 16 bytes each. Window addresses are the GPU addresses of host-mapped memory
//! (offload.zig). Prefill is not here: a prompt crosses the window one position at a time, rows = 1.

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

/// dst (n values of dst.kind) = the fp32 row at the window address `src_f32`, rounded to nearest even (src 4-byte aligned).
pub fn importRow(o: Ops, src_f32: u64, dst: Tensor, n: usize) Error!void {
    try launches.importRow(o.l, src_f32, dst.ptr, @backingInt(dst.kind), n, o.stream);
}

/// The fp32 row at the window address `dst_f32` = src (n values of src.kind) widened, exactly (dst 4-byte aligned).
pub fn exportRow(o: Ops, src: Tensor, dst_f32: u64, n: usize) Error!void {
    try launches.exportRow(o.l, src.ptr, dst_f32, @backingInt(src.kind), n, o.stream);
}

/// dst (n values of dst.kind) = the stamped row at `src` (16-byte aligned) once every group's stamp is `seq`.
/// A stale group is read again after a short sleep, at most `retries` more times, so a group costs at most `retries + 1`
/// reads. `status` is the device address of a u32 (window word): written once, 0 when every group was current and 1 when
/// one stayed stale (dst then holds that group as last read). The word describes the kernel's whole run: read it after
/// the stream's later release write, not before.
pub fn consumeStamped(o: Ops, src: u64, seq: u32, dst: Tensor, n: usize, status: u64, retries: u32) Error!void {
    try launches.consumeStamped(o.l, src, seq, dst.ptr, @backingInt(dst.kind), n, status, retries, o.stream);
}

// The semantics the kernels implement, on the host: what the tests below and the contract entry check them against.

/// One value of the row in the activation kind's bits (fp16, bf16) or fp32's, rounded to nearest even; NaN stays NaN
/// (bf16: the quiet NaN 0x7fc0, as c10 makes it).
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

/// What `consumeStamped` leaves: the row's values from `region` (groups of three fp32 values and a stamp), the status
/// word, and the reads in all. Group g shows a stale stamp for its first `stale_reads[g]` reads and then `region[g]`; a
/// region stamp other than `seq` is never current. A group takes `retries + 1` reads at most.
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
