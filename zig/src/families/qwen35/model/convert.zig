//! Host-side element conversions the Python loader does with torch: `.float()`, bf16 rounding and the affine unpack.

const std = @import("std");
const table = @import("table.zig");

const Tensor = table.Tensor;
const DType = table.DType;

pub const Error = error{ UnexpectedTensor, UnsupportedQuantization } || std.mem.Allocator.Error;

/// One element as fp32 (bf16 and fp16 widen exactly).
pub fn load(dtype: DType, bytes: []const u8, i: usize) f32 {
    return switch (dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)))),
        else => unreachable,
    };
}

pub fn isFloat(dtype: DType) bool {
    return switch (dtype) {
        .f16, .bf16, .f32, .f64 => true,
        else => false,
    };
}

/// The group-table dtypes the kernels read as stored.
pub fn tableDtype(dtype: DType) bool {
    return dtype == .f32 or dtype == .bf16 or dtype == .f16;
}

/// torch's fp32 -> bf16: round to nearest even, NaN stays a NaN.
pub fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

/// Writes `src` as little-endian fp32 into `dst` (4 bytes an element).
pub fn widenInto(dst: []u8, src: Tensor) Error!void {
    const n = src.numel();
    if (dst.len != n * 4) return error.UnexpectedTensor;
    switch (src.dtype) {
        .f32, .bf16, .f16 => for (0..n) |i| std.mem.writeInt(u32, dst[4 * i ..][0..4], @bitCast(load(src.dtype, src.bytes, i)), .little),
        .f64 => for (0..n) |i| {
            const v: f64 = @bitCast(std.mem.readInt(u64, src.bytes[8 * i ..][0..8], .little));
            std.mem.writeInt(u32, dst[4 * i ..][0..4], @bitCast(@as(f32, @floatCast(v))), .little);
        },
        else => return error.UnexpectedTensor,
    }
}

/// `.float().contiguous()` of a floating tensor into memory of `a` (an fp32 tensor is shared, not copied).
pub fn float32(a: std.mem.Allocator, src: Tensor) Error!Tensor {
    if (src.dtype == .f32) return src;
    const out = try a.alloc(u8, src.numel() * 4);
    try widenInto(out, src);
    return .{ .dtype = .f32, .rank = src.rank, .shape = src.shape, .bytes = out };
}

/// The fp32 scale and bias tables when the pair is not one of the dtypes the kernels read as stored.
pub fn tables(a: std.mem.Allocator, scale: Tensor, bias: Tensor) Error![2]Tensor {
    if (scale.dtype == bias.dtype and tableDtype(scale.dtype)) return .{ scale, bias };
    return .{ try float32(a, scale), try float32(a, bias) };
}

/// MLX affine words [R, K * bits / 32] with scale and bias [R, K / group] -> fp32 [R, K] (s * q + b), bits 2, 4 or 8.
pub fn dequant(a: std.mem.Allocator, words: Tensor, scale: Tensor, bias: Tensor, group: usize) Error![]f32 {
    if (words.rank != 2 or scale.rank != 2 or bias.rank != 2 or !std.mem.eql(usize, scale.shape[0..2], bias.shape[0..2])) return error.UnexpectedTensor;
    if (words.dtype != .u32 and words.dtype != .i32) return error.UnexpectedTensor;
    if (!isFloat(scale.dtype) or scale.dtype == .f64 or !isFloat(bias.dtype) or bias.dtype == .f64) return error.UnexpectedTensor;
    const rows = words.shape[0];
    const k = scale.shape[1] * group;
    if (k == 0 or 32 * words.shape[1] / k == 0) return error.UnexpectedTensor;
    const bits = 32 * words.shape[1] / k;
    if (bits != 2 and bits != 4 and bits != 8) return error.UnsupportedQuantization;
    const per = 32 / bits;
    if (words.shape[1] * per != k or scale.shape[0] != rows) return error.UnexpectedTensor;
    const out = try a.alloc(f32, rows * k);
    const mask: u32 = (@as(u32, 1) << @intCast(bits)) - 1;
    for (0..rows) |r| for (0..k) |c| {
        const word = std.mem.readInt(u32, words.bytes[4 * (r * words.shape[1] + c / per) ..][0..4], .little);
        const code: f32 = @floatFromInt((word >> @intCast(bits * (c % per))) & mask);
        const g = r * scale.shape[1] + c / group;
        const s = load(scale.dtype, scale.bytes, g);
        const b = load(bias.dtype, bias.bytes, g);
        // torch multiplies, then adds: two roundings
        const prod = code * s;
        out[r * k + c] = prod + b;
    };
    return out;
}

test "bf16 rounds to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3F80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3F80), bf16(@bitCast(@as(u32, 0x3F807FFF))));
    try std.testing.expectEqual(@as(u16, 0x3F82), bf16(@bitCast(@as(u32, 0x3F818000))));
    try std.testing.expectEqual(@as(u16, 0x3F80), bf16(@bitCast(@as(u32, 0x3F808000))));
    try std.testing.expect(std.math.isNan(@as(f32, @bitCast(@as(u32, bf16(std.math.nan(f32))) << 16))));
}

test "dequant unpacks 4-bit words low nibble first" {
    var words = [_]u32{0x76543210};
    var scale = [_]u16{0x4000}; // bf16 2.0
    var bias = [_]u16{0xBF80}; // bf16 -1.0
    const w: Tensor = .{ .dtype = .u32, .rank = 2, .shape = .{ 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&words) };
    const s: Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&scale) };
    const b: Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&bias) };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const out = try dequant(arena.allocator(), w, s, b, 8);
    try std.testing.expectEqualSlices(f32, &.{ -1, 1, 3, 5, 7, 9, 11, 13 }, out);
}
