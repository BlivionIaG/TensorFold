//! The Python loader's `_packed`, `_float`, `_conv` and `_halves`: one tensor, or one projection's tensors, read and checked.

const std = @import("std");
const table = @import("table.zig");
const convert = @import("convert.zig");
const host = @import("host.zig");

const Tensor = table.Tensor;
const Table = table.Table;
pub const Error = table.Error || convert.Error || error{UnexpectedTensor};

/// `parts` joined in `buf`.
pub fn join(buf: []u8, parts: []const []const u8) []const u8 {
    var n: usize = 0;
    for (parts) |p| {
        @memcpy(buf[n..][0..p.len], p);
        n += p.len;
    }
    return buf[0..n];
}

fn refuse(key: []const u8, what: []const u8) error{UnexpectedTensor} {
    std.log.err("{s}: {s}", .{ key, what });
    return error.UnexpectedTensor;
}

/// `_float`: the tensor widened to fp32.
pub fn float(a: std.mem.Allocator, t: *const Table, key: []const u8) Error!Tensor {
    const src = try t.get(key);
    if (!convert.isFloat(src.dtype)) return refuse(key, "not a floating tensor");
    return convert.float32(a, src);
}

/// `_conv`: a depthwise conv weight as fp32 (channels, kernel); a trailing 1 is squeezed.
pub fn conv(a: std.mem.Allocator, t: *const Table, key: []const u8) Error!Tensor {
    var w = try float(a, t, key);
    if (w.rank == 3 and w.shape[2] == 1) {
        w.rank = 2;
    } else if (w.rank == 3 and w.shape[1] == 1) {
        return refuse(key, "still in the unsanitized (channels, 1, kernel) layout");
    }
    if (w.rank != 2) return refuse(key, "conv weight must be (channels, kernel)");
    return w;
}

/// `_packed`: an affine projection as stored (group tables widened to fp32 only when not fp32, bf16 or fp16 pairs), or a float one.
pub fn projection(a: std.mem.Allocator, t: *const Table, key: []const u8) Error!host.Projection {
    var buf: [256]u8 = undefined;
    var other: [256]u8 = undefined;
    var words = try t.get(join(&buf, &.{ key, ".weight" }));
    if (convert.isFloat(words.dtype) and words.rank == 2 and !t.has(join(&other, &.{ key, ".scales" }))) {
        return .{ .dense = .{ .weight = try convert.float32(a, words) } };
    }
    const w = try t.width(key);
    if ((words.dtype != .u32 and words.dtype != .i32) or words.rank != 2) return refuse(key, "weight is not packed int32 words");
    words.dtype = .i32;
    const scale = try t.get(join(&buf, &.{ key, ".scales" }));
    const bias = try t.get(join(&other, &.{ key, ".biases" }));
    if (scale.rank != 2 or bias.rank != 2 or !std.mem.eql(usize, scale.shape[0..2], bias.shape[0..2])) return refuse(key, "scale and bias must share shape (N, K / group)");
    const k = scale.shape[1] * w.group;
    if (words.shape[1] != k * w.bits / 32 or words.shape[0] != scale.shape[0]) return refuse(key, "packed shape does not match K, bits and group");
    const pair = try convert.tables(a, scale, bias);
    return .{ .affine = .{ .words = words, .scales = pair[0], .biases = pair[1], .bits = w.bits, .group = w.group } };
}

/// Columns [from, to) of a rank-2 tensor, copied.
fn columns(a: std.mem.Allocator, src: Tensor, from: usize, to: usize) Error!Tensor {
    const size = src.dtype.size();
    const rows = src.shape[0];
    const cols = src.shape[1];
    const out = try a.alloc(u8, rows * (to - from) * size);
    for (0..rows) |r| @memcpy(out[r * (to - from) * size ..][0 .. (to - from) * size], src.bytes[(r * cols + from) * size ..][0 .. (to - from) * size]);
    var t = src;
    t.shape[1] = to - from;
    t.bytes = out;
    return t;
}

/// `_halves`: a fused [embedding | hidden] projection split in two along K, words and group tables alike.
pub fn halves(a: std.mem.Allocator, fused: host.Projection) Error![2]host.Projection {
    switch (fused) {
        .dense => |d| {
            const half = d.weight.shape[1] / 2;
            return .{
                .{ .dense = .{ .weight = try columns(a, d.weight, 0, half) } },
                .{ .dense = .{ .weight = try columns(a, d.weight, half, d.weight.shape[1]) } },
            };
        },
        .affine => |p| {
            const groups = p.scales.shape[1];
            const half = groups * p.group / 2;
            if (groups % 2 != 0 or half * p.bits % 32 != 0) return refuse("fc", "does not split on a word boundary");
            const word = p.words.shape[1] / 2;
            const tab = groups / 2;
            var out: [2]host.Projection = undefined;
            for (&out, 0..) |*o, i| {
                const w = if (i == 0) [2]usize{ 0, word } else [2]usize{ word, p.words.shape[1] };
                const g = if (i == 0) [2]usize{ 0, tab } else [2]usize{ tab, groups };
                o.* = .{ .affine = .{
                    .words = try columns(a, p.words, w[0], w[1]),
                    .scales = try columns(a, p.scales, g[0], g[1]),
                    .biases = try columns(a, p.biases, g[0], g[1]),
                    .bits = p.bits,
                    .group = p.group,
                } };
            }
            return out;
        },
    }
}

test "halves split words and group tables on the same boundary" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var words: [2 * 4]u32 = undefined;
    for (&words, 0..) |*w, i| w.* = @intCast(i);
    var scales: [2 * 2]u16 = .{ 10, 11, 12, 13 };
    const p: host.Projection = .{ .affine = .{
        .words = .{ .dtype = .i32, .rank = 2, .shape = .{ 2, 4, 1, 1 }, .bytes = std.mem.sliceAsBytes(&words) },
        .scales = .{ .dtype = .bf16, .rank = 2, .shape = .{ 2, 2, 1, 1 }, .bytes = std.mem.sliceAsBytes(&scales) },
        .biases = .{ .dtype = .bf16, .rank = 2, .shape = .{ 2, 2, 1, 1 }, .bytes = std.mem.sliceAsBytes(&scales) },
        .bits = 4,
        .group = 32,
    } };
    const pair = try halves(arena.allocator(), p);
    try std.testing.expectEqual(@as(usize, 2), pair[0].affine.words.shape[1]);
    try std.testing.expectEqual(@as(usize, 1), pair[1].affine.scales.shape[1]);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 6, 7 }, @alignCast(std.mem.bytesAsSlice(u32, pair[1].affine.words.bytes)));
    try std.testing.expectEqualSlices(u16, &.{ 11, 13 }, @alignCast(std.mem.bytesAsSlice(u16, pair[1].affine.scales.bytes)));
}
