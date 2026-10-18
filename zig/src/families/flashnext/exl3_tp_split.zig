//! Host slicing of one stored-order EXL3 matrix for a tensor-parallel rank.
//! Stored tiles are [K/16, N/16, words] (exl3_strips.zig reorders them for the kernel afterwards). A K range keeps whole 128-row
//! Hadamard blocks of K (suh is sliced with it, svh stays whole); one or more N ranges keep whole 128-column blocks (svh is
//! gathered with them, suh stays whole). Ranges are element units. The result owns its memory (`free`).
//!
//! Why this is exact for N: an output column block is independent of the others (its own K sum, its own 128-point Hadamard, its own
//! svh), so a rank's columns equal the same columns of the whole product. A K range is a partial sum: the output Hadamard is linear,
//! so the ranks' outputs add up to the whole, up to the fp16 rounding of each partial.

const std = @import("std");
const shape = @import("exl3_shape.zig");

pub const Range = struct { lo: usize, hi: usize };

pub const Stored = struct {
    trellis: []const i16,
    suh: []const f16,
    svh: []const f16,
    tiles_k: usize,
    tiles_n: usize,
    halves: u8,

    pub fn k(m: Stored) usize {
        return m.tiles_k * 16;
    }
    pub fn n(m: Stored) usize {
        return m.tiles_n * 16;
    }
};

pub const Error = error{ Misaligned, OutOfRange, Length } || std.mem.Allocator.Error;

pub fn free(gpa: std.mem.Allocator, m: Stored) void {
    gpa.free(m.trellis);
    gpa.free(m.suh);
    gpa.free(m.svh);
}

/// Rows `kr` of K and the columns `nr` (in order) of N.
pub fn slice(gpa: std.mem.Allocator, m: Stored, kr: Range, nr: []const Range) Error!Stored {
    const h = shape.hadamard;
    if (m.suh.len != m.k() or m.svh.len != m.n() or m.halves == 0) return error.Length;
    const words: usize = @as(usize, m.halves) * 8;
    if (m.trellis.len != m.tiles_k * m.tiles_n * words) return error.Length;
    if (kr.lo % h != 0 or kr.hi % h != 0 or kr.hi <= kr.lo) return error.Misaligned;
    if (kr.hi > m.k()) return error.OutOfRange;
    var n_total: usize = 0;
    for (nr) |r| {
        if (r.lo % h != 0 or r.hi % h != 0 or r.hi <= r.lo) return error.Misaligned;
        if (r.hi > m.n()) return error.OutOfRange;
        n_total += r.hi - r.lo;
    }
    if (n_total == 0) return error.Length;
    const tk = (kr.hi - kr.lo) / 16;
    const tn = n_total / 16;
    const trellis = try gpa.alloc(i16, tk * tn * words);
    errdefer gpa.free(trellis);
    const suh = try gpa.alloc(f16, kr.hi - kr.lo);
    errdefer gpa.free(suh);
    const svh = try gpa.alloc(f16, n_total);
    errdefer gpa.free(svh);
    @memcpy(suh, m.suh[kr.lo..kr.hi]);
    var dst_col: usize = 0;
    for (nr) |r| {
        const cols = (r.hi - r.lo) / 16;
        @memcpy(svh[dst_col * 16 ..][0 .. r.hi - r.lo], m.svh[r.lo..r.hi]);
        for (0..tk) |row| {
            const src = ((kr.lo / 16 + row) * m.tiles_n + r.lo / 16) * words;
            const dst = (row * tn + dst_col) * words;
            @memcpy(trellis[dst..][0 .. cols * words], m.trellis[src..][0 .. cols * words]);
        }
        dst_col += cols;
    }
    return .{ .trellis = trellis, .suh = suh, .svh = svh, .tiles_k = tk, .tiles_n = tn, .halves = m.halves };
}

/// Rows of `row_bytes` bytes each, the ranges in order, into one new buffer (dense per-channel tensors such as conv1d).
pub fn gatherRows(gpa: std.mem.Allocator, bytes: []const u8, row_bytes: usize, rows: []const Range) Error![]u8 {
    var total: usize = 0;
    for (rows) |r| {
        if (r.hi <= r.lo or r.hi * row_bytes > bytes.len) return error.OutOfRange;
        total += r.hi - r.lo;
    }
    const out = try gpa.alloc(u8, total * row_bytes);
    var at: usize = 0;
    for (rows) |r| {
        const n = (r.hi - r.lo) * row_bytes;
        @memcpy(out[at..][0..n], bytes[r.lo * row_bytes ..][0..n]);
        at += n;
    }
    return out;
}

fn fixture(gpa: std.mem.Allocator, tiles_k: usize, tiles_n: usize, halves: u8) !Stored {
    const words = @as(usize, halves) * 8;
    const t = try gpa.alloc(i16, tiles_k * tiles_n * words);
    for (t, 0..) |*w, i| w.* = @intCast(i % 30011);
    const su = try gpa.alloc(f16, tiles_k * 16);
    for (su, 0..) |*w, i| w.* = @floatFromInt(i);
    const sv = try gpa.alloc(f16, tiles_n * 16);
    for (sv, 0..) |*w, i| w.* = @floatFromInt(1000 + i);
    return .{ .trellis = t, .suh = su, .svh = sv, .tiles_k = tiles_k, .tiles_n = tiles_n, .halves = halves };
}

test "an N slice takes whole columns of every K row and the matching svh" {
    const gpa = std.testing.allocator;
    const m = try fixture(gpa, 16, 32, 2); // K = 256, N = 512
    defer free(gpa, m);
    const part = try slice(gpa, m, .{ .lo = 0, .hi = 256 }, &.{.{ .lo = 128, .hi = 384 }});
    defer free(gpa, part);
    try std.testing.expectEqual(@as(usize, 16), part.tiles_k);
    try std.testing.expectEqual(@as(usize, 16), part.tiles_n);
    const words = 16;
    for (0..16) |row| for (0..16) |col| {
        const got = part.trellis[(row * 16 + col) * words ..][0..words];
        const want = m.trellis[(row * 32 + 8 + col) * words ..][0..words];
        try std.testing.expectEqualSlices(i16, want, got);
    };
    try std.testing.expectEqual(@as(f16, 1128), part.svh[0]);
    try std.testing.expectEqualSlices(f16, m.suh, part.suh);
}

test "a K slice keeps whole rows of tiles and the matching suh" {
    const gpa = std.testing.allocator;
    const m = try fixture(gpa, 16, 16, 1); // K = 256, N = 256
    defer free(gpa, m);
    const part = try slice(gpa, m, .{ .lo = 128, .hi = 256 }, &.{.{ .lo = 0, .hi = 256 }});
    defer free(gpa, part);
    try std.testing.expectEqual(@as(usize, 8), part.tiles_k);
    try std.testing.expectEqualSlices(i16, m.trellis[8 * 16 * 8 ..], part.trellis);
    try std.testing.expectEqual(@as(f16, 128), part.suh[0]);
    try std.testing.expectEqualSlices(f16, m.svh, part.svh);
}

test "several N ranges are gathered in order (the q|k|v cut of in_proj_qkv)" {
    const gpa = std.testing.allocator;
    const m = try fixture(gpa, 8, 64, 1); // N = 1024
    defer free(gpa, m);
    const part = try slice(gpa, m, .{ .lo = 0, .hi = 128 }, &.{ .{ .lo = 0, .hi = 128 }, .{ .lo = 512, .hi = 768 } });
    defer free(gpa, part);
    try std.testing.expectEqual(@as(usize, 24), part.tiles_n);
    try std.testing.expectEqual(@as(f16, 1512), part.svh[128]);
    try std.testing.expectEqualSlices(i16, m.trellis[(0 * 64 + 32) * 8 ..][0 .. 16 * 8], part.trellis[(0 * 24 + 8) * 8 ..][0 .. 16 * 8]);
}

test "misaligned and out of range cuts are refused before any allocation" {
    const gpa = std.testing.allocator;
    const m = try fixture(gpa, 16, 16, 1);
    defer free(gpa, m);
    try std.testing.expectError(error.Misaligned, slice(gpa, m, .{ .lo = 0, .hi = 256 }, &.{.{ .lo = 0, .hi = 160 }}));
    try std.testing.expectError(error.Misaligned, slice(gpa, m, .{ .lo = 16, .hi = 256 }, &.{.{ .lo = 0, .hi = 128 }}));
    try std.testing.expectError(error.OutOfRange, slice(gpa, m, .{ .lo = 0, .hi = 384 }, &.{.{ .lo = 0, .hi = 128 }}));
    try std.testing.expectError(error.OutOfRange, slice(gpa, m, .{ .lo = 0, .hi = 256 }, &.{.{ .lo = 128, .hi = 512 }}));
}

test "gatherRows concatenates row ranges" {
    const gpa = std.testing.allocator;
    var src: [10 * 4]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = @intCast(i);
    const out = try gatherRows(gpa, &src, 4, &.{ .{ .lo = 1, .hi = 3 }, .{ .lo = 7, .hi = 8 } });
    defer gpa.free(out);
    try std.testing.expectEqualSlices(u8, &.{ 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31 }, out);
}
