//! A reader of NumPy `.npy` files (format 1.0, 2.0, 3.0): C-order little-endian arrays of bool, signed / unsigned integers and float16/32/64.
//! The data is borrowed from the bytes given to `parse`; nothing is copied or converted until `at` / `atInt` / `toF64` ask for an element.
const std = @import("std");

pub const max_dims = 4;

pub const Dtype = enum {
    bool,
    i1,
    u1,
    i2,
    u2,
    i4,
    u4,
    i8,
    u8,
    f2,
    f4,
    f8,

    pub fn size(d: Dtype) usize {
        return switch (d) {
            .bool, .i1, .u1 => 1,
            .i2, .u2, .f2 => 2,
            .i4, .u4, .f4 => 4,
            .i8, .u8, .f8 => 8,
        };
    }

    pub fn isFloat(d: Dtype) bool {
        return d == .f2 or d == .f4 or d == .f8;
    }
};

pub const Error = error{ NotNpy, BadHeader, UnsupportedDtype, FortranOrder, TooManyDims, Truncated };

pub const Array = struct {
    dtype: Dtype,
    ndim: usize,
    shape: [max_dims]usize,
    data: []const u8,

    pub fn len(self: Array) usize {
        var n: usize = 1;
        for (self.shape[0..self.ndim]) |s| n *= s;
        return n;
    }

    /// Element `i` (flat, C order) as an integer; floats are truncated toward zero.
    pub fn atInt(self: Array, i: usize) i64 {
        const w = self.dtype.size();
        const raw = self.data[i * w ..][0..w];
        return switch (self.dtype) {
            .bool, .u1 => raw[0],
            .i1 => @as(i8, @bitCast(raw[0])),
            .i2 => std.mem.readInt(i16, raw[0..2], .little),
            .u2 => std.mem.readInt(u16, raw[0..2], .little),
            .i4 => std.mem.readInt(i32, raw[0..4], .little),
            .u4 => std.mem.readInt(u32, raw[0..4], .little),
            .i8 => std.mem.readInt(i64, raw[0..8], .little),
            .u8 => @bitCast(std.mem.readInt(u64, raw[0..8], .little)),
            .f2, .f4, .f8 => @intFromFloat(self.at(i)),
        };
    }

    /// Element `i` (flat, C order) as an f64 (what `astype(np.float64)` gives).
    pub fn at(self: Array, i: usize) f64 {
        const w = self.dtype.size();
        const raw = self.data[i * w ..][0..w];
        return switch (self.dtype) {
            .f2 => @as(f16, @bitCast(std.mem.readInt(u16, raw[0..2], .little))),
            .f4 => @as(f32, @bitCast(std.mem.readInt(u32, raw[0..4], .little))),
            .f8 => @bitCast(std.mem.readInt(u64, raw[0..8], .little)),
            .u8 => @floatFromInt(@as(u64, @bitCast(self.atInt(i)))),
            else => @floatFromInt(self.atInt(i)),
        };
    }
};

const magic = "\x93NUMPY";

/// The array in `bytes` (a whole `.npy` file).
pub fn parse(bytes: []const u8) Error!Array {
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], magic)) return error.NotNpy;
    const major = bytes[6];
    var start: usize = undefined;
    var hlen: usize = undefined;
    switch (major) {
        1 => {
            hlen = std.mem.readInt(u16, bytes[8..10], .little);
            start = 10;
        },
        2, 3 => {
            if (bytes.len < 12) return error.Truncated;
            hlen = std.mem.readInt(u32, bytes[8..12], .little);
            start = 12;
        },
        else => return error.NotNpy,
    }
    if (bytes.len < start + hlen) return error.Truncated;
    const header = bytes[start .. start + hlen];

    const descr = try quoted(header, "'descr'");
    if (descr.len < 2) return error.UnsupportedDtype;
    const dtype = try dtypeOf(descr);

    const fo = try valueAfter(header, "'fortran_order'");
    if (!std.mem.startsWith(u8, fo, "False")) return error.FortranOrder;

    const sh = try valueAfter(header, "'shape'");
    if (sh.len == 0 or sh[0] != '(') return error.BadHeader;
    const close = std.mem.indexOfScalar(u8, sh, ')') orelse return error.BadHeader;
    var arr: Array = .{ .dtype = dtype, .ndim = 0, .shape = @splat(0), .data = undefined };
    var it = std.mem.tokenizeAny(u8, sh[1..close], ", ");
    while (it.next()) |tok| {
        if (arr.ndim == max_dims) return error.TooManyDims;
        arr.shape[arr.ndim] = std.fmt.parseInt(usize, tok, 10) catch return error.BadHeader;
        arr.ndim += 1;
    }
    const need = arr.len() * dtype.size();
    const data = bytes[start + hlen ..];
    if (data.len < need) return error.Truncated;
    arr.data = data[0..need];
    return arr;
}

fn dtypeOf(descr: []const u8) Error!Dtype {
    // byte order: '<' little, '|' not applicable; '>' big is refused
    if (descr[0] == '>') return error.UnsupportedDtype;
    const code = descr[1..];
    const table = [_]struct { []const u8, Dtype }{
        .{ "b1", .bool }, .{ "i1", .i1 }, .{ "u1", .u1 }, .{ "i2", .i2 }, .{ "u2", .u2 }, .{ "i4", .i4 },
        .{ "u4", .u4 },   .{ "i8", .i8 }, .{ "u8", .u8 }, .{ "f2", .f2 }, .{ "f4", .f4 }, .{ "f8", .f8 },
    };
    for (table) |e| if (std.mem.eql(u8, e[0], code)) return e[1];
    return error.UnsupportedDtype;
}

/// The text after `key` and its colon, spaces skipped.
fn valueAfter(header: []const u8, key: []const u8) Error![]const u8 {
    const at = std.mem.indexOf(u8, header, key) orelse return error.BadHeader;
    var rest = header[at + key.len ..];
    rest = std.mem.trimStart(u8, rest, " ");
    if (rest.len == 0 or rest[0] != ':') return error.BadHeader;
    return std.mem.trimStart(u8, rest[1..], " ");
}

/// The quoted string value of `key`.
fn quoted(header: []const u8, key: []const u8) Error![]const u8 {
    const v = try valueAfter(header, key);
    if (v.len < 2 or (v[0] != '\'' and v[0] != '"')) return error.BadHeader;
    const end = std.mem.indexOfScalar(u8, v[1..], v[0]) orelse return error.BadHeader;
    return v[1 .. 1 + end];
}

/// An in-memory `.npy` (format 1.0) of `dtype` with the given little-endian `data`, for tests and fixtures.
pub fn encode(a: std.mem.Allocator, descr: []const u8, shape: []const usize, data: []const u8) ![]u8 {
    var dict: std.ArrayList(u8) = .empty;
    defer dict.deinit(a);
    try dict.print(a, "{{'descr': '{s}', 'fortran_order': False, 'shape': (", .{descr});
    for (shape) |s| try dict.print(a, "{d},", .{s});
    if (shape.len != 1) _ = dict.pop(); // a one-element tuple keeps its comma
    try dict.appendSlice(a, "), }");
    const total = 10 + dict.items.len + 1;
    const pad = (64 - total % 64) % 64;
    const hlen = dict.items.len + pad + 1;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, magic ++ "\x01\x00");
    try out.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u16, @intCast(hlen))));
    try out.appendSlice(a, dict.items);
    try out.appendNTimes(a, ' ', pad);
    try out.append(a, '\n');
    try out.appendSlice(a, data);
    return out.toOwnedSlice(a);
}

test "a float32 matrix round-trips through encode and parse" {
    const a = std.testing.allocator;
    const vals = [_]f32{ 1.5, -2, 0.25, 8, 3, 7 };
    const bytes = try encode(a, "<f4", &.{ 2, 3 }, std.mem.sliceAsBytes(&vals));
    defer a.free(bytes);
    try std.testing.expectEqual(@as(usize, 0), (bytes.len - 24) % 64); // the header ends on a 64-byte boundary
    const arr = try parse(bytes);
    try std.testing.expectEqual(Dtype.f4, arr.dtype);
    try std.testing.expectEqual(@as(usize, 2), arr.ndim);
    try std.testing.expectEqual(@as(usize, 3), arr.shape[1]);
    try std.testing.expectEqual(@as(f64, -2), arr.at(1));
    try std.testing.expectEqual(@as(f64, 7), arr.at(5));
}

test "integers, booleans and a one-dimensional shape" {
    const a = std.testing.allocator;
    const ids = [_]i64{ 4, -1, 300 };
    const bytes = try encode(a, "<i8", &.{3}, std.mem.sliceAsBytes(&ids));
    defer a.free(bytes);
    const arr = try parse(bytes);
    try std.testing.expectEqual(@as(usize, 1), arr.ndim);
    try std.testing.expectEqual(@as(i64, -1), arr.atInt(1));
    try std.testing.expectEqual(@as(i64, 300), arr.atInt(2));
    const flags_bytes = try encode(a, "|b1", &.{2}, &.{ 1, 0 });
    defer a.free(flags_bytes);
    const flags = try parse(flags_bytes);
    try std.testing.expectEqual(Dtype.bool, flags.dtype);
    try std.testing.expectEqual(@as(i64, 1), flags.atInt(0));
    try std.testing.expectEqual(@as(i64, 0), flags.atInt(1));
}

test "half precision, refusals" {
    const a = std.testing.allocator;
    const h = [_]u16{ 0x3c00, 0xc000 }; // 1, -2
    const bytes = try encode(a, "<f2", &.{2}, std.mem.sliceAsBytes(&h));
    defer a.free(bytes);
    const arr = try parse(bytes);
    try std.testing.expectEqual(@as(f64, 1), arr.at(0));
    try std.testing.expectEqual(@as(f64, -2), arr.at(1));
    try std.testing.expectError(error.NotNpy, parse("hello world, not numpy"));
    const be = try encode(a, ">f4", &.{1}, &.{ 0, 0, 0, 0 });
    defer a.free(be);
    try std.testing.expectError(error.UnsupportedDtype, parse(be));
    const short = try encode(a, "<f4", &.{4}, &.{ 0, 0, 0, 0 });
    defer a.free(short);
    try std.testing.expectError(error.Truncated, parse(short));
}
