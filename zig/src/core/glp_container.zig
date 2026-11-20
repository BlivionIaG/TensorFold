//! The GGUF v3 subset a GLP file uses; implements the GLP format defined by the `weightless` project (@msuiche).
//! Ported from the TensorFold-native reader of upstream #1835 (head c5947a8d2e, BlivionIaG); GLP spec pin
//! b181a07d3980fc48abe39d95ddb044459dd0b868 (https://github.com/msuiche/weightless, docs/spec/GLP.md).
//! The container convention (`direction.<N>` tensors, fp32, 1-D) originated in llama.cpp's control vectors.

const std = @import("std");

pub const Error = error{BadGguf} || std.mem.Allocator.Error;

/// The alignment a container pads its tensor data to when its metadata names none.
pub const default_alignment = 32;

/// The largest alignment a container may name: a power of two far past any tensor payload a GLP file carries.
pub const max_alignment = 1 << 20;

/// The ggml type a tensor of floats carries.
pub const f32_type = 0;

/// The gguf value types the container may hold.
pub const Type = enum(u32) {
    u8v = 0,
    i8v = 1,
    u16v = 2,
    i16v = 3,
    u32v = 4,
    i32v = 5,
    f32v = 6,
    boolv = 7,
    string = 8,
    array = 9,
    u64v = 10,
    i64v = 11,
    f64v = 12,
};

/// An array's element type and its items, so an empty array still names the type it holds. A file's array is read
/// past, not kept: `skipped` counts the items its bytes held and `items` stays empty; a writer gives `items`.
pub const Array = struct {
    elem: Type,
    items: []const Value = &.{},
    skipped: u64 = 0,
};

/// A metadata value; strings and arrays borrow from the file's bytes or the arena.
pub const Value = union(Type) {
    u8v: u8,
    i8v: i8,
    u16v: u16,
    i16v: i16,
    u32v: u32,
    i32v: i32,
    f32v: f32,
    boolv: bool,
    string: []const u8,
    array: Array,
    u64v: u64,
    i64v: i64,
    f64v: f64,
};

/// One metadata key and the value it holds.
pub const Entry = struct { key: []const u8, value: Value };

/// One tensor's header: its name, its shape, its ggml type and its offset in the data section.
pub const Tensor = struct {
    name: []const u8,
    dims: []const u64,
    kind: u32,
    offset: u64,
};

/// A tensor the writer emits: its name, its shape, its ggml type and its raw bytes.
pub const OutTensor = struct {
    name: []const u8,
    dims: []const u64,
    kind: u32,
    data: []const u8,
};

/// Why a container was refused, in one sentence.
pub const Problem = struct {
    buf: [200]u8 = undefined,
    len: usize = 0,

    pub fn set(self: *Problem, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Problem) []const u8 {
        return self.buf[0..self.len];
    }
};

/// A parsed container: its metadata, its tensor headers and the file's bytes.
pub const File = struct {
    entries: []Entry,
    tensors: []Tensor,
    data_start: usize,
    alignment: usize,
    bytes: []const u8,

    /// The value a key holds, or null when the key is absent or holds another type.
    pub fn find(self: File, key: []const u8) ?Value {
        for (self.entries) |e| if (std.mem.eql(u8, e.key, key)) return e.value;
        return null;
    }

    /// The string a key holds, or null when the key is absent or holds another type.
    pub fn str(self: File, key: []const u8) ?[]const u8 {
        const v = self.find(key) orelse return null;
        return switch (v) {
            .string => |s| s,
            else => null,
        };
    }

    /// The integer a key holds, or null; a negative value reads as nothing rather than as its wrap.
    pub fn num(self: File, key: []const u8) ?u64 {
        const v = self.find(key) orelse return null;
        return switch (v) {
            .u8v => |x| x,
            .u16v => |x| x,
            .u32v => |x| x,
            .u64v => |x| x,
            .i8v => |x| if (x < 0) null else @intCast(x),
            .i16v => |x| if (x < 0) null else @intCast(x),
            .i32v => |x| if (x < 0) null else @intCast(x),
            .i64v => |x| if (x < 0) null else @intCast(x),
            else => null,
        };
    }

    /// The tensor header a name names, or null.
    pub fn tensor(self: File, name: []const u8) ?Tensor {
        for (self.tensors) |t| if (std.mem.eql(u8, t.name, name)) return t;
        return null;
    }

    /// The tensor's bytes, by its offset in the data section; null when its type's size or its place is unknown.
    pub fn bytesOf(self: File, t: Tensor) ?[]const u8 {
        const count = elemCount(t.dims) orelse return null;
        const width = elemSize(t.kind) orelse return null;
        const size = std.math.mul(usize, count, width) catch return null;
        const offset = std.math.cast(usize, t.offset) orelse return null;
        const start = std.math.add(usize, self.data_start, offset) catch return null;
        if (start < self.data_start or size > self.bytes.len or start > self.bytes.len - size) return null;
        return self.bytes[start..][0..size];
    }
};

/// The bytes one element of a ggml type takes; the quantized types are not sized here.
fn elemSize(kind: u32) ?usize {
    return switch (kind) {
        0 => 4, // F32
        1 => 2, // F16
        24 => 1, // I8
        25 => 2, // I16
        26 => 4, // I32
        27 => 8, // I64
        28 => 8, // F64
        30 => 2, // BF16
        else => null,
    };
}

/// The elements a shape holds, or null when the shape overflows.
fn elemCount(dims: []const u64) ?usize {
    var n: u64 = 1;
    for (dims) |d| {
        if (d == 0 or n > (1 << 40) / d) return null;
        n *= d;
    }
    return std.math.cast(usize, n);
}

/// A cursor over the container's bytes that refuses to read past its end.
const Cursor = struct {
    b: []const u8,
    o: usize = 0,

    fn need(self: *Cursor, n: usize) Error!void {
        if (n > self.b.len or self.o + n > self.b.len) return error.BadGguf;
    }

    fn u8v(self: *Cursor) Error!u8 {
        try self.need(1);
        defer self.o += 1;
        return self.b[self.o];
    }

    fn u16v(self: *Cursor) Error!u16 {
        try self.need(2);
        defer self.o += 2;
        return std.mem.readInt(u16, self.b[self.o..][0..2], .little);
    }

    fn u32v(self: *Cursor) Error!u32 {
        try self.need(4);
        defer self.o += 4;
        return std.mem.readInt(u32, self.b[self.o..][0..4], .little);
    }

    fn u64v(self: *Cursor) Error!u64 {
        try self.need(8);
        defer self.o += 8;
        return std.mem.readInt(u64, self.b[self.o..][0..8], .little);
    }

    fn f32v(self: *Cursor) Error!f32 {
        try self.need(4);
        defer self.o += 4;
        return @bitCast(std.mem.readInt(u32, self.b[self.o..][0..4], .little));
    }

    fn string(self: *Cursor) Error![]const u8 {
        const n = try self.u64v();
        if (n > self.b.len) return error.BadGguf;
        try self.need(@intCast(n));
        defer self.o += @intCast(n);
        return self.b[self.o..][0..@intCast(n)];
    }

    /// Advance past `n` array items of one type without keeping them; each item takes at least one byte of the file.
    fn skipItems(self: *Cursor, elem: Type, n: u64) Error!void {
        const size: u64 = switch (elem) {
            .u8v, .i8v, .boolv => 1,
            .u16v, .i16v => 2,
            .u32v, .i32v, .f32v => 4,
            .u64v, .i64v, .f64v => 8,
            .string => {
                // each string is a length word and its bytes, so a count past the bytes left is a lie
                if (n > (self.b.len - self.o) / 8) return error.BadGguf;
                for (0..@as(usize, @intCast(n))) |_| _ = try self.string();
                return;
            },
            .array => return error.BadGguf,
        };
        const total = std.math.mul(u64, n, size) catch return error.BadGguf;
        const bytes = std.math.cast(usize, total) orelse return error.BadGguf;
        try self.need(bytes);
        self.o += bytes;
    }
};

fn refuse(problem: *Problem, comptime fmt: []const u8, args: anytype) Error {
    problem.set(fmt, args);
    return error.BadGguf;
}

/// Parse the header, the metadata and the tensor headers; the arena keeps every allocation.
pub fn parse(a: std.mem.Allocator, bytes: []const u8, problem: *Problem) Error!File {
    if (bytes.len < 8) return refuse(problem, "the file is too short to be a container", .{});
    var c: Cursor = .{ .b = bytes };
    if (!std.mem.eql(u8, c.b[0..4], "GGUF")) return refuse(problem, "not a GGUF container", .{});
    c.o = 4;
    const version = try c.u32v();
    if (version != 3) return refuse(problem, "GGUF version {d}; this container reader implements version 3", .{version});
    const tensor_count = try c.u64v();
    const kv_count = try c.u64v();
    if (tensor_count > 1 << 20 or kv_count > 1 << 16) return refuse(problem, "the container's counts are out of range", .{});

    var entries: std.ArrayList(Entry) = .empty;
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    for (0..kv_count) |_| {
        const key = try c.string();
        const kind = std.enums.fromInt(Type, try c.u32v()) orelse
            return refuse(problem, "the container holds an unknown metadata type", .{});
        const value = try readValue(&c, kind);
        // A key named twice has no single meaning: which value applies would depend on the reader.
        if ((try seen.getOrPut(a, key)).found_existing) return refuse(problem, "the container holds the key \"{f}\" twice", .{std.zig.fmtString(key)});
        try entries.append(a, .{ .key = key, .value = value });
    }

    var tensors: std.ArrayList(Tensor) = .empty;
    for (0..tensor_count) |_| {
        const name = try c.string();
        const n_dims = try c.u32v();
        if (n_dims > 8) return refuse(problem, "a tensor has {d} dimensions", .{n_dims});
        const dims = try a.alloc(u64, n_dims);
        for (dims) |*d| d.* = try c.u64v();
        const kind = try c.u32v();
        const offset = try c.u64v();
        try tensors.append(a, .{ .name = name, .dims = dims, .kind = kind, .offset = offset });
    }

    const alignment = (try alignmentOf(entries.items, problem)) orelse default_alignment;
    if (alignment == 0 or alignment > max_alignment or !std.math.isPowerOfTwo(alignment)) {
        return refuse(problem, "the container's alignment is {d}", .{alignment});
    }

    return .{
        .entries = entries.items,
        .tensors = tensors.items,
        .data_start = alignUp(c.o, alignment),
        .alignment = alignment,
        .bytes = bytes,
    };
}

fn readValue(c: *Cursor, kind: Type) Error!Value {
    return switch (kind) {
        .u8v => .{ .u8v = try c.u8v() },
        .i8v => .{ .i8v = @bitCast(try c.u8v()) },
        .u16v => .{ .u16v = try c.u16v() },
        .i16v => .{ .i16v = @bitCast(try c.u16v()) },
        .u32v => .{ .u32v = try c.u32v() },
        .i32v => .{ .i32v = @bitCast(try c.u32v()) },
        .f32v => .{ .f32v = try c.f32v() },
        .boolv => .{ .boolv = (try c.u8v()) != 0 },
        .string => .{ .string = try c.string() },
        .u64v => .{ .u64v = try c.u64v() },
        .i64v => .{ .i64v = @bitCast(try c.u64v()) },
        .f64v => .{ .f64v = @bitCast(try c.u64v()) },
        .array => blk: {
            const elem = std.enums.fromInt(Type, try c.u32v()) orelse return error.BadGguf;
            // A nested array would recurse as deep as the file allows; no GLP key holds one.
            if (elem == .array) return error.BadGguf;
            const n = try c.u64v();
            // No GLP key reads an array: its items are checked against the file's bytes and walked past, never kept,
            // so a small file cannot make the reader hold more memory than the file is long.
            try c.skipItems(elem, n);
            break :blk .{ .array = .{ .elem = elem, .skipped = n } };
        },
    };
}

/// Write a container the way llama.cpp's convention lays it out; the returned bytes are owned by the caller.
pub fn write(a: std.mem.Allocator, entries: []const Entry, tensors: []const OutTensor) Error![]u8 {
    const alignment = (try alignmentOf(entries, null)) orelse default_alignment;
    if (alignment == 0 or alignment > max_alignment or !std.math.isPowerOfTwo(alignment)) return error.BadGguf;

    // The tensor offsets are relative to the data section, so their place follows from the bytes alone.
    const offsets = try a.alloc(u64, tensors.len);
    defer a.free(offsets);
    var data_len: usize = 0;
    for (tensors, offsets) |t, *off| {
        data_len = alignUp(data_len, alignment);
        off.* = data_len;
        data_len = std.math.add(usize, data_len, t.data.len) catch return error.BadGguf;
    }

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "GGUF");
    try putU32(a, &out, 3);
    try putU64(a, &out, tensors.len);
    try putU64(a, &out, entries.len);
    for (entries) |e| try writeEntry(a, &out, e);
    for (tensors, offsets) |t, off| {
        try putString(a, &out, t.name);
        try putU32(a, &out, @intCast(t.dims.len));
        for (t.dims) |d| try putU64(a, &out, d);
        try putU32(a, &out, t.kind);
        try putU64(a, &out, off);
    }
    while (out.items.len % alignment != 0) try out.append(a, 0);
    for (tensors) |t| {
        while (out.items.len % alignment != 0) try out.append(a, 0);
        try out.appendSlice(a, t.data);
    }
    return out.toOwnedSlice(a);
}

/// The alignment a container's metadata names, or null when it names none; a key of another type is refused.
fn alignmentOf(entries: []const Entry, problem: ?*Problem) Error!?usize {
    for (entries) |e| {
        if (!std.mem.eql(u8, e.key, "general.alignment")) continue;
        return switch (e.value) {
            .u32v => |x| x,
            .u64v => |x| std.math.cast(usize, x) orelse return error.BadGguf,
            else => if (problem) |p| refuse(p, "general.alignment is not an unsigned integer", .{}) else error.BadGguf,
        };
    }
    return null;
}

fn writeEntry(a: std.mem.Allocator, out: *std.ArrayList(u8), e: Entry) Error!void {
    try putString(a, out, e.key);
    try putU32(a, out, @backingInt(std.meta.activeTag(e.value)));
    try writeValue(a, out, e.value);
}

/// Write a value without its type tag; an array's items carry the array's element type instead.
fn writeValue(a: std.mem.Allocator, out: *std.ArrayList(u8), v: Value) Error!void {
    switch (v) {
        .u8v => |x| try out.append(a, x),
        .i8v => |x| try out.append(a, @bitCast(x)),
        .u16v => |x| try putU16(a, out, x),
        .i16v => |x| try putU16(a, out, @bitCast(x)),
        .u32v => |x| try putU32(a, out, x),
        .i32v => |x| try putU32(a, out, @bitCast(x)),
        .f32v => |x| try putU32(a, out, @bitCast(x)),
        .boolv => |x| try out.append(a, if (x) 1 else 0),
        .string => |s| try putString(a, out, s),
        .array => |arr| {
            if (arr.skipped != 0) return error.BadGguf;
            for (arr.items) |it| {
                if (std.meta.activeTag(it) != arr.elem) return error.BadGguf;
            }
            try putU32(a, out, @backingInt(arr.elem));
            try putU64(a, out, arr.items.len);
            for (arr.items) |it| try writeValue(a, out, it);
        },
        .u64v => |x| try putU64(a, out, x),
        .i64v => |x| try putU64(a, out, @bitCast(x)),
        .f64v => |x| try putU64(a, out, @bitCast(x)),
    }
}

fn alignUp(x: usize, a: usize) usize {
    return (x + a - 1) / a * a;
}

fn putU16(a: std.mem.Allocator, out: *std.ArrayList(u8), x: u16) Error!void {
    var buf: [2]u8 = undefined;
    std.mem.writeInt(u16, &buf, x, .little);
    try out.appendSlice(a, &buf);
}

fn putU32(a: std.mem.Allocator, out: *std.ArrayList(u8), x: u32) Error!void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, x, .little);
    try out.appendSlice(a, &buf);
}

fn putU64(a: std.mem.Allocator, out: *std.ArrayList(u8), x: u64) Error!void {
    var buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &buf, x, .little);
    try out.appendSlice(a, &buf);
}

fn putString(a: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) Error!void {
    try putU64(a, out, s.len);
    try out.appendSlice(a, s);
}
