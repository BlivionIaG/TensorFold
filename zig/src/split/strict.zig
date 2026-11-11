//! Strict JSON reading shared by the probe receipt and placement.json: every struct field is required (an optional field
//! may be absent or null), unknown keys are refused, and types and ranges are checked; nothing is defaulted silently.
//! Errors are named; `Diag` carries the dotted path of the offending value for the message.
const std = @import("std");

const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const Error = error{
    InvalidJson,
    /// A required field is absent.
    MissingField,
    /// A value has the wrong JSON type.
    WrongType,
    /// An object has a key the schema does not define.
    UnknownField,
    /// The type is right but the value is not allowed: out of range, not finite, not one of the enum's words.
    InvalidValue,
} || Allocator.Error;

/// The dotted path of the value that failed, e.g. `link.hop_to_host_ns.p99`.
pub const Diag = struct {
    buf: [160]u8 = undefined,
    len: usize = 0,

    pub fn path(d: *const Diag) []const u8 {
        return d.buf[0..d.len];
    }

    pub fn set(d: *Diag, p: []const u8) void {
        d.len = @min(p.len, d.buf.len);
        @memcpy(d.buf[0..d.len], p[0..d.len]);
    }
};

/// Record `path` in `diag` and return `err`.
pub fn fail(diag: *Diag, path: []const u8, err: anytype) @TypeOf(err) {
    diag.set(path);
    return err;
}

/// `parent.key`, or `key` at the top; the result lives in `buf`.
pub fn join(buf: []u8, parent: []const u8, key: []const u8) []const u8 {
    if (parent.len == 0) return std.fmt.bufPrint(buf, "{s}", .{key}) catch key;
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ parent, key }) catch parent;
}

pub fn parseDocument(arena: Allocator, bytes: []const u8) Error!Value {
    return std.json.parseFromSliceLeaky(Value, arena, bytes, .{}) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidJson,
    };
}

/// The object at `v`, or WrongType.
pub fn object(v: Value, path: []const u8, diag: *Diag) Error!std.json.ObjectMap {
    if (v != .object) return fail(diag, path, error.WrongType);
    return v.object;
}

/// The member `name` of `obj`, or MissingField.
pub fn member(obj: std.json.ObjectMap, name: []const u8, path: []const u8, diag: *Diag) Error!Value {
    return obj.get(name) orelse {
        var buf: [160]u8 = undefined;
        return fail(diag, join(&buf, path, name), error.MissingField);
    };
}

/// Read a JSON value as `T` (bool, integer, finite float, enum by tag name, string, slice, optional or struct), allocating
/// strings and slices from `a`.
pub fn parseValue(comptime T: type, a: Allocator, v: Value, path: []const u8, diag: *Diag) Error!T {
    switch (@typeInfo(T)) {
        .bool => {
            if (v != .bool) return fail(diag, path, error.WrongType);
            return v.bool;
        },
        .int => {
            if (v != .integer) return fail(diag, path, error.WrongType);
            return std.math.cast(T, v.integer) orelse fail(diag, path, error.InvalidValue);
        },
        .float => {
            const f: f64 = switch (v) {
                .float => |f| f,
                .integer => |i| @floatFromInt(i),
                else => return fail(diag, path, error.WrongType),
            };
            if (!std.math.isFinite(f)) return fail(diag, path, error.InvalidValue);
            return @floatCast(f);
        },
        .@"enum" => |e| {
            if (v != .string) return fail(diag, path, error.WrongType);
            inline for (e.field_names, e.field_values) |name, value| if (std.mem.eql(u8, name, v.string)) return @fromBackingInt(@intCast(value));
            return fail(diag, path, error.InvalidValue);
        },
        .optional => |o| {
            if (v == .null) return null;
            return try parseValue(o.child, a, v, path, diag);
        },
        .pointer => |p| {
            if (p.size != .slice) @compileError("strict: unsupported type " ++ @typeName(T));
            if (p.child == u8) {
                if (v != .string) return fail(diag, path, error.WrongType);
                return try a.dupe(u8, v.string);
            }
            if (v != .array) return fail(diag, path, error.WrongType);
            const out = try a.alloc(p.child, v.array.items.len);
            for (out, v.array.items, 0..) |*o, item, i| {
                var buf: [160]u8 = undefined;
                const sub = std.fmt.bufPrint(&buf, "{s}[{d}]", .{ path, i }) catch path;
                o.* = try parseValue(p.child, a, item, sub, diag);
            }
            return out;
        },
        .@"struct" => |s| {
            const obj = try object(v, path, diag);
            var out: T = undefined;
            inline for (s.field_names, s.field_types) |name, FT| {
                var buf: [160]u8 = undefined;
                const sub = join(&buf, path, name);
                if (obj.get(name)) |fv| {
                    @field(out, name) = try parseValue(FT, a, fv, sub, diag);
                } else if (@typeInfo(FT) == .optional) {
                    @field(out, name) = null;
                } else return fail(diag, sub, error.MissingField);
            }
            var it = obj.iterator();
            next: while (it.next()) |kv| {
                inline for (s.field_names) |name| if (std.mem.eql(u8, name, kv.key_ptr.*)) continue :next;
                var buf: [160]u8 = undefined;
                return fail(diag, join(&buf, path, kv.key_ptr.*), error.UnknownField);
            }
            return out;
        },
        else => @compileError("strict: unsupported type " ++ @typeName(T)),
    }
}

test "parseValue reads structs, enums, slices and optionals" {
    const E = enum { a, b };
    const T = struct { n: u32, f: f64, e: E, s: []const u8, xs: []const u16, o: ?bool, inner: struct { k: i8 } };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var diag: Diag = .{};
    const doc = try parseDocument(arena.allocator(), "{\"n\":3,\"f\":2,\"e\":\"b\",\"s\":\"hi\",\"xs\":[1,2],\"inner\":{\"k\":-4}}");
    const t = try parseValue(T, arena.allocator(), doc, "", &diag);
    try std.testing.expectEqual(@as(u32, 3), t.n);
    try std.testing.expectEqual(@as(f64, 2), t.f);
    try std.testing.expectEqual(E.b, t.e);
    try std.testing.expectEqualStrings("hi", t.s);
    try std.testing.expectEqualSlices(u16, &.{ 1, 2 }, t.xs);
    try std.testing.expectEqual(@as(?bool, null), t.o);
    try std.testing.expectEqual(@as(i8, -4), t.inner.k);
}

test "parseValue names the offending path" {
    const T = struct { inner: struct { n: u8, e: enum { x } } };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var diag: Diag = .{};
    const cases = [_]struct { json: []const u8, err: Error, path: []const u8 }{
        .{ .json = "{}", .err = error.MissingField, .path = "inner" },
        .{ .json = "{\"inner\":{\"e\":\"x\"}}", .err = error.MissingField, .path = "inner.n" },
        .{ .json = "{\"inner\":{\"n\":\"1\",\"e\":\"x\"}}", .err = error.WrongType, .path = "inner.n" },
        .{ .json = "{\"inner\":{\"n\":300,\"e\":\"x\"}}", .err = error.InvalidValue, .path = "inner.n" },
        .{ .json = "{\"inner\":{\"n\":1,\"e\":\"y\"}}", .err = error.InvalidValue, .path = "inner.e" },
        .{ .json = "{\"inner\":{\"n\":1,\"e\":\"x\",\"z\":0}}", .err = error.UnknownField, .path = "inner.z" },
        .{ .json = "{\"inner\":{\"n\":1.5,\"e\":\"x\"}}", .err = error.WrongType, .path = "inner.n" },
    };
    for (cases) |c| {
        const doc = try parseDocument(a, c.json);
        try std.testing.expectError(c.err, parseValue(T, a, doc, "", &diag));
        try std.testing.expectEqualStrings(c.path, diag.path());
    }
    try std.testing.expectError(error.InvalidJson, parseDocument(a, "{\"a\":"));
}
