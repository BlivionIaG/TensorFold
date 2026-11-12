//! `std.json.Value`s printed the way Python prints them, so the Zig gates write the same `results.jsonl` lines and console text as the Python gates:
//! `dumps` is json.dumps(value) (", " and ": " separators, ensure_ascii) and `repr` is Python's str() of the parsed dict / list.
const std = @import("std");
const pf = @import("pyfmt.zig");

pub const Value = std.json.Value;

/// json.dumps(v) with Python's defaults.
pub fn dumps(w: *std.Io.Writer, v: Value) std.Io.Writer.Error!void {
    switch (v) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .integer => |i| try w.print("{d}", .{i}),
        .number_string => |s| try w.writeAll(s),
        .float => |f| {
            var buf: [pf.buf_len]u8 = undefined;
            // json.dumps writes float.__repr__, with NaN / Infinity for the non-finite values
            try w.writeAll(if (std.math.isNan(f)) "NaN" else if (std.math.isInf(f)) (if (f < 0) "-Infinity" else "Infinity") else pf.repr(&buf, f));
        },
        .string => |s| try dumpsString(w, s),
        .array => |arr| {
            try w.writeByte('[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try w.writeAll(", ");
                try dumps(w, item);
            }
            try w.writeByte(']');
        },
        .object => |obj| {
            try w.writeByte('{');
            for (obj.keys(), obj.values(), 0..) |k, item, i| {
                if (i > 0) try w.writeAll(", ");
                try dumpsString(w, k);
                try w.writeAll(": ");
                try dumps(w, item);
            }
            try w.writeByte('}');
        },
    }
}

/// A JSON string literal as json.dumps writes it: `"`, `\\` and the control characters escaped, every non-ASCII code point as \uXXXX (a surrogate pair above the BMP).
pub fn dumpsString(w: *std.Io.Writer, s: []const u8) std.Io.Writer.Error!void {
    try w.writeByte('"');
    var it = std.unicode.Utf8Iterator{ .bytes = s, .i = 0 };
    while (true) {
        const cp = it.nextCodepoint() orelse break;
        switch (cp) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            0x08 => try w.writeAll("\\b"),
            0x0c => try w.writeAll("\\f"),
            0x20...0x21, 0x23...0x5b, 0x5d...0x7e => try w.writeByte(@intCast(cp)),
            else => if (cp < 0x10000) {
                try w.print("\\u{x:0>4}", .{cp});
            } else {
                const c = cp - 0x10000;
                try w.print("\\u{x:0>4}\\u{x:0>4}", .{ 0xd800 + (c >> 10), 0xdc00 + (c & 0x3ff) });
            },
        }
    }
    try w.writeByte('"');
}

/// Python's repr of the value json.loads would give: `{'a': True, 'b': [1, 2.5, None]}`.
pub fn repr(w: *std.Io.Writer, v: Value) std.Io.Writer.Error!void {
    switch (v) {
        .null => try w.writeAll("None"),
        .bool => |b| try w.writeAll(if (b) "True" else "False"),
        .integer => |i| try w.print("{d}", .{i}),
        .number_string => |s| try w.writeAll(s),
        .float => |f| {
            var buf: [pf.buf_len]u8 = undefined;
            try w.writeAll(pf.repr(&buf, f));
        },
        .string => |s| try reprString(w, s),
        .array => |arr| {
            try w.writeByte('[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try w.writeAll(", ");
                try repr(w, item);
            }
            try w.writeByte(']');
        },
        .object => |obj| {
            try w.writeByte('{');
            for (obj.keys(), obj.values(), 0..) |k, item, i| {
                if (i > 0) try w.writeAll(", ");
                try reprString(w, k);
                try w.writeAll(": ");
                try repr(w, item);
            }
            try w.writeByte('}');
        },
    }
}

/// repr(str): single quotes unless the text has a `'` and no `"`; `\\`, the quote, \n \r \t and other control characters escaped; printable non-ASCII kept.
pub fn reprString(w: *std.Io.Writer, s: []const u8) std.Io.Writer.Error!void {
    const quote: u8 = if (std.mem.indexOfScalar(u8, s, '\'') != null and std.mem.indexOfScalar(u8, s, '"') == null) '"' else '\'';
    try w.writeByte(quote);
    for (s) |c| switch (c) {
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        0...8, 11, 12, 14...31, 127 => try w.print("\\x{x:0>2}", .{c}),
        else => if (c == quote) {
            try w.writeByte('\\');
            try w.writeByte(c);
        } else try w.writeByte(c),
    };
    try w.writeByte(quote);
}

test "dumps and repr of a server reply's tensorfold dict" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text =
        \\{"enable_thinking": false, "engine": "lanes", "tokens_per_second": 81.51208119758576, "prefill_widths": [], "n": 3, "x": null, "r": 80.0, "s": "it's \"q\"\n", "u": "é€😀"}
    ;
    const v = try std.json.parseFromSliceLeaky(Value, a, text, .{});
    var w: std.Io.Writer.Allocating = .init(a);
    try dumps(&w.writer, v);
    try std.testing.expectEqualStrings(
        \\{"enable_thinking": false, "engine": "lanes", "tokens_per_second": 81.51208119758576, "prefill_widths": [], "n": 3, "x": null, "r": 80.0, "s": "it's \"q\"\n", "u": "\u00e9\u20ac\ud83d\ude00"}
    , w.written());
    var r: std.Io.Writer.Allocating = .init(a);
    try repr(&r.writer, v);
    try std.testing.expectEqualStrings(
        \\{'enable_thinking': False, 'engine': 'lanes', 'tokens_per_second': 81.51208119758576, 'prefill_widths': [], 'n': 3, 'x': None, 'r': 80.0, 's': 'it\'s "q"\n', 'u': 'é€😀'}
    , r.written());
    var q: std.Io.Writer.Allocating = .init(a);
    try reprString(&q.writer, "it's");
    try reprString(&q.writer, "plain");
    try std.testing.expectEqualStrings("\"it's\"'plain'", q.written());
}
