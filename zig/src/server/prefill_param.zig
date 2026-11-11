//! `prefill` on a completion request: where a server with a remote prefill source prefills this prompt.
const std = @import("std");
const json = @import("json");
const Directive = @import("engine_api").LaneHost.Remote.Directive;

/// The request's directive (`auto` without one); anything but "auto", "force" or "never" is refused with `why`.
pub fn parse(f: json.Value, why: *[]const u8) error{BadPrefill}!Directive {
    const v = f.field("prefill") orelse return .auto;
    const word = v.str() orelse return bad(why);
    return Directive.parse(word) orelse bad(why);
}

fn bad(why: *[]const u8) error{BadPrefill} {
    why.* = "prefill: expected \"auto\", \"force\" or \"never\"";
    return error.BadPrefill;
}

test "prefill: a request names auto, force or never, and any other value is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    const body = try json.newObject(a);
    try std.testing.expectEqual(Directive.auto, try parse(.{ .object = body }, &why));
    for ([_]struct { []const u8, Directive }{ .{ "auto", .auto }, .{ "force", .force }, .{ "never", .never } }) |c| {
        try body.put(a, "prefill", .{ .string = c[0] });
        try std.testing.expectEqual(c[1], try parse(.{ .object = body }, &why));
    }
    for ([_]json.Value{ .{ .string = "Force" }, .{ .string = "" }, .{ .bool = true }, .{ .int = "1" } }) |bad_value| {
        try body.put(a, "prefill", bad_value);
        try std.testing.expectError(error.BadPrefill, parse(.{ .object = body }, &why));
        try std.testing.expectEqualStrings("prefill: expected \"auto\", \"force\" or \"never\"", why);
    }
}
