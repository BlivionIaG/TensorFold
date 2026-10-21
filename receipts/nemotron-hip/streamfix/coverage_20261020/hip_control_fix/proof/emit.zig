const std = @import("std");
const r = @import("records_emitter.zig");
pub fn main() !void {
    const bytes = try std.json.Stringify.valueAlloc(std.heap.page_allocator, r.records ++ r.controls, .{});
    std.debug.print("{s}\n", .{bytes});
}
