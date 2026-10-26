//! A verify cost table from tf-lfm2-verify-costs: shared forward ms by total rows, for the lane core's shared costs.
const std = @import("std");
const lanes = @import("lanes");
const st = @import("state.zig");

const Cost = lanes.config.Cost;

/// Lines `streams rows_each total_rows ms` ('#' comments): each total's ms at the fewest streams that reach it.
pub fn parse(gpa: std.mem.Allocator, text: []const u8) ![]Cost {
    var best: [st.batch_rows + 1]?struct { streams: u32, ms: f64 } = @splat(null);
    var lines = std.mem.tokenizeAny(u8, text, "\r\n");
    while (lines.next()) |line| {
        const body = std.mem.trim(u8, line, " \t");
        if (body.len == 0 or body[0] == '#') continue;
        var fields = std.mem.tokenizeAny(u8, body, " \t");
        var v: [4][]const u8 = undefined;
        for (&v) |*x| x.* = fields.next() orelse return error.BadVerifyCosts;
        if (fields.next() != null) return error.BadVerifyCosts;
        const streams = try std.fmt.parseInt(u32, v[0], 10);
        const each = try std.fmt.parseInt(u32, v[1], 10);
        const total = try std.fmt.parseInt(u32, v[2], 10);
        const ms = try std.fmt.parseFloat(f64, v[3]);
        if (streams == 0 or each == 0 or total != streams * each or total > st.batch_rows or !(ms > 0)) return error.BadVerifyCosts;
        if (best[total]) |b| if (b.streams <= streams) continue;
        best[total] = .{ .streams = streams, .ms = ms };
    }
    var out: std.ArrayList(Cost) = .empty;
    errdefer out.deinit(gpa);
    for (best, 0..) |b, total| if (b) |x| try out.append(gpa, .{ .width = @intCast(total), .ms = x.ms });
    if (out.items.len == 0) return error.BadVerifyCosts;
    return out.toOwnedSlice(gpa);
}

test "lfm2 dspark: a verify cost table gives each total its fewest streams' ms and refuses malformed lines" {
    const gpa = std.testing.allocator;
    const t = try parse(gpa, "# streams rows_each total_rows ms\n1 1 1 3.7\n2 1 2 4.9\n1 2 2 4.6\n\n4 8 32 40.5\n");
    defer gpa.free(t);
    try std.testing.expectEqual(@as(usize, 3), t.len);
    try std.testing.expectEqual(@as(f64, 4.6), t[1].ms); // one stream of two rows, not two of one
    try std.testing.expectEqual(@as(u32, 32), t[2].width);
    try std.testing.expectError(error.BadVerifyCosts, parse(gpa, "2 2 5 4.0\n"));
    try std.testing.expectError(error.BadVerifyCosts, parse(gpa, "8 8 64 9.0\n"));
    try std.testing.expectError(error.BadVerifyCosts, parse(gpa, "# nothing\n"));
}
