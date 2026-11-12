//! The split rows' stamped layout against the offload layer's own packer (wire.zig), in the builds that carry `split`.

const std = @import("std");
const wire = @import("split").wire;
const rows = @import("split_rows.zig");

test "split rows: the reference's layout is wire.zig's stamped row" {
    var f: [7]f32 = undefined;
    for (&f, 0..) |*v, i| v.* = @floatFromInt(i + 1);
    var region: [wire.stampedLen(7 * 4)]u8 align(16) = undefined;
    wire.packStamped(&region, std.mem.sliceAsBytes(&f), 77);
    const groups_in = std.mem.bytesAsSlice([4]u32, &region);
    try std.testing.expectEqual(rows.groups(7), groups_in.len);
    try std.testing.expectEqual(rows.stampedBytes(7), region.len);
    var out: [7]u32 = undefined;
    const r = rows.consumeReference(&out, .f32, groups_in, 77, &.{ 0, 0, 0 }, 0);
    try std.testing.expectEqual(@as(u32, 0), r.status);
    for (out, f) |got, want| try std.testing.expectEqual(@as(u32, @bitCast(want)), got);
    // a stale stamp after packing is refused whatever the data
    wire.packStamped(&region, std.mem.sliceAsBytes(&f), 76);
    const stale = rows.consumeReference(&out, .f32, groups_in, 77, &.{ 0, 0, 0 }, 2);
    try std.testing.expectEqual(@as(u32, 1), stale.status);
}
