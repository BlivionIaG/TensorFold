//! Metal GLP launches reserve 256 threads and 32 static threadgroup bytes; see the GLP recipe.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "dsv41 glp fold tf_dsv41_glp_fold", .threads = 256, .tg_memory = 32, .reserved = 256 },
    .{ .name = "dsv41 glp writer tf_dsv41_glp_writer", .threads = 256, .tg_memory = 32, .reserved = 256 },
};

test "the DeepSeek V4.1 steering launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
