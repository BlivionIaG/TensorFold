//! The row store and load a layer-range stage uses to carry a hidden row across the split window (qwen27/copy.metal).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "qwen27 copy q27_row_store", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen27 copy q27_row_load", .threads = 256, .tg_memory = 0, .reserved = 256 },
};

test "the split stage's row store and load fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
