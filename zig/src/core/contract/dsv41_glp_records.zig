//! DeepSeek V4.1's steering launches on Metal: each reserves its thread count, so none needs an M1/M2 receipt.
//! Both kernels hold one `threadgroup float part[8]` (a simdgroup's partial a slot): 32 bytes of static memory.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "dsv41 glp fold tf_dsv41_glp_fold", .threads = 256, .tg_memory = 32, .reserved = 256 },
    .{ .name = "dsv41 glp writer tf_dsv41_glp_writer", .threads = 256, .tg_memory = 32, .reserved = 256 },
};

test "the DeepSeek V4.1 steering launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
