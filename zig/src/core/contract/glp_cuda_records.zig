//! CUDA GLP nvcc 13.0 sm_121 resource records; later compilers remain pending.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

const served_width = 5120;

fn record(comptime symbol: []const u8, dynamic: u32, regs: u32, static: u32) registry.CudaRecord {
    var usage = registry.none;
    for (limits.sms, 0..) |target, i| {
        if (target.sm == 121) usage[i] = .{ .regs = regs, .static = static, .bound = 320 };
    }
    return .{
        .launch = .{ .name = "dsv41_glp " ++ symbol, .symbol = symbol, .image = "dsv41_glp", .threads = 320, .dynamic = dynamic, .cluster = 1, .opt_in = false, .min_sm = 121 },
        .usage = usage,
        .later_pending = true,
    };
}

pub const records = [_]registry.CudaRecord{
    record("dsv41_glp_stream", 0, 113, 48),
    record("dsv41_glp_writer", 0, 61, 48),
    record("dsv41_glp_final", 2 * served_width, 48, 336),
};

test "the GLP registered launches fit their served targets" {
    for (records) |r| {
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
