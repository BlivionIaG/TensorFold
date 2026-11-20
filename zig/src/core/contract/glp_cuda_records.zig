//! CUDA GLP launch records remain pending until kernel-contract-limits measures the actual image.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

const served_width = 5120;

fn record(comptime symbol: []const u8, dynamic: u32) registry.CudaRecord {
    return .{
        .launch = .{ .name = "dsv41_glp " ++ symbol, .symbol = symbol, .image = "dsv41_glp", .threads = 320, .dynamic = dynamic, .cluster = 1, .opt_in = false, .min_sm = 121 },
        .usage = registry.none,
        .later_pending = true,
    };
}

pub const records = [_]registry.CudaRecord{
    record("dsv41_glp_stream", 0),
    record("dsv41_glp_writer", 0),
    record("dsv41_glp_final", 2 * served_width),
};

test "the GLP launches are one 320-thread CTA a row and fit every SM's block and dynamic shared limits" {
    for (records) |r| {
        try std.testing.expectEqual(@as(u32, 320), r.launch.threads);
        try std.testing.expectEqualStrings("dsv41_glp", r.launch.image);
        for (limits.sms) |t| {
            try std.testing.expect(r.launch.threads <= t.threads);
            try std.testing.expect(r.launch.dynamic <= t.smem_optin);
        }
    }
}

test "unmeasured GLP records cannot pass a compiled resource check" {
    for (records) |r| {
        try std.testing.expect(std.meta.eql(registry.none, r.usage));
        try std.testing.expect(try registry.checkCuda(std.testing.allocator, &.{r}, true) > 0);
    }
}
