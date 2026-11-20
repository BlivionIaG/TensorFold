//! DeepSeek V4.1's GLP steering launches (kernels/cuda/deepseek_v41/glp.cu, image "dsv41_glp"), for gate 5's limits.
//! One CTA of 320 threads a row: the fold and the writer keep one set of 10 warp partials (40 B static), the final
//! keeps 80 chain partials and its norm scale (324 B static) and takes 2 bytes a column of dynamic memory for the
//! collapsed row (10240 B at the served width 5120). No register receipt is recorded: the usage is derived by
//! `tf-cuda-test kernel-contract-limits` on the built image, and an unmeasured record blocks qualification (it never
//! passes a compiled resource check). The planted-fault controls (dsv41_glp_controls) carry no record.
//!
//! The shape of the declarations (threads and group memory of the steering launch listed for the limits gate) follows
//! the earlier steering records at git ref c5947a8d2e (zig/src/core/contract/steer_records.zig); the control-vector
//! steering idea itself is credited in kernels/cuda/deepseek_v41/glp.cuh.
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
