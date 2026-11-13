//! Offload of a model's layers to an external GPU on the same host: no GPU imports, everything runs on the CPU mock.
const std = @import("std");

pub const device = @import("device.zig");
pub const wire = @import("wire.zig");
pub const mock = @import("mock.zig");
pub const pipeline = @import("pipeline.zig");
pub const toy = @import("toy.zig");
pub const geometry = @import("geometry.zig");
pub const receipt = @import("receipt.zig");
pub const placement = @import("placement.zig");
pub const planner = @import("planner.zig");
pub const probe = @import("probe.zig");
pub const cpu_host = @import("cpu_host.zig");
pub const cli = @import("cli.zig");

test {
    std.testing.refAllDecls(@This());
}
