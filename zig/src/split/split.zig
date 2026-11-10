//! TB4 offload: one model across the Mac's Metal GPU and an external GPU on the same host.
//! This module has no GPU imports: the device interface, the slot protocol, a CPU stand-in device, the layer pipeline
//! and the planner. Metal's side (windows, the command-buffer relay) is in `zig/src/metal`; `metal_toy.zig` joins them.
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

test {
    std.testing.refAllDecls(@This());
}
