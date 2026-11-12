//! Offload of a model's layers to an external GPU on the same host: no GPU imports, everything runs on the CPU mock.
const std = @import("std");

pub const device = @import("device.zig");
pub const wire = @import("wire.zig");
pub const mock = @import("mock.zig");

test {
    std.testing.refAllDecls(@This());
}
