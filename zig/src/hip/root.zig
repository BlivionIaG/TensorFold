//! TensorFold's HIP runtime: libamdhip64 through dlopen and our kernels as embedded code objects, no Python.

pub const abi = @import("runtime/abi.zig");
pub const Driver = @import("runtime/driver.zig").Driver;
pub const Error = @import("runtime/driver.zig").Error;
pub const Context = @import("runtime/context.zig").Context;
pub const DeviceBuffer = @import("runtime/memory.zig").DeviceBuffer;
pub const HostBuffer = @import("runtime/memory.zig").HostBuffer;
pub const Stream = @import("runtime/stream.zig").Stream;
pub const Event = @import("runtime/stream.zig").Event;
pub const Module = @import("runtime/module.zig").Module;
pub const Function = @import("runtime/module.zig").Function;
pub const launch = @import("runtime/launch.zig");
pub const Args = launch.Args;
pub const Config = launch.Config;
pub const Dim3 = launch.Dim3;
pub const graph = @import("runtime/graph.zig");
pub const kernels = @import("kernels.zig");
pub const rocm = @import("rocm.zig");
pub const ops = @import("ops.zig");
pub const affine = @import("launch/affine.zig");
pub const Arena = @import("runtime/arena.zig").Arena;
pub const rccl = @import("comm/rccl.zig");
pub const link = @import("comm/link.zig");

test {
    _ = launch;
    _ = abi;
    _ = rocm;
    _ = ops;
    _ = rccl;
    _ = link;
}
