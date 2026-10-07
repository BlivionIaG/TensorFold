//! TensorFold's HIP runtime: libamdhip64 through dlopen and our kernels as embedded code objects, no Python.

pub const abi = @import("runtime/abi.zig");
pub const Driver = @import("runtime/driver.zig").Driver;
pub const Error = @import("runtime/driver.zig").Error;
pub const Context = @import("runtime/context.zig").Context;
pub const DeviceBuffer = @import("runtime/memory.zig").DeviceBuffer;
pub const HostBuffer = @import("runtime/memory.zig").HostBuffer;
pub const usage = @import("runtime/memory.zig").usage;
pub const Usage = @import("runtime/memory.zig").Usage;
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
pub const policy = @import("policy.zig");
pub const Policy = policy.Policy;
pub const caps = @import("caps.zig");
pub const admission = @import("admission.zig");
pub const Caps = caps.Caps;
pub const rocm = @import("rocm.zig");
pub const ops = @import("ops/ops.zig");
pub const plan_ops = @import("ops/plan.zig");
pub const quant = @import("core").quant;
pub const Upload = @import("upload.zig").Upload;
pub const affine = @import("launch/affine.zig");
pub const registry = @import("core").registry;
pub const tuning = @import("tuning/tuning.zig");
pub const Arena = @import("runtime/arena.zig").Arena;
pub const rccl = @import("comm/rccl.zig");
pub const link = @import("comm/link.zig");
pub const Group = @import("comm/group.zig").Group;
pub const Device = @import("device.zig").Device;

test {
    _ = launch;
    _ = abi;
    _ = policy;
    _ = caps;
    _ = rocm;
    _ = ops;
    _ = quant;
    _ = tuning;
    _ = rccl;
    _ = link;
    _ = Group;
    _ = Device;
}
