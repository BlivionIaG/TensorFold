//! The runtime symbols hipprof forwards unchanged: zig/src/hip/abi.zig's table less the five it interposes, plus the
//! lookups made beside it (the device properties; the stream value ops of the expert offload, expert_store.zig).
const std = @import("std");

/// Forwarded as they are, with up to 11 integer or pointer arguments.
pub const names = [_][:0]const u8{
    "hipInit",
    "hipGetDeviceCount",
    "hipDeviceGet",
    "hipDevicePrimaryCtxRetain",
    "hipDevicePrimaryCtxRelease",
    "hipCtxSetCurrent",
    "hipDeviceSynchronize",
    "hipMalloc",
    "hipFree",
    "hipHostMalloc",
    "hipHostFree",
    "hipMemcpyHtoDAsync",
    "hipMemcpyDtoHAsync",
    "hipMemcpyHtoD",
    "hipMemcpyDtoH",
    "hipMemset",
    "hipMemsetAsync",
    "hipStreamCreateWithFlags",
    "hipStreamDestroy",
    "hipStreamSynchronize",
    "hipModuleLoadData",
    "hipModuleUnload",
    "hipGetErrorName",
    "hipRuntimeGetVersion",
    "hipDeviceGetName",
    "hipDeviceGetAttribute",
    "hipHostGetDevicePointer",
    "hipMemcpyDtoD",
    "hipMemcpyDtoDAsync",
    "hipMemsetD32",
    "hipMemsetD32Async",
    "hipModuleGetGlobal",
    "hipFuncGetAttribute",
    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
    "hipModuleLaunchCooperativeKernel",
    "hipStreamIsCapturing",
    "hipGraphCreate",
    "hipGraphDestroy",
    "hipGraphAddKernelNode",
    "hipGraphKernelNodeSetParams",
    "hipGraphAddDependencies",
    "hipGraphGetNodes",
    "hipGraphInstantiateWithFlags",
    "hipGraphUpload",
    "hipGraphExecDestroy",
    "hipGraphExecUpdate",
    "hipGraphExecKernelNodeSetParams",
    "hipGetDevicePropertiesR0600",
    "hipMemGetInfo",
    "hipEventCreateWithFlags",
    "hipEventDestroy",
    "hipEventElapsedTime",
    "hipEventQuery",
    "hipEventRecord",
    "hipEventSynchronize",
    "hipStreamQuery",
    "hipStreamWaitEvent",
    "hipStreamWaitValue32",
    "hipStreamWriteValue32",
};

/// Entry points hipprof defines itself; each calls the runtime's own after its bookkeeping.
pub const interposed = [_][:0]const u8{
    "hipModuleGetFunction",
    "hipStreamBeginCapture",
    "hipStreamEndCapture",
    "hipGraphLaunch",
    "hipModuleLaunchKernel",
};

fn listed(list: []const [:0]const u8, name: []const u8) bool {
    for (list) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

test "every symbol zig/src/hip/abi.zig binds is forwarded or interposed" {
    const abi = @import("hip_abi");
    inline for (@typeInfo(abi.Api).@"struct".field_names) |name| {
        try std.testing.expect(listed(&names, name) != listed(&interposed, name));
    }
}

test "no symbol is listed twice and none is both forwarded and interposed" {
    const all = names ++ interposed;
    for (all, 0..) |a, i| {
        for (all[i + 1 ..]) |b| try std.testing.expect(!std.mem.eql(u8, a, b));
    }
}
