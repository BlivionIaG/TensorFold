//! HIP runtime types and entry points as ROCm documents them (HIP 6.0+ ABI, checked on 7.14), declared by hand.

pub const Result = c_int;
pub const Device = c_int;
pub const DevicePtr = u64;

pub const Module = ?*opaque {};
pub const Function = ?*opaque {};
pub const Stream = ?*opaque {};
pub const Event = ?*opaque {};
pub const Graph = ?*opaque {};
pub const GraphNode = ?*opaque {};
pub const GraphExec = ?*opaque {};

pub const success: Result = 0;
pub const error_out_of_memory: Result = 2;
pub const error_not_found: Result = 500;
pub const error_not_ready: Result = 600;

pub const stream_non_blocking: c_uint = 1;
pub const event_disable_timing: c_uint = 2;
pub const host_malloc_portable: c_uint = 1;
pub const host_malloc_mapped: c_uint = 2;

pub const CaptureMode = enum(c_int) { global = 0, thread_local = 1, relaxed = 2 };
pub const CaptureStatus = enum(c_int) { none = 0, active = 1, invalidated = 2, _ };

/// hipDeviceAttribute_t: HIP's own numbering, not CUDA's.
pub const DeviceAttribute = enum(c_int) {
    clock_rate = 5,
    cooperative_launch = 10,
    integrated = 16,
    l2_cache_size = 19,
    compute_capability_major = 23,
    max_threads_per_block = 56,
    compute_capability_minor = 61,
    multiprocessor_count = 63,
    max_shared_memory_per_block = 74,
    shared_mem_per_block_optin = 75,
    warp_size = 87,
    max_shared_memory_per_multiprocessor = 10002,
};

pub const FunctionAttribute = enum(c_int) {
    max_threads_per_block = 0,
    shared_size_bytes = 1,
    const_size_bytes = 2,
    local_size_bytes = 3,
    num_regs = 4,
    max_dynamic_shared_size_bytes = 8,
};

pub const Dim3 = extern struct { x: c_uint = 1, y: c_uint = 1, z: c_uint = 1 };

/// hipKernelNodeParams: its own field order (block first), 64 bytes.
pub const KernelNodeParams = extern struct {
    block: Dim3,
    extra: ?[*]?*anyopaque,
    func: Function,
    grid: Dim3,
    params: ?[*]?*anyopaque,
    shared_bytes: c_uint,
};

pub const ExecUpdateResult = enum(c_int) {
    success = 0,
    @"error" = 1,
    topology_changed = 2,
    node_type_changed = 3,
    function_changed = 4,
    parameters_changed = 5,
    not_supported = 6,
    unsupported_function_change = 7,
    _,
};

const R = Result;
const Ptr = ?*anyopaque;
const CPtr = ?*const anyopaque;
const Params = ?[*]?*anyopaque;

/// Each field is the exact exported symbol (the library's default version of it).
pub const Api = struct {
    hipInit: *const fn (c_uint) callconv(.c) R,
    hipDriverGetVersion: *const fn (*c_int) callconv(.c) R,
    hipRuntimeGetVersion: *const fn (*c_int) callconv(.c) R,
    hipGetDeviceCount: *const fn (*c_int) callconv(.c) R,
    hipDeviceGet: *const fn (*Device, c_int) callconv(.c) R,
    hipDeviceGetName: *const fn ([*]u8, c_int, Device) callconv(.c) R,
    hipDeviceGetAttribute: *const fn (*c_int, DeviceAttribute, c_int) callconv(.c) R,
    hipGetDevicePropertiesR0600: *const fn ([*]u8, c_int) callconv(.c) R,
    hipDeviceTotalMem: *const fn (*usize, Device) callconv(.c) R,
    hipSetDevice: *const fn (c_int) callconv(.c) R,
    hipGetDevice: *const fn (*c_int) callconv(.c) R,
    hipDeviceSynchronize: *const fn () callconv(.c) R,
    hipMemGetInfo: *const fn (*usize, *usize) callconv(.c) R,
    hipMalloc: *const fn (*DevicePtr, usize) callconv(.c) R,
    hipFree: *const fn (DevicePtr) callconv(.c) R,
    hipHostMalloc: *const fn (*Ptr, usize, c_uint) callconv(.c) R,
    hipHostGetDevicePointer: *const fn (*DevicePtr, Ptr, c_uint) callconv(.c) R,
    hipHostFree: *const fn (Ptr) callconv(.c) R,
    hipMemcpyHtoD: *const fn (DevicePtr, CPtr, usize) callconv(.c) R,
    hipMemcpyDtoH: *const fn (Ptr, DevicePtr, usize) callconv(.c) R,
    hipMemcpyDtoD: *const fn (DevicePtr, DevicePtr, usize) callconv(.c) R,
    hipMemcpyHtoDAsync: *const fn (DevicePtr, CPtr, usize, Stream) callconv(.c) R,
    hipMemcpyDtoHAsync: *const fn (Ptr, DevicePtr, usize, Stream) callconv(.c) R,
    hipMemcpyDtoDAsync: *const fn (DevicePtr, DevicePtr, usize, Stream) callconv(.c) R,
    hipMemsetD8: *const fn (DevicePtr, u8, usize) callconv(.c) R,
    hipMemsetD32: *const fn (DevicePtr, c_int, usize) callconv(.c) R,
    hipMemsetD8Async: *const fn (DevicePtr, u8, usize, Stream) callconv(.c) R,
    hipMemsetD32Async: *const fn (DevicePtr, c_int, usize, Stream) callconv(.c) R,
    hipStreamCreateWithFlags: *const fn (*Stream, c_uint) callconv(.c) R,
    hipStreamDestroy: *const fn (Stream) callconv(.c) R,
    hipStreamSynchronize: *const fn (Stream) callconv(.c) R,
    hipStreamWaitEvent: *const fn (Stream, Event, c_uint) callconv(.c) R,
    hipStreamQuery: *const fn (Stream) callconv(.c) R,
    hipEventCreateWithFlags: *const fn (*Event, c_uint) callconv(.c) R,
    hipEventDestroy: *const fn (Event) callconv(.c) R,
    hipEventRecord: *const fn (Event, Stream) callconv(.c) R,
    hipEventSynchronize: *const fn (Event) callconv(.c) R,
    hipEventQuery: *const fn (Event) callconv(.c) R,
    hipEventElapsedTime: *const fn (*f32, Event, Event) callconv(.c) R,
    hipModuleLoadData: *const fn (*Module, CPtr) callconv(.c) R,
    hipModuleUnload: *const fn (Module) callconv(.c) R,
    hipModuleGetFunction: *const fn (*Function, Module, [*:0]const u8) callconv(.c) R,
    hipModuleGetGlobal: *const fn (*DevicePtr, *usize, Module, [*:0]const u8) callconv(.c) R,
    hipFuncGetAttribute: *const fn (*c_int, FunctionAttribute, Function) callconv(.c) R,
    hipModuleOccupancyMaxActiveBlocksPerMultiprocessor: *const fn (*c_int, Function, c_int, usize) callconv(.c) R,
    hipModuleLaunchKernel: *const fn (Function, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Stream, Params, Params) callconv(.c) R,
    hipModuleLaunchCooperativeKernel: *const fn (Function, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Stream, Params) callconv(.c) R,
    hipStreamBeginCapture: *const fn (Stream, CaptureMode) callconv(.c) R,
    hipStreamEndCapture: *const fn (Stream, *Graph) callconv(.c) R,
    hipStreamIsCapturing: *const fn (Stream, *CaptureStatus) callconv(.c) R,
    hipGraphCreate: *const fn (*Graph, c_uint) callconv(.c) R,
    hipGraphDestroy: *const fn (Graph) callconv(.c) R,
    hipGraphAddKernelNode: *const fn (*GraphNode, Graph, ?[*]const GraphNode, usize, *const KernelNodeParams) callconv(.c) R,
    hipGraphKernelNodeGetParams: *const fn (GraphNode, *KernelNodeParams) callconv(.c) R,
    hipGraphKernelNodeSetParams: *const fn (GraphNode, *const KernelNodeParams) callconv(.c) R,
    hipGraphAddDependencies: *const fn (Graph, [*]const GraphNode, [*]const GraphNode, usize) callconv(.c) R,
    hipGraphGetNodes: *const fn (Graph, ?[*]GraphNode, *usize) callconv(.c) R,
    hipGraphInstantiateWithFlags: *const fn (*GraphExec, Graph, c_ulonglong) callconv(.c) R,
    hipGraphUpload: *const fn (GraphExec, Stream) callconv(.c) R,
    hipGraphLaunch: *const fn (GraphExec, Stream) callconv(.c) R,
    hipGraphExecDestroy: *const fn (GraphExec) callconv(.c) R,
    hipGraphExecUpdate: *const fn (GraphExec, Graph, *GraphNode, *ExecUpdateResult) callconv(.c) R,
    hipGraphExecKernelNodeSetParams: *const fn (GraphExec, GraphNode, *const KernelNodeParams) callconv(.c) R,
    hipGetErrorName: *const fn (R) callconv(.c) ?[*:0]const u8,
    hipGetErrorString: *const fn (R) callconv(.c) ?[*:0]const u8,
};

comptime {
    const std = @import("std");
    std.debug.assert(@sizeOf(Dim3) == 12);
    std.debug.assert(@sizeOf(KernelNodeParams) == 64);
    std.debug.assert(@offsetOf(KernelNodeParams, "extra") == 16 and @offsetOf(KernelNodeParams, "func") == 24);
    std.debug.assert(@offsetOf(KernelNodeParams, "grid") == 32 and @offsetOf(KernelNodeParams, "params") == 48);
    std.debug.assert(@offsetOf(KernelNodeParams, "shared_bytes") == 56);
}
