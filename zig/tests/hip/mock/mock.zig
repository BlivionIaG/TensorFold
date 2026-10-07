//! A stand-in libamdhip64 for the host tests: one device in host memory, every other entry point a no-op success.

const std = @import("std");
const abi = @import("abi");
const options = @import("mock_options");

const Api = abi.Api;

fn ok() callconv(.c) abi.Result {
    return abi.success;
}

pub fn hipGetDeviceCount(n: *c_int) callconv(.c) abi.Result {
    n.* = 1;
    return abi.success;
}

pub fn hipDeviceGet(dev: *abi.Device, ordinal: c_int) callconv(.c) abi.Result {
    if (ordinal != 0) return 101;
    dev.* = 0;
    return abi.success;
}

pub fn hipSetDevice(ordinal: c_int) callconv(.c) abi.Result {
    return if (ordinal == 0) abi.success else 101;
}

pub fn hipGetDevicePropertiesR0600(props: [*]u8, ordinal: c_int) callconv(.c) abi.Result {
    if (ordinal != 0) return 101;
    @memcpy(props[512..][0..options.arch.len], options.arch);
    props[512 + options.arch.len] = 0;
    return abi.success;
}

pub fn hipMemGetInfo(free: *usize, total: *usize) callconv(.c) abi.Result {
    free.* = 8 << 30;
    total.* = 16 << 30;
    return abi.success;
}

pub fn hipMalloc(p: *abi.DevicePtr, n: usize) callconv(.c) abi.Result {
    p.* = @intFromPtr(std.c.malloc(n) orelse return abi.error_out_of_memory);
    return abi.success;
}

pub fn hipFree(p: abi.DevicePtr) callconv(.c) abi.Result {
    std.c.free(@ptrFromInt(p));
    return abi.success;
}

pub fn hipHostMalloc(p: *?*anyopaque, n: usize, _: c_uint) callconv(.c) abi.Result {
    p.* = std.c.malloc(n) orelse return abi.error_out_of_memory;
    return abi.success;
}

pub fn hipHostGetDevicePointer(dev: *abi.DevicePtr, host: ?*anyopaque, _: c_uint) callconv(.c) abi.Result {
    dev.* = @intFromPtr(host);
    return abi.success;
}

pub fn hipHostFree(p: ?*anyopaque) callconv(.c) abi.Result {
    std.c.free(p);
    return abi.success;
}

fn copy(dst: usize, src: usize, n: usize) abi.Result {
    @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
    return abi.success;
}

pub fn hipMemcpyHtoD(dst: abi.DevicePtr, src: ?*const anyopaque, n: usize) callconv(.c) abi.Result {
    return copy(dst, @intFromPtr(src), n);
}

pub fn hipMemcpyDtoH(dst: ?*anyopaque, src: abi.DevicePtr, n: usize) callconv(.c) abi.Result {
    return copy(@intFromPtr(dst), src, n);
}

pub fn hipMemcpyDtoD(dst: abi.DevicePtr, src: abi.DevicePtr, n: usize) callconv(.c) abi.Result {
    return copy(dst, src, n);
}

pub fn hipMemcpyHtoDAsync(dst: abi.DevicePtr, src: ?*const anyopaque, n: usize, _: abi.Stream) callconv(.c) abi.Result {
    return copy(dst, @intFromPtr(src), n);
}

pub fn hipMemcpyDtoHAsync(dst: ?*anyopaque, src: abi.DevicePtr, n: usize, _: abi.Stream) callconv(.c) abi.Result {
    return copy(@intFromPtr(dst), src, n);
}

pub fn hipMemcpyDtoDAsync(dst: abi.DevicePtr, src: abi.DevicePtr, n: usize, _: abi.Stream) callconv(.c) abi.Result {
    return copy(dst, src, n);
}

pub fn hipMemsetD8(dst: abi.DevicePtr, v: u8, n: usize) callconv(.c) abi.Result {
    @memset(@as([*]u8, @ptrFromInt(dst))[0..n], v);
    return abi.success;
}

pub fn hipMemsetD32(dst: abi.DevicePtr, v: c_int, n: usize) callconv(.c) abi.Result {
    @memset(@as([*]c_int, @ptrFromInt(dst))[0..n], v);
    return abi.success;
}

pub fn hipMemsetD8Async(dst: abi.DevicePtr, v: u8, n: usize, _: abi.Stream) callconv(.c) abi.Result {
    return hipMemsetD8(dst, v, n);
}

pub fn hipMemsetD32Async(dst: abi.DevicePtr, v: c_int, n: usize, _: abi.Stream) callconv(.c) abi.Result {
    return hipMemsetD32(dst, v, n);
}

pub fn hipGetErrorName(_: abi.Result) callconv(.c) ?[*:0]const u8 {
    return "hipErrorMock";
}

pub const hipGetErrorString = hipGetErrorName;

// The functions above under their HIP names; every other entry point in `abi.Api` returns success and writes nothing.
comptime {
    for (@typeInfo(Api).@"struct".field_names) |name| {
        if (std.mem.eql(u8, name, options.omit)) continue;
        if (@hasDecl(@This(), name)) @export(&@field(@This(), name), .{ .name = name }) else @export(&ok, .{ .name = name });
    }
}
