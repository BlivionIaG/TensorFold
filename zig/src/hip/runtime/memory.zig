//! Device memory and pinned host memory, each owned by one value; copies and fills, blocking or on a stream.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const DeviceBuffer = struct {
    d: *const Driver,
    ptr: abi.DevicePtr,
    len: usize,

    /// `len` bytes, 256-byte aligned by HIP; zero bytes allocate nothing and hold address 0.
    pub fn alloc(d: *const Driver, len: usize) Error!DeviceBuffer {
        var p: abi.DevicePtr = 0;
        if (len > 0) try d.check(d.api.hipMalloc(&p, len), "hipMalloc");
        return .{ .d = d, .ptr = p, .len = len };
    }

    /// Allocates and fills from host bytes in one call.
    pub fn fromHost(d: *const Driver, bytes: []const u8) Error!DeviceBuffer {
        var b = try alloc(d, bytes.len);
        errdefer b.free();
        try b.upload(0, bytes);
        return b;
    }

    pub fn free(self: *DeviceBuffer) void {
        if (self.ptr != 0) _ = self.d.api.hipFree(self.ptr);
        self.* = undefined;
    }

    /// The device address `offset` bytes in; out of range is refused.
    pub fn at(self: DeviceBuffer, offset: usize) Error!abi.DevicePtr {
        if (offset > self.len) return error.Invalid;
        return self.ptr + offset;
    }

    fn span(self: DeviceBuffer, offset: usize, n: usize) Error!abi.DevicePtr {
        if (offset > self.len or n > self.len - offset) return error.Invalid;
        return self.ptr + offset;
    }

    pub fn upload(self: DeviceBuffer, offset: usize, bytes: []const u8) Error!void {
        const dst = try self.span(offset, bytes.len);
        if (bytes.len == 0) return;
        try self.d.check(self.d.api.hipMemcpyHtoD(dst, bytes.ptr, bytes.len), "hipMemcpyHtoD");
    }

    pub fn download(self: DeviceBuffer, offset: usize, out: []u8) Error!void {
        const src = try self.span(offset, out.len);
        if (out.len == 0) return;
        try self.d.check(self.d.api.hipMemcpyDtoH(out.ptr, src, out.len), "hipMemcpyDtoH");
    }

    /// Asynchronous only when `bytes` is pinned (HostBuffer); pageable memory makes HIP stage it.
    pub fn uploadAsync(self: DeviceBuffer, offset: usize, bytes: []const u8, stream: abi.Stream) Error!void {
        const dst = try self.span(offset, bytes.len);
        if (bytes.len == 0) return;
        try self.d.check(self.d.api.hipMemcpyHtoDAsync(dst, bytes.ptr, bytes.len, stream), "hipMemcpyHtoDAsync");
    }

    pub fn downloadAsync(self: DeviceBuffer, offset: usize, out: []u8, stream: abi.Stream) Error!void {
        const src = try self.span(offset, out.len);
        if (out.len == 0) return;
        try self.d.check(self.d.api.hipMemcpyDtoHAsync(out.ptr, src, out.len, stream), "hipMemcpyDtoHAsync");
    }

    /// Without a stream the copy has finished on return (a device-to-device copy alone may not have).
    pub fn copyFrom(self: DeviceBuffer, offset: usize, src: abi.DevicePtr, n: usize, stream: ?abi.Stream) Error!void {
        const dst = try self.span(offset, n);
        if (n == 0) return;
        if (stream) |s| {
            try self.d.check(self.d.api.hipMemcpyDtoDAsync(dst, src, n, s), "hipMemcpyDtoDAsync");
        } else {
            try self.d.check(self.d.api.hipMemcpyDtoD(dst, src, n), "hipMemcpyDtoD");
            try self.settle();
        }
    }

    /// Without a stream the fill has finished on return, so a non-blocking stream's next kernel cannot race it.
    pub fn fill8(self: DeviceBuffer, value: u8, stream: ?abi.Stream) Error!void {
        if (self.len == 0) return;
        if (stream) |s| {
            try self.d.check(self.d.api.hipMemsetD8Async(self.ptr, value, self.len, s), "hipMemsetD8Async");
        } else {
            try self.d.check(self.d.api.hipMemsetD8(self.ptr, value, self.len), "hipMemsetD8");
            try self.settle();
        }
    }

    /// Fills whole 32-bit words (finished on return without a stream); the length must be a multiple of four.
    pub fn fill32(self: DeviceBuffer, value: u32, stream: ?abi.Stream) Error!void {
        if (self.len % 4 != 0) return error.Invalid;
        if (self.len == 0) return;
        const v: c_int = @bitCast(value);
        if (stream) |s| {
            try self.d.check(self.d.api.hipMemsetD32Async(self.ptr, v, self.len / 4, s), "hipMemsetD32Async");
        } else {
            try self.d.check(self.d.api.hipMemsetD32(self.ptr, v, self.len / 4), "hipMemsetD32");
            try self.settle();
        }
    }

    /// Waits for the null stream, where HIP's stream-less fills and device copies run.
    fn settle(self: DeviceBuffer) Error!void {
        try self.d.check(self.d.api.hipStreamSynchronize(null), "hipStreamSynchronize");
    }
};

/// Page-locked host memory: the only host memory an asynchronous copy can read or write without staging.
pub const HostBuffer = struct {
    d: *const Driver,
    bytes: []align(16) u8,

    pub fn alloc(d: *const Driver, len: usize) Error!HostBuffer {
        return allocFlags(d, len, abi.host_malloc_portable);
    }

    /// Pinned memory kernels write directly (mapped into the device's address space): no copy reads it back.
    pub fn allocMapped(d: *const Driver, len: usize) Error!HostBuffer {
        return allocFlags(d, len, abi.host_malloc_portable | abi.host_malloc_mapped);
    }

    /// The device address of a mapped buffer.
    pub fn device(self: HostBuffer) Error!abi.DevicePtr {
        var p: abi.DevicePtr = 0;
        try self.d.check(self.d.api.hipHostGetDevicePointer(&p, self.bytes.ptr, 0), "hipHostGetDevicePointer");
        return p;
    }

    fn allocFlags(d: *const Driver, len: usize, flags: c_uint) Error!HostBuffer {
        if (len == 0) return error.Invalid;
        var p: ?*anyopaque = null;
        try d.check(d.api.hipHostMalloc(&p, len, flags), "hipHostMalloc");
        const base: [*]align(16) u8 = @ptrCast(@alignCast(p.?));
        return .{ .d = d, .bytes = base[0..len] };
    }

    pub fn free(self: *HostBuffer) void {
        _ = self.d.api.hipHostFree(self.bytes.ptr);
        self.* = undefined;
    }

    pub fn slice(self: HostBuffer, comptime T: type) []T {
        return std.mem.bytesAsSlice(T, self.bytes[0 .. self.bytes.len / @sizeOf(T) * @sizeOf(T)]);
    }
};
