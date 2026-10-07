//! Device memory and pinned host memory, each owned by one value; copies and fills, blocking or on a stream.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

/// Bytes this process holds in DeviceBuffers and HostBuffers, and the device peak since the last reset.
pub const Usage = struct { device: u64, host: u64, peak: u64 };

var device_bytes: std.atomic.Value(u64) = .init(0);
var host_bytes: std.atomic.Value(u64) = .init(0);
var peak_bytes: std.atomic.Value(u64) = .init(0);
var counts_mutex: std.atomic.Mutex = .unlocked;

/// The counts now; `reset_peak` starts a new peak at the current device bytes. Safe from any thread.
pub fn usage(reset_peak: bool) Usage {
    const NoHook = struct {
        fn afterRead(_: @This()) void {}
    };
    return usageWithHook(reset_peak, NoHook{});
}

fn usageWithHook(reset_peak: bool, hook: anytype) Usage {
    lockCounts();
    defer counts_mutex.unlock();
    const now = device_bytes.load(.monotonic);
    hook.afterRead();
    if (reset_peak) peak_bytes.store(now, .monotonic);
    return .{ .device = now, .host = host_bytes.load(.monotonic), .peak = @max(now, peak_bytes.load(.monotonic)) };
}

fn held(counter: *std.atomic.Value(u64), n: usize) void {
    const device = counter == &device_bytes;
    if (device) lockCounts();
    defer if (device) counts_mutex.unlock();
    const now = counter.fetchAdd(n, .monotonic) + n;
    if (counter == &device_bytes) _ = peak_bytes.fetchMax(now, .monotonic);
}

fn freed(counter: *std.atomic.Value(u64), n: usize) void {
    const device = counter == &device_bytes;
    if (device) lockCounts();
    defer if (device) counts_mutex.unlock();
    _ = counter.fetchSub(n, .monotonic);
}

fn lockCounts() void {
    while (!counts_mutex.tryLock()) std.Thread.yield() catch {};
}

pub const DeviceBuffer = struct {
    d: *const Driver,
    ptr: abi.DevicePtr,
    len: usize,

    /// `len` bytes, 256-byte aligned by HIP; zero bytes allocate nothing and hold address 0.
    pub fn alloc(d: *const Driver, len: usize) Error!DeviceBuffer {
        var p: abi.DevicePtr = 0;
        if (len > 0) try d.check(d.api.hipMalloc(&p, len), "hipMalloc");
        held(&device_bytes, len);
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
        freed(&device_bytes, self.len);
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
        held(&host_bytes, len);
        return .{ .d = d, .bytes = base[0..len] };
    }

    pub fn free(self: *HostBuffer) void {
        _ = self.d.api.hipHostFree(self.bytes.ptr);
        freed(&host_bytes, self.bytes.len);
        self.* = undefined;
    }

    pub fn slice(self: HostBuffer, comptime T: type) []T {
        return std.mem.bytesAsSlice(T, self.bytes[0 .. self.bytes.len / @sizeOf(T) * @sizeOf(T)]);
    }
};

test "usage counts held bytes and the device peak" {
    const before = usage(true);
    held(&device_bytes, 1000);
    held(&host_bytes, 24);
    freed(&device_bytes, 1000);
    const after = usage(false);
    try std.testing.expectEqual(before.device, after.device);
    try std.testing.expectEqual(before.host + 24, after.host);
    try std.testing.expectEqual(before.device + 1000, after.peak);
    try std.testing.expectEqual(before.device, usage(true).peak);
    freed(&host_bytes, 24);
}

test "peak reset preserves an allocation between its read and store" {
    const before = usage(true);
    held(&device_bytes, 100);
    defer freed(&device_bytes, 100);
    const Probe = struct {
        read: std.atomic.Value(bool) = .init(false),
        attempted: std.atomic.Value(bool) = .init(false),
        blocked: std.atomic.Value(bool) = .init(false),
        allocated: std.atomic.Value(bool) = .init(false),
        finish: std.atomic.Value(bool) = .init(false),

        fn wait(flag: *std.atomic.Value(bool)) void {
            while (!flag.load(.acquire)) std.Thread.yield() catch {};
        }

        fn afterRead(probe: *@This()) void {
            probe.read.store(true, .release);
            wait(&probe.attempted);
            if (!probe.blocked.load(.acquire)) wait(&probe.allocated);
        }

        fn allocate(probe: *@This()) void {
            wait(&probe.read);
            const acquired = counts_mutex.tryLock();
            if (acquired) counts_mutex.unlock();
            probe.blocked.store(!acquired, .release);
            probe.attempted.store(true, .release);
            held(&device_bytes, 300);
            probe.allocated.store(true, .release);
            wait(&probe.finish);
            freed(&device_bytes, 300);
        }
    };
    var probe: Probe = .{};
    const thread = try std.Thread.spawn(.{}, Probe.allocate, .{&probe});
    _ = usageWithHook(true, &probe);
    probe.finish.store(true, .release);
    thread.join();
    const after = usage(false);
    try std.testing.expectEqual(before.device + 100, after.device);
    try std.testing.expectEqual(before.device + 400, after.peak);
}
