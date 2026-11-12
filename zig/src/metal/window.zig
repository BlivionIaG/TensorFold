//! HostWindow: Metal's view of pinned host memory an external GPU DMAs into (zero copy, shared storage).
const std = @import("std");
const objc = @import("objc.zig");
const types = @import("types.zig");
const dev = @import("device.zig");
const Device = dev.Device;
const Buffer = dev.Buffer;

/// The granule wrap() accepts: Apple silicon's 16 KiB page, also a multiple of the 4 KiB page of an Intel Mac.
pub const alignment: usize = 16 * 1024;

pub const Error = dev.Error || error{ Misaligned, MapFailed };

var shm_serial = std.atomic.Value(u32).init(0);

pub const HostWindow = struct {
    buffer: Buffer,
    ptr: [*]u8,
    len: usize,
    /// True when deinit unmaps `ptr` (createShm); false for memory another owner keeps.
    owned: bool,

    /// Wrap memory another owner keeps (a pinned sysmem block); deinit leaves it mapped. Shared, no hazard tracking.
    pub fn wrap(device: Device, ptr: [*]u8, len: usize) Error!HostWindow {
        if (len == 0 or len % alignment != 0 or @intFromPtr(ptr) % alignment != 0) return error.Misaligned;
        const options = types.ResourceOptions.shared | types.ResourceOptions.untracked;
        const buffer = try device.bufferNoCopy(ptr, len, options);
        // a copy instead of a wrap would leave the DMA target invisible to the kernels
        if (buffer.contents() != ptr) {
            buffer.deinit();
            return error.NoBuffer;
        }
        return .{ .buffer = buffer, .ptr = ptr, .len = len, .owned = false };
    }

    /// `len` bytes rounded up to `alignment`, zeroed, unlinked POSIX shared memory; deinit unmaps it.
    pub fn createShm(device: Device, len: usize) Error!HostWindow {
        const size = std.mem.alignForward(usize, @max(len, 1), alignment);
        const mem = try mapShm(size);
        var window = wrap(device, mem.ptr, size) catch |err| {
            std.posix.munmap(mem);
            return err;
        };
        window.owned = true;
        return window;
    }

    /// The window as 32-bit words for flag access; every load and store goes to memory.
    pub fn words(self: HostWindow) []volatile u32 {
        return @as([*]volatile u32, @ptrCast(@alignCast(self.ptr)))[0 .. self.len / 4];
    }

    pub fn deinit(self: HostWindow) void {
        self.buffer.deinit();
        if (self.owned) std.posix.munmap(@as([*]align(std.heap.page_size_min) u8, @alignCast(self.ptr))[0..self.len]);
    }
};

/// A POSIX shm object of `size` zero bytes, already unlinked: the descriptor is its only handle.
fn shmFd(size: usize) Error!c_int {
    var name: [32]u8 = undefined;
    const flags: c_int = @bitCast(std.c.O{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true });
    var fd: c_int = -1;
    var attempt: usize = 0;
    // macOS caps a shm name at 31 bytes: "/tfw" + pid + "." + serial fits
    while (fd < 0 and attempt < 16) : (attempt += 1) {
        const serial = shm_serial.fetchAdd(1, .monotonic);
        const z = std.fmt.bufPrintSentinel(&name, "/tfw{d}.{d}", .{ std.c.getpid(), serial }, 0) catch return error.MapFailed;
        fd = std.c.shm_open(z.ptr, flags, @as(c_uint, 0o600));
        if (fd >= 0) _ = std.c.shm_unlink(z.ptr);
    }
    if (fd < 0) return error.MapFailed;
    if (std.c.ftruncate(fd, @intCast(size)) != 0) {
        _ = std.c.close(fd);
        return error.MapFailed;
    }
    return fd;
}

/// A fresh shared mapping of `size` bytes (zero filled by the kernel); the object has no name.
fn mapShm(size: usize) Error![]align(std.heap.page_size_min) u8 {
    const fd = try shmFd(size);
    defer _ = std.c.close(fd);
    return std.posix.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0) catch error.MapFailed;
}

const testing = std.testing;

const probe_src =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\kernel void fill(device uint* p [[buffer(0)]], constant uint& v [[buffer(1)]], uint t [[thread_position_in_grid]]) { p[t] = v + t; }
;

fn pattern(i: usize) u32 {
    return 0xA5A50000 +% @as(u32, @truncate(i));
}

test "host window: wrap accepts shm windows of 16 KiB, 1 MiB and 64 MiB" {
    const pool = objc.Pool.push();
    defer pool.pop();
    const device = Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    for ([_]usize{ 16 << 10, 1 << 20, 64 << 20 }) |len| {
        const w = try HostWindow.createShm(device, len);
        defer w.deinit();
        try testing.expectEqual(len, w.len);
        try testing.expect(w.owned);
        try testing.expect(w.buffer.contents() == w.ptr);
        try testing.expect(w.buffer.length() >= len);
        const words = w.words();
        try testing.expectEqual(len / 4, words.len);
        try testing.expectEqual(@as(u32, 0), words[0]);
        try testing.expectEqual(@as(u32, 0), words[words.len - 1]);
    }
}

test "host window: createShm rounds its length up to the alignment" {
    const device = Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const w = try HostWindow.createShm(device, 20 * 1024);
    defer w.deinit();
    try testing.expectEqual(2 * alignment, w.len);
    const tiny = try HostWindow.createShm(device, 1);
    defer tiny.deinit();
    try testing.expectEqual(alignment, tiny.len);
}

test "host window: wrap refuses an unaligned pointer and an unaligned length" {
    const device = Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const w = try HostWindow.createShm(device, 4 * alignment);
    defer w.deinit();
    try testing.expectError(error.Misaligned, HostWindow.wrap(device, w.ptr + 4096, alignment));
    try testing.expectError(error.Misaligned, HostWindow.wrap(device, w.ptr, 20 * 1024));
    try testing.expectError(error.Misaligned, HostWindow.wrap(device, w.ptr, 0));
    const inner = try HostWindow.wrap(device, w.ptr + alignment, 2 * alignment);
    defer inner.deinit();
    try testing.expect(!inner.owned);
    try testing.expect(inner.buffer.contents() == w.ptr + alignment);
}

test "host window: wrap takes a MAP_FIXED mapping inside a PROT_NONE reservation" {
    const device = Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const arena: usize = 64 << 20;
    const block: usize = 16 << 20;
    const reservation = try std.posix.mmap(null, arena, .{}, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(reservation);
    const fd = try shmFd(block);
    defer _ = std.c.close(fd);
    const fixed_at: [*]align(std.heap.page_size_min) u8 = @alignCast(reservation.ptr + (8 << 20));
    const fixed = try std.posix.mmap(fixed_at, block, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED, .FIXED = true }, fd, 0);
    try testing.expect(fixed.ptr == fixed_at);
    const w = try HostWindow.wrap(device, fixed.ptr, block);
    defer w.deinit();
    w.words()[3] = 0x1234;
    try testing.expectEqual(@as(u32, 0x1234), w.buffer.slice(u32, 4)[3]);
}

test "host window: an aliased second mapping of the same shm is CPU-coherent with the window" {
    const device = Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const fd = try shmFd(4 * alignment);
    defer _ = std.c.close(fd);
    const prot: std.posix.PROT = .{ .READ = true, .WRITE = true };
    const a = try std.posix.mmap(null, 4 * alignment, prot, .{ .TYPE = .SHARED }, fd, 0);
    defer std.posix.munmap(a);
    const b = try std.posix.mmap(null, 4 * alignment, prot, .{ .TYPE = .SHARED }, fd, 0);
    defer std.posix.munmap(b);
    const w = try HostWindow.wrap(device, a.ptr, a.len);
    defer w.deinit();
    @as(*volatile u32, @ptrCast(@alignCast(b.ptr + 20))).* = 0xC0FFEE;
    try testing.expectEqual(@as(u32, 0xC0FFEE), w.words()[5]);
    w.words()[6] = 0xBEEF;
    try testing.expectEqual(@as(u32, 0xBEEF), @as(*volatile u32, @ptrCast(@alignCast(b.ptr + 24))).*);
}

test "host window: a kernel's pattern is visible to the CPU after the command buffer completes" {
    const pool = objc.Pool.push();
    defer pool.pop();
    const device = Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const lib = try @import("library.zig").Library.fromSource(device, probe_src, .{});
    defer lib.deinit();
    const pipeline = try @import("library.zig").Pipeline.init(device, lib, "fill", false);
    defer pipeline.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const w = try HostWindow.createShm(device, 1 << 20);
    defer w.deinit();
    const n: usize = 4096;
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(pipeline);
    enc.setBuffer(w.buffer, 0, 0);
    enc.setValue(@as(u32, pattern(0)), 1);
    enc.dispatchThreads(types.Size.of(n, 1, 1), types.Size.of(256, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    try testing.expect(cb.failure() == null);
    const words = w.words();
    for (0..n) |i| try testing.expectEqual(pattern(i), words[i]);
    try testing.expectEqual(@as(u32, 0), words[n]);
}
