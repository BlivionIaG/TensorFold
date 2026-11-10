//! The device-agnostic side of TB4 offload: what a Metal host needs from an external GPU, and nothing vendor specific.
//!
//! The device has one in-order stream. Work, waits on a host word and writes of a host word are queued on it; the host
//! sees results through windows (pinned host memory the device reads and writes), never through the stream.
const std = @import("std");

/// The activation type a device's kernels keep between layers (gfx900 has no bf16 dot, so it keeps f16).
pub const Dtype = enum(u8) { f16, bf16 };

pub const Error = error{ Unsupported, OutOfMemory, Timeout, DeviceLost, BadWindow, Failed };

pub const Caps = struct {
    name: []const u8,
    arch: []const u8,
    /// Memory the device keeps for tensors, after what the runtime reserves.
    vram_bytes: u64,
    /// Peak memory bandwidth in bytes a second (datasheet); the planner scales it by a measured efficiency.
    peak_bandwidth: u64,
    act: Dtype,
    /// Cost of one launch that cannot overlap the previous one.
    launch_ns: u32,
    max_windows: u32,
    /// Window length and address multiple (Metal wraps whole 16 KiB pages).
    window_align: usize = 16 * 1024,
    /// The queue can stall on a host word and write one after earlier work (stream wait/write value).
    flag_wait: bool = true,
    flag_write: bool = true,
    timestamps: bool = true,
};

/// Pinned host memory the device can read and write and the host can map. The host pointer never moves while the
/// window exists; `len` and the address are multiples of `Caps.window_align`.
pub const Window = struct {
    ptr: [*]u8,
    len: usize,
    id: u32 = 0,

    pub fn bytes(w: Window) []u8 {
        return w.ptr[0..w.len];
    }

    /// The 32-bit word at byte `offset`, for flag access with acquire/release ordering.
    pub fn word(w: Window, offset: usize) *u32 {
        std.debug.assert(offset % 4 == 0 and offset + 4 <= w.len);
        return @ptrCast(@alignCast(w.ptr + offset));
    }
};

/// A recorded launch plan: `run` is called in stream order (a HIP graph launch, a captured indirect buffer, or, on a
/// mock, plain Zig code on the device's thread).
pub const Work = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque) anyerror!void,
};

pub const DevPtr = u64;

pub const VTable = struct {
    caps: *const fn (self: *anyopaque) Caps,
    alloc: *const fn (self: *anyopaque, len: usize) Error!DevPtr,
    free: *const fn (self: *anyopaque, p: DevPtr) void,
    upload: *const fn (self: *anyopaque, dst: DevPtr, src: []const u8) Error!void,
    download: *const fn (self: *anyopaque, dst: []u8, src: DevPtr) Error!void,
    alloc_window: *const fn (self: *anyopaque, len: usize) Error!Window,
    free_window: *const fn (self: *anyopaque, w: Window) void,
    enqueue: *const fn (self: *anyopaque, work: Work) Error!void,
    /// Stall the stream until the 32-bit word at `w + offset` equals `value`; `timeout_ns` 0 waits without bound.
    enqueue_wait: *const fn (self: *anyopaque, w: Window, offset: usize, value: u32, timeout_ns: u64) Error!void,
    /// After every earlier operation on the stream is complete and visible, store `value` at `w + offset`.
    enqueue_write: *const fn (self: *anyopaque, w: Window, offset: usize, value: u32) Error!void,
    /// Block the host until the stream is idle; the first error a queued operation hit is returned.
    sync: *const fn (self: *anyopaque, timeout_ns: u64) Error!void,
    timestamp_ns: *const fn (self: *anyopaque) Error!u64,
    deinit: *const fn (self: *anyopaque) void,
};

pub const ExternalDevice = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub fn caps(d: ExternalDevice) Caps {
        return d.vtable.caps(d.ptr);
    }

    pub fn alloc(d: ExternalDevice, len: usize) Error!DevPtr {
        return d.vtable.alloc(d.ptr, len);
    }

    pub fn free(d: ExternalDevice, p: DevPtr) void {
        d.vtable.free(d.ptr, p);
    }

    pub fn upload(d: ExternalDevice, dst: DevPtr, src: []const u8) Error!void {
        return d.vtable.upload(d.ptr, dst, src);
    }

    pub fn download(d: ExternalDevice, dst: []u8, src: DevPtr) Error!void {
        return d.vtable.download(d.ptr, dst, src);
    }

    pub fn allocWindow(d: ExternalDevice, len: usize) Error!Window {
        return d.vtable.alloc_window(d.ptr, len);
    }

    pub fn freeWindow(d: ExternalDevice, w: Window) void {
        d.vtable.free_window(d.ptr, w);
    }

    pub fn enqueue(d: ExternalDevice, work: Work) Error!void {
        return d.vtable.enqueue(d.ptr, work);
    }

    pub fn enqueueWait(d: ExternalDevice, w: Window, offset: usize, value: u32, timeout_ns: u64) Error!void {
        return d.vtable.enqueue_wait(d.ptr, w, offset, value, timeout_ns);
    }

    pub fn enqueueWrite(d: ExternalDevice, w: Window, offset: usize, value: u32) Error!void {
        return d.vtable.enqueue_write(d.ptr, w, offset, value);
    }

    pub fn sync(d: ExternalDevice, timeout_ns: u64) Error!void {
        return d.vtable.sync(d.ptr, timeout_ns);
    }

    pub fn timestampNs(d: ExternalDevice) Error!u64 {
        return d.vtable.timestamp_ns(d.ptr);
    }

    pub fn deinit(d: ExternalDevice) void {
        d.vtable.deinit(d.ptr);
    }
};

/// CLOCK_MONOTONIC in nanoseconds.
pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Sleep about `ns` nanoseconds.
pub fn pause(ns: u64) void {
    const ts: std.c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = std.c.nanosleep(&ts, null);
}
