//! A HIP card as the offload layer's external device: mapped pinned windows, stream wait/write on host words, sync.
const std = @import("std");
const split = @import("split");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const memory = @import("memory.zig");
const Context = @import("context.zig").Context;
const Stream = @import("stream.zig").Stream;
const Caps = @import("caps.zig").Caps;
const Device = @import("device.zig").Device;

const device = split.device;

/// hipStreamWaitValue32's compare for equality (hipStreamWaitValueEq).
const wait_eq: c_uint = 1;

pub const Error = runtime.Error || error{ NoValueOps, UnsupportedGpu, OutOfMemory };

pub const WaitValue32 = *const fn (abi.Stream, ?*anyopaque, u32, c_uint, u32) callconv(.c) abi.Result;
pub const WriteValue32 = *const fn (abi.Stream, ?*anyopaque, u32, c_uint) callconv(.c) abi.Result;

/// What the adapter needs beyond the engine's admitted symbols: the stream value calls and the card's architecture.
pub const Calls = struct { wait32: WaitValue32, write32: WriteValue32, arch: []const u8 };

pub const Options = struct {
    /// Datasheet memory bandwidth in bytes a second; the planner scales it by a measured efficiency.
    peak_bandwidth: u64,
    /// Bytes the runtime keeps for itself out of the card's total.
    reserved_bytes: u64 = 128 << 20,
    max_windows: u32 = 128,
    /// Queue the waits and writes on this stream (the engine's) instead of an own one; the caller destroys it.
    stream: ?Stream = null,
};

pub const HipExternal = struct {
    gpa: std.mem.Allocator,
    rt: *const runtime.Runtime,
    ctx: Context,
    stream: Stream,
    owns_stream: bool = true,
    wait32: WaitValue32,
    write32: WriteValue32,
    caps: device.Caps,
    name: [128]u8 = @splat(0),
    arch: [64]u8 = @splat(0),
    fence_window: memory.HostBuffer,
    fence_next: u32 = 0,
    windows: std.AutoHashMapUnmanaged(usize, memory.HostBuffer) = .empty,
    allocs: std.AutoHashMapUnmanaged(u64, memory.DeviceBuffer) = .empty,
    lost: bool = false,

    /// The card at `index` of an open runtime; a runtime without the stream value calls is refused with `NoValueOps`.
    pub fn create(gpa: std.mem.Allocator, rt: *runtime.Runtime, index: c_int, options: Options) Error!*HipExternal {
        const wait32 = rt.lib.lookup(WaitValue32, "hipStreamWaitValue32") orelse return error.NoValueOps;
        const write32 = rt.lib.lookup(WriteValue32, "hipStreamWriteValue32") orelse return error.NoValueOps;
        var arch_buf: [64]u8 = undefined;
        const arch = try Device.archName(rt, index, &arch_buf);
        return createWith(gpa, rt, index, options, .{ .wait32 = wait32, .write32 = write32, .arch = arch });
    }

    /// `create` with the calls and the architecture name given, for a runtime that is not a loaded library.
    pub fn createWith(gpa: std.mem.Allocator, rt: *const runtime.Runtime, index: c_int, options: Options, calls: Calls) Error!*HipExternal {
        if (calls.arch.len > 63) return error.Invalid;
        const card = Caps.of(calls.arch) orelse return error.UnsupportedGpu;
        const self = try gpa.create(HipExternal);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .rt = rt, .ctx = undefined, .stream = undefined, .wait32 = calls.wait32, .write32 = calls.write32, .caps = undefined, .fence_window = undefined };
        self.ctx = try Context.init(rt, index);
        errdefer self.ctx.deinit();
        @memcpy(self.arch[0..calls.arch.len], calls.arch);
        const name = try self.ctx.name(&self.name);
        const mem = try self.ctx.memInfo();
        self.owns_stream = options.stream == null;
        self.stream = options.stream orelse try Stream.init(rt);
        errdefer if (self.owns_stream) self.stream.deinit();
        self.fence_window = try memory.HostBuffer.allocMapped(rt, window_align);
        @memset(self.fence_window.bytes, 0);
        self.caps = .{
            .backend = "hip",
            .name = self.name[0..name.len],
            .arch = self.arch[0..calls.arch.len],
            .vram_bytes = mem.total -| options.reserved_bytes,
            .peak_bandwidth = options.peak_bandwidth,
            .act = switch (card.act) {
                .f16 => .f16,
                .bf16 => .bf16,
            },
            .launch_ns = 4_400,
            .max_windows = options.max_windows,
        };
        return self;
    }

    pub fn external(self: *HipExternal) device.ExternalDevice {
        return .{ .ptr = self, .vtable = &vtable };
    }

    pub const window_align: usize = 16 * 1024;

    fn me(ptr: *anyopaque) *HipExternal {
        return @ptrCast(@alignCast(ptr));
    }

    fn map(err: runtime.Error) device.Error {
        return switch (err) {
            error.Invalid => error.BadWindow,
            else => error.Failed,
        };
    }

    fn capsFn(ptr: *anyopaque) device.Caps {
        return me(ptr).caps;
    }

    fn allocFn(ptr: *anyopaque, len: usize) device.Error!device.DevPtr {
        const self = me(ptr);
        var buf = memory.DeviceBuffer.alloc(self.rt, len) catch return error.OutOfMemory;
        errdefer buf.free();
        self.allocs.put(self.gpa, buf.base(), buf) catch return error.OutOfMemory;
        return buf.base();
    }

    fn freeFn(ptr: *anyopaque, p: device.DevPtr) void {
        const self = me(ptr);
        if (self.allocs.fetchRemove(p)) |kv| {
            var buf = kv.value;
            buf.free();
        }
    }

    fn upload(ptr: *anyopaque, dst: device.DevPtr, src: []const u8) device.Error!void {
        const buf = me(ptr).allocs.get(dst) orelse return error.Failed;
        buf.upload(0, src) catch |err| return map(err);
    }

    fn download(ptr: *anyopaque, dst: []u8, src: device.DevPtr) device.Error!void {
        const buf = me(ptr).allocs.get(src) orelse return error.Failed;
        buf.download(0, dst) catch |err| return map(err);
    }

    fn allocWindow(ptr: *anyopaque, len: usize) device.Error!device.Window {
        const self = me(ptr);
        const n = (len + window_align - 1) / window_align * window_align;
        var buf = memory.HostBuffer.allocMapped(self.rt, n) catch |err| return map(err);
        errdefer buf.free();
        // a host that wraps whole pages cannot serve memory the runtime returned less aligned
        if (@intFromPtr(buf.bytes.ptr) % window_align != 0) return error.BadWindow;
        @memset(buf.bytes, 0);
        self.windows.put(self.gpa, @intFromPtr(buf.bytes.ptr), buf) catch return error.OutOfMemory;
        return .{ .ptr = buf.bytes.ptr, .len = n, .id = @intCast(self.windows.count()) };
    }

    fn freeWindow(ptr: *anyopaque, w: device.Window) void {
        const self = me(ptr);
        // queued operations may still hold the window's words
        syncFn(ptr, 5_000_000_000) catch {};
        if (self.windows.fetchRemove(@intFromPtr(w.ptr))) |kv| {
            var buf = kv.value;
            buf.free();
        }
    }

    fn enqueue(ptr: *anyopaque, work: device.Work) device.Error!void {
        const self = me(ptr);
        if (self.lost) return error.DeviceLost;
        // a stream device records the launches now; they run in stream order behind the waits already queued
        work.run(work.ctx) catch return error.Failed;
    }

    fn enqueueWait(ptr: *anyopaque, w: device.Window, offset: usize, value: u32, _: u64) device.Error!void {
        const self = me(ptr);
        if (self.lost) return error.DeviceLost;
        if (offset % 4 != 0 or offset + 4 > w.len) return error.BadWindow;
        // the stream has no timeout: a host that never stores the value is released by the pipeline's abort store
        if (self.wait32(self.stream.handle, w.ptr + offset, value, wait_eq, 0xFFFF_FFFF) != 0) return error.Failed;
    }

    fn enqueueWrite(ptr: *anyopaque, w: device.Window, offset: usize, value: u32) device.Error!void {
        const self = me(ptr);
        if (self.lost) return error.DeviceLost;
        if (offset % 4 != 0 or offset + 4 > w.len) return error.BadWindow;
        if (self.write32(self.stream.handle, w.ptr + offset, value, 0) != 0) return error.Failed;
    }

    fn syncFn(ptr: *anyopaque, timeout_ns: u64) device.Error!void {
        const self = me(ptr);
        if (self.lost) return error.DeviceLost;
        self.fence_next +%= 1;
        const word: *u32 = @ptrCast(@alignCast(self.fence_window.bytes.ptr));
        if (self.write32(self.stream.handle, word, self.fence_next, 0) != 0) return error.Failed;
        const start = device.nowNs();
        while (@atomicLoad(u32, word, .acquire) != self.fence_next) {
            if (timeout_ns != 0 and device.nowNs() - start >= timeout_ns) return error.Timeout;
            std.atomic.spinLoopHint();
        }
    }

    fn timestamp(_: *anyopaque) device.Error!u64 {
        return device.nowNs();
    }

    fn deinitFn(ptr: *anyopaque) void {
        const self = me(ptr);
        self.stream.synchronize() catch {};
        var wit = self.windows.valueIterator();
        while (wit.next()) |b| {
            var buf = b.*;
            buf.free();
        }
        self.windows.deinit(self.gpa);
        var ait = self.allocs.valueIterator();
        while (ait.next()) |b| {
            var buf = b.*;
            buf.free();
        }
        self.allocs.deinit(self.gpa);
        self.fence_window.free();
        if (self.owns_stream) self.stream.deinit();
        self.ctx.deinit();
        self.gpa.destroy(self);
    }

    const vtable: device.VTable = .{
        .caps = capsFn,
        .alloc = allocFn,
        .free = freeFn,
        .upload = upload,
        .download = download,
        .alloc_window = allocWindow,
        .free_window = freeWindow,
        .enqueue = enqueue,
        .enqueue_wait = enqueueWait,
        .enqueue_write = enqueueWrite,
        .sync = syncFn,
        .timestamp_ns = timestamp,
        .deinit = deinitFn,
    };
};

test {
    _ = @import("offload_test.zig");
}
