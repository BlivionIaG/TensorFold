//! HipExternal against a stand-in runtime whose stream is a queue that a thread runs in order.
const std = @import("std");
const split = @import("split");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const offload = @import("offload.zig");
const Stream = @import("stream.zig").Stream;

const T = std.testing;

/// The stand-in device: stream operations run in order on a thread, and a wait holds the queue until its word matches.
const Queue = struct {
    const Op = struct { word: *u32, value: u32, wait: bool };
    ops: [64]Op = undefined,
    head: std.atomic.Value(usize) = .init(0),
    tail: std.atomic.Value(usize) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    /// The stream the last operation was queued on, and the streams destroyed.
    last_stream: abi.Stream = null,
    destroyed: usize = 0,

    fn start(q: *Queue) !void {
        q.* = .{};
        q.thread = try std.Thread.spawn(.{}, run, .{q});
    }

    fn finish(q: *Queue) void {
        q.stop.store(true, .release);
        if (q.thread) |t| t.join();
        q.thread = null;
    }

    fn push(q: *Queue, op: Op) bool {
        const h = q.head.load(.monotonic);
        if (h - q.tail.load(.acquire) == q.ops.len) return false;
        q.ops[h % q.ops.len] = op;
        q.head.store(h + 1, .release);
        return true;
    }

    fn run(q: *Queue) void {
        while (!q.stop.load(.acquire)) {
            const t = q.tail.load(.monotonic);
            if (t == q.head.load(.acquire)) {
                split.device.pause(20_000);
                continue;
            }
            const op = q.ops[t % q.ops.len];
            if (op.wait) {
                while (@atomicLoad(u32, op.word, .acquire) != op.value) {
                    if (q.stop.load(.acquire)) return;
                    std.atomic.spinLoopHint();
                }
            } else {
                @atomicStore(u32, op.word, op.value, .release);
            }
            q.tail.store(t + 1, .release);
        }
    }
};

var queue: Queue = .{};

fn waitValue(stream: abi.Stream, p: ?*anyopaque, value: u32, flags: c_uint, mask: u32) callconv(.c) abi.Result {
    if (flags != 1 or mask != 0xFFFF_FFFF) return 1;
    queue.last_stream = stream;
    return if (queue.push(.{ .word = @ptrCast(@alignCast(p.?)), .value = value, .wait = true })) 0 else 1;
}

fn writeValue(stream: abi.Stream, p: ?*anyopaque, value: u32, flags: c_uint) callconv(.c) abi.Result {
    if (flags != 0) return 1;
    queue.last_stream = stream;
    return if (queue.push(.{ .word = @ptrCast(@alignCast(p.?)), .value = value, .wait = false })) 0 else 1;
}

fn ok() callconv(.c) abi.Result {
    return 0;
}

fn retain(handle: *abi.Context, _: abi.Device) callconv(.c) abi.Result {
    handle.* = @ptrFromInt(0x10);
    return 0;
}

fn createStream(out: *abi.Stream, _: c_uint) callconv(.c) abi.Result {
    out.* = @ptrFromInt(0x20 + 0x10 * (queue.destroyed + 1));
    return 0;
}

fn destroyStream(_: abi.Stream) callconv(.c) abi.Result {
    queue.destroyed += 1;
    return 0;
}

fn deviceName(buf: [*]u8, len: c_int, _: abi.Device) callconv(.c) abi.Result {
    const name = "Stand-in";
    if (len < name.len + 1) return 1;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    return 0;
}

fn memInfo(free_bytes: *usize, total: *usize) callconv(.c) abi.Result {
    free_bytes.* = 1 << 30;
    total.* = 1 << 30;
    return 0;
}

fn alignedAlloc(out: *abi.DevicePtr, n: usize, alignment: usize) abi.Result {
    var p: ?*anyopaque = null;
    if (std.c.posix_memalign(&p, alignment, n) != 0) return 1;
    out.* = p;
    return 0;
}

fn malloc(out: *abi.DevicePtr, n: usize) callconv(.c) abi.Result {
    return alignedAlloc(out, n, 64);
}

fn hostMalloc(out: *abi.DevicePtr, n: usize, _: c_uint) callconv(.c) abi.Result {
    return alignedAlloc(out, n, offload.HipExternal.window_align);
}

fn release(p: abi.DevicePtr) callconv(.c) abi.Result {
    std.c.free(p);
    return 0;
}

fn toDevice(dst: abi.DevicePtr, src: [*]const u8, n: usize) callconv(.c) abi.Result {
    @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], src[0..n]);
    return 0;
}

fn toHost(dst: [*]u8, src: abi.DevicePtr, n: usize) callconv(.c) abi.Result {
    @memcpy(dst[0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
    return 0;
}

/// A runtime whose every call succeeds and does nothing, but for the few the adapter's memory and context need.
fn standIn() runtime.Runtime {
    var r: runtime.Runtime = .{ .lib = undefined, .api = undefined };
    inline for (@typeInfo(abi.Api).@"struct".field_names) |name| @field(r.api, name) = @ptrCast(&ok);
    r.api.hipDevicePrimaryCtxRetain = retain;
    r.api.hipStreamCreateWithFlags = createStream;
    r.api.hipStreamDestroy = destroyStream;
    r.api.hipDeviceGetName = deviceName;
    r.api.hipMemGetInfo = memInfo;
    r.api.hipMalloc = malloc;
    r.api.hipHostMalloc = hostMalloc;
    r.api.hipFree = release;
    r.api.hipHostFree = release;
    r.api.hipMemcpyHtoD = toDevice;
    r.api.hipMemcpyDtoH = toHost;
    return r;
}

const calls: offload.Calls = .{ .wait32 = waitValue, .write32 = writeValue, .arch = "gfx1030" };
const options: offload.Options = .{ .peak_bandwidth = 400_000_000_000, .reserved_bytes = 1 << 20 };

test "HipExternal: caps come from the runtime, the architecture table and the options" {
    const rt = standIn();
    const card = try offload.HipExternal.createWith(T.allocator, &rt, 0, options, calls);
    const dev = card.external();
    defer dev.deinit();
    const caps = dev.caps();
    try T.expectEqualStrings("hip", caps.backend);
    try T.expectEqualStrings("Stand-in", caps.name);
    try T.expectEqualStrings("gfx1030", caps.arch);
    try T.expectEqual(@as(u64, (1 << 30) - (1 << 20)), caps.vram_bytes);
    try T.expectEqual(@as(u64, 400_000_000_000), caps.peak_bandwidth);
    try T.expectEqual(split.device.Dtype.f16, caps.act);
}

test "HipExternal: an architecture outside the caps table is refused" {
    const rt = standIn();
    var other = calls;
    other.arch = "gfx0";
    try T.expectError(error.UnsupportedGpu, offload.HipExternal.createWith(T.allocator, &rt, 0, options, other));
}

test "HipExternal: the window protocol runs through stream wait and write" {
    try queue.start();
    defer queue.finish();
    const rt = standIn();
    const dev = (try offload.HipExternal.createWith(T.allocator, &rt, 0, options, calls)).external();
    defer dev.deinit();
    const w = try dev.allocWindow(4096);
    defer dev.freeWindow(w);
    try T.expectEqual(offload.HipExternal.window_align, w.len);
    try T.expect(@intFromPtr(w.ptr) % offload.HipExternal.window_align == 0);
    try dev.enqueueWait(w, 0, 7, 0);
    try dev.enqueueWrite(w, 64, 9);
    // the stream holds at its wait: a bounded sync times out, and the write behind it has not landed
    try T.expectError(error.Timeout, dev.sync(5_000_000));
    try T.expectEqual(@as(u32, 0), @atomicLoad(u32, w.word(64), .acquire));
    @atomicStore(u32, w.word(0), 7, .release);
    try dev.sync(2_000_000_000);
    try T.expectEqual(@as(u32, 9), @atomicLoad(u32, w.word(64), .acquire));
}

test "HipExternal: waits and writes outside the window or off a word are refused" {
    try queue.start();
    defer queue.finish();
    const rt = standIn();
    const dev = (try offload.HipExternal.createWith(T.allocator, &rt, 0, options, calls)).external();
    defer dev.deinit();
    const w = try dev.allocWindow(1);
    defer dev.freeWindow(w);
    try T.expectError(error.BadWindow, dev.enqueueWait(w, 2, 1, 0));
    try T.expectError(error.BadWindow, dev.enqueueWrite(w, w.len, 1));
    try T.expectEqual(@as(usize, 0), queue.head.load(.acquire));
}

test "HipExternal: device memory is uploaded, downloaded and freed by address" {
    const rt = standIn();
    const dev = (try offload.HipExternal.createWith(T.allocator, &rt, 0, options, calls)).external();
    defer dev.deinit();
    const p = try dev.alloc(16);
    try dev.upload(p, "0123456789abcdef");
    var back: [16]u8 = undefined;
    try dev.download(&back, p);
    try T.expectEqualStrings("0123456789abcdef", &back);
    dev.free(p);
    try T.expectError(error.Failed, dev.upload(p, "x"));
}

test "HipExternal: it queues on the stream it is given and leaves that stream to the caller" {
    try queue.start();
    defer queue.finish();
    const rt = standIn();
    var shared = try Stream.init(&rt);
    var own_options = options;
    own_options.stream = shared;
    const card = try offload.HipExternal.createWith(T.allocator, &rt, 0, own_options, calls);
    try T.expectEqual(shared.handle, card.stream.handle);
    const dev = card.external();
    const w = try dev.allocWindow(4096);
    try dev.enqueueWrite(w, 64, 5);
    try dev.sync(2_000_000_000);
    try T.expectEqual(shared.handle, queue.last_stream);
    try T.expectEqual(@as(u32, 5), @atomicLoad(u32, w.word(64), .acquire));
    dev.freeWindow(w);
    dev.deinit();
    try T.expectEqual(@as(usize, 0), queue.destroyed);
    shared.deinit();
    try T.expectEqual(@as(usize, 1), queue.destroyed);
}

test "HipExternal: a stream of its own is destroyed with it" {
    queue = .{};
    const rt = standIn();
    const dev = (try offload.HipExternal.createWith(T.allocator, &rt, 0, options, calls)).external();
    dev.deinit();
    try T.expectEqual(@as(usize, 1), queue.destroyed);
}
