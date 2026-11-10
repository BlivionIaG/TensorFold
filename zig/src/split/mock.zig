//! A CPU stand-in for an external GPU: one thread runs the device's in-order stream, windows are anonymous page-aligned
//! memory the host (and Metal) can wrap, and waits and writes obey the same rules a queue-side wait/write value has.
//! Tests and the offline pipeline use it; it is also the reference for what a real adapter must provide.
const std = @import("std");
const device = @import("device.zig");

const Op = union(enum) {
    work: device.Work,
    wait: struct { word: *u32, value: u32, timeout_ns: u64 },
    write: struct { word: *u32, value: u32 },
    fence: *std.atomic.Value(u64),
};

pub const Options = struct {
    name: []const u8 = "mock",
    vram_bytes: u64 = 8 << 30,
    peak_bandwidth: u64 = 410_000_000_000,
    act: device.Dtype = .f16,
    /// Delay between a stream write being issued and the host seeing it, standing in for the link.
    write_delay_ns: u64 = 0,
    /// Delay before a satisfied wait releases the stream, standing in for the link and the poll interval.
    wait_delay_ns: u64 = 0,
};

const queue_len = 1024;

pub const Mock = struct {
    gpa: std.mem.Allocator,
    options: Options,
    thread: ?std.Thread = null,
    lock: std.atomic.Mutex = .unlocked,
    ring: [queue_len]Op = undefined,
    head: usize = 0,
    tail: usize = 0,
    stop: std.atomic.Value(bool) = .init(false),
    lost: std.atomic.Value(bool) = .init(false),
    issued: std.atomic.Value(u64) = .init(0),
    done: std.atomic.Value(u64) = .init(0),
    failure: std.atomic.Value(u32) = .init(0),
    allocs: std.AutoHashMapUnmanaged(u64, []u8) = .empty,
    windows: std.AutoHashMapUnmanaged(u64, usize) = .empty,

    pub fn create(gpa: std.mem.Allocator, options: Options) !*Mock {
        const m = try gpa.create(Mock);
        m.* = .{ .gpa = gpa, .options = options };
        m.thread = try std.Thread.spawn(.{}, run, .{m});
        return m;
    }

    pub fn external(m: *Mock) device.ExternalDevice {
        return .{ .ptr = m, .vtable = &vtable };
    }

    /// Simulate the cable being pulled: queued and later operations fail, waits stop waiting.
    pub fn kill(m: *Mock) void {
        m.lost.store(true, .release);
    }

    fn push(m: *Mock, op: Op) device.Error!void {
        if (m.lost.load(.acquire)) return error.DeviceLost;
        while (true) {
            while (!m.lock.tryLock()) std.atomic.spinLoopHint();
            if (m.head - m.tail < queue_len) {
                m.ring[m.head % queue_len] = op;
                m.head += 1;
                _ = m.issued.fetchAdd(1, .release);
                m.lock.unlock();
                return;
            }
            m.lock.unlock();
            device.pause(10_000);
        }
    }

    fn pop(m: *Mock) ?Op {
        while (!m.lock.tryLock()) std.atomic.spinLoopHint();
        defer m.lock.unlock();
        if (m.tail == m.head) return null;
        const op = m.ring[m.tail % queue_len];
        m.tail += 1;
        return op;
    }

    fn fail(m: *Mock, err: device.Error) void {
        const code: u32 = switch (err) {
            error.Timeout => 1,
            error.DeviceLost => 2,
            else => 3,
        };
        _ = m.failure.cmpxchgStrong(0, code, .acq_rel, .acquire);
    }

    fn run(m: *Mock) void {
        while (true) {
            const op = m.pop() orelse {
                if (m.stop.load(.acquire)) return;
                std.atomic.spinLoopHint();
                device.pause(2_000);
                continue;
            };
            m.exec(op);
            _ = m.done.fetchAdd(1, .release);
        }
    }

    fn exec(m: *Mock, op: Op) void {
        if (m.lost.load(.acquire)) {
            m.fail(error.DeviceLost);
            if (op == .fence) op.fence.store(1, .release);
            return;
        }
        switch (op) {
            .work => |w| w.run(w.ctx) catch m.fail(error.Failed),
            .wait => |w| {
                const start = device.nowNs();
                while (@atomicLoad(u32, w.word, .acquire) != w.value) {
                    if (m.lost.load(.acquire)) return m.fail(error.DeviceLost);
                    if (w.timeout_ns != 0 and device.nowNs() - start >= w.timeout_ns) return m.fail(error.Timeout);
                    std.atomic.spinLoopHint();
                }
                if (m.options.wait_delay_ns != 0) device.pause(m.options.wait_delay_ns);
            },
            .write => |w| {
                if (m.options.write_delay_ns != 0) device.pause(m.options.write_delay_ns);
                @atomicStore(u32, w.word, w.value, .release);
            },
            .fence => |f| f.store(1, .release),
        }
    }

    fn self(ptr: *anyopaque) *Mock {
        return @ptrCast(@alignCast(ptr));
    }

    fn capsFn(ptr: *anyopaque) device.Caps {
        const m = self(ptr);
        return .{
            .name = m.options.name,
            .arch = "mock",
            .vram_bytes = m.options.vram_bytes,
            .peak_bandwidth = m.options.peak_bandwidth,
            .act = m.options.act,
            .launch_ns = 0,
            .max_windows = 128,
        };
    }

    fn allocFn(ptr: *anyopaque, len: usize) device.Error!device.DevPtr {
        const m = self(ptr);
        if (len == 0) return error.OutOfMemory;
        const mem = m.gpa.alloc(u8, len) catch return error.OutOfMemory;
        errdefer m.gpa.free(mem);
        @memset(mem, 0);
        m.allocs.put(m.gpa, @intFromPtr(mem.ptr), mem) catch return error.OutOfMemory;
        return @intFromPtr(mem.ptr);
    }

    fn freeFn(ptr: *anyopaque, p: device.DevPtr) void {
        const m = self(ptr);
        if (m.allocs.fetchRemove(p)) |kv| m.gpa.free(kv.value);
    }

    fn upload(ptr: *anyopaque, dst: device.DevPtr, src: []const u8) device.Error!void {
        const mem = self(ptr).allocs.get(dst) orelse return error.Failed;
        if (src.len > mem.len) return error.Failed;
        @memcpy(mem[0..src.len], src);
    }

    fn download(ptr: *anyopaque, dst: []u8, src: device.DevPtr) device.Error!void {
        const mem = self(ptr).allocs.get(src) orelse return error.Failed;
        if (dst.len > mem.len) return error.Failed;
        @memcpy(dst, mem[0..dst.len]);
    }

    fn allocWindow(ptr: *anyopaque, len: usize) device.Error!device.Window {
        const m = self(ptr);
        const page: usize = 16 * 1024;
        const n = (len + page - 1) / page * page;
        const mem = std.posix.mmap(null, n, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0) catch return error.OutOfMemory;
        errdefer std.posix.munmap(mem);
        m.windows.put(m.gpa, @intFromPtr(mem.ptr), n) catch return error.OutOfMemory;
        return .{ .ptr = mem.ptr, .len = n, .id = @intCast(m.windows.count()) };
    }

    fn freeWindow(ptr: *anyopaque, w: device.Window) void {
        const m = self(ptr);
        // a queued operation may still hold the window's words: unmapping waits for the stream like a runtime's free does
        syncFn(ptr, 5_000_000_000) catch {};
        if (m.windows.fetchRemove(@intFromPtr(w.ptr))) |kv| {
            const mem: []align(16 * 1024) u8 = @alignCast(w.ptr[0..kv.value]);
            std.posix.munmap(mem);
        }
    }

    fn enqueue(ptr: *anyopaque, work: device.Work) device.Error!void {
        return self(ptr).push(.{ .work = work });
    }

    fn enqueueWait(ptr: *anyopaque, w: device.Window, offset: usize, value: u32, timeout_ns: u64) device.Error!void {
        if (offset % 4 != 0 or offset + 4 > w.len) return error.BadWindow;
        return self(ptr).push(.{ .wait = .{ .word = w.word(offset), .value = value, .timeout_ns = timeout_ns } });
    }

    fn enqueueWrite(ptr: *anyopaque, w: device.Window, offset: usize, value: u32) device.Error!void {
        if (offset % 4 != 0 or offset + 4 > w.len) return error.BadWindow;
        return self(ptr).push(.{ .write = .{ .word = w.word(offset), .value = value } });
    }

    fn syncFn(ptr: *anyopaque, timeout_ns: u64) device.Error!void {
        const m = self(ptr);
        const start = device.nowNs();
        while (m.done.load(.acquire) != m.issued.load(.acquire)) {
            if (timeout_ns != 0 and device.nowNs() - start >= timeout_ns) return error.Timeout;
            std.atomic.spinLoopHint();
            device.pause(2_000);
        }
        const f = m.failure.swap(0, .acq_rel);
        return switch (f) {
            0 => {},
            1 => error.Timeout,
            2 => error.DeviceLost,
            else => error.Failed,
        };
    }

    fn timestamp(_: *anyopaque) device.Error!u64 {
        return device.nowNs();
    }

    fn deinitFn(ptr: *anyopaque) void {
        const m = self(ptr);
        m.stop.store(true, .release);
        if (m.thread) |t| t.join();
        var it = m.allocs.valueIterator();
        while (it.next()) |mem| m.gpa.free(mem.*);
        m.allocs.deinit(m.gpa);
        var wit = m.windows.iterator();
        while (wit.next()) |e| {
            const mem: []align(16 * 1024) u8 = @alignCast(@as([*]u8, @ptrFromInt(e.key_ptr.*))[0..e.value_ptr.*]);
            std.posix.munmap(mem);
        }
        m.windows.deinit(m.gpa);
        m.gpa.destroy(m);
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

test "stream operations run in order and a write follows earlier work" {
    const m = try Mock.create(std.testing.allocator, .{});
    const d = m.external();
    defer d.deinit();
    const w = try d.allocWindow(4096);
    try std.testing.expect(w.len % (16 * 1024) == 0 and @intFromPtr(w.ptr) % (16 * 1024) == 0);
    const Ctx = struct {
        var log: [4]u8 = undefined;
        var n: usize = 0;
        fn step(c: *anyopaque) anyerror!void {
            const tag: *u8 = @ptrCast(c);
            log[n] = tag.*;
            n += 1;
        }
    };
    var a: u8 = 'a';
    var b: u8 = 'b';
    try d.enqueue(.{ .ctx = &a, .run = Ctx.step });
    try d.enqueue(.{ .ctx = &b, .run = Ctx.step });
    try d.enqueueWrite(w, 0, 7);
    try d.sync(1_000_000_000);
    try std.testing.expectEqualSlices(u8, "ab", Ctx.log[0..2]);
    try std.testing.expectEqual(@as(u32, 7), @atomicLoad(u32, w.word(0), .acquire));
}

test "a wait stalls the stream until the host stores the value" {
    const m = try Mock.create(std.testing.allocator, .{});
    const d = m.external();
    defer d.deinit();
    const w = try d.allocWindow(4096);
    try d.enqueueWait(w, 64, 5, 0);
    try d.enqueueWrite(w, 128, 9);
    device.pause(2_000_000);
    try std.testing.expectEqual(@as(u32, 0), @atomicLoad(u32, w.word(128), .acquire));
    try std.testing.expectError(error.Timeout, d.sync(1_000_000));
    @atomicStore(u32, w.word(64), 5, .release);
    try d.sync(1_000_000_000);
    try std.testing.expectEqual(@as(u32, 9), @atomicLoad(u32, w.word(128), .acquire));
}

test "a wait that is never satisfied times out and the loss of the device is reported" {
    const m = try Mock.create(std.testing.allocator, .{});
    const d = m.external();
    defer d.deinit();
    const w = try d.allocWindow(4096);
    try d.enqueueWait(w, 0, 1, 2_000_000);
    try std.testing.expectError(error.Timeout, d.sync(1_000_000_000));
    try d.enqueueWait(w, 0, 1, 0);
    m.kill();
    try std.testing.expectError(error.DeviceLost, d.sync(1_000_000_000));
    try std.testing.expectError(error.DeviceLost, d.enqueueWrite(w, 0, 1));
}

test "device memory round trips and out-of-range copies are refused" {
    const m = try Mock.create(std.testing.allocator, .{});
    const d = m.external();
    defer d.deinit();
    const p = try d.alloc(64);
    try d.upload(p, "tensorfold");
    var out: [10]u8 = undefined;
    try d.download(&out, p);
    try std.testing.expectEqualSlices(u8, "tensorfold", &out);
    var big: [65]u8 = undefined;
    try std.testing.expectError(error.Failed, d.upload(p, &big));
    d.free(p);
}
