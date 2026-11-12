//! A copy chunk drains the quarantined copies of its context, each by its own event or stream, never the context.
const std = @import("std");
const t = std.testing;
const source = @import("copy_source.zig");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");

const Kind = enum { stream_sync, event_sync, ctx_sync, copy };
const Fake = struct {
    var busy: [16]bool = @splat(false); // streams whose work has not finished
    var event_stream: [64]usize = @splat(0);
    var events: usize = 0;
    var log: [32]struct { kind: Kind, id: usize } = undefined;
    var logged: usize = 0;
    var failed_at_copy: usize = 0;
    var host: [8]?[]align(16) u8 = @splat(null);
    var record_fails = false;
    fn note(kind: Kind, id: usize) void {
        log[logged] = .{ .kind = kind, .id = id };
        logged += 1;
    }
    fn count(kind: Kind, id: ?usize) usize {
        var n: usize = 0;
        for (log[0..logged]) |e| n += @intFromBool(e.kind == kind and (id == null or e.id == id.?));
        return n;
    }
    fn at(kind: Kind, id: ?usize) usize {
        for (log[0..logged], 0..) |e, i| if (e.kind == kind and (id == null or e.id == id.?)) return i;
        return logged;
    }
    fn state(running: bool) abi.Result {
        return if (running) abi.error_not_ready else 0;
    }
    fn copy(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) abi.Result {
        failed_at_copy = source.failedCount();
        note(.copy, 0);
        return 0;
    }
    fn asyncCopy(dst: u64, ptr: ?*const anyopaque, len: usize, _: abi.Stream) callconv(.c) abi.Result {
        return copy(dst, ptr, len);
    }
    fn streamSync(s: abi.Stream) callconv(.c) abi.Result {
        note(.stream_sync, @intFromPtr(s));
        return state(busy[@intFromPtr(s)]);
    }
    fn ctxSync() callconv(.c) abi.Result {
        note(.ctx_sync, 0);
        return 0;
    }
    fn eventSync(e: abi.Event) callconv(.c) abi.Result {
        note(.event_sync, @intFromPtr(e));
        return state(busy[event_stream[@intFromPtr(e)]]);
    }
    fn eventQuery(e: abi.Event) callconv(.c) abi.Result {
        return state(busy[event_stream[@intFromPtr(e)]]);
    }
    fn eventCreate(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        events += 1;
        out.* = @ptrFromInt(events);
        return 0;
    }
    fn record(e: abi.Event, s: abi.Stream) callconv(.c) abi.Result {
        if (record_fails) return abi.error_not_ready;
        event_stream[@intFromPtr(e)] = @intFromPtr(s);
        return 0;
    }
    fn eventDestroy(_: abi.Event) callconv(.c) abi.Result {
        return 0;
    }
    fn hostAlloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
        const bytes = t.allocator.alignedAlloc(u8, .@"16", len) catch return abi.error_not_ready;
        for (&host) |*slot| if (slot.* == null) {
            slot.* = bytes;
            out.* = bytes.ptr;
            return 0;
        };
        unreachable;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) abi.Result {
        for (&host) |*slot| if (slot.*) |bytes| if (@intFromPtr(bytes.ptr) == @intFromPtr(ptr)) {
            t.allocator.free(bytes);
            slot.* = null;
            return 0;
        };
        unreachable;
    }
    fn init() Driver {
        busy = @splat(false);
        events = 0;
        logged = 0;
        failed_at_copy = 0;
        record_fails = false;
        var d: Driver = undefined;
        d.api.cuMemcpyHtoD_v2 = copy;
        d.api.cuMemcpyHtoDAsync_v2 = asyncCopy;
        d.api.cuStreamSynchronize = streamSync;
        d.api.cuCtxSynchronize = ctxSync;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventRecord = record;
        d.api.cuEventSynchronize = eventSync;
        d.api.cuEventQuery = eventQuery;
        d.api.cuEventDestroy_v2 = eventDestroy;
        return d;
    }
};

/// Async uploads on the streams, then a failed wait on each: the copies stay in quarantine as failed.
fn quarantine(d: *Driver, on: []const usize) !void {
    const b: memory.DeviceBuffer = .{ .d = d, .ptr = 256, .len = 16, .borrowed = true };
    const bytes: [16]u8 = @splat(0x44);
    for (on) |stream| {
        Fake.busy[stream] = true;
        try b.uploadAsync(0, &bytes, @ptrFromInt(stream));
    }
    for (on) |stream| try t.expectError(error.NotReady, source.synchronizeStream(d, @ptrFromInt(stream)));
}

fn cleanup(d: *Driver) void {
    Fake.busy = @splat(false);
    for ([_]usize{ 9, 10 }) |c| {
        source.setCurrent(@ptrFromInt(c));
        source.synchronizeContext(d) catch {};
    }
    source.setCurrent(@ptrFromInt(9));
}

test "a chunk drains each quarantined copy on its own event and leaves the context and other streams alone" {
    var d = Fake.init();
    defer cleanup(&d);
    source.setCurrent(@ptrFromInt(9));
    try quarantine(&d, &.{ 5, 7 });
    try t.expectEqual(@as(usize, 2), source.failedCount());
    Fake.busy[5] = true; // stream 5 is still running: it keeps admission closed
    Fake.busy[7] = false;
    Fake.busy[3] = true; // an unrelated stream with long work in flight
    Fake.logged = 0;
    const bytes: [16]u8 = @splat(7);
    try t.expectError(error.NotReady, source.synchronousCopy(&d, 512, &bytes, t.allocator));
    try t.expectEqual(@as(usize, 0), Fake.count(.copy, null)); // refused before the copy was issued
    try t.expectEqual(@as(usize, 1), source.failedCount()); // stream 7's copy is proven and released
    Fake.busy[5] = false;
    Fake.logged = 0;
    try source.synchronousCopy(&d, 512, &bytes, t.allocator);
    try t.expectEqual(@as(usize, 0), source.failedCount());
    try t.expectEqual(@as(usize, 0), Fake.failed_at_copy); // the copy ran only once nothing was quarantined
    try t.expectEqual(@as(usize, 0), Fake.count(.ctx_sync, null));
    try t.expectEqual(@as(usize, 0), Fake.count(.stream_sync, 3)); // the busy bystander was never waited on
    try t.expectEqual(@as(usize, 1), Fake.count(.event_sync, null)); // one pending copy, one event wait
    try t.expect(Fake.at(.event_sync, null) < Fake.at(.copy, null)); // the drain finished before the copy
}

test "a copy without an event is proven on its own stream only" {
    var d = Fake.init();
    defer cleanup(&d);
    source.setCurrent(@ptrFromInt(9));
    var stage: source.Stage = .{ .gpa = t.allocator };
    defer stage.deinit();
    @memset(try stage.take(&d, 16), 0x21);
    Fake.record_fails = true; // the event cannot be recorded, so only the stream can prove the copy
    Fake.busy[2] = true;
    const dst: memory.DeviceBuffer = .{ .d = &d, .ptr = 256, .len = 16, .borrowed = true };
    try t.expectError(error.NotReady, stage.copyTo(dst, 0, .{ .d = &d, .handle = @ptrFromInt(2) }));
    stage.deinit();
    try t.expectEqual(@as(usize, 1), source.failedCount());
    Fake.busy[2] = false;
    Fake.busy[4] = true; // another stream's work must not be waited for
    Fake.logged = 0;
    const bytes: [16]u8 = @splat(9);
    try source.synchronousCopy(&d, 512, &bytes, t.allocator);
    try t.expectEqual(@as(usize, 0), source.failedCount());
    try t.expectEqual(@as(usize, 1), Fake.count(.stream_sync, 2));
    try t.expectEqual(@as(usize, 0), Fake.count(.stream_sync, 4));
    try t.expectEqual(@as(usize, 0), Fake.count(.ctx_sync, null));
    try t.expect(Fake.at(.stream_sync, 2) < Fake.at(.copy, null));
}

test "a chunk never waits for the quarantine of another context" {
    var d = Fake.init();
    defer cleanup(&d);
    source.setCurrent(@ptrFromInt(10));
    try quarantine(&d, &.{6});
    source.setCurrent(@ptrFromInt(9));
    Fake.logged = 0;
    const bytes: [16]u8 = @splat(5);
    try source.synchronousCopy(&d, 512, &bytes, t.allocator);
    try t.expectEqual(@as(usize, 0), Fake.count(.event_sync, null));
    try t.expectEqual(@as(usize, 0), Fake.count(.ctx_sync, null));
    try t.expectEqual(@as(usize, 1), Fake.count(.copy, null));
    source.setCurrent(@ptrFromInt(10));
    try t.expectEqual(@as(usize, 1), source.failedCount()); // still quarantined for its own context
}

test "an async chunk drains the same way" {
    var d = Fake.init();
    defer cleanup(&d);
    source.setCurrent(@ptrFromInt(9));
    try quarantine(&d, &.{5});
    Fake.busy[5] = false;
    Fake.busy[8] = true;
    Fake.logged = 0;
    const b: memory.DeviceBuffer = .{ .d = &d, .ptr = 256, .len = 16, .borrowed = true };
    const bytes: [16]u8 = @splat(6);
    try b.uploadAsync(0, &bytes, @ptrFromInt(8));
    try t.expectEqual(@as(usize, 0), source.failedCount());
    try t.expectEqual(@as(usize, 0), Fake.count(.ctx_sync, null));
    try t.expectEqual(@as(usize, 0), Fake.count(.stream_sync, 8));
    try t.expectEqual(@as(usize, 1), Fake.count(.copy, null));
}
