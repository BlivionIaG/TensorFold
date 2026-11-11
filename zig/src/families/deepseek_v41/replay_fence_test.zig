//! Host tests of the exact replay's end fence on a timeline driver: one stream's events, the host's clock apart.
const std = @import("std");
const testing = std.testing;
const cuda = @import("cuda");
const abi = cuda.abi;
const rf = @import("replay_fence.zig");

/// One stream's queue and the host's clock: an item ends at max(stream, host) + its ms; an event reads that time.
const Fake = struct {
    const Op = enum { record_start, record_stop, launch, query, sync };
    var ops: [64]Op = undefined;
    var n: usize = 0;
    var host: f64 = 0;
    var queue: f64 = 0;
    var times: [2]f64 = @splat(-1);
    var made: usize = 0;
    var gone: usize = 0;

    fn reset() void {
        n = 0;
        host = 0;
        queue = 0;
        times = @splat(-1);
        made = 0;
        gone = 0;
    }

    fn note(op: Op) void {
        ops[n] = op;
        n += 1;
    }

    fn at(e: abi.Event) usize {
        return @intFromPtr(e) - 1;
    }

    /// A program of `ms` queued behind the stream's work.
    fn launch(ms: f64) void {
        note(.launch);
        queue = @max(queue, host) + ms;
    }

    fn clock() f64 {
        return host;
    }

    fn create(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        made += 1;
        out.* = @ptrFromInt(made);
        return abi.success;
    }

    fn destroy(_: abi.Event) callconv(.c) abi.Result {
        gone += 1;
        return abi.success;
    }

    fn record(e: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        note(if (at(e) == 0) .record_start else .record_stop);
        times[at(e)] = @max(queue, host);
        return abi.success;
    }

    fn query(e: abi.Event) callconv(.c) abi.Result {
        note(.query);
        return if (host >= times[at(e)]) abi.success else abi.error_not_ready;
    }

    fn sync(e: abi.Event) callconv(.c) abi.Result {
        note(.sync);
        host = @max(host, times[at(e)]);
        return abi.success;
    }

    fn elapsed(ms: *f32, s: abi.Event, e: abi.Event) callconv(.c) abi.Result {
        ms.* = @floatCast(times[at(e)] - times[at(s)]);
        return abi.success;
    }

    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuEventCreate = create;
        d.api.cuEventDestroy_v2 = destroy;
        d.api.cuEventRecord = record;
        d.api.cuEventQuery = query;
        d.api.cuEventSynchronize = sync;
        d.api.cuEventElapsedTime = elapsed;
        return d;
    }
};

const replay: rf.Replay = .{ .end = 11292, .rows = 2541, .programs = 5, .group = 4, .layer_rows = 30500, .issue_ms = 5 };

/// A replay of five 100 ms programs issued 1 ms apart: the fence armed before the first, marked after the last.
fn issue(fence: *rf.Fence, d: *const cuda.Driver) !void {
    const s = cuda.Stream.borrowed(d, null);
    try fence.arm(d, s);
    for (0..5) |_| {
        Fake.launch(100);
        Fake.host += 1;
    }
    try fence.mark(s, replay);
}

test "the fence issues its start before every program and its stop after, and the host does not wait to issue them" {
    Fake.reset();
    var d = Fake.driver();
    var fence: rf.Fence = .{};
    defer fence.deinit();
    try issue(&fence, &d);
    const want = [_]Fake.Op{.record_start} ++ @as([5]Fake.Op, @splat(.launch)) ++ [_]Fake.Op{.record_stop};
    try testing.expectEqualSlices(Fake.Op, &want, Fake.ops[0..Fake.n]);
    try testing.expect(fence.pending != null);
    try testing.expectEqual(@as(usize, 2), Fake.made);
}

test "a polled fence reports nothing until the stop event ran, then the replay's GPU time, all of it overlapped" {
    Fake.reset();
    var d = Fake.driver();
    var fence: rf.Fence = .{};
    defer fence.deinit();
    try issue(&fence, &d);
    try testing.expect(try fence.settle(false, Fake.clock) == null);
    Fake.host = 300;
    try testing.expect(try fence.settle(false, Fake.clock) == null);
    try testing.expect(fence.pending != null);
    Fake.host = 600;
    const r = (try fence.settle(false, Fake.clock)).?;
    try testing.expectEqual(@as(f64, 500), r.gpu_ms);
    try testing.expectEqual(@as(f64, 0), r.waited_ms);
    try testing.expectEqual(@as(f64, 500), r.overlapped_ms);
    try testing.expectEqual(replay, r.replay);
    try testing.expect(fence.pending == null);
    try testing.expect(try fence.settle(false, Fake.clock) == null);
    for (Fake.ops[0..Fake.n]) |op| try testing.expect(op != .sync);
}

test "the first reader waits for the stop event: the GPU time the host spent elsewhere is the overlap" {
    Fake.reset();
    var d = Fake.driver();
    var fence: rf.Fence = .{};
    defer fence.deinit();
    try issue(&fence, &d);
    Fake.host = 200; // the host's other work until the reader
    const r = (try fence.settle(true, Fake.clock)).?;
    try testing.expectEqual(@as(f64, 500), r.gpu_ms);
    try testing.expectEqual(@as(f64, 300), r.waited_ms);
    try testing.expectEqual(@as(f64, 200), r.overlapped_ms);
    try testing.expectEqual(@as(f64, 500), Fake.host);
    try testing.expectEqual(Fake.Op.sync, Fake.ops[Fake.n - 1]);
}

test "TF_DSV41_REPLAY_OVERLAP=0 waits at the replay's end: only the issue time overlaps" {
    Fake.reset();
    var d = Fake.driver();
    var fence: rf.Fence = .{};
    defer fence.deinit();
    try issue(&fence, &d);
    const r = (try fence.settle(true, Fake.clock)).?;
    try testing.expectEqual(@as(f64, 495), r.waited_ms);
    try testing.expectEqual(@as(f64, 5), r.overlapped_ms);
}

test "a settled fence arms again on the same two events, and deinit frees them" {
    Fake.reset();
    var d = Fake.driver();
    var fence: rf.Fence = .{};
    try issue(&fence, &d);
    _ = try fence.settle(true, Fake.clock);
    Fake.queue = 0;
    Fake.host = 0;
    try issue(&fence, &d);
    const r = (try fence.settle(true, Fake.clock)).?;
    try testing.expectEqual(@as(f64, 500), r.gpu_ms);
    try testing.expectEqual(@as(usize, 2), Fake.made);
    fence.deinit();
    try testing.expectEqual(@as(usize, 2), Fake.gone);
    try testing.expect(fence.start == null and fence.pending == null);
}
