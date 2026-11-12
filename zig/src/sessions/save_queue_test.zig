//! The save queue's contract on a host thread: order, bounds, drops, read-your-writes, flush, shutdown, failures.

const std = @import("std");
const testing = std.testing;
const sq = @import("save_queue.zig");
const Job = sq.Job;
const SaveQueue = sq.SaveQueue;

/// One save: it logs its id when it runs, optionally waits on a gate first and optionally fails.
const Save = struct {
    job: Job,
    id: u32,
    log: *std.ArrayList(u32),
    gate: ?*std.Io.Event = null,
    fail: ?anyerror = null,

    fn make(id: u32, key: u64, bytes: u64, log: *std.ArrayList(u32)) Save {
        return .{ .job = .{ .run = run, .key = key, .bytes = bytes }, .id = id, .log = log };
    }

    fn run(j: *Job) anyerror!void {
        const s: *Save = @fieldParentPtr("job", j);
        if (s.gate) |g| try g.wait(testing.io);
        if (s.fail) |e| return e;
        s.log.appendAssumeCapacity(s.id);
    }
};

fn queue(opts: sq.Options) !*SaveQueue {
    const q = try SaveQueue.create(testing.allocator, testing.io, opts);
    errdefer q.destroy();
    try q.start();
    return q;
}

test "save queue: saves run on the worker in submission order, strict per key" {
    const q = try queue(.{});
    defer q.destroy();
    var log: std.ArrayList(u32) = .empty;
    defer log.deinit(testing.allocator);
    try log.ensureTotalCapacity(testing.allocator, 8);
    var saves: [5]Save = undefined;
    for (&saves, 0..) |*s, i| s.* = Save.make(@intCast(i), if (i % 2 == 0) 7 else 8, 10, &log);
    for (&saves) |*s| try testing.expectEqual(sq.Verdict.queued, q.submit(&s.job));
    q.flush();
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4 }, log.items);
    // the owner takes them in the same order
    for (&saves) |*s| try testing.expectEqual(&s.job, q.reapWait().?);
    try testing.expectEqual(@as(?*Job, null), q.reapWait());
    const c = q.stats();
    try testing.expectEqual(@as(u64, 5), c.queued);
    try testing.expectEqual(@as(u64, 5), c.written);
    try testing.expectEqual(@as(u64, 0), c.dropped + c.failed);
    try testing.expectEqual(@as(u64, 0), q.heldBytes());
}

test "save queue: a full queue drops and counts, never blocks, and the dropped save never runs" {
    const q = try queue(.{ .entries = 2 });
    defer q.destroy();
    var log: std.ArrayList(u32) = .empty;
    defer log.deinit(testing.allocator);
    try log.ensureTotalCapacity(testing.allocator, 8);
    var gate: std.Io.Event = .unset;
    var saves: [4]Save = undefined;
    for (&saves, 0..) |*s, i| s.* = Save.make(@intCast(i), i, 1, &log);
    saves[0].gate = &gate; // the worker holds the first; the next one waits behind it
    try testing.expectEqual(sq.Verdict.queued, q.submit(&saves[0].job));
    try testing.expectEqual(sq.Verdict.queued, q.submit(&saves[1].job));
    try testing.expectEqual(sq.Verdict.dropped, q.submit(&saves[2].job));
    try testing.expectEqual(sq.Verdict.dropped, q.submit(&saves[3].job));
    try testing.expectEqual(@as(usize, 2), q.pending());
    gate.set(testing.io);
    q.flush();
    try testing.expectEqualSlices(u32, &.{ 0, 1 }, log.items); // the dropped two never ran
    const c = q.stats();
    try testing.expectEqual(@as(u64, 2), c.queued);
    try testing.expectEqual(@as(u64, 2), c.dropped);
    try testing.expectEqual(sq.State.queued, saves[2].job.state.load(.acquire)); // never entered
    // room again once the owner takes what finished
    while (q.reap()) |_| {}
    try testing.expectEqual(sq.Verdict.queued, q.submit(&saves[2].job));
    q.flush();
}

test "save queue: bytes bound the backlog, and a save alone is accepted whatever its size" {
    const q = try queue(.{ .entries = 8, .bytes = 100 });
    defer q.destroy();
    var log: std.ArrayList(u32) = .empty;
    defer log.deinit(testing.allocator);
    try log.ensureTotalCapacity(testing.allocator, 8);
    var gate: std.Io.Event = .unset;
    var saves: [4]Save = undefined;
    for (&saves, 0..) |*s, i| s.* = Save.make(@intCast(i), i, 60, &log);
    saves[0].job.bytes = 500; // alone in the queue: over the bound, taken
    saves[0].gate = &gate;
    try testing.expectEqual(sq.Verdict.queued, q.submit(&saves[0].job));
    try testing.expectEqual(sq.Verdict.dropped, q.submit(&saves[1].job)); // 500 + 60 > 100
    try testing.expectEqual(@as(u64, 500), q.heldBytes());
    gate.set(testing.io);
    q.flush();
    while (q.reap()) |_| {}
    try testing.expectEqual(sq.Verdict.queued, q.submit(&saves[1].job)); // 60
    try testing.expectEqual(sq.Verdict.dropped, q.submit(&saves[2].job)); // 60 + 60 > 100
    try testing.expectEqual(@as(u64, 60), q.heldBytes());
    q.flush();
    try testing.expectEqual(@as(u64, 2), q.stats().dropped);
}

test "save queue: a read finds the save that is queued or running, newest first, until the owner takes it" {
    const q = try queue(.{});
    defer q.destroy();
    var log: std.ArrayList(u32) = .empty;
    defer log.deinit(testing.allocator);
    try log.ensureTotalCapacity(testing.allocator, 8);
    var gate: std.Io.Event = .unset;
    var a = Save.make(0, 42, 1, &log);
    var b = Save.make(1, 42, 1, &log);
    var other = Save.make(2, 43, 1, &log);
    a.gate = &gate;
    _ = q.submit(&a.job);
    _ = q.submit(&b.job);
    _ = q.submit(&other.job);
    try testing.expectEqual(&b.job, q.find(42).?); // the newest save of the key
    try testing.expectEqual(&other.job, q.find(43).?);
    try testing.expectEqual(@as(?*Job, null), q.find(44));
    try testing.expect(!a.job.finished()); // its bytes are not on disk yet: a reader waits for this save only
    gate.set(testing.io);
    q.wait(&b.job);
    try testing.expect(b.job.finished() and a.job.finished());
    try testing.expectEqual(&b.job, q.find(42).?); // finished but not settled: still visible
    q.flush();
    while (q.reap()) |_| {}
    try testing.expectEqual(@as(?*Job, null), q.find(42));
}

test "save queue: the owner settles the finished prefix without waiting" {
    const q = try queue(.{});
    defer q.destroy();
    var log: std.ArrayList(u32) = .empty;
    defer log.deinit(testing.allocator);
    try log.ensureTotalCapacity(testing.allocator, 8);
    var gate: std.Io.Event = .unset;
    var first = Save.make(0, 1, 1, &log);
    var second = Save.make(1, 2, 1, &log);
    var third = Save.make(2, 3, 1, &log);
    second.gate = &gate;
    _ = q.submit(&first.job);
    _ = q.submit(&second.job);
    _ = q.submit(&third.job);
    q.wait(&first.job);
    try testing.expectEqual(@as(usize, 1), q.ready());
    try testing.expectEqual(&first.job, q.reap().?);
    try testing.expectEqual(@as(?*Job, null), q.reap()); // second still holds its gate; third waits behind it
    try testing.expectEqual(@as(usize, 0), q.ready());
    try testing.io.sleep(.fromMilliseconds(3), .awake);
    gate.set(testing.io);
    q.flush();
    try testing.expectEqual(@as(usize, 2), q.ready());
    try testing.expectEqual(&second.job, q.reap().?);
    try testing.expectEqual(&third.job, q.reap().?);
    try testing.expect(q.stats().max_queue_ns >= 2 * std.time.ns_per_ms); // the third waited behind the gated one
}

test "save queue: shutdown while busy runs what is queued, joins, and leaves the saves to the owner" {
    const q = try queue(.{});
    defer q.destroy();
    var log: std.ArrayList(u32) = .empty;
    defer log.deinit(testing.allocator);
    try log.ensureTotalCapacity(testing.allocator, 8);
    var gate: std.Io.Event = .unset;
    var saves: [3]Save = undefined;
    for (&saves, 0..) |*s, i| s.* = Save.make(@intCast(i), i, 1, &log);
    saves[0].gate = &gate;
    for (&saves) |*s| _ = q.submit(&s.job);
    const opener = try std.Thread.spawn(.{}, struct {
        fn go(g: *std.Io.Event) void {
            testing.io.sleep(.fromMilliseconds(20), .awake) catch {};
            g.set(testing.io);
        }
    }.go, .{&gate});
    q.close(); // returns only once the busy save and the two behind it ran
    opener.join();
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, log.items);
    try testing.expectEqual(@as(u64, 3), q.stats().written);
    for (&saves) |*s| try testing.expect(s.job.finished());
    // closed: nothing more is admitted, and the owner still takes the three
    var late = Save.make(9, 9, 1, &log);
    try testing.expectEqual(sq.Verdict.dropped, q.submit(&late.job));
    for (&saves) |*s| try testing.expectEqual(&s.job, q.reapWait().?);
}

test "save queue: a failed write is counted, its error kept, only the first one logged" {
    const q = try queue(.{});
    defer q.destroy();
    var log: std.ArrayList(u32) = .empty;
    defer log.deinit(testing.allocator);
    try log.ensureTotalCapacity(testing.allocator, 8);
    var bad = Save.make(0, 1, 1, &log);
    var worse = Save.make(1, 2, 1, &log);
    var good = Save.make(2, 3, 1, &log);
    bad.fail = error.DiskFull;
    worse.fail = error.IoError;
    _ = q.submit(&bad.job);
    _ = q.submit(&worse.job);
    _ = q.submit(&good.job);
    q.flush();
    const c = q.stats();
    try testing.expectEqual(@as(u64, 2), c.failed);
    try testing.expectEqual(@as(u64, 1), c.written);
    try testing.expectEqual(error.DiskFull, q.first_error.?);
    try testing.expectEqual(sq.State.failed, bad.job.state.load(.acquire));
    try testing.expectEqual(error.IoError, worse.job.err.?);
    try testing.expectEqual(@as(?anyerror, null), good.job.err);
    try testing.expectEqualSlices(u32, &.{2}, log.items); // a failure does not stop the saves behind it
    while (q.reap()) |_| {}
}

test "save queue: a queue never started drops everything and costs nothing" {
    const q = try SaveQueue.create(testing.allocator, testing.io, .{});
    defer q.destroy();
    var log: std.ArrayList(u32) = .empty;
    defer log.deinit(testing.allocator);
    var s = Save.make(0, 1, 1, &log);
    try testing.expectEqual(sq.Verdict.dropped, q.submit(&s.job));
    try testing.expectEqual(@as(u64, 1), q.stats().dropped);
    try testing.expectEqual(@as(?*Job, null), q.reapWait());
}
