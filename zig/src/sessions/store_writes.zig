//! Write-through saves behind the engine thread: the entry stays in RAM and the save queue's worker writes its file.

const std = @import("std");
const Io = std.Io;
const pool_mod = @import("pool.zig");
const sq = @import("save_queue.zig");
const jobs = @import("jobs.zig");
const store_mod = @import("store.zig");
const Store = store_mod.Store;
const Job = jobs.Job;

/// Saves queued at once at most: the device's fence ring (kv/device.zig) is this long.
pub const max_entries = 8;

/// The queue a policy asks for, its worker started.
pub fn open(gpa: std.mem.Allocator, io: Io, policy: store_mod.Policy) !*sq.SaveQueue {
    const q = try sq.SaveQueue.create(gpa, io, .{ .entries = max_entries, .bytes = policy.save_queue_bytes, .name = "sessions: write-through saves" });
    errdefer q.destroy();
    try q.start();
    return q;
}

/// Closes the queue (it writes what it holds); unsettled saves are freed and their files found at the next boot.
pub fn close(s: *Store) void {
    const q = s.writes orelse return;
    q.close();
    while (q.reap()) |sj| from(sj).deinit(s.gpa);
    q.destroy();
    s.writes = null;
}

fn from(sj: *sq.Job) *Job {
    return @fieldParentPtr("save", sj);
}

/// What a write of entry `id` charges the queue, alike on every rank: its ids, its pages whole, its bounded state.
fn bytesOf(s: *Store, id: u32) u64 {
    const e = s.index.get(id);
    var n: u64 = @as(u64, e.pos) * 4;
    if (e.extra.bounded) |b| n += b.logical_bytes orelse b.vtable.nbytes(b.ptr);
    const pages = pool_mod.pagesFor(e.pos, s.pool.page);
    for (s.pool.layout.families) |f| n += @as(u64, pages) * f.pageBytes(s.pool.page);
    return n;
}

/// A saved entry's file now, the entry kept in RAM; a failed write or a full save queue leaves it RAM-only, counted.
pub fn writeThrough(s: *Store, id: u32) void {
    const d = s.disk orelse return;
    const e = s.index.get(id);
    if (!e.live or !e.ram or e.extra.job != null or !d.accepts(e.pos)) return;
    if (s.writes) |q| {
        if (!q.admit(bytesOf(s, id))) return;
        const j = s.beginFile(id, true, true) catch {
            s.stats.write_fails += 1;
            return;
        };
        if (j != null) s.stats.write_throughs += 1;
        return;
    }
    const j = s.beginWrite(id) catch {
        s.stats.write_fails += 1;
        return;
    } orelse return;
    s.stats.write_throughs += 1;
    if (!s.policy.async_io) _ = s.settle(j) catch {
        s.stats.write_fails += 1;
    };
}

/// Hands a built job to the queue; its pin (`extra.job`) holds the entry in RAM until the job is settled.
pub fn submit(s: *Store, j: *Job, id: u32) !void {
    j.disk = s.disk.?;
    j.io = s.io;
    j.save = .{ .run = runQueued, .key = std.mem.readInt(u64, j.src.key[0..8], .little), .bytes = bytesOf(s, id) };
    if (s.writes.?.submit(&j.save) == .dropped) return error.SaveQueueClosed;
}

fn runQueued(sj: *sq.Job) anyerror!void {
    const j = from(sj);
    j.run(j.disk, j.io);
    if (j.err) |e| return e;
}

/// How many of the oldest write-throughs have finished: the leader's count, taken without waiting.
pub fn finished(s: *Store) u32 {
    const q = s.writes orelse return 0;
    return @intCast(q.ready());
}

/// Settles the `n` oldest write-throughs (file registered, pin gone, budget trimmed); a rank waits for its own write.
pub fn settleOldest(s: *Store, n: u32) void {
    const q = s.writes orelse return;
    var i: u32 = 0;
    while (i < n) : (i += 1) settleOne(s, q.reapWait() orelse return);
}

/// Every write-through settled, oldest first (a drain or a stop).
pub fn settleEvery(s: *Store) void {
    const q = s.writes orelse return;
    while (q.reapWait()) |sj| settleOne(s, sj);
}

/// The saves through `j` settled, oldest first: an entry may not give its pages back while its file is being written.
pub fn settleThrough(s: *Store, j: *Job) void {
    const q = s.writes.?;
    const target = &j.save;
    while (q.reapWait()) |sj| {
        settleOne(s, sj);
        if (sj == target) return;
    }
}

fn settleOne(s: *Store, sj: *sq.Job) void {
    if (s.settle(from(sj))) |_| {} else |_| s.stats.write_fails += 1;
}

/// The queue's counters (null: writes are inline).
pub fn counters(s: *Store) ?sq.Counters {
    const q = s.writes orelse return null;
    return q.stats();
}
