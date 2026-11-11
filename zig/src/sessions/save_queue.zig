//! A bounded queue of saves behind the engine thread: one worker runs them in order; a full queue drops, never blocks.

const std = @import("std");
const Io = std.Io;

pub const State = enum(u8) { queued, running, done, failed };

/// A write the worker runs. Embed it in the owner's job and recover the job with @fieldParentPtr.
pub const Job = struct {
    /// Runs on the worker thread, once, in submission order.
    run: *const fn (job: *Job) anyerror!void,
    /// What a read that must see this write looks up by (an entry's digest).
    key: u64,
    /// The charge against the byte bound.
    bytes: u64,
    state: std.atomic.Value(State) = .init(.queued),
    /// Set before `state` leaves `running`: read it once `finished` is true.
    err: ?anyerror = null,
    submitted_ns: i96 = 0,
    started_ns: i96 = 0,
    finished_ns: i96 = 0,

    pub fn finished(j: *const Job) bool {
        const s = j.state.load(.acquire);
        return s == .done or s == .failed;
    }
};

pub const Options = struct {
    /// Saves queued or running (and finished ones the owner has not taken) at most.
    entries: u32 = 8,
    /// Their bytes at most; a save alone in the queue is accepted whatever its size.
    bytes: u64 = 256 << 20,
    /// Names the queue in its one log line.
    name: []const u8 = "saves",
};

/// What the queue did, for the owner's log lines.
pub const Counters = struct {
    queued: u64 = 0,
    written: u64 = 0,
    failed: u64 = 0,
    dropped: u64 = 0,
    /// The longest a save waited behind others before the worker started it.
    max_queue_ns: u64 = 0,
    /// The longest one write ran.
    max_run_ns: u64 = 0,
};

pub const Verdict = enum { queued, dropped };

pub const SaveQueue = struct {
    gpa: std.mem.Allocator,
    io: Io,
    opts: Options,
    mutex: Io.Mutex = .init,
    cond: Io.Condition = .init,
    /// Unsettled saves, oldest first; `next` is the first the worker has not started.
    items: std.ArrayList(*Job) = .empty,
    next: usize = 0,
    held_bytes: u64 = 0,
    thread: ?std.Thread = null,
    stop: bool = false,
    counters: Counters = .{},
    /// The first failure: logged once, the rest only counted.
    first_error: ?anyerror = null,

    /// A queue whose worker has not started (`start`); it lives on the heap, the worker keeps its address.
    pub fn create(gpa: std.mem.Allocator, io: Io, opts: Options) !*SaveQueue {
        const q = try gpa.create(SaveQueue);
        errdefer gpa.destroy(q);
        q.* = .{ .gpa = gpa, .io = io, .opts = opts };
        try q.items.ensureTotalCapacity(gpa, opts.entries);
        return q;
    }

    /// Starts the worker; until then every save is dropped.
    pub fn start(q: *SaveQueue) !void {
        if (q.thread != null) return;
        q.thread = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, loop, .{q});
    }

    /// Stops admitting, runs what is queued, joins the worker. Finished saves stay for the owner to take.
    pub fn close(q: *SaveQueue) void {
        q.mutex.lockUncancelable(q.io);
        q.stop = true;
        q.cond.broadcast(q.io);
        q.mutex.unlock(q.io);
        if (q.thread) |t| t.join();
        q.thread = null;
    }

    /// Closes (the queued saves run first) and frees the queue; the owner has taken every save it wants settled.
    pub fn destroy(q: *SaveQueue) void {
        q.close();
        q.items.deinit(q.gpa);
        q.gpa.destroy(q);
    }

    /// Whether a save of `bytes` would be queued now; a refusal is counted as a drop (the caller then skips the save).
    pub fn admit(q: *SaveQueue, bytes: u64) bool {
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        if (q.fits(bytes)) return true;
        q.counters.dropped += 1;
        return false;
    }

    /// The lock held: the queue is open, running, and has room for `bytes` by entries and bytes.
    fn fits(q: *const SaveQueue, bytes: u64) bool {
        if (q.stop or q.thread == null or q.items.items.len >= q.opts.entries) return false;
        return q.items.items.len == 0 or q.held_bytes +| bytes <= q.opts.bytes;
    }

    /// Queues `j` unless the queue is full or closing; never waits. A dropped save stays the caller's.
    pub fn submit(q: *SaveQueue, j: *Job) Verdict {
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        if (!q.fits(j.bytes)) {
            q.counters.dropped += 1;
            return .dropped;
        }
        j.state.store(.queued, .release);
        j.submitted_ns = q.now();
        q.items.appendAssumeCapacity(j);
        q.held_bytes += j.bytes;
        q.counters.queued += 1;
        q.cond.broadcast(q.io);
        return .queued;
    }

    /// The newest unsettled save of `key`: a read that must see a write finds it here, in memory, before its file.
    pub fn find(q: *SaveQueue, key: u64) ?*Job {
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        var i = q.items.items.len;
        while (i > 0) : (i -= 1) if (q.items.items[i - 1].key == key) return q.items.items[i - 1];
        return null;
    }

    /// Blocks until `j` has finished (the one wait a read that needs this very save takes).
    pub fn wait(q: *SaveQueue, j: *Job) void {
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        while (!j.finished()) q.cond.waitUncancelable(q.io, &q.mutex);
    }

    /// How many of the oldest saves have finished; never waits.
    pub fn ready(q: *SaveQueue) usize {
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        var n: usize = 0;
        for (q.items.items) |j| {
            if (!j.finished()) break;
            n += 1;
        }
        return n;
    }

    /// Takes the oldest save if it has finished.
    pub fn reap(q: *SaveQueue) ?*Job {
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        if (q.items.items.len == 0 or !q.items.items[0].finished()) return null;
        return q.take();
    }

    /// Takes the oldest save, waiting for it to finish; null when none is queued.
    pub fn reapWait(q: *SaveQueue) ?*Job {
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        if (q.items.items.len == 0) return null;
        const j = q.items.items[0];
        while (!j.finished()) q.cond.waitUncancelable(q.io, &q.mutex);
        return q.take();
    }

    /// Blocks until the worker has run every queued save.
    pub fn flush(q: *SaveQueue) void {
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        for (q.items.items) |j| while (!j.finished()) q.cond.waitUncancelable(q.io, &q.mutex);
    }

    pub fn pending(q: *SaveQueue) usize {
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        return q.items.items.len;
    }

    pub fn heldBytes(q: *SaveQueue) u64 {
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        return q.held_bytes;
    }

    pub fn stats(q: *SaveQueue) Counters {
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        return q.counters;
    }

    /// The lock held, the oldest save finished.
    fn take(q: *SaveQueue) *Job {
        const j = q.items.orderedRemove(0);
        q.next -= 1;
        q.held_bytes -= j.bytes;
        return j;
    }

    fn now(q: *const SaveQueue) i96 {
        return Io.Clock.awake.now(q.io).toNanoseconds();
    }

    fn loop(q: *SaveQueue) void {
        while (true) {
            const j = blk: {
                q.mutex.lockUncancelable(q.io);
                defer q.mutex.unlock(q.io);
                while (q.next >= q.items.items.len and !q.stop) q.cond.waitUncancelable(q.io, &q.mutex);
                if (q.next >= q.items.items.len) return;
                q.next += 1;
                break :blk q.items.items[q.next - 1];
            };
            j.started_ns = q.now();
            j.state.store(.running, .release);
            const failure: ?anyerror = if (j.run(j)) |_| null else |e| e;
            j.finished_ns = q.now();
            j.err = failure;
            q.mutex.lockUncancelable(q.io);
            defer q.mutex.unlock(q.io);
            q.counters.max_queue_ns = @max(q.counters.max_queue_ns, @as(u64, @intCast(@max(0, j.started_ns - j.submitted_ns))));
            q.counters.max_run_ns = @max(q.counters.max_run_ns, @as(u64, @intCast(@max(0, j.finished_ns - j.started_ns))));
            if (failure) |e| {
                q.counters.failed += 1;
                if (q.first_error == null) {
                    q.first_error = e;
                    std.log.scoped(.save_queue).warn("{s}: a write failed ({t}); later failures are counted, not logged", .{ q.opts.name, e });
                }
            } else q.counters.written += 1;
            j.state.store(if (failure == null) .done else .failed, .release);
            q.cond.broadcast(q.io);
        }
    }
};
