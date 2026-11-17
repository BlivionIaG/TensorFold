//! Layer-split prefill, the M5's side of the seam: the boundary rows another engine streams in, as the backend's pass
//! sees them (no fabric types here; native/ wraps the pull's `row_stage` in a `Feed`). Opt-in, mixed CUDA/Metal.
const std = @import("std");

/// The boundary row: four residual planes of 5120 bf16 (post applied), then the next site's four f32 pre values.
pub const row_bytes: usize = 4 * 5120 * 2 + 4 * 4;

/// Whether the layers [0, cut) state (the producer's neutral hand-off) has landed in a blank snapshot.
pub const State = enum(u8) { pulling, landed, failed };
pub const Wait = enum { ready, timeout, failed };

/// What the engine's pass spent, for the reply's runtime block (ns; the engine thread fills it as the pass runs).
pub const Stats = struct {
    /// boundary rows the pass ran, and its passes over them
    rows: u64 = 0,
    passes: u32 = 0,
    /// waiting for rows the producer had not streamed yet
    wait_ns: u64 = 0,
    /// forward passes (layers cut..19, the cone's publish) over the rows
    compute_ns: u64 = 0,
    /// waiting for the state of the layers below the cut, and restoring it into the slot
    state_wait_ns: u64 = 0,
    restore_ns: u64 = 0,
};

/// What the pull side saw, ns from the trigger (0: not yet): the first boundary row, the last, the state landed; bytes.
pub const Pull = struct { first_row_ns: u64 = 0, rows_ns: u64 = 0, state_ns: u64 = 0, bytes: u64 = 0 };

pub const Feed = struct {
    ptr: *anyopaque,
    /// the Metal pass enters at layer `cut`: layers below it ran on the producer
    cut: u32,
    /// boundary rows the stream carries: prompt rows [0, upto)
    upto: u64,
    vtable: *const VTable,

    pub const VTable = struct {
        /// rows [0, n) have arrived, complete and unchanging (acquire)
        ready: *const fn (ptr: *anyopaque) u64,
        /// rows [row0, row0 + n) as `row_bytes` bytes each; only rows below `ready` may be asked
        rows: *const fn (ptr: *anyopaque, row0: u64, n: u64) []const u8,
        /// block up to `timeout_ns` for rows [0, end) or the stream's failure
        wait: *const fn (ptr: *anyopaque, end: u64, timeout_ns: u64) Wait,
        state: *const fn (ptr: *anyopaque) State,
        /// the pull's own timings so far
        pull: *const fn (ptr: *anyopaque) Pull,
        /// the blank `Model.Snapshot` holding layers [0, cut) once `state` is `.landed` (type-erased), else null
        snapshot: *const fn (ptr: *anyopaque) ?*anyopaque,
        /// the pass is over (`ok`: it ran every row and restored the state), or the job ended: the puller may stop
        finish: *const fn (ptr: *anyopaque, ok: bool, stats: Stats) void,
    };

    pub fn ready(f: Feed) u64 {
        return f.vtable.ready(f.ptr);
    }

    pub fn rows(f: Feed, row0: u64, n: u64) []const u8 {
        return f.vtable.rows(f.ptr, row0, n);
    }

    pub fn wait(f: Feed, end: u64, timeout_ns: u64) Wait {
        return f.vtable.wait(f.ptr, end, timeout_ns);
    }

    pub fn state(f: Feed) State {
        return f.vtable.state(f.ptr);
    }

    pub fn pull(f: Feed) Pull {
        return f.vtable.pull(f.ptr);
    }

    pub fn snapshot(f: Feed) ?*anyopaque {
        return f.vtable.snapshot(f.ptr);
    }

    pub fn finish(f: Feed, ok: bool, stats: Stats) void {
        f.vtable.finish(f.ptr, ok, stats);
    }
};

/// A feed over plain memory for tests: rows arrive when `ready_rows` is raised.
pub const Fake = struct {
    bytes: []const u8,
    cut: u32,
    upto: u64,
    ready_rows: std.atomic.Value(u64) = .init(0),
    landed: std.atomic.Value(u8) = .init(@backingInt(State.pulling)),
    snap: ?*anyopaque = null,
    finished: ?bool = null,
    stats: Stats = .{},
    times: Pull = .{},

    pub fn feed(f: *Fake) Feed {
        return .{ .ptr = f, .cut = f.cut, .upto = f.upto, .vtable = &.{ .ready = ready, .rows = rows, .wait = wait, .state = state, .pull = pull, .snapshot = snapshot, .finish = finish } };
    }

    fn self(p: *anyopaque) *Fake {
        return @ptrCast(@alignCast(p));
    }

    fn ready(p: *anyopaque) u64 {
        return self(p).ready_rows.load(.acquire);
    }

    fn rows(p: *anyopaque, row0: u64, n: u64) []const u8 {
        return self(p).bytes[@intCast(row0 * row_bytes)..][0..@intCast(n * row_bytes)];
    }

    fn wait(p: *anyopaque, end: u64, _: u64) Wait {
        const f = self(p);
        if (f.landed.load(.acquire) == @backingInt(State.failed)) return .failed;
        return if (f.ready_rows.load(.acquire) >= end) .ready else .timeout;
    }

    fn state(p: *anyopaque) State {
        return @fromBackingInt(@intCast(self(p).landed.load(.acquire)));
    }

    fn pull(p: *anyopaque) Pull {
        return self(p).times;
    }

    fn snapshot(p: *anyopaque) ?*anyopaque {
        const f = self(p);
        return if (f.landed.load(.acquire) == @backingInt(State.landed)) f.snap else null;
    }

    fn finish(p: *anyopaque, ok: bool, stats: Stats) void {
        self(p).finished = ok;
        self(p).stats = stats;
    }
};

test "a fake feed hands rows out as they are raised" {
    var bytes: [3 * row_bytes]u8 = undefined;
    @memset(&bytes, 7);
    var fake: Fake = .{ .bytes = &bytes, .cut = 14, .upto = 3 };
    const f = fake.feed();
    try std.testing.expectEqual(Wait.timeout, f.wait(1, 0));
    fake.ready_rows.store(2, .release);
    try std.testing.expectEqual(Wait.ready, f.wait(2, 0));
    try std.testing.expectEqual(@as(usize, 2 * row_bytes), f.rows(0, 2).len);
    try std.testing.expectEqual(State.pulling, f.state());
    try std.testing.expectEqual(@as(?*anyopaque, null), f.snapshot());
    f.finish(true, .{ .rows = 3, .passes = 2 });
    try std.testing.expectEqual(@as(?bool, true), fake.finished);
    try std.testing.expectEqual(@as(u64, 3), fake.stats.rows);
}
