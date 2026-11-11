//! The exact decoder replay's end as two events on its stream: the host waits where it reads the replay's rows.

const std = @import("std");
const cuda = @import("cuda");

/// The host clock in milliseconds (a parameter, so a test supplies its own).
pub const Clock = *const fn () f64;

/// What the replay was, kept until its end is read.
pub const Replay = struct { end: u64, rows: u32, programs: usize, group: u32, layer_rows: u64, issue_ms: f64 };

/// A read replay: the GPU time between its two events, the host's wait for the end, the GPU time the host did not wait.
pub const Report = struct { replay: Replay, gpu_ms: f64, waited_ms: f64, overlapped_ms: f64 };

pub const Fence = struct {
    start: ?cuda.Event = null,
    stop: ?cuda.Event = null,
    open: bool = false,
    pending: ?Replay = null,

    /// Records the start event on `stream`, before the replay's first program; the last replay's report is read first.
    pub fn arm(self: *Fence, d: *const cuda.Driver, stream: cuda.Stream) !void {
        std.debug.assert(self.pending == null);
        if (self.start == null) self.start = try cuda.Event.init(d, true);
        if (self.stop == null) self.stop = try cuda.Event.init(d, true);
        try self.start.?.record(stream);
        self.open = true;
    }

    /// Records the stop event after the replay's last program: the replay is in flight, its report pending.
    pub fn mark(self: *Fence, stream: cuda.Stream, replay: Replay) !void {
        std.debug.assert(self.open);
        try self.stop.?.record(stream);
        self.open = false;
        self.pending = replay;
    }

    /// The pending replay's report once its stop event has run: `wait` blocks the host for it, else null while it runs.
    pub fn settle(self: *Fence, wait: bool, clock: Clock) !?Report {
        const replay = self.pending orelse return null;
        const stop = self.stop.?;
        var waited: f64 = 0;
        if (wait) {
            const t = clock();
            try stop.synchronize();
            waited = clock() - t;
        } else if (!try stop.done()) return null;
        const gpu: f64 = try cuda.Event.elapsedMs(self.start.?, stop);
        self.pending = null;
        return .{ .replay = replay, .gpu_ms = gpu, .waited_ms = waited, .overlapped_ms = @max(0, gpu - waited) };
    }

    pub fn deinit(self: *Fence) void {
        if (self.start) |*e| e.deinit();
        if (self.stop) |*e| e.deinit();
        self.* = .{};
    }
};
