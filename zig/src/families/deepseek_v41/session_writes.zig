//! The leader's settle point for write-through saves behind the engine thread, and the counters the logs carry.

const std = @import("std");
const kv = @import("kv");
const sg = @import("sessions_gpu.zig");

const through = kv.sessions.store.through;

/// Finished write-through saves settle, the same count on every rank (the leader's); a follower gets op_sess_settle.
pub fn retire(s: *sg.Sessions) !void {
    if (s.f.link != null and s.f.comm.rank() != 0) return;
    const n = through.finished(&s.store);
    if (n == 0) return;
    try s.f.sendSessSettle(n);
    through.settleOldest(&s.store, n);
}

/// Every write-through save settled, waiting for the ones still running (a rank drains; nothing is decided after it).
pub fn flush(s: *sg.Sessions) void {
    through.settleEvery(&s.store);
}

/// The leader's count settled on this rank (op_sess_settle).
pub fn settleFn(ptr: *anyopaque, n: u32) anyerror!void {
    const s: *sg.Sessions = @ptrCast(@alignCast(ptr));
    through.settleOldest(&s.store, n);
}

/// The save queue's counters as the session log lines carry them; nothing when the saves are written inline.
pub const Report = struct {
    counters: ?kv.sessions.save_queue.Counters,

    pub fn format(r: Report, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const c = r.counters orelse return;
        try w.print(" [saves: {d} queued, {d} written, {d} dropped, {d} failed, queue wait max {d:.1} ms, write max {d:.1} ms]", .{ c.queued, c.written, c.dropped, c.failed, ms(c.max_queue_ns), ms(c.max_run_ns) });
    }
};

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

pub fn report(s: *sg.Sessions) Report {
    return .{ .counters = through.counters(&s.store) };
}

test "the save counters print in the session log lines, and nothing when saves are inline" {
    var buf: [160]u8 = undefined;
    const none = try std.fmt.bufPrint(&buf, "{f}", .{Report{ .counters = null }});
    try std.testing.expectEqualStrings("", none);
    const c: kv.sessions.save_queue.Counters = .{ .queued = 5, .written = 4, .dropped = 1, .failed = 0, .max_queue_ns = 2_500_000, .max_run_ns = 1_300_000 };
    const line = try std.fmt.bufPrint(&buf, "{f}", .{Report{ .counters = c }});
    try std.testing.expectEqualStrings(" [saves: 5 queued, 4 written, 1 dropped, 0 failed, queue wait max 2.5 ms, write max 1.3 ms]", line);
}
