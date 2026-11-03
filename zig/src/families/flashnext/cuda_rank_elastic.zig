//! Elastic memory on a rank pair: the leader moves its own side first, then the follower's loop applies the same step.
const std = @import("std");
const wire = @import("cuda_rank_wire.zig");
const Session = @import("cuda_rank_engine.zig").Session;
const kvl = @import("cuda_kv_lanes.zig");
const Lines = @import("exl3/elastic/cuda_exl3_lines.zig").Lines;

/// The epoch of a lines answer: steps applied and not taken back.
pub fn epochOf(arg: u64) u32 {
    return @intCast(arg >> 32);
}

/// The free bytes of a lines answer (whole MiB).
pub fn freeOf(arg: u64) u64 {
    return @as(u64, @as(u32, @truncate(arg))) << 20;
}

/// Both ranks' K/V pools at `target` mapped pages. A refusal on either rank leaves both as they were.
pub fn resizeKv(s: *Session, target: usize) !void {
    const pool = s.pool orelse return error.BadRankFrame;
    const old = pool.mappedPages();
    pool.resizeMapped(target) catch return error.ElasticRefused; // it changes nothing when it fails; the peer has heard nothing
    if (!try kvStep(s, pool, old, target)) return error.ElasticRefused;
    logKv(s.rank, target, "done", pool.mappedPages()); // the peer's census agrees
}

/// The peer's side of a K/V resize; false when it refused and this rank went back to `old`.
fn kvStep(s: *Session, pool: *kvl.Pool, old: usize, target: usize) !bool {
    const h = try s.command(.{ .kind = .resize, .arg = target }, &.{});
    errdefer s.poison();
    const r = try s.ackReply(h);
    if (!r.refused) {
        if (r.arg != pool.tally()) return error.RemoteRank; // one rank mapped what the other did not
        return true;
    }
    try pool.resizeMapped(old); // growing back may fail: the ranks would then differ
    if (r.arg != pool.tally()) return error.RemoteRank;
    return false;
}

/// Both ranks' expert pools `delta` lines more, each clamped to its bounds; a refusal leaves both as they were.
pub fn resizeLines(s: *Session, delta: i32) !void {
    const l = s.lines orelse return error.BadRankFrame;
    l.apply(delta) catch |e| { // refused here: the peer has heard nothing
        if (e == error.ElasticRollbackFailed) s.poison();
        return e;
    };
    if (!try linesStep(s, l, delta)) return error.ElasticRefused;
    logLines(s.rank, 0, delta, "done", l.epoch); // the peer's ack agrees on the epoch
}

/// The peer's side of a lines step; false when it refused and this rank took the step back.
fn linesStep(s: *Session, l: *Lines, delta: i32) !bool {
    const h = try s.command(.{ .kind = .lines, .arg = @as(u32, @bitCast(delta)) }, &.{});
    errdefer s.poison();
    const r = try s.ackReply(h);
    if (r.refused) try l.undoLast();
    if (epochOf(r.arg) != l.epoch) return error.RemoteRank; // the ranks did not do the same steps
    return !r.refused;
}

/// The last lines step taken back on both ranks, exactly.
pub fn undoLines(s: *Session) !void {
    const l = s.lines orelse return error.BadRankFrame;
    l.undoLast() catch |e| {
        if (e == error.ElasticRollbackFailed) s.poison();
        return e;
    };
    const h = try s.command(.{ .kind = .lines, .flags = 1 }, &.{});
    errdefer s.poison();
    const r = try s.ackReply(h);
    if (epochOf(r.arg) != l.epoch) return error.RemoteRank;
    logLines(s.rank, 1, 0, "done", l.epoch); // the peer's ack agrees, so the undo is on both ranks
}

/// The peer's free device bytes (its lines answer to a probe, which changes nothing).
pub fn peerFree(s: *Session) !u64 {
    const h = try s.command(.{ .kind = .lines, .flags = 2 }, &.{});
    errdefer s.poison();
    return freeOf((try s.ackReply(h)).arg);
}

/// The follower's K/V resize: a failed one leaves its pages and accounting whole, so it refuses and says its census.
pub fn applyKv(s: *Session, h: wire.Header) !void {
    const pool = s.pool orelse return error.BadRankFrame;
    pool.resizeMapped(@intCast(h.arg)) catch {
        s.refused = true;
    };
    s.reply = pool.tally();
    logKv(s.rank, h.arg, if (s.refused) "refused" else "done", pool.mappedPages());
}

/// One record per step a rank applies, the same on both ranks, so their ordered tuples compare exactly.
fn logKv(rank: usize, target: u64, status: []const u8, mapped: usize) void {
    std.log.info("flashnext elastic rank {d}: K/V resize to {d} pages, {s}, {d} mapped", .{ rank, target, status, mapped });
}

fn logLines(rank: usize, flags: u32, delta: i32, status: []const u8, epoch: u32) void {
    std.log.info("flashnext elastic rank {d}: lines {s} {d}, {s}, epoch {d}", .{ rank, if (flags == 0) "step" else "undo", delta, status, epoch });
}

/// The follower's lines command: apply (flags 0), take the last step back (1) or just answer (2).
pub fn applyLines(s: *Session, h: wire.Header) !void {
    const l = s.lines orelse return error.BadRankFrame;
    switch (h.flags) {
        0 => l.apply(@bitCast(@as(u32, @intCast(h.arg)))) catch |e| switch (e) {
            error.ElasticRefused => s.refused = true,
            else => return e,
        },
        1 => try l.undoLast(),
        else => {},
    }
    s.reply = l.ack();
    if (h.flags < 2) logLines(s.rank, h.flags, @bitCast(@as(u32, @intCast(h.arg))), if (s.refused) "refused" else "done", epochOf(s.reply));
}
