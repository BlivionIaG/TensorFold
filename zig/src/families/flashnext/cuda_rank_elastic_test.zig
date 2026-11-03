//! The lockstep commands of elastic memory: a leader session and a follower's real `follow` loop on a socket pair.
const std = @import("std");
const cuda = @import("cuda");
const kvt = @import("cuda_kv_elastic_test.zig");
const pages = @import("cuda_kv_lanes.zig");
const ranks = @import("cuda_rank_engine.zig");
const wire = @import("cuda_rank_wire.zig");
const lines_mod = @import("exl3/elastic/cuda_exl3_lines.zig");
const Tri = @import("cuda_launch.zig").Tri;
const t = std.testing;

const mib: u64 = 1 << 20;

fn fakeLines(f: *lines_mod.Fake) !lines_mod.Lines {
    f.* = .{ .n = 4 };
    for (f.pools[0..4], 0..) |*p, i| p.* = .{ .active = 20 + @as(u32, @intCast(i)), .floor = 0, .cap = 60, .bytes = 1_843_200, .chunk = 2 << 20 };
    return lines_mod.Lines.init(f.backend(), 10);
}

fn actives(f: *const lines_mod.Fake) [4]u32 {
    return .{ f.pools[0].active, f.pools[1].active, f.pools[2].active, f.pools[3].active };
}

/// The follower rank: its session's `follow` loop on its own thread until the link closes or a stop arrives.
const Follower = struct {
    s: *ranks.Session,
    thread: ?std.Thread = null,
    err: ?anyerror = null,

    fn run(f: *Follower) void {
        f.s.follow(Tri, undefined, 0) catch |e| {
            f.err = e;
        };
    }
};

/// Both ranks, wired. `deinit` stops the pair the way a server would and joins the follower.
const Pair = struct {
    d: cuda.Driver,
    e: @import("cuda_prompt.zig").Engine,
    a: *pages.Pool,
    b: *pages.Pool,
    count: usize,
    fa: lines_mod.Fake,
    fb: lines_mod.Fake,
    la: lines_mod.Lines,
    lb: lines_mod.Lines,
    leader: ranks.Session,
    follower: ranks.Session,
    runner: Follower,
    fds: [2]std.posix.socket_t,

    fn start(p: *Pair) !void {
        p.d = kvt.driver();
        p.e = kvt.engine(&p.d);
        const m0 = try kvt.made(&p.e, kvt.Range(0), 1);
        const m1 = try kvt.made(&p.e, kvt.Range(1), 1);
        p.a = m0.pool;
        p.b = m1.pool;
        p.count = m0.count;
        p.la = try fakeLines(&p.fa);
        p.lb = try fakeLines(&p.fb);
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &p.fds) != 0) return error.SocketPair;
        const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
        for (p.fds) |fd| {
            const flags = std.c.fcntl(fd, std.c.F.GETFL);
            if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
        }
        p.leader = .{ .gpa = t.allocator, .rank = 0, .link = .{ .fd = p.fds[0] }, .collective = undefined, .engine = &p.e, .pool = p.a, .lines = &p.la };
        p.follower = .{ .gpa = t.allocator, .rank = 1, .link = .{ .fd = p.fds[1] }, .collective = undefined, .engine = &p.e, .pool = p.b, .lines = &p.lb };
        for ([_]*ranks.Session{ &p.leader, &p.follower }) |s| {
            s.collective.failed = false;
            s.collective.comm = null;
        }
        p.runner = .{ .s = &p.follower };
        p.runner.thread = try std.Thread.spawn(.{}, Follower.run, .{&p.runner});
    }

    fn deinit(p: *Pair) void {
        if (!p.leader.stopped) { // a clean stop: the follower returns from `follow`
            const h = p.leader.command(.{ .kind = .stop }, &.{}) catch null;
            if (h) |sent| p.leader.ack(sent) catch p.leader.poison();
        } else _ = std.c.shutdown(p.fds[0], std.posix.SHUT.RDWR);
        p.runner.thread.?.join();
        _ = std.c.close(p.fds[0]);
        _ = std.c.close(p.fds[1]);
        p.a.deinit();
        p.b.deinit();
        cuda.completion.drain(&p.d, p.e.stream.*, "the test driver must drain before it leaves");
    }
};

test "exl3: elastic ranks: the leader's K/V and lines steps are applied by the follower's loop and the ranks agree" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // the peer rank is a Unix socket pair
    var p: Pair = undefined;
    try p.start();
    defer p.deinit();
    const target = p.count - 10;
    try p.leader.resizeKv(target);
    try t.expectEqual(target, p.a.mappedPages());
    try t.expectEqual(p.a.tally(), p.b.tally());
    try p.leader.resizeKv(p.count);
    try t.expectEqual(p.a.tally(), p.b.tally());
    try p.leader.resizeLines(-4);
    try t.expectEqual([4]u32{ 16, 17, 18, 19 }, actives(&p.fa));
    try t.expectEqual(actives(&p.fa), actives(&p.fb));
    try t.expectEqual(p.la.epoch, p.lb.epoch);
    try p.leader.resizeLines(100); // clamped to each pool's cap on each rank
    try t.expectEqual([4]u32{ 60, 60, 60, 60 }, actives(&p.fb));
    try p.leader.undoLines();
    try t.expectEqual([4]u32{ 16, 17, 18, 19 }, actives(&p.fa));
    try t.expectEqual(actives(&p.fa), actives(&p.fb));
    try p.leader.undoLines();
    try t.expectEqual([4]u32{ 20, 21, 22, 23 }, actives(&p.fb));
    try t.expectEqual(@as(u32, 0), p.lb.epoch);
    p.fb.free_bytes = 777 * mib;
    try t.expectEqual(777 * mib, try p.leader.peerFree());
    try t.expect(!p.leader.stopped and p.leader.healthy());
}

test "exl3: elastic ranks: a follower that refuses sends the leader back to where it was and nothing is poisoned" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var p: Pair = undefined;
    try p.start();
    defer p.deinit();
    // K/V: the follower's range fails a map the leader's made; the leader had grown first and shrinks back
    const R1 = kvt.Range(1);
    try p.leader.resizeKv(p.count - 6);
    const tally = p.a.tally();
    R1.fail_map_call = R1.calls + 2; // the second chunk it maps
    try t.expectError(error.ElasticRefused, p.leader.resizeKv(p.count));
    R1.fail_map_call = 0;
    try t.expectEqual(p.count - 6, p.a.mappedPages());
    try t.expectEqual(tally, p.a.tally());
    try t.expectEqual(p.a.tally(), p.b.tally());
    try t.expect(!p.leader.stopped);
    try p.leader.resizeKv(p.count); // the ranks still answer, in order
    try t.expectEqual(p.a.tally(), p.b.tally());
    // lines: the follower's pools refuse the third pool; the leader had applied the step and takes it back
    const was = actives(&p.fa);
    p.fb.fail_after = p.fb.calls + 2; // only that one attempt: the put backs go through
    p.fb.fail_until = p.fb.calls + 3;
    try t.expectError(error.ElasticRefused, p.leader.resizeLines(-3));
    try t.expectEqual(was, actives(&p.fa));
    try t.expectEqual(was, actives(&p.fb));
    try t.expectEqual(@as(u32, 0), p.la.epoch);
    try t.expectEqual(@as(u32, 0), p.lb.epoch);
    try t.expect(!p.leader.stopped);
    // the leader's own pools refuse: nothing is sent, and the follower's loop sees the next command in sequence
    p.fb.fail_after = null;
    p.fa.fail_after = p.fa.calls + 1;
    p.fa.fail_until = p.fa.calls + 2;
    try t.expectError(error.ElasticRefused, p.leader.resizeLines(-3));
    try t.expectEqual(was, actives(&p.fa));
    p.fa.fail_after = null;
    try p.leader.resizeLines(-3);
    try t.expectEqual(actives(&p.fa), actives(&p.fb));
    try t.expect(!p.leader.stopped and p.leader.healthy());
}

test "exl3: elastic ranks: ranks whose lines differ poison the session instead of going on" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var p: Pair = undefined;
    try p.start();
    defer p.deinit();
    try p.leader.resizeLines(-2);
    try p.lb.apply(-1); // the follower did a step the leader did not
    try t.expectError(error.RemoteRank, p.leader.resizeLines(-2));
    try t.expect(p.leader.stopped);
    try t.expect(!p.leader.healthy());
    try t.expectError(error.RankDisconnected, p.leader.resizeLines(-1)); // nothing more is commanded
}

test "exl3: elastic ranks: a lines command is shaped like its neighbours" {
    try ranks.shape(.{ .kind = .lines, .arg = 3 });
    try ranks.shape(.{ .kind = .lines, .flags = 1 });
    try ranks.shape(.{ .kind = .lines, .flags = 2 });
    try ranks.shape(.{ .kind = .lines, .arg = std.math.maxInt(u32) });
    try t.expectError(error.BadRankFrame, ranks.shape(.{ .kind = .lines, .flags = 3 }));
    try t.expectError(error.BadRankFrame, ranks.shape(.{ .kind = .lines, .object = 1 }));
    try t.expectError(error.BadRankFrame, ranks.shape(.{ .kind = .lines, .bytes = 4 }));
    try t.expectError(error.BadRankFrame, ranks.shape(.{ .kind = .lines, .arg = @as(u64, std.math.maxInt(u32)) + 1 }));
    try t.expectError(error.BadRankFrame, ranks.shape(.{ .kind = .resize, .flags = 1 }));
}
