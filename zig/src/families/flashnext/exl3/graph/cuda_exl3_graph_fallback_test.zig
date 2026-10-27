//! The capture-failure fallback: a failed capture costs one eager round; two ranks decide it together.

const std = @import("std");
const collective = @import("../../cuda_collective.zig");
const wire = @import("../../cuda_rank_wire.zig");
const gfall = @import("cuda_exl3_graph_fallback.zig");

const FakeGraph = struct {
    var live: i32 = 0;
    id: u32,

    fn make(id: u32) FakeGraph {
        live += 1;
        return .{ .id = id };
    }

    pub fn deinit(_: *FakeGraph) void {
        live -= 1;
    }
};

/// One rank's capture attempts: `fail[i]` is whether its i-th attempt fails.
const Attempt = struct {
    fail: []const bool,
    calls: *usize,

    pub fn run(a: Attempt) ?FakeGraph {
        defer a.calls.* += 1;
        return if (a.fail[a.calls.*]) null else FakeGraph.make(@intCast(a.calls.*));
    }
};

/// A rank's side of the wire: `graphAgree` is the exchange `Collective.graphAgree` makes.
const Rank = struct {
    link: wire.Link,
    step: u64 = 0,

    pub fn graphAgree(r: *Rank, signature: u64, vote: collective.GraphVote) !collective.GraphDecision {
        r.step += 1;
        return collective.graphExchange(r.link, r.step, signature, vote, wire.deadline(2000));
    }
};

fn socketPair() ![2]std.posix.socket_t {
    var pair: [2]std.posix.socket_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    errdefer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
    for (pair) |fd| {
        const flags = std.c.fcntl(fd, std.c.F.GETFL);
        if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
    }
    return pair;
}

const rounds = 4;

/// What a rank saw of each round: whether it kept a graph.
const Outcome = struct { kept: [rounds]bool = @splat(false), err: ?anyerror = null };

fn rankMain(link: wire.Link, fail: []const bool, out: *Outcome) void {
    var rank: Rank = .{ .link = link };
    var calls: usize = 0;
    for (0..rounds) |round| {
        const got = gfall.agreedCapture(&rank, 0x5eed + round, 7, Attempt{ .fail = fail, .calls = &calls }) catch |err| {
            out.err = err;
            return;
        };
        out.kept[round] = got != null;
        if (got) |g| {
            var graph = g;
            graph.deinit(); // the round's replay is done with it
        }
    }
}

fn runPair(fail0: []const bool, fail1: []const bool) !struct { Outcome, Outcome } {
    const pair = try socketPair();
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    var peer: Outcome = .{};
    var mine: Outcome = .{};
    const thread = try std.Thread.spawn(.{}, rankMain, .{ wire.Link{ .fd = pair[1] }, fail1, &peer });
    rankMain(.{ .fd = pair[0] }, fail0, &mine);
    thread.join();
    return .{ mine, peer };
}

test "exl3: both ranks keep a capture only when both captured" {
    FakeGraph.live = 0;
    const none = [_]bool{ false, false, false, false };
    const second = [_]bool{ false, true, false, false };
    const cases = [_][2][]const bool{ .{ &none, &none }, .{ &second, &none }, .{ &none, &second }, .{ &second, &second } };
    for (cases) |c| {
        const a, const b = try runPair(c[0], c[1]);
        try std.testing.expectEqual(@as(?anyerror, null), a.err);
        try std.testing.expectEqual(@as(?anyerror, null), b.err);
        // lockstep: both ranks decided every round the same way
        try std.testing.expectEqualSlices(bool, &a.kept, &b.kept);
        // round 2 failed on at least one rank in three of the cases: both ran it eagerly; later rounds are not poisoned
        const failed = c[0][1] or c[1][1];
        try std.testing.expectEqualSlices(bool, &.{ true, !failed, true, true }, &a.kept);
        // no capture is left behind on either rank
        try std.testing.expectEqual(@as(i32, 0), FakeGraph.live);
    }
}

test "exl3: a rank that fails every capture leaves its peer eager every round, the communicator untouched" {
    FakeGraph.live = 0;
    const all = [_]bool{ true, true, true, true };
    const none = [_]bool{ false, false, false, false };
    const a, const b = try runPair(&all, &none);
    try std.testing.expectEqual(@as(?anyerror, null), a.err);
    try std.testing.expectEqual(@as(?anyerror, null), b.err);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, &a.kept);
    try std.testing.expectEqualSlices(bool, &a.kept, &b.kept);
    try std.testing.expectEqual(@as(i32, 0), FakeGraph.live);
}

test "exl3: a lost peer during the capture agreement drops the local capture and reports the loss" {
    FakeGraph.live = 0;
    const pair = try socketPair();
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    _ = std.c.shutdown(pair[1], std.posix.SHUT.RDWR);
    var rank: Rank = .{ .link = .{ .fd = pair[0] } };
    var calls: usize = 0;
    const none = [_]bool{false};
    try std.testing.expectError(error.RankDisconnected, gfall.agreedCapture(&rank, 1, 7, Attempt{ .fail = &none, .calls = &calls }));
    try std.testing.expectEqual(@as(i32, 0), FakeGraph.live);
}
