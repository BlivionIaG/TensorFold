//! Agree on loaded GLP policy before execution; mismatches must fail on every TP rank.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const tp = @import("tp");
const glp_plan = @import("glp_plan.zig");

const timeout_ns = 5 * std.time.ns_per_s;
const Words = [8]i64;

fn send(plan: *tp.planlink.PlanLink, peer: usize, words: []const i64, deadline: u64) !void {
    try tp.sock.sendBefore(plan.fds[peer], std.mem.sliceAsBytes(words), deadline);
}

fn recv(plan: *tp.planlink.PlanLink, peer: usize, words: []i64, deadline: u64) !void {
    try tp.sock.recvExact(plan.fds[peer], std.mem.sliceAsBytes(words), deadline);
}

fn refuse(rank: usize, own: Words, leader: Words) void {
    if (quiet) return;
    const have = struct {
        fn of(w: Words) []const u8 {
            return if (w[0] == 1) "a vector" else "no vector";
        }
    }.of;
    std.log.scoped(.dsv41).err("glp: rank {d} serves {s} and rank 0 serves {s}, and they differ (file, mode, strength or presence): every rank serves the same --glp file and --glp-alpha", .{ rank, have(own), have(leader) });
}

/// Compare presence, digest, mode, normalized strength and layer count, including zero-strength files.
pub fn agree(plan: *tp.planlink.PlanLink, mine: Words) !void {
    if (plan.world <= 1) return;
    const deadline = tp.sock.nowNs() + timeout_ns;
    if (plan.rank != 0) {
        var leader: Words = undefined;
        try recv(plan, 0, &leader, deadline);
        try send(plan, 0, &mine, deadline);
        var verdict: [1]i64 = undefined;
        try recv(plan, 0, &verdict, deadline);
        const same = std.mem.eql(i64, &leader, &mine);
        if (!same) refuse(plan.rank, mine, leader);
        if (!same or verdict[0] != 1) return error.GlpMismatch;
        return;
    }
    var ok = true;
    for (1..plan.world) |rank| {
        try send(plan, rank, &mine, deadline);
        var peer: Words = undefined;
        try recv(plan, rank, &peer, deadline);
        if (!std.mem.eql(i64, &peer, &mine)) {
            refuse(rank, peer, mine);
            ok = false;
        }
    }
    const verdict = [_]i64{@intFromBool(ok)};
    for (1..plan.world) |rank| try send(plan, rank, &verdict, deadline);
    if (!ok) return error.GlpMismatch;
}

const Ranks = struct {
    err: [2]?anyerror = .{ null, null },

    fn side(plan: *tp.planlink.PlanLink, own: Words, err: *?anyerror) void {
        agree(plan, own) catch |e| {
            err.* = e;
        };
    }

    fn run(own: [2]Words) !Ranks {
        const listener = try tp.bootstrap.testListener();
        defer tp.sock.close(listener.fd);
        const deadline = tp.sock.nowNs() + timeout_ns;
        const follower_fd = try tp.sock.connect("localhost", listener.port, deadline);
        const root_fd = try tp.sock.accept(listener.fd, deadline);
        var root_fds: [tp.planlink.max_ranks]tp.sock.Fd = @splat(-1);
        var follower_fds = root_fds;
        root_fds[1] = root_fd;
        follower_fds[0] = follower_fd;
        var root = tp.planlink.PlanLink.init(0, 2, root_fds);
        defer root.close();
        var follower = tp.planlink.PlanLink.init(1, 2, follower_fds);
        defer follower.close();
        var r: Ranks = .{};
        const thread = try std.Thread.spawn(.{}, side, .{ &follower, own[1], &r.err[1] });
        side(&root, own[0], &r.err[0]);
        thread.join();
        return r;
    }
};

const test_dir = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
const test_layers = [_]struct { id: u32, dir: []const f32 }{.{ .id = 1, .dir = &test_dir }};

const Source = struct {
    alpha: f32,
    vector: struct { mode: enum { add, project }, layers: []const @TypeOf(test_layers[0]), file_sha256: [32]u8 },
};

/// The words of a loaded vector (any strength, 0 and -0 included), as the engine computes them from `api.glp.Glp`.
fn policy(mode: @FieldType(@FieldType(Source, "vector"), "mode"), alpha: f32, tag: u8) Words {
    const source: ?Source = .{ .alpha = alpha, .vector = .{ .mode = mode, .layers = &test_layers, .file_sha256 = @splat(tag) } };
    return glp_plan.policyWords(source);
}

test "glp: two ranks with the same vector and strength agree, none at all agree too" {
    const t = std.testing;
    const a = policy(.project, 0.5, 1);
    try t.expectEqual([2]?anyerror{ null, null }, (try Ranks.run(.{ a, policy(.project, 0.5, 1) })).err);
    const none = glp_plan.policyWords(@as(?Source, null));
    try t.expectEqual([2]?anyerror{ null, null }, (try Ranks.run(.{ none, none })).err);
}

test "glp: a different file, mode or strength, or a rank without the vector, refuses on both ranks" {
    const t = std.testing;
    const a = policy(.project, 0.5, 1);
    const none = glp_plan.policyWords(@as(?Source, null));
    const both = [2]?anyerror{ error.GlpMismatch, error.GlpMismatch };
    for ([_]Words{ policy(.project, 0.5, 2), policy(.add, 0.5, 1), policy(.project, 0.25, 1), none }) |other| {
        try t.expectEqual(both, (try Ranks.run(.{ a, other })).err);
        try t.expectEqual(both, (try Ranks.run(.{ other, a })).err);
    }
}

test "glp: an inactive zero-strength file is a policy of its own: it differs from none and from another file" {
    const t = std.testing;
    const none = glp_plan.policyWords(@as(?Source, null));
    const zero = policy(.add, 0, 1);
    const both = [2]?anyerror{ error.GlpMismatch, error.GlpMismatch };
    try t.expectEqual(both, (try Ranks.run(.{ zero, none })).err);
    try t.expectEqual(both, (try Ranks.run(.{ none, zero })).err);
    try t.expectEqual(both, (try Ranks.run(.{ zero, policy(.add, 0, 2) })).err);
    try t.expectEqual([2]?anyerror{ null, null }, (try Ranks.run(.{ zero, policy(.add, 0, 1) })).err);
    try t.expectEqual([2]?anyerror{ null, null }, (try Ranks.run(.{ zero, policy(.add, -0.0, 1) })).err); // -0 is 0
}

test "glp: one rank agrees with itself and needs no plan socket" {
    var plan = tp.planlink.PlanLink.init(0, 1, @splat(-1));
    try agree(&plan, @splat(0));
}
