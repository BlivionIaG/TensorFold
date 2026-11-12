const std = @import("std");
const policy_mod = @import("expert_offload.zig");

const Id = policy_mod.Id;
const Decision = policy_mod.Decision;
const Policy = policy_mod.Policy;
const Line = policy_mod.Line;
const Vram = policy_mod.Vram;
const Copy = policy_mod.Copy;
const stage = policy_mod.stage;
const reap = policy_mod.reap;
const touch = policy_mod.touch;
const onDevice = policy_mod.onDevice;
const schedule = policy_mod.schedule;
const observe = policy_mod.observe;
const seedHottest = policy_mod.seedHottest;

fn testVram(comptime slots: usize, comptime keys: usize) struct {
    slots: [slots]Id = undefined,
    line: [slots]Line = undefined,
    stamp: [slots]u32 = undefined,
    where: [keys]i32 = undefined,
    unique: [keys * 4]Id = undefined,
    tally: [keys]u16 = @splat(0),
    decisions: [keys * 4]Decision = undefined,
    copies: [slots]Copy = undefined,
} {
    return .{};
}

test "a staged copy is not on device until reap, and LRU drops the older resident" {
    var t = testVram(2, 16);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    try std.testing.expectEqual(@as(u32, 0), (try stage(&vram, 4)).slot);
    try std.testing.expectEqual(@as(u32, 1), (try stage(&vram, 7)).slot);
    try std.testing.expect(!onDevice(vram, 4));
    try std.testing.expectError(error.NotInflight, reap(&vram, 9, 1));
    try reap(&vram, 4, 1);
    try reap(&vram, 7, 2);
    try std.testing.expect(onDevice(vram, 4) and onDevice(vram, 7));
    const c = try stage(&vram, 9);
    try std.testing.expectEqual(@as(u32, 0), c.slot);
    try std.testing.expectEqual(@as(?Id, 4), c.evicted);
    try std.testing.expect(!onDevice(vram, 4) and !onDevice(vram, 9) and onDevice(vram, 7));
    try std.testing.expectEqual(@as(i32, -1), vram.where[4]);
}

test "an SLRU keeps a line that was hit again over a newer line that was not, and demotes the oldest protected past its cap" {
    var t = testVram(3, 16);
    var flags: [3]bool = undefined;
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    vram.enableSegments(&flags, 1);
    for ([_]Id{ 1, 2, 3 }, 1..) |id, now| {
        _ = try stage(&vram, id);
        try reap(&vram, id, @intCast(now));
    }
    // 1 is hit again: protected (stamp 4). 2 is now the oldest probation line though 1 was the oldest stamp
    try std.testing.expect(touch(&vram, 1, 4));
    try std.testing.expectEqual(@as(usize, 1), vram.protected_count);
    const c = try stage(&vram, 9);
    try std.testing.expectEqual(@as(?Id, 2), c.evicted);
    try reap(&vram, 9, 5);
    // a second promotion past the cap of one sends the oldest protected line (1) back to probation
    try std.testing.expect(touch(&vram, 3, 6));
    try std.testing.expectEqual(@as(usize, 1), vram.protected_count);
    const d = try stage(&vram, 10);
    try std.testing.expectEqual(@as(?Id, 1), d.evicted);
}

test "an in-flight line is not reused, and a full in-flight set is busy" {
    var t = testVram(2, 16);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    _ = try stage(&vram, 1);
    _ = try stage(&vram, 2);
    try std.testing.expectError(error.Busy, stage(&vram, 3));
    try reap(&vram, 2, 5);
    const c = try stage(&vram, 3);
    try std.testing.expectEqual(@as(u32, 1), c.slot);
    try std.testing.expectEqual(@as(?Id, 2), c.evicted);
    try std.testing.expectError(error.NotInflight, reap(&vram, 2, 6));
}

test "schedule touches residents, stages the hottest miss, and will not evict a hit" {
    var t = testVram(2, 16);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    _ = try stage(&vram, 1);
    _ = try stage(&vram, 2);
    try reap(&vram, 1, 1);
    try reap(&vram, 2, 2);
    const policy: Policy = .{ .slots = 2, .expert_bytes = 10, .copy_bytes = 10 };
    const ids = [_]Id{ 1, 5, 5, 6 };
    const n = try schedule(&vram, &ids, policy, 3, t.decisions[0..4], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqualSlices(Decision, &.{ .hit, .copy, .copy, .cpu }, t.decisions[0..4]);
    // 5 (two rows) is the hottest miss; it took the line of 2, which was older than the touched 1
    try std.testing.expectEqual(@as(?Id, 2), t.copies[0].evicted);
    try std.testing.expect(onDevice(vram, 1) and !onDevice(vram, 5));
    try std.testing.expectEqual(@as(u16, 0), t.tally[5]);
}

test "a zero copy budget stages nothing, and an older resident is the one that moves" {
    var t = testVram(2, 16);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    _ = try stage(&vram, 1);
    _ = try stage(&vram, 2);
    try reap(&vram, 1, 1);
    try reap(&vram, 2, 2);
    var policy: Policy = .{ .slots = 2, .expert_bytes = 10, .copy_bytes = 0 };
    const ids = [_]Id{ 7, 1 };
    var n = try schedule(&vram, &ids, policy, 3, t.decisions[0..2], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies);
    try std.testing.expectEqual(@as(usize, 0), n);
    try std.testing.expectEqualSlices(Decision, &.{ .cpu, .hit }, t.decisions[0..2]);
    policy.copy_bytes = 10;
    n = try schedule(&vram, &ids, policy, 4, t.decisions[0..2], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(?Id, 2), t.copies[0].evicted);
    policy.expert_bytes = 0;
    const again = [_]Id{8};
    try std.testing.expectEqual(@as(usize, 0), try schedule(&vram, &again, policy, 5, t.decisions[0..1], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies));
}

test "misses past the copy budget are cpu, equal counts take the lower id, duplicates share a decision" {
    var t = testVram(4, 16);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    const policy: Policy = .{ .slots = 4, .expert_bytes = 4, .copy_bytes = 8 };
    const ids = [_]Id{ 9, 3, 9, 5, 3, 7 };
    const n = try schedule(&vram, &ids, policy, 1, t.decisions[0..6], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies);
    try std.testing.expectEqual(@as(usize, 2), n);
    // 3 and 9 appear twice (lower id first), 5 and 7 once: only two copies fit
    try std.testing.expectEqual(@as(Id, 3), t.copies[0].id);
    try std.testing.expectEqual(@as(Id, 9), t.copies[1].id);
    try std.testing.expectEqualSlices(Decision, &.{ .copy, .copy, .copy, .cpu, .copy, .cpu }, t.decisions[0..6]);
    // an in-flight id is cpu until reap and is not staged twice
    const next = [_]Id{3};
    try std.testing.expectEqual(@as(usize, 0), try schedule(&vram, &next, policy, 2, t.decisions[0..1], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies));
    try std.testing.expectEqual(Decision.cpu, t.decisions[0]);
}

test "a resident named this round is never the victim even when it is the oldest" {
    var t = testVram(2, 16);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    _ = try stage(&vram, 1);
    _ = try stage(&vram, 2);
    try reap(&vram, 1, 1);
    try reap(&vram, 2, 2);
    const policy: Policy = .{ .slots = 2, .expert_bytes = 1, .copy_bytes = 100 };
    // both residents and two misses in one round: nothing can be evicted, so no copy starts
    const ids = [_]Id{ 1, 2, 8, 9 };
    const n = try schedule(&vram, &ids, policy, 3, t.decisions[0..4], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies);
    try std.testing.expectEqual(@as(usize, 0), n);
    try std.testing.expectEqualSlices(Decision, &.{ .hit, .hit, .cpu, .cpu }, t.decisions[0..4]);
}

test "observed counts seed empty lines hottest first, without eviction" {
    var t = testVram(3, 8);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    var counts: [8]u32 = @splat(0);
    try observe(&counts, &.{ 5, 5, 5, 2, 2, 6, 1 });
    var order: [8]Id = undefined;
    const n = try seedHottest(&vram, &counts, &order, &t.copies);
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqual(@as(Id, 5), t.copies[0].id);
    try std.testing.expectEqual(@as(Id, 2), t.copies[1].id);
    try std.testing.expectEqual(@as(Id, 1), t.copies[2].id); // 1 and 6 tie on one: the lower
    try std.testing.expectEqual(@as(?Id, null), t.copies[0].evicted);
    try std.testing.expectEqual(@as(usize, 0), try seedHottest(&vram, &counts, &order, &t.copies));
    try std.testing.expectError(error.Length, observe(&counts, &.{9}));
}

test "a short decision buffer or an unknown key is refused without writes" {
    var t = testVram(2, 4);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    const policy: Policy = .{ .slots = 2, .expert_bytes = 1, .copy_bytes = 1 };
    try std.testing.expectError(error.Length, schedule(&vram, &.{ 1, 2 }, policy, 1, t.decisions[0..1], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies));
    try std.testing.expectError(error.Scratch, schedule(&vram, &.{9}, policy, 1, t.decisions[0..1], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies));
    try std.testing.expectEqual(@as(usize, 0), vram.used);
}
