const std = @import("std");
const policy = @import("expert_offload.zig");
const tier_mod = @import("expert_tier.zig");

const Tier = tier_mod.Tier;
const Shape = tier_mod.Shape;
const Window = tier_mod.Window;
const Ports = tier_mod.Ports;
const parts = tier_mod.parts;

const win_dev: u64 = 0x4000_0000;
const lines_base: u64 = 0x1000_0000;
const scratch_base: u64 = 0x8000_0000;
const shape: Shape = .{ .layers = 2, .experts = 4, .picks = 3, .trace_rows = 2, .part_bytes = .{ 64, 16, 16, 32, 16, 16 } };
const blob = 160;

/// The device's memory as host bytes: cache lines and the prompt scratch, addressed from fixed bases.
const Fake = struct {
    lines: [8 * blob]u8 = undefined,
    scratch: [4 * blob]u8 = undefined,
    window: []u8,
    scratch_copies: usize = 0,

    fn lineAddr(ctx: *anyopaque, slot: usize) u64 {
        _ = ctx;
        return lines_base + slot * blob;
    }

    fn copy(ctx: *anyopaque, at: usize, window_off: usize, len: usize) error{CopyFailed}!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        @memcpy(f.scratch[at..][0..len], f.window[window_off..][0..len]);
        f.scratch_copies += 1;
    }

    fn wait(ctx: *anyopaque) error{CopyFailed}!void {
        _ = ctx;
    }

    fn ports(f: *Fake) Ports {
        return .{ .ctx = f, .line_addr = lineAddr, .scratch_base = scratch_base, .scratch_copy = copy, .scratch_wait = wait };
    }

    /// What a stream tile does for an admitted miss: store the blob it read at the entry's line address.
    fn store(f: *Fake, t: *const Tier, entry: usize, layer: usize) void {
        const row = t.win.tableWords()[layer * shape.entries() * 2 ..];
        const src = row[2 * entry];
        const dst = row[2 * entry + 1];
        if (dst == 0) return;
        @memcpy(f.lines[dst - lines_base ..][0..blob], t.win.bytes[src - win_dev ..][0..blob]);
    }
};

fn tensors(gpa: std.mem.Allocator, seed: u8) ![parts][]u8 {
    var out: [parts][]u8 = undefined;
    for (&out, shape.part_bytes, 0..) |*t, b, i| {
        t.* = try gpa.alloc(u8, b * (shape.experts + 1));
        for (t.*, 0..) |*x, j| x.* = @truncate(j *% 31 +% i *% 7 +% seed);
    }
    return out;
}

fn blobOf(ts: [parts][]u8, expert: usize, out: []u8) void {
    var at: usize = 0;
    for (ts, shape.part_bytes) |t, b| {
        @memcpy(out[at..][0..b], t[expert * b ..][0..b]);
        at += b;
    }
}

fn setPicks(t: *Tier, row: usize, ids: []const i32) void {
    @memcpy(t.win.traceWords()[row * shape.picks ..][0..ids.len], ids);
}

const Rig = struct {
    gpa: std.mem.Allocator,
    memory: []align(64) u8,
    t: Tier,
    t0: [parts][]u8,
    t1: [parts][]u8,
    fake: Fake,

    fn open(gpa: std.mem.Allocator) !*Rig {
        const r = try gpa.create(Rig);
        errdefer gpa.destroy(r);
        const bytes = Window.bytesFor(shape, Window.leastSlots(shape));
        r.gpa = gpa;
        r.memory = try gpa.alignedAlloc(u8, .@"64", bytes);
        errdefer gpa.free(r.memory);
        r.t = try Tier.init(gpa, shape, r.memory, win_dev, scratch_base + 0x100_0000);
        errdefer r.t.deinit();
        r.t0 = try tensors(gpa, 1);
        r.t1 = try tensors(gpa, 2);
        try r.t.addLayer(0, r.t0);
        try r.t.addLayer(1, r.t1);
        r.fake = .{ .window = r.memory };
        return r;
    }

    fn close(r: *Rig) void {
        for (r.t0 ++ r.t1) |t| r.gpa.free(t);
        r.t.deinit();
        r.gpa.free(r.memory);
        r.gpa.destroy(r);
    }
};

test "a tier holds each routed expert's six parts back to back, and the shared expert apart" {
    const r = try Rig.open(std.testing.allocator);
    defer r.close();
    try std.testing.expectEqual(@as(usize, 160), shape.blobBytes());
    try std.testing.expectEqual(@as(usize, 96), shape.downOffset());
    try std.testing.expect(r.t.win.nslots >= Window.leastSlots(shape) and r.t.win.nslots % 2 == 0);
    var want: [blob]u8 = undefined;
    blobOf(r.t1, 2, &want);
    try std.testing.expectEqualSlices(u8, &want, r.t.host[1].?[2 * blob ..][0..blob]);
    blobOf(r.t1, 4, &want);
    var shared: [blob]u8 = undefined;
    Tier.sharedBlob(shape, r.t1, &shared);
    try std.testing.expectEqualSlices(u8, &want, &shared);
    // the shared expert's entry names its own line from the start, a routed expert's the first slot
    const words = r.t.win.tableWords();
    try std.testing.expectEqual(scratch_base + 0x100_0000 + blob, words[(1 * 5 + 4) * 2]);
    try std.testing.expectEqual(@as(u64, 0), words[(1 * 5 + 4) * 2 + 1]);
    try std.testing.expectEqual(win_dev + r.t.win.slots_off, words[(1 * 5 + 2) * 2]);
    try std.testing.expectError(error.BadShape, r.t.addLayer(2, r.t0));
}

test "stages fill window slots, admitted misses name their lines, and the table names where each expert sits" {
    const r = try Rig.open(std.testing.allocator);
    defer r.close();
    const t = &r.t;
    // 8 lines of cache; a round admits at most 2 misses
    try t.attachPool(8, 2, 0);
    try std.testing.expectEqual(@as(usize, 8), t.lineCount());
    var want: [blob]u8 = undefined;
    // layer 0: key 1 is named twice, 2 and 3 once: the hottest and the lowest id of the rest are admitted
    setPicks(t, 0, &.{ 1, 2, 4 });
    setPicks(t, 1, &.{ 1, 3, 4 });
    try t.stage(.{ .layer = 0, .rows = 2, .prefill = false }, r.fake.ports());
    try std.testing.expectEqual(@as(u64, 0), t.stats.hits);
    try std.testing.expectEqual(@as(u64, 2), t.stats.admissions);
    // every miss is read where it sits: the three experts have window slots holding their bytes
    const slot1 = t.win.tableWords()[2];
    try std.testing.expect(t.win.contains(slot1));
    blobOf(r.t0, 1, &want);
    try std.testing.expectEqualSlices(u8, &want, t.win.bytes[slot1 - win_dev ..][0..blob]);
    var named: usize = 0;
    for (1..4) |x| named += @intFromBool(t.win.tableWords()[2 * x + 1] != 0);
    try std.testing.expectEqual(@as(usize, 2), named);
    const line_of_1 = t.win.tableWords()[3];
    try std.testing.expect(line_of_1 != 0 and !t.win.contains(line_of_1));
    for (1..4) |x| r.fake.store(t, x, 0);
    t.endRound(2, 1000);
    try std.testing.expect(policy.onDevice(t.pool.?.vram, 1));
    // layer 0 again: the resident expert is a hit and its entry names its line, with the bytes the GEMVs stored
    setPicks(t, 0, &.{ 1, 2, 4 });
    setPicks(t, 1, &.{ 1, 2, 4 });
    try t.stage(.{ .layer = 0, .rows = 2, .prefill = false }, r.fake.ports());
    try std.testing.expect(t.stats.hits >= 2);
    const line = t.win.tableWords()[2];
    try std.testing.expect(!t.win.contains(line));
    try std.testing.expectEqual(@as(u64, 0), t.win.tableWords()[3]);
    try std.testing.expectEqualSlices(u8, &want, r.fake.lines[line - lines_base ..][0..blob]);
    // a prompt span copies each named expert not resident into the scratch, before it answers, and admits nothing
    const before = t.stats.admissions;
    setPicks(t, 0, &.{ 0, 3, 4 });
    setPicks(t, 1, &.{ 2, 3, 4 });
    try t.stage(.{ .layer = 1, .rows = 2, .prefill = true }, r.fake.ports());
    try std.testing.expectEqual(before, t.stats.admissions);
    const e0 = t.win.tableWords()[(1 * 5 + 0) * 2];
    try std.testing.expect(e0 >= scratch_base and e0 < scratch_base + 4 * blob);
    blobOf(r.t1, 0, &want);
    try std.testing.expectEqualSlices(u8, &want, r.fake.scratch[e0 - scratch_base ..][0..blob]);
    try std.testing.expectEqual(@as(u64, 3 * blob), t.stats.scratch_bytes);
    try std.testing.expectEqual(@as(usize, 3), r.fake.scratch_copies);
}

test "without a cache every pick is read from a window slot and nothing is admitted" {
    const r = try Rig.open(std.testing.allocator);
    defer r.close();
    const t = &r.t;
    setPicks(t, 0, &.{ 0, 3, 4 });
    setPicks(t, 1, &.{ 3, 2, 4 });
    try t.stage(.{ .layer = 1, .rows = 2, .prefill = false }, r.fake.ports());
    try std.testing.expectEqual(@as(u64, 0), t.stats.admissions);
    var want: [blob]u8 = undefined;
    for ([_]usize{ 0, 2, 3 }) |x| {
        const entry = t.win.tableWords()[(1 * 5 + x) * 2 ..][0..2];
        try std.testing.expect(t.win.contains(entry[0]) and entry[1] == 0);
        blobOf(r.t1, x, &want);
        try std.testing.expectEqualSlices(u8, &want, t.win.bytes[entry[0] - win_dev ..][0..blob]);
    }
    // the other expert's entry is untouched
    try std.testing.expectEqual(win_dev + t.win.slots_off, t.win.tableWords()[(1 * 5 + 1) * 2]);
    // the two halves of the window alternate by layer parity
    const per_half = t.win.nslots / 2;
    const slot0 = (t.win.tableWords()[(1 * 5 + 0) * 2] - win_dev - t.win.slots_off) / blob;
    try std.testing.expect(slot0 >= per_half);
}

test "a stage with more rows than the trace holds, or a layer without experts, is refused" {
    const r = try Rig.open(std.testing.allocator);
    defer r.close();
    const t = &r.t;
    try std.testing.expectError(error.StageFailed, t.stage(.{ .layer = 0, .rows = 3, .prefill = false }, r.fake.ports()));
    std.heap.page_allocator.free(t.host[0].?);
    t.host[0] = null;
    setPicks(t, 0, &.{ 1, 2, 4 });
    try std.testing.expectError(error.StageFailed, t.stage(.{ .layer = 0, .rows = 1, .prefill = false }, r.fake.ports()));
}

test "a window that cannot hold the least slots is refused" {
    const gpa = std.testing.allocator;
    const bytes = Window.bytesFor(shape, Window.leastSlots(shape) - 2);
    const memory = try gpa.alignedAlloc(u8, .@"64", bytes);
    defer gpa.free(memory);
    try std.testing.expectError(error.BadShape, Tier.init(gpa, shape, memory, win_dev, 0));
    var bad = shape;
    bad.part_bytes[2] = 12;
    try std.testing.expectError(error.BadShape, Tier.init(gpa, bad, memory, win_dev, 0));
}
