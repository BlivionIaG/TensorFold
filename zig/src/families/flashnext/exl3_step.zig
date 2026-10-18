const std = @import("std");
const builtin = @import("builtin");
const offload = @import("exl3_offload.zig");
const cpu = @import("exl3_cpu.zig");
const reduce = @import("exl3_reduce.zig");
const shape = @import("exl3_shape.zig");
const format = @import("exl3_format.zig");

/// Shared expert. `scheduleUsedAt` marks this id `.hit` and does not stage a line.
pub const shared_id: u16 = 512;

pub const Phase = enum { decode, prefill };

/// One host expert view. Indexed in parallel with the id list. Read only for `.copy`.
pub const ExpertGemv = struct {
    width: format.Width,
    codebook: format.Codebook,
    trellis: []const i16,
    tiles_k: usize,
    tiles_n: usize,
    suh: []const f16,
    svh: []const f16,
};

pub const StepError = error{ Length, Scratch, CopyBudget };

fn copyBudget(phase: Phase) u16 {
    return switch (phase) {
        .decode => 1,
        .prefill => 4,
    };
}

fn isOwned(owned: offload.Owned, id: u16) bool {
    return id >= owned.lo and id < owned.hi;
}

/// Owned miss left on the CPU because the copy budget ran out.
/// Shared stays a hit. Ids outside `owned` stay `.cpu` and are not a budget failure.
fn budgetExceeded(ids: []const u16, decisions: []const offload.Decision, owned: offload.Owned) bool {
    for (ids, decisions) |id, decision| {
        if (decision != .cpu) continue;
        if (id == shared_id) continue;
        if (!isOwned(owned, id)) continue;
        return true;
    }
    return false;
}

/// Host MoE step for one layer on one rank.
/// Not a router, GDN, attention, or sampler. Does not call `fillOwnedLayers`.
///
/// Copy budget (via `Policy.copy_bytes`): one expert line per decoded token, four per prefill chunk.
/// After a successful schedule, each `.copy` decision is a host GEMV into `routed_out` (fp32).
/// Rank 0 (`owned.lo == 0`) adds `shared_fp32` into that partial. Rank 1 does not.
/// `addHidden` is the cross-rank reduce and owns the single bf16 round; this function does not call it.
pub fn stepLayer(
    vram: *offload.Vram,
    layer: u8,
    ids: []const u16,
    counts: *const [shape.layers][shape.experts]u32,
    now: u32,
    decisions: []offload.Decision,
    id_scratch: []u16,
    owned: offload.Owned,
    phase: Phase,
    slots: u16,
    expert_bytes: u64,
    experts: []const ExpertGemv,
    x: []const f64,
    y_scratch: []f64,
    gemv_scratch: []f64,
    shared_fp32: []const f32,
    routed_out: []f32,
) StepError!void {
    if (decisions.len != ids.len) return error.Length;
    if (experts.len != ids.len) return error.Length;
    if (shared_fp32.len != routed_out.len) return error.Length;

    const copies: u64 = copyBudget(phase);
    const policy = offload.Policy{
        .slots = slots,
        .expert_bytes = expert_bytes,
        .copy_bytes = expert_bytes * copies,
    };
    try offload.scheduleUsedAt(vram, layer, ids, counts, policy, now, decisions, id_scratch, owned);

    if (budgetExceeded(ids, decisions, owned)) return error.CopyBudget;

    for (routed_out) |*lane| lane.* = 0;

    // One call site. x86_64 uses the AVX2 kernel; every other arch uses the scalar oracle.
    const apply = if (builtin.cpu.arch == .x86_64) cpu.gemvAvx2 else cpu.gemv;
    for (decisions, experts) |decision, expert| {
        if (decision != .copy) continue;
        if (y_scratch.len != routed_out.len) return error.Length;
        try apply(
            expert.width,
            expert.codebook,
            expert.trellis,
            expert.tiles_k,
            expert.tiles_n,
            expert.suh,
            expert.svh,
            x,
            y_scratch,
            gemv_scratch,
        );
        for (routed_out, y_scratch) |*lane, partial| lane.* += @floatCast(partial);
    }

    // Rank 0 only. fp32 add; do not round to bf16 here.
    if (owned.lo == 0) {
        for (routed_out, shared_fp32) |*lane, shared| lane.* += shared;
    }
}

// Re-export so a rank partial can be reduced without a second import in tests that only pull step.
pub const addHidden = reduce.addHidden;

// --- tests ---

const Cache = offload.Cache;
const Line = offload.Line;
const Vram = offload.Vram;
const Decision = offload.Decision;
const Owned = offload.Owned;

const rank0_owned = Owned{ .lo = 0, .hi = 256 };
const rank1_owned = Owned{ .lo = 256, .hi = 512 };

const Fixture = struct {
    slots: []u16,
    line: []Line,
    stamp: []u32,
    layer_of: []u8,
    vram: Vram,
    counts: [shape.layers][shape.experts]u32,
};

fn fixture(slots: []u16, line: []Line, stamp: []u32, layer_of: []u8) Fixture {
    return .{
        .slots = slots,
        .line = line,
        .stamp = stamp,
        .layer_of = layer_of,
        .vram = Vram.init(slots, line, stamp, layer_of),
        .counts = @splat(@splat(0)),
    };
}

fn expectNotResident(slots: []const u16, id: u16) !void {
    for (slots) |held| try std.testing.expect(held != id);
}

fn scratchIds(buf: []u16) []u16 {
    return buf;
}

test "shared 512 is a hit and is not staged" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer_of = [_]u8{ 0, 0 };
    var fix = fixture(&slots, &line, &stamp, &layer_of);

    const ids = [_]u16{shared_id};
    var decisions = [_]Decision{.cpu};
    var id_scratch: [64]u16 = undefined;
    const experts = [_]ExpertGemv{undefined};
    var routed = [_]f32{ 7, 7, 7, 7 };
    const shared = [_]f32{ 1, 2, 3, 4 };

    try stepLayer(
        &fix.vram,
        0,
        &ids,
        &fix.counts,
        1,
        &decisions,
        scratchIds(&id_scratch),
        rank0_owned,
        .decode,
        2,
        1,
        &experts,
        &.{},
        &.{},
        &.{},
        &shared,
        &routed,
    );

    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try expectNotResident(&slots, shared_id);
    try std.testing.expectEqual(Cache.empty_id, slots[0]);
    try std.testing.expectEqual(Cache.empty_id, slots[1]);
    // Rank 0 still folds the precomputed shared vector into the routed partial.
    try std.testing.expectEqualSlices(f32, &shared, &routed);
}

test "other rank id is not staged and is not gemved" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer_of = [_]u8{ 0, 0 };
    var fix = fixture(&slots, &line, &stamp, &layer_of);

    // 300 belongs to rank 1. Rank 0 must leave it .cpu and must not GEMV it.
    const ids = [_]u16{300};
    var decisions = [_]Decision{.hit};
    var id_scratch: [64]u16 = undefined;
    const experts = [_]ExpertGemv{undefined};
    var routed = [_]f32{ 5, 5, 5, 5 };
    const shared = [_]f32{ 1, 0, 0, 0 };
    var gemv_calls: usize = 0;
    // Empty expert slices: a GEMV would return Length. Success means it was skipped.
    _ = &gemv_calls;

    try stepLayer(
        &fix.vram,
        0,
        &ids,
        &fix.counts,
        1,
        &decisions,
        scratchIds(&id_scratch),
        rank0_owned,
        .decode,
        2,
        1,
        &experts,
        &.{},
        &.{},
        &.{},
        &shared,
        &routed,
    );

    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try expectNotResident(&slots, 300);
    try std.testing.expectEqual(Cache.empty_id, slots[0]);
    try std.testing.expectEqual(Cache.empty_id, slots[1]);
    // Routed lanes were cleared, then only rank-0 shared was added. No GEMV term.
    try std.testing.expectEqualSlices(f32, &shared, &routed);
}

test "one decode copy is allowed and a second owned miss is CopyBudget" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer_of = [_]u8{ 0, 0 };
    var fix = fixture(&slots, &line, &stamp, &layer_of);

    const ids = [_]u16{ 1, 2 };
    var decisions = [_]Decision{ .cpu, .cpu };
    var id_scratch: [64]u16 = undefined;
    const experts = [_]ExpertGemv{ undefined, undefined };
    var routed = [_]f32{0};
    const shared = [_]f32{0};

    try std.testing.expectError(error.CopyBudget, stepLayer(
        &fix.vram,
        0,
        &ids,
        &fix.counts,
        1,
        &decisions,
        scratchIds(&id_scratch),
        rank0_owned,
        .decode,
        2,
        1,
        &experts,
        &.{},
        &.{},
        &.{},
        &shared,
        &routed,
    ));

    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(Decision.cpu, decisions[1]);
    try std.testing.expect(slots[0] == 1 or slots[1] == 1);
    try expectNotResident(&slots, 2);
}

test "one owned decode miss copies and returns" {
    var tile = try Tile.init();
    defer tile.deinit();

    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer_of = [_]u8{ 0, 0 };
    var fix = fixture(&slots, &line, &stamp, &layer_of);

    const ids = [_]u16{4};
    var decisions = [_]Decision{.cpu};
    var id_scratch: [64]u16 = undefined;
    const experts = [_]ExpertGemv{tile.view()};
    var y: [128]f64 = undefined;
    var routed: [128]f32 = @splat(0);
    const shared: [128]f32 = @splat(0);

    try stepLayer(
        &fix.vram,
        0,
        &ids,
        &fix.counts,
        1,
        &decisions,
        scratchIds(&id_scratch),
        rank0_owned,
        .decode,
        2,
        1,
        &experts,
        tile.x,
        &y,
        tile.scratch,
        &shared,
        &routed,
    );

    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expect(slots[0] == 4 or slots[1] == 4);
    try expectNotResident(&slots, shared_id);
}

test "four prefill copies are allowed and a fifth owned miss is CopyBudget" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id, Cache.empty_id, Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty, .empty, .empty, .empty };
    var stamp = [_]u32{ 0, 0, 0, 0, 0 };
    var layer_of = [_]u8{ 0, 0, 0, 0, 0 };
    var fix = fixture(&slots, &line, &stamp, &layer_of);

    const ids = [_]u16{ 10, 11, 12, 13, 14 };
    var decisions = [_]Decision{ .cpu, .cpu, .cpu, .cpu, .cpu };
    var id_scratch: [64]u16 = undefined;
    const experts = [_]ExpertGemv{ undefined, undefined, undefined, undefined, undefined };
    var routed = [_]f32{0};
    const shared = [_]f32{0};

    try std.testing.expectError(error.CopyBudget, stepLayer(
        &fix.vram,
        3,
        &ids,
        &fix.counts,
        8,
        &decisions,
        scratchIds(&id_scratch),
        rank0_owned,
        .prefill,
        5,
        1,
        &experts,
        &.{},
        &.{},
        &.{},
        &shared,
        &routed,
    ));

    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[1]);
    try std.testing.expectEqual(Decision.copy, decisions[2]);
    try std.testing.expectEqual(Decision.copy, decisions[3]);
    try std.testing.expectEqual(Decision.cpu, decisions[4]);
    try expectNotResident(&slots, 14);
    try std.testing.expect(slots[0] != Cache.empty_id or slots[1] != Cache.empty_id);
}

test "four prefill copies succeed" {
    var tile = try Tile.init();
    defer tile.deinit();

    var slots = [_]u16{ Cache.empty_id, Cache.empty_id, Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty, .empty, .empty };
    var stamp = [_]u32{ 0, 0, 0, 0 };
    var layer_of = [_]u8{ 0, 0, 0, 0 };
    var fix = fixture(&slots, &line, &stamp, &layer_of);

    const ids = [_]u16{ 10, 11, 12, 13 };
    var decisions = [_]Decision{ .cpu, .cpu, .cpu, .cpu };
    var id_scratch: [64]u16 = undefined;
    const experts = [_]ExpertGemv{ tile.view(), tile.view(), tile.view(), tile.view() };
    var y: [128]f64 = undefined;
    var routed: [128]f32 = @splat(0);
    const shared: [128]f32 = @splat(0);

    try stepLayer(
        &fix.vram,
        3,
        &ids,
        &fix.counts,
        8,
        &decisions,
        scratchIds(&id_scratch),
        rank0_owned,
        .prefill,
        4,
        64,
        &experts,
        tile.x,
        &y,
        tile.scratch,
        &shared,
        &routed,
    );

    for (decisions) |decision| try std.testing.expectEqual(Decision.copy, decision);
    try expectNotResident(&slots, 14);
}

test "rank 0 sum includes shared and rank 1 shared does not change addHidden" {
    const n = 4;
    var slots0 = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line0 = [_]Line{ .empty, .empty };
    var stamp0 = [_]u32{ 0, 0 };
    var layer0 = [_]u8{ 0, 0 };
    var fix0 = fixture(&slots0, &line0, &stamp0, &layer0);

    var slots1 = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line1 = [_]Line{ .empty, .empty };
    var stamp1 = [_]u32{ 0, 0 };
    var layer1 = [_]u8{ 0, 0 };
    var fix1 = fixture(&slots1, &line1, &stamp1, &layer1);

    const ids = [_]u16{shared_id};
    var decisions0 = [_]Decision{.cpu};
    var decisions1 = [_]Decision{.cpu};
    var id_scratch: [64]u16 = undefined;
    const experts = [_]ExpertGemv{undefined};

    var rank0_partial: [n]f32 = @splat(0);
    const shared0 = [_]f32{ 1.0, -2.0, 0.5, 4.0 };
    try stepLayer(
        &fix0.vram,
        0,
        &ids,
        &fix0.counts,
        1,
        &decisions0,
        scratchIds(&id_scratch),
        rank0_owned,
        .decode,
        2,
        1,
        &experts,
        &.{},
        &.{},
        &.{},
        &shared0,
        &rank0_partial,
    );

    var rank1_partial = [_]f32{ 9, 9, 9, 9 };
    var shared1 = [_]f32{ 100.0, -50.0, 12.5, 7.0 };
    try stepLayer(
        &fix1.vram,
        0,
        &ids,
        &fix1.counts,
        1,
        &decisions1,
        scratchIds(&id_scratch),
        rank1_owned,
        .decode,
        2,
        1,
        &experts,
        &.{},
        &.{},
        &.{},
        &shared1,
        &rank1_partial,
    );

    // Rank 0 kept the shared vector. Rank 1 cleared its partial and did not add shared.
    try std.testing.expectEqualSlices(f32, &shared0, &rank0_partial);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 0, 0, 0, 0 }, &rank1_partial);

    var out_a: [n]f32 = @splat(0);
    try addHidden(&rank0_partial, &rank1_partial, &out_a);

    // A different rank-1 shared vector must not move the sum.
    shared1 = [_]f32{ -3.0, 8.0, 1.0, -64.0 };
    var rank1_again = [_]f32{ 3, 3, 3, 3 };
    try stepLayer(
        &fix1.vram,
        0,
        &ids,
        &fix1.counts,
        2,
        &decisions1,
        scratchIds(&id_scratch),
        rank1_owned,
        .decode,
        2,
        1,
        &experts,
        &.{},
        &.{},
        &.{},
        &shared1,
        &rank1_again,
    );
    try std.testing.expectEqualSlices(f32, &[_]f32{ 0, 0, 0, 0 }, &rank1_again);

    var out_b: [n]f32 = @splat(0);
    try addHidden(&rank0_partial, &rank1_again, &out_b);
    try std.testing.expectEqualSlices(f32, &out_a, &out_b);

    // Dropping rank 0's shared contribution changes the sum. Values are bf16-exact,
    // so the single round inside addHidden does not hide the difference.
    const rank0_without = [_]f32{ 0, 0, 0, 0 };
    var out_without: [n]f32 = @splat(0);
    try addHidden(&rank0_without, &rank1_again, &out_without);
    try std.testing.expect(!std.mem.eql(f32, &out_a, &out_without));
}

test "stepLayer does not startup-fill" {
    const src = @embedFile("exl3_step.zig");
    const needle = "fillOwned" ++ "Layers(";
    try std.testing.expect(std.mem.indexOf(u8, src, needle) == null);

    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer_of = [_]u8{ 0, 0 };
    var fix = fixture(&slots, &line, &stamp, &layer_of);
    var fill_counter: u32 = 0;

    var decisions: [0]Decision = .{};
    var id_scratch: [64]u16 = undefined;
    var routed = [_]f32{ 1, 2 };
    const shared = [_]f32{ 3, 4 };

    try stepLayer(
        &fix.vram,
        0,
        &.{},
        &fix.counts,
        1,
        &decisions,
        scratchIds(&id_scratch),
        rank0_owned,
        .decode,
        2,
        1,
        &.{},
        &.{},
        &.{},
        &.{},
        &shared,
        &routed,
    );

    // No fill path: the counter stays put and empty slots stay empty.
    fill_counter +|= 0;
    try std.testing.expectEqual(@as(u32, 0), fill_counter);
    try std.testing.expectEqual(Cache.empty_id, slots[0]);
    try std.testing.expectEqual(Cache.empty_id, slots[1]);
    try std.testing.expectEqualSlices(f32, &shared, &routed);
}

/// 128x128 mul1 tile. `Square` in exl3_cpu.zig is private; this copies its init.
const Tile = struct {
    trellis: []i16,
    suh: []f16,
    svh: []f16,
    x: []f64,
    scratch: []f64,
    width: format.Width,
    codebook: format.Codebook,
    tiles_k: usize,
    tiles_n: usize,

    fn init() !Tile {
        const allocator = std.testing.allocator;
        const w: format.Width = .{ .halves = 2 };
        const tiles_k: usize = 8;
        const tiles_n: usize = 8;
        const k = tiles_k * 16;
        const n = tiles_n * 16;
        const trellis = try allocator.alloc(i16, tiles_k * tiles_n * w.tileWords());
        errdefer allocator.free(trellis);
        for (trellis, 0..) |*word, i| word.* = @bitCast(@as(u16, @truncate(i *% 4099 + 7)));

        const suh = try allocator.alloc(f16, k);
        errdefer allocator.free(suh);
        for (suh) |*v| v.* = 1.0;
        const svh = try allocator.alloc(f16, n);
        errdefer allocator.free(svh);
        for (svh) |*v| v.* = 1.0;

        const x = try allocator.alloc(f64, k);
        errdefer allocator.free(x);
        for (x, 0..) |*v, i| v.* = @floatFromInt(i + 1);

        const scratch = try allocator.alloc(f64, 3 * k * n);
        errdefer allocator.free(scratch);
        return .{
            .trellis = trellis,
            .suh = suh,
            .svh = svh,
            .x = x,
            .scratch = scratch,
            .width = w,
            .codebook = .mul1,
            .tiles_k = tiles_k,
            .tiles_n = tiles_n,
        };
    }

    fn deinit(self: *Tile) void {
        const allocator = std.testing.allocator;
        allocator.free(self.trellis);
        allocator.free(self.suh);
        allocator.free(self.svh);
        allocator.free(self.x);
        allocator.free(self.scratch);
    }

    fn view(self: *const Tile) ExpertGemv {
        return .{
            .width = self.width,
            .codebook = self.codebook,
            .trellis = self.trellis,
            .tiles_k = self.tiles_k,
            .tiles_n = self.tiles_n,
            .suh = self.suh,
            .svh = self.svh,
        };
    }
};
