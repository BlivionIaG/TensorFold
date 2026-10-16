//! Host-side description of what a CUDA graph may bake in versus what replay overwrites.
//! No CUDA calls. A replay binding is overwritten on each replay; a fixed binding stays
//! for the life of that executable graph. One capture per rank, layer, and window.

const std = @import("std");
const cache = @import("exl3_cache.zig");
const offload = @import("exl3_offload.zig");

pub const Binding = struct {
    name: []const u8,
    address: u64,
    /// True: replay input. False: fixed for the life of the executable graph.
    replay: bool,
};

pub const Capture = struct {
    window: u32,
    bindings: []const Binding,
};

/// One executable graph. Rank, layer, and window are the whole key.
pub const Key = struct {
    rank: u8,
    layer: u8,
    window: u32,
};

const replay_names = [_][]const u8{
    "expert_ids",
    "ple_rows",
    "kv_index",
};

const fixed_names = [_][]const u8{
    "dense_weights",
    "shared_weights",
};

fn listed(name: []const u8, table: []const []const u8) bool {
    for (table) |entry| if (std.mem.eql(u8, name, entry)) return true;
    return false;
}

/// Expert ids, PLE rows, and the KV index must be replay inputs.
/// Slot pointers and the hit mask are not published.
pub fn isReplayName(name: []const u8) bool {
    return listed(name, &replay_names);
}

fn isFixedName(name: []const u8) bool {
    return listed(name, &fixed_names);
}

/// Known names must carry the flag this module requires. Unknown names keep the caller's flag.
pub fn valid(c: Capture) bool {
    for (c.bindings) |b| {
        if (isReplayName(b.name)) {
            if (!b.replay) return false;
        } else if (isFixedName(b.name)) {
            if (b.replay) return false;
        }
    }
    return true;
}

/// Same window, same binding order, same replay flags, and the same address on every fixed binding.
/// Replay addresses may differ.
pub fn sameTopology(a: Capture, b: Capture) bool {
    if (a.window != b.window or a.bindings.len != b.bindings.len) return false;
    for (a.bindings, b.bindings) |ba, bb| {
        if (!std.mem.eql(u8, ba.name, bb.name) or ba.replay != bb.replay) return false;
        if (!ba.replay and ba.address != bb.address) return false;
    }
    return true;
}

/// `sameTopology`, and every binding address matches. Eager and the graph stand-in share ids and weights.
pub fn argumentsEqual(a: Capture, b: Capture) bool {
    if (!sameTopology(a, b)) return false;
    for (a.bindings, b.bindings) |ba, bb| {
        if (ba.address != bb.address) return false;
    }
    return true;
}

fn sameKey(a: Key, b: Key) bool {
    return a.rank == b.rank and a.layer == b.layer and a.window == b.window;
}

/// Cold start stays eager through the first reap wave. That is the only eager window.
/// Ready only when `startup` is false, no decision is `.copy`, and no line is `.inflight`.
pub fn captureReady(startup: bool, decisions: []const offload.Decision, lines: []const offload.Line) bool {
    if (startup) return false;
    for (decisions) |d| if (d == .copy) return false;
    for (lines) |line| if (line == .inflight) return false;
    return true;
}

pub const Promote = enum {
    /// Rewrite the expert_ids line-index buffer and replay.
    rewrite_ids,
    /// This key has no capture yet.
    recapture,
};

/// One capture per key. A stored key rewrites expert_ids and does not recapture.
pub fn promote(keys: []const Key, key: Key) Promote {
    for (keys) |have| if (sameKey(have, key)) return .rewrite_ids;
    return .recapture;
}

/// One routed line. `count` is counts[layer][id].
pub const LineUse = struct {
    layer: u8,
    id: u16,
    count: u32,
};

fn colder(a: LineUse, b: LineUse) bool {
    if (a.count != b.count) return a.count < b.count;
    if (a.layer != b.layer) return a.layer < b.layer;
    return a.id < b.id;
}

/// `rank_pinned + kv_rank + n * stride + workspace` fits on the card.
/// Neither pin nor KV is reduced. Overflow does not fit.
fn cardHolds(rank_pinned: u64, kv_rank: u64, n: usize, stride: u64, workspace: u64) bool {
    const rows: u64 = @intCast(n);
    const line_bytes = std.math.mul(u64, rows, stride) catch return false;
    const fixed = std.math.add(u64, rank_pinned, kv_rank) catch return false;
    const with_lines = std.math.add(u64, fixed, line_bytes) catch return false;
    const total = std.math.add(u64, with_lines, workspace) catch return false;
    return total <= cache.card_bytes;
}

fn dropColdest(lines: []LineUse) void {
    var cold: usize = 0;
    for (lines[1..], 1..) |line, i| {
        if (colder(line, lines[cold])) cold = i;
    }
    var i = cold;
    while (i + 1 < lines.len) : (i += 1) lines[i] = lines[i + 1];
}

/// Lines kept once the batch workspace is placed. Kept rows are packed at the front.
/// `workspace <= device_reserve` keeps every line.
/// A larger workspace drops the coldest rows until
/// `rank_pinned + kv_rank + kept * stride + workspace <= card_bytes`.
/// Coldest is the smallest count, then the smaller layer, then the smaller id.
/// `rank_pinned` and `kv_rank` are not reduced. Zero rows that still miss the card
/// return `error.GraphWorkspace`.
pub fn fitWorkspace(
    rank_pinned: u64,
    kv_rank: u64,
    lines: []LineUse,
    stride: u64,
    workspace: u64,
) error{GraphWorkspace}!usize {
    if (workspace <= cache.device_reserve) return lines.len;
    var active: usize = lines.len;
    while (active > 0 and !cardHolds(rank_pinned, kv_rank, active, stride, workspace)) {
        dropColdest(lines[0..active]);
        active -= 1;
    }
    if (!cardHolds(rank_pinned, kv_rank, active, stride, workspace)) return error.GraphWorkspace;
    return active;
}

fn capture(window: u32, bindings: []const Binding) Capture {
    return .{ .window = window, .bindings = bindings };
}

test "replay address change keeps the topology and drops argument equality" {
    const fixed: u64 = 0x2000;
    const a = [_]Binding{
        .{ .name = "expert_ids", .address = 0x1000, .replay = true },
        .{ .name = "dense_weights", .address = fixed, .replay = false },
    };
    const b = [_]Binding{
        .{ .name = "expert_ids", .address = 0x3000, .replay = true },
        .{ .name = "dense_weights", .address = fixed, .replay = false },
    };
    const left = capture(4, &a);
    const right = capture(4, &b);
    try std.testing.expect(sameTopology(left, right));
    try std.testing.expect(!argumentsEqual(left, right));
}

test "a new fixed address is a different topology" {
    const a = [_]Binding{
        .{ .name = "expert_ids", .address = 0x1000, .replay = true },
        .{ .name = "dense_weights", .address = 0x2000, .replay = false },
    };
    const b = [_]Binding{
        .{ .name = "expert_ids", .address = 0x1000, .replay = true },
        .{ .name = "dense_weights", .address = 0x4000, .replay = false },
    };
    try std.testing.expect(!sameTopology(capture(4, &a), capture(4, &b)));
    try std.testing.expect(!argumentsEqual(capture(4, &a), capture(4, &b)));
}

test "a known replay name marked fixed is not a valid capture" {
    const bad = [_]Binding{
        .{ .name = "expert_ids", .address = 0x1000, .replay = false },
    };
    try std.testing.expect(!valid(capture(1, &bad)));
    try std.testing.expect(isReplayName("expert_ids"));
}

test "the three replay names and the two fixed names" {
    const good = [_]Binding{
        .{ .name = "expert_ids", .address = 1, .replay = true },
        .{ .name = "ple_rows", .address = 3, .replay = true },
        .{ .name = "kv_index", .address = 5, .replay = true },
        .{ .name = "dense_weights", .address = 6, .replay = false },
        .{ .name = "shared_weights", .address = 7, .replay = false },
    };
    try std.testing.expect(valid(capture(2, &good)));
    for (replay_names) |name| try std.testing.expect(isReplayName(name));
    try std.testing.expect(!isReplayName("dense_weights"));
    try std.testing.expect(!isReplayName("shared_weights"));

    const shared_replay = [_]Binding{
        .{ .name = "shared_weights", .address = 7, .replay = true },
    };
    try std.testing.expect(!valid(capture(2, &shared_replay)));

    const unknown = [_]Binding{
        .{ .name = "token_ids", .address = 9, .replay = true },
    };
    try std.testing.expect(valid(capture(2, &unknown)));
    try std.testing.expect(!isReplayName("token_ids"));
}

test "slot_ptrs and hit_mask are not replay names" {
    try std.testing.expect(!isReplayName("slot_ptrs"));
    try std.testing.expect(!isReplayName("hit_mask"));
    try std.testing.expect(isReplayName("ple_rows"));
    try std.testing.expect(isReplayName("kv_index"));
}

test "matching addresses are equal arguments" {
    const rows = [_]Binding{
        .{ .name = "expert_ids", .address = 0x11, .replay = true },
        .{ .name = "slot_ptrs", .address = 0x22, .replay = true },
        .{ .name = "dense_weights", .address = 0x33, .replay = false },
        .{ .name = "shared_weights", .address = 0x44, .replay = false },
    };
    const again = [_]Binding{
        .{ .name = "expert_ids", .address = 0x11, .replay = true },
        .{ .name = "slot_ptrs", .address = 0x22, .replay = true },
        .{ .name = "dense_weights", .address = 0x33, .replay = false },
        .{ .name = "shared_weights", .address = 0x44, .replay = false },
    };
    try std.testing.expect(argumentsEqual(capture(1, &rows), capture(1, &again)));

    const other_window = capture(2, &again);
    try std.testing.expect(!sameTopology(capture(1, &rows), other_window));
}

test "startup blocks capture when every decision is hit and every line is resident" {
    const decisions = [_]offload.Decision{ .hit, .hit };
    const lines = [_]offload.Line{ .resident, .resident };
    try std.testing.expect(!captureReady(true, &decisions, &lines));
    try std.testing.expect(captureReady(false, &decisions, &lines));
}

test "a copy decision or an inflight line is not capture ready" {
    const copy = [_]offload.Decision{ .hit, .copy, .cpu };
    const resident = [_]offload.Line{.resident};
    try std.testing.expect(!captureReady(false, &copy, &resident));

    const hits = [_]offload.Decision{.hit};
    const inflight = [_]offload.Line{ .resident, .inflight, .empty };
    try std.testing.expect(!captureReady(false, &hits, &inflight));

    const cpu = [_]offload.Decision{.cpu};
    const quiet = [_]offload.Line{ .resident, .empty };
    try std.testing.expect(captureReady(false, &cpu, &quiet));
}

test "promote on a ready key is rewrite_ids, not recapture" {
    const decisions = [_]offload.Decision{.hit};
    const resident = [_]offload.Line{.resident};
    const key: Key = .{ .rank = 2, .layer = 7, .window = 4 };
    try std.testing.expect(captureReady(false, &decisions, &resident));

    const have = [_]Key{key};
    try std.testing.expectEqual(Promote.rewrite_ids, promote(&have, key));
    try std.testing.expect(promote(&have, key) != .recapture);

    const other_rank: Key = .{ .rank = 3, .layer = 7, .window = 4 };
    const other_layer: Key = .{ .rank = 2, .layer = 8, .window = 4 };
    const other_window: Key = .{ .rank = 2, .layer = 7, .window = 5 };
    try std.testing.expectEqual(Promote.recapture, promote(&have, other_rank));
    try std.testing.expectEqual(Promote.recapture, promote(&have, other_layer));
    try std.testing.expectEqual(Promote.recapture, promote(&have, other_window));
}

test "workspace at device_reserve keeps every line" {
    var lines = [_]LineUse{
        .{ .layer = 1, .id = 2, .count = 1 },
        .{ .layer = 0, .id = 9, .count = 50 },
    };
    const stride: u64 = 8 << 30;
    try std.testing.expect(2 * stride + cache.device_reserve > cache.card_bytes);
    const kept = try fitWorkspace(0, 0, &lines, stride, cache.device_reserve);
    try std.testing.expectEqual(@as(usize, 2), kept);
    try std.testing.expectEqual(@as(u8, 1), lines[0].layer);
    try std.testing.expectEqual(@as(u16, 2), lines[0].id);
    try std.testing.expectEqual(@as(u8, 0), lines[1].layer);
}

test "workspace above device_reserve drops the coldest line first" {
    var lines = [_]LineUse{
        .{ .layer = 2, .id = 1, .count = 10 },
        .{ .layer = 4, .id = 3, .count = 1 },
        .{ .layer = 1, .id = 0, .count = 7 },
    };
    const stride: u64 = 6 << 30;
    const workspace = cache.device_reserve + 1;
    try std.testing.expect(3 * stride + workspace > cache.card_bytes);
    try std.testing.expect(2 * stride + workspace <= cache.card_bytes);
    const kept = try fitWorkspace(0, 0, &lines, stride, workspace);
    try std.testing.expectEqual(@as(usize, 2), kept);
    try std.testing.expectEqual(@as(u32, 10), lines[0].count);
    try std.testing.expectEqual(@as(u8, 2), lines[0].layer);
    try std.testing.expectEqual(@as(u32, 7), lines[1].count);
    try std.testing.expectEqual(@as(u8, 1), lines[1].layer);
}

test "equal counts drop the smaller layer first" {
    var lines = [_]LineUse{
        .{ .layer = 5, .id = 1, .count = 4 },
        .{ .layer = 2, .id = 9, .count = 4 },
    };
    const stride: u64 = 8 << 30;
    const workspace = cache.device_reserve + 1;
    try std.testing.expect(2 * stride + workspace > cache.card_bytes);
    try std.testing.expect(stride + workspace <= cache.card_bytes);
    const kept = try fitWorkspace(0, 0, &lines, stride, workspace);
    try std.testing.expectEqual(@as(usize, 1), kept);
    try std.testing.expectEqual(@as(u8, 5), lines[0].layer);
    try std.testing.expectEqual(@as(u16, 1), lines[0].id);
    try std.testing.expectEqual(@as(u32, 4), lines[0].count);
}

test "pins and kv stay in the sum" {
    const pin: u64 = 8 << 30;
    const kv: u64 = 6 << 30;
    const stride: u64 = 1 << 30;
    const workspace = cache.device_reserve + 1;
    // Original pin and kv. Two lines miss the card; one line fits. Dropping either input would keep both.
    try std.testing.expect(pin + kv + 2 * stride + workspace > cache.card_bytes);
    try std.testing.expect(pin + kv + stride + workspace <= cache.card_bytes);
    try std.testing.expect(2 * stride + workspace <= cache.card_bytes);
    var lines = [_]LineUse{
        .{ .layer = 1, .id = 1, .count = 3 },
        .{ .layer = 0, .id = 2, .count = 1 },
    };
    const kept = try fitWorkspace(pin, kv, &lines, stride, workspace);
    try std.testing.expectEqual(@as(usize, 1), kept);
    try std.testing.expectEqual(@as(u8, 1), lines[0].layer);
    try std.testing.expectEqual(@as(u16, 1), lines[0].id);
    try std.testing.expectEqual(@as(u32, 3), lines[0].count);
}

test "zero lines past the card return GraphWorkspace" {
    const pin: u64 = 10 << 30;
    const kv: u64 = 6 << 30;
    const workspace = cache.device_reserve + 1;
    try std.testing.expect(pin + kv + workspace > cache.card_bytes);
    var lines = [_]LineUse{};
    try std.testing.expectError(error.GraphWorkspace, fitWorkspace(pin, kv, &lines, 1, workspace));
}
