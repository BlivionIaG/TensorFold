//! Host-side description of what a CUDA graph may bake in versus what replay overwrites.
//! No CUDA calls. A replay binding is overwritten on each replay; a fixed binding stays
//! for the life of that executable graph.

const std = @import("std");

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

const replay_names = [_][]const u8{
    "expert_ids",
    "slot_ptrs",
    "ple_rows",
    "hit_mask",
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

/// Expert ids, slot pointers, PLE rows, the hit mask, and the KV index must be replay inputs.
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

test "the five replay names and the two fixed names" {
    const good = [_]Binding{
        .{ .name = "expert_ids", .address = 1, .replay = true },
        .{ .name = "slot_ptrs", .address = 2, .replay = true },
        .{ .name = "ple_rows", .address = 3, .replay = true },
        .{ .name = "hit_mask", .address = 4, .replay = true },
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
