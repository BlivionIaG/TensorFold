//! Capture rules that the Python ROCm path does not keep.
const std = @import("std");

pub const Stream = enum { launch, side };
pub const RingUpdate = enum { gdn, kda, kv_slot };

pub const Error = error{
    LutNotWarm,
    LutWarmDuringCapture,
    SyncOnCaptureStream,
    PinnedCopyRefused,
    GraphReplayOnHybridRing,
    RcclEagerBroadcastMixed,
};

pub const State = struct {
    lut_warm: bool = false,
    capturing: bool = false,
    graph_has_collective: bool = false,
    hybrid_ring_dirty: bool = false,
};

pub fn warmLut(state: *State, stream: Stream) Error!void {
    if (state.capturing) return error.LutWarmDuringCapture;
    _ = stream;
    state.lut_warm = true;
}

pub fn beginCapture(state: *State) Error!void {
    if (!state.lut_warm) return error.LutNotWarm;
    if (state.hybrid_ring_dirty) return error.GraphReplayOnHybridRing;
    state.capturing = true;
}

pub fn synchronize(state: *const State, stream: Stream) Error!void {
    if (state.capturing and stream == .launch) return error.SyncOnCaptureStream;
}

pub fn pinnedCopy(state: *const State, stream: Stream) Error!void {
    _ = state;
    if (stream != .side) return error.PinnedCopyRefused;
}

pub fn noteRingUpdate(state: *State, update: RingUpdate) void {
    switch (update) {
        .gdn, .kda => state.hybrid_ring_dirty = true,
        .kv_slot => {},
    }
}

pub fn replay(state: *const State) Error!void {
    if (!state.lut_warm) return error.LutNotWarm;
    if (state.hybrid_ring_dirty) return error.GraphReplayOnHybridRing;
}

pub fn recordCollective(state: *State) Error!void {
    if (!state.capturing) return error.RcclEagerBroadcastMixed;
    state.graph_has_collective = true;
}

pub fn eagerBroadcast(state: *const State) Error!void {
    if (state.capturing or state.graph_has_collective) return error.RcclEagerBroadcastMixed;
}

test "a device LUT is warm before capture, and the launch stream is not synchronized inside it" {
    var state = State{};
    try std.testing.expectError(error.LutNotWarm, beginCapture(&state));
    try warmLut(&state, .launch);
    try beginCapture(&state);
    try std.testing.expectError(error.SyncOnCaptureStream, synchronize(&state, .launch));
    try synchronize(&state, .side);
    try std.testing.expectError(error.LutWarmDuringCapture, warmLut(&state, .launch));
}

test "pinned copies stay on a side stream and out of the graph" {
    var state = State{};
    try warmLut(&state, .side);
    try std.testing.expectError(error.PinnedCopyRefused, pinnedCopy(&state, .launch));
    try pinnedCopy(&state, .side);
    try beginCapture(&state);
    try pinnedCopy(&state, .side);
    try std.testing.expectError(error.PinnedCopyRefused, pinnedCopy(&state, .launch));
}

test "GDN and KDA ring updates are not replayed; a KV slot rewrite can be" {
    var gdn = State{ .lut_warm = true };
    noteRingUpdate(&gdn, .gdn);
    try std.testing.expectError(error.GraphReplayOnHybridRing, replay(&gdn));
    try std.testing.expectError(error.GraphReplayOnHybridRing, beginCapture(&gdn));

    var kda = State{ .lut_warm = true };
    noteRingUpdate(&kda, .kda);
    try std.testing.expectError(error.GraphReplayOnHybridRing, replay(&kda));

    var kv = State{ .lut_warm = true };
    noteRingUpdate(&kv, .kv_slot);
    try replay(&kv);
    try beginCapture(&kv);
}

test "an RCCL collective inside the graph is not paired with an eager broadcast" {
    var state = State{};
    try warmLut(&state, .side);
    try eagerBroadcast(&state);
    try beginCapture(&state);
    try std.testing.expectError(error.RcclEagerBroadcastMixed, eagerBroadcast(&state));
    try recordCollective(&state);
    state.capturing = false;
    try std.testing.expectError(error.RcclEagerBroadcastMixed, eagerBroadcast(&state));
}
