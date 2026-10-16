//! Host-side lane for one EXL3 rank pair. No driver, no peer access, no serve loop.
//! `pinned` is `pinRank` and `kv` is `contextKvRank`; this file does not walk headers or split heads.
//! Rank 0's footprint is those two plus the learned lines. A failed walk stays the file total so
//! admission can refuse before open. Rank 1's context id is recorded on its own thread.

const std = @import("std");

/// Current context id on this thread. Rank 0 leaves 0 on the caller. The CUDA lane is not linked here.
threadlocal var current_context: u32 = 0;

pub const Loaded = struct {
    backend: struct {} = .{},
    facts: struct {} = .{},
    rows: u32 = 0,
    stream_bytes: usize = 0,
    free_streams: u32 = 0,
    learn_bytes: usize = 0,
    /// Empty and zero: this host stand-in does not serve.
    model_type: []const u8 = "",
    formats: []const []const u8 = &.{},
    default_context: i64 = 0,
    max_segments: u32 = 0,
    prompt_rows: u32 = 0,
    /// Rank 1's context id. Its thread writes this; rank 0 stays current on the caller.
    context_id: u32 = 0,
    /// Peer access is unsupported on the pair. This stays false; nothing enables it.
    peer_access: bool = false,
    dead: bool = false,

    pub fn deinit(self: *Loaded) void {
        self.dead = true;
    }
};

/// Rank 0's device footprint when the walk succeeded: pin + KV + `lines` * `stride`.
/// Not a fraction of the checkpoint. `lines` and `stride` stay arguments so a measured line is not frozen.
/// A failed walk returns `file_total` unchanged. Adding the pin or the KV there would hide the size refused before open.
pub fn weightShare(pinned: u64, kv: u64, lines: u64, stride: u64, walk_ok: bool, file_total: u64) u64 {
    if (!walk_ok) return file_total;
    return pinned + kv + lines * stride;
}

/// True when the share does not fit the card. A file total from a failed walk refuses the same way.
pub fn openRefuses(share: u64, card_bytes: u64) bool {
    return share > card_bytes;
}

/// Refuse before any `Loaded` exists. A share that fits returns without opening ranks.
pub fn openWith(share: u64, card_bytes: u64) error{Refused}!void {
    if (openRefuses(share, card_bytes)) return error.Refused;
}

/// Rank 0 is context 0 on this thread. Rank 1's id is stored from a joined thread. `peer_access` is left false.
/// The host stand-in only records ids, so `gpa` waits for the CUDA lane that will actually allocate.
pub fn openRanks(gpa: std.mem.Allocator) !Loaded {
    _ = gpa;
    current_context = 0;
    var loaded: Loaded = .{};
    const thread = try std.Thread.spawn(.{}, rank1, .{&loaded});
    thread.join();
    return loaded;
}

fn rank1(loaded: *Loaded) void {
    // The CUDA lane will call `cuCtxSetCurrent` here.
    current_context = 1;
    loaded.context_id = current_context;
}

test "weightShare is the pin, the KV, and the lines, or the file total when the walk failed" {
    try std.testing.expectEqual(@as(u64, 10 + 20 + 12), weightShare(10, 20, 3, 4, true, 999));
    try std.testing.expectEqual(@as(u64, 999), weightShare(10, 20, 3, 4, false, 999));
}

test "openWith refuses a file total bigger than the card and accepts a share that fits" {
    const missed = weightShare(10, 20, 3, 4, false, 999);
    try std.testing.expectError(error.Refused, openWith(missed, 100));
    try openWith(42, 100);
}

test "openRanks joins with rank 1's context id, no peer access, and deinit marks the lane dead" {
    var loaded = try openRanks(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 0), current_context);
    try std.testing.expectEqual(@as(u32, 1), loaded.context_id);
    try std.testing.expect(!loaded.peer_access);
    try std.testing.expect(!loaded.dead);
    loaded.deinit();
    try std.testing.expect(loaded.dead);
}
