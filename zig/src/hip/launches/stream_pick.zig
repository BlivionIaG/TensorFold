//! The stream tile's lanes a column and columns a lane: the widest columns that fill the card (4 at most on gfx900), the narrowest at RDNA3.5's one and two rows.

const std = @import("std");
const affine = @import("affine.zig");
const Generation = @import("../caps.zig").Generation;

comptime {
    // a column's start round, (col0 >> 3) % rounds, stays the same at every width only while each width divides 8
    for (affine.stream_cbs) |cb| std.debug.assert(@mod(8, cb) == 0);
}

fn cdiv(n: c_int, by: c_int) c_uint {
    return @intCast(@divTrunc(n + by - 1, by));
}

// Output columns a block of the `c`th width: its waves' lane groups times a lane's outputs.
fn perBlock(c: usize, lpc_log2: u5, halves: c_int) c_int {
    return affine.stream_waves * (@as(c_int, 32) >> lpc_log2) * @divExact(affine.stream_cbs[c], halves);
}

/// The row tile of `m` rows, an index into `stream_rows`: 1, 2, then 4 rows a block.
pub fn rowTile(m: c_int) usize {
    return if (m == 1) 0 else if (m == 2) 1 else 2;
}

/// A launch's columns a lane, an index into `stream_cbs`, and its output columns a block.
pub const Tile = struct { at: usize, per_block: c_int };

/// The lanes a column takes, as a power of two's log2: the most that are not past the column's last chunk (k / 32 of them), at most 32.
/// gfx900 takes the most lanes that leave none idle in the last round: k = 2560 is 80 chunks, 16 lanes and 5 rounds, not 32 lanes with
/// 3 rounds of 2.5.
pub fn lanesLog2(generation: Generation, k: c_int) u5 {
    var lpc_log2: u5 = 0;
    while ((@as(c_int, 1) << lpc_log2) < (k >> 5) and lpc_log2 < 5) lpc_log2 += 1;
    if (generation == .gcn5) {
        while (lpc_log2 > 2 and @rem(k >> 5, @as(c_int, 1) << lpc_log2) != 0) lpc_log2 -= 1;
    }
    return lpc_log2;
}

/// The tile for `m` rows of `n` outputs over `items`; a `pair` lane's columns are the gate and up halves of outputs.
pub fn pick(generation: Generation, m: c_int, lpc_log2: u5, bits: c_int, n: c_int, items: c_int, pair: bool) Tile {
    const ri = rowTile(m);
    // RDNA3.5's one- and two-row launches keep more loads in flight with the extra waves of the narrowest
    const narrowest = generation == .rdna3_5 and ri < 2;
    const halves: c_int = if (pair) 2 else 1;
    // the narrowest when no width fits the lane's registers
    var tile: Tile = .{ .at = 0, .per_block = perBlock(0, lpc_log2, halves) };
    var c: usize = affine.stream_cbs.len;
    while (c > 0) {
        c -= 1;
        const cb = affine.stream_cbs[c];
        // within the registers the lane's code words take
        if (cb * affine.stream_rows[ri] > 16 or cb * bits > affine.stream_code_words) continue;
        // gfx900: 8 columns a lane take 215 VGPRs, one wave a SIMD; 4 or 2 keep two or four waves (measured: lm_head 2363 -> 1529 us)
        if (generation == .gcn5 and cb > 4) continue;
        tile = .{ .at = c, .per_block = perBlock(c, lpc_log2, halves) };
        const waves = @as(u64, cdiv(n, tile.per_block)) * affine.stream_waves * @as(u64, @intCast(items)) * cdiv(m, affine.stream_rows[ri]);
        if (!narrowest and waves >= affine.streams_wanted) break;
    }
    return tile;
}

/// `pick`, but at `want` columns a lane (a TF_STREAM_CB experiment; 0 keeps the pick) when that is one of `stream_cbs` and fits the
/// lane's registers.
pub fn pickWanted(want: c_int, generation: Generation, m: c_int, lpc_log2: u5, bits: c_int, n: c_int, items: c_int, pair: bool) Tile {
    var tile = pick(generation, m, lpc_log2, bits, n, items, pair);
    const ri = rowTile(m);
    for (affine.stream_cbs, 0..) |cb, c| {
        if (cb != want or cb * affine.stream_rows[ri] > 16 or cb * bits > affine.stream_code_words) continue;
        tile = .{ .at = c, .per_block = perBlock(c, lpc_log2, if (pair) 2 else 1) };
    }
    return tile;
}

test "the widest columns that fill the card, and RDNA3.5's narrowest at one and two rows" {
    // 4-bit, K = 2048 (32 lanes a column): 12352 and 248320 columns
    try std.testing.expectEqual(@as(usize, 1), pick(.rdna3, 1, 5, 4, 12352, 1, false).at);
    try std.testing.expectEqual(@as(usize, 2), pick(.rdna3, 1, 5, 4, 248320, 1, false).at);
    try std.testing.expectEqual(@as(usize, 2), pick(.rdna3, 2, 5, 4, 248320, 1, false).at);
    try std.testing.expectEqual(@as(usize, 0), pick(.rdna3_5, 1, 5, 4, 12352, 1, false).at);
    try std.testing.expectEqual(@as(usize, 0), pick(.rdna3_5, 1, 5, 4, 248320, 1, false).at);
    try std.testing.expectEqual(@as(usize, 0), pick(.rdna3_5, 2, 5, 4, 248320, 1, false).at);
}

test "three rows and more keep the widest that fills the card on every generation" {
    try std.testing.expectEqual(@as(usize, 1), pick(.rdna3, 4, 5, 4, 248320, 1, false).at);
    try std.testing.expectEqual(@as(usize, 1), pick(.rdna3_5, 4, 5, 4, 248320, 1, false).at);
    try std.testing.expectEqual(@as(usize, 1), pick(.rdna3_5, 3, 5, 4, 248320, 1, false).at);
}

test "a routed pair counts its gate and up as one column, over every item" {
    // 512 gate and up pairs over 40 items fill the card at 8 columns a lane, 4 outputs a wave
    try std.testing.expectEqual(Tile{ .at = 2, .per_block = 16 }, pick(.rdna3, 1, 5, 4, 512, 40, true));
    try std.testing.expectEqual(Tile{ .at = 0, .per_block = 4 }, pick(.rdna3_5, 1, 5, 4, 512, 40, true));
}

test "a width no column count fits falls back to the narrowest tile and its block width" {
    try std.testing.expectEqual(Tile{ .at = 0, .per_block = 8 }, pick(.rdna3, 1, 5, 32, 12352, 1, false));
    try std.testing.expectEqual(Tile{ .at = 0, .per_block = 4 }, pick(.rdna3_5, 1, 5, 32, 512, 9, true));
}

test "gfx900 takes at most 4 columns a lane where RDNA3 takes 8" {
    try std.testing.expectEqual(@as(usize, 2), pick(.rdna3, 1, 5, 4, 248320, 1, false).at);
    try std.testing.expectEqual(@as(usize, 1), pick(.gcn5, 1, 5, 4, 248320, 1, false).at);
    // a launch no width fills ends at the narrowest, as on RDNA3
    try std.testing.expectEqual(pick(.rdna3, 1, 5, 4, 64, 1, false).at, pick(.gcn5, 1, 5, 4, 64, 1, false).at);
}

test "the lanes a column take cover its chunks, and on gfx900 leave none idle in the last round" {
    // k = 2560 is 80 chunks
    try std.testing.expectEqual(@as(u5, 5), lanesLog2(.rdna3, 2560));
    try std.testing.expectEqual(@as(u5, 4), lanesLog2(.gcn5, 2560));
    // 128 and 288 chunks are multiples of 32
    try std.testing.expectEqual(@as(u5, 5), lanesLog2(.gcn5, 4096));
    try std.testing.expectEqual(@as(u5, 5), lanesLog2(.gcn5, 9216));
    // 7 chunks: 8 lanes cover them, 4 leave none idle in the last round
    try std.testing.expectEqual(@as(u5, 3), lanesLog2(.rdna3, 224));
    try std.testing.expectEqual(@as(u5, 2), lanesLog2(.gcn5, 224));
    // never below 4 lanes
    try std.testing.expectEqual(@as(u5, 2), lanesLog2(.gcn5, 96));
    try std.testing.expectEqual(@as(u5, 0), lanesLog2(.rdna3, 32));
}

test "a wanted width replaces the pick where the registers allow it, and 0 keeps the pick" {
    try std.testing.expectEqual(pick(.rdna3, 1, 5, 4, 12352, 1, false), pickWanted(0, .rdna3, 1, 5, 4, 12352, 1, false));
    try std.testing.expectEqual(@as(usize, 2), pickWanted(8, .rdna3, 1, 5, 4, 12352, 1, false).at);
    try std.testing.expectEqual(@as(usize, 0), pickWanted(2, .rdna3, 1, 5, 4, 248320, 1, false).at);
    // 8 columns of 32-bit codes take more words than a lane has: the pick stays
    try std.testing.expectEqual(pick(.rdna3, 1, 5, 32, 12352, 1, false), pickWanted(8, .rdna3, 1, 5, 32, 12352, 1, false));
}

test "the row tile takes 1, 2, then 4 rows a block" {
    try std.testing.expectEqual(@as(usize, 0), rowTile(1));
    try std.testing.expectEqual(@as(usize, 1), rowTile(2));
    try std.testing.expectEqual(@as(usize, 2), rowTile(3));
    try std.testing.expectEqual(@as(usize, 2), rowTile(16));
}
