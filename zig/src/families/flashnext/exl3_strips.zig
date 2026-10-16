//! Checkpoint trellis order versus the kernel's strip order.
//! Stored tiles are [K/16, N/16, words]. Strips are [N/128, K/16, 8, words]:
//! one 128-column block, then every K tile, then the eight 16-wide tiles.
//! Same permutation as v0.6.6 `strips` in `src/tensorfold/cuda/exl3/linear.py`.
//! No allocation. The caller owns `out`.

const std = @import("std");

pub const tiles_per_strip: usize = 8;

/// Tile index of `(tk, tn)` after the strips permutation. `tn / 8` is the 128-column block.
pub fn stripsTile(tiles_k: usize, tk: usize, tn: usize) usize {
    const strip = tn / tiles_per_strip;
    const local = tn % tiles_per_strip;
    return strip * tiles_k * tiles_per_strip + tk * tiles_per_strip + local;
}

/// Stored tile index, row-major over K then N.
pub fn storedTile(tiles_n: usize, tk: usize, tn: usize) usize {
    return tk * tiles_n + tn;
}

/// Copy `stored` int16 words into strip order. `words` is the int16 count of one tile.
/// `tiles_n` must be a multiple of 8 (N is a multiple of 128). Lengths must be `tiles_k * tiles_n * words`.
pub fn reorder(stored: []const i16, tiles_k: usize, tiles_n: usize, words: usize, out: []i16) error{Length}!void {
    if (words == 0 or tiles_k == 0 or tiles_n == 0 or tiles_n % tiles_per_strip != 0) return error.Length;
    const ntiles = tiles_k * tiles_n;
    const need = ntiles * words;
    if (stored.len != need or out.len != need) return error.Length;
    var tk: usize = 0;
    while (tk < tiles_k) : (tk += 1) {
        var tn: usize = 0;
        while (tn < tiles_n) : (tn += 1) {
            const from = storedTile(tiles_n, tk, tn) * words;
            const to = stripsTile(tiles_k, tk, tn) * words;
            @memcpy(out[to .. to + words], stored[from .. from + words]);
        }
    }
}

test "a 128-column block walks K before the next block" {
    const tiles_k: usize = 4;
    try std.testing.expectEqual(@as(usize, 0), stripsTile(tiles_k, 0, 0));
    try std.testing.expectEqual(@as(usize, 8), stripsTile(tiles_k, 1, 0));
    try std.testing.expectEqual(@as(usize, 1), stripsTile(tiles_k, 0, 1));
    // Strip 1 starts after every K tile of strip 0: 4 * 8.
    try std.testing.expectEqual(@as(usize, 32), stripsTile(tiles_k, 0, 8));
    try std.testing.expectEqual(@as(usize, 32 + 8 + 3), stripsTile(tiles_k, 1, 11));
    // Eight local tiles of one K step are contiguous, so a warp's next column is the next word group.
    try std.testing.expectEqual(@as(usize, 7), stripsTile(tiles_k, 0, 7));
}

test "reorder matches the view-permute and refuses a short N" {
    const tiles_k: usize = 2;
    const tiles_n: usize = 16;
    const words: usize = 4;
    var stored: [2 * 16 * 4]i16 = undefined;
    for (&stored, 0..) |*w, i| w.* = @intCast(i);
    var out: [2 * 16 * 4]i16 = @splat(-1);
    try reorder(&stored, tiles_k, tiles_n, words, &out);

    var tk: usize = 0;
    while (tk < tiles_k) : (tk += 1) {
        var tn: usize = 0;
        while (tn < tiles_n) : (tn += 1) {
            const from = storedTile(tiles_n, tk, tn) * words;
            const to = stripsTile(tiles_k, tk, tn) * words;
            try std.testing.expectEqualSlices(i16, stored[from .. from + words], out[to .. to + words]);
        }
    }
    // Word 0 of stored tile (1, 9) lands at strip 1, k 1, local 1.
    const dest = stripsTile(tiles_k, 1, 9) * words;
    try std.testing.expectEqual(stored[storedTile(tiles_n, 1, 9) * words], out[dest]);

    var y: [8]i16 = @splat(0);
    var short: [8]i16 = @splat(7);
    try std.testing.expectError(error.Length, reorder(&y, 1, 4, 2, &short));
    try std.testing.expectEqual(@as(i16, 7), short[0]);
    try std.testing.expectError(error.Length, reorder(stored[0..8], tiles_k, tiles_n, words, &out));
}
