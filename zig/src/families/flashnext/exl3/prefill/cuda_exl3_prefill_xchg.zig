//! Deep prefill at two ranks: a rank sends the slots it computed to the other, not an all-reduce of the padded block.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const state = @import("cuda_exl3_prefill_state.zig");

const dims = wt.dims;
const slots = wt.slots;
const table_n = wt.experts_n + 1;

const Deep = state.Deep;

/// The pairs of `picks` in [lo, hi) go to `own`, other table entries to `peer`; row counts to `own_row`, `peer_row`.
pub fn lists(picks: []const i32, rows: usize, lo: usize, hi: usize, own: []u32, peer: []u32, own_row: []u32, peer_row: []u32) [2]usize {
    var a: usize = 0;
    var b: usize = 0;
    for (0..rows) |r| {
        own_row[r] = @intCast(a);
        peer_row[r] = @intCast(b);
        for (0..slots) |k| {
            const pair = r * slots + k;
            const pk = picks[pair];
            if (pk < 0 or pk >= table_n) continue;
            if (pk >= lo and pk < hi) {
                own[a] = @intCast(pair);
                a += 1;
            } else {
                peer[b] = @intCast(pair);
                b += 1;
            }
        }
    }
    own_row[rows] = @intCast(a);
    peer_row[rows] = @intCast(b);
    return .{ a, b };
}

/// Whether this deep chunk can use the exchange: two ranks, exact mode, its own buffers (not a two-chunk window's).
pub fn usable(dp: *const Deep, world: usize) bool {
    const o = state.opts();
    return world == 2 and !o.full_reduce and !o.rank_partials and !dp.reduced_branch and dp.shared == null and dp.second == null;
}

/// After a layer's picks reached the host (before another rank's are masked): the pair lists on the device.
pub fn prepare(dp: *Deep, st: cuda.Stream, picks: []const i32, r: ?@import("../tp/cuda_exl3_tp.zig").Rank, n: usize) !void {
    dp.xs.ready = false;
    const rk = r orelse return;
    if (!usable(dp, rk.world)) return;
    if (dp.xs.own_row.len < n + 1) {
        if (dp.xs.own_row.len != 0) dp.gpa.free(dp.xs.own_row);
        if (dp.xs.peer_row.len != 0) dp.gpa.free(dp.xs.peer_row);
        dp.xs.own_row = try dp.gpa.alloc(u32, dp.rows + 1);
        dp.xs.peer_row = try dp.gpa.alloc(u32, dp.rows + 1);
    }
    const pairs = n * slots;
    const own: []u32 = @alignCast(std.mem.bytesAsSlice(u32, dp.hbytes(dp.hl.xs_own, pairs * 4)));
    const peer: []u32 = @alignCast(std.mem.bytesAsSlice(u32, dp.hbytes(dp.hl.xs_peer, pairs * 4)));
    const counts = lists(picks[0..pairs], n, rk.lo, rk.hi, own, peer, dp.xs.own_row, dp.xs.peer_row);
    if (counts[0] != 0) try state.upload(st, dp.dptr(dp.lay.xs_own), std.mem.sliceAsBytes(own[0..counts[0]]));
    if (counts[1] != 0) try state.upload(st, dp.dptr(dp.lay.xs_peer), std.mem.sliceAsBytes(peer[0..counts[1]]));
    dp.xs.ready = true;
}

/// Rows [r0, r0 + m): the owned slots to the peer, the peer's into place, in rounds of at most `dp.rows` slots a way.
pub fn exchange(dp: *Deep, c: anytype, s: cuda.Stream, r0: usize, m: usize) !void {
    const own_lo: usize = dp.xs.own_row[r0];
    const own_n: usize = dp.xs.own_row[r0 + m] - own_lo;
    const peer_lo: usize = dp.xs.peer_row[r0];
    const peer_n: usize = dp.xs.peer_row[r0 + m] - peer_lo;
    const cap = dp.rows;
    const rounds = @max((own_n + cap - 1) / cap, (peer_n + cap - 1) / cap);
    const y = dp.dptr(dp.lay.y);
    const send = dp.dptr(dp.lay.xs_buf);
    const recv = send + cap * dims * 4;
    for (0..rounds) |i| {
        const sent = @min(cap, own_n -| i * cap);
        const got = @min(cap, peer_n -| i * cap);
        if (sent != 0) try move(&dp.kern, s, y, send, dp.dptr(dp.lay.xs_own) + (own_lo + i * cap) * 4, sent, true);
        try c.exchange(s, send, sent * dims, recv, got * dims);
        if (got != 0) try move(&dp.kern, s, y, recv, dp.dptr(dp.lay.xs_peer) + (peer_lo + i * cap) * 4, got, false);
    }
}

fn move(k: *const @import("cuda_exl3_prefill_k.zig").Kernels, s: cuda.Stream, y: u64, buf: u64, idx: u64, count: usize, pack: bool) !void {
    var a: cuda.Args = .{};
    a.add(y);
    a.add(buf);
    a.add(idx);
    a.add(@as(i32, @intCast(count)));
    a.add(@as(i32, @intCast(dims)));
    a.add(@as(i32, @intFromBool(!pack)));
    try cuda.launch.launch(k.xs_move, .{ .grid = .{ .x = @intCast(@min(count, 4096)) }, .block = .{ .x = 256 } }, s, &a);
}

test "xchg: lists split a row's slots by owner, skip empty picks and keep row prefixes" {
    // 3 rows of 4 slots (the module's slots is larger: the rest are -1); rank owns ids [0, 256)
    var picks: [3 * slots]i32 = @splat(-1);
    picks[0] = 5;
    picks[1] = 300;
    picks[2] = 512; // the shared expert: the other rank's here
    picks[slots + 0] = 700; // outside the table: nobody's
    picks[slots + 3] = 255;
    picks[2 * slots + 1] = 256;
    var own: [3 * slots]u32 = undefined;
    var peer: [3 * slots]u32 = undefined;
    var own_row: [4]u32 = undefined;
    var peer_row: [4]u32 = undefined;
    const n = lists(&picks, 3, 0, 256, &own, &peer, &own_row, &peer_row);
    try std.testing.expectEqual([2]usize{ 2, 3 }, n);
    try std.testing.expectEqualSlices(u32, &.{ 0, @intCast(slots + 3) }, own[0..2]);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, @intCast(2 * slots + 1) }, peer[0..3]);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 2 }, &own_row);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 2, 3 }, &peer_row);
}
