//! Which pool lines the prefill ring owns: lines [0, active) of every pool, the ones that have memory behind them.
const std = @import("std");
const cuda = @import("cuda");
const pool_ = @import("../offload/cuda_exl3_pool.zig");

/// Fills the slot arrays with the first `active` lines of every pool (all lines unless elastic); returns the count.
pub fn fill(pools: []const *pool_.Pool, addr: []u64, off: [][3]u32, room_of: [][3]u32) usize {
    var k: usize = 0;
    for (pools) |p| {
        const lay = p.lines.layout;
        for (0..p.active) |i| {
            addr[k] = p.lines.at(i, 0);
            for (0..3) |w| {
                const stop = if (w < 2) lay.off[w + 1] else lay.line_bytes;
                off[k][w] = @intCast(lay.off[w]);
                room_of[k][w] = @intCast(stop - lay.off[w]);
            }
            k += 1;
        }
    }
    return k;
}

/// Slots the pools hold at most (every line of capacity): what the ring's arrays are sized for.
pub fn capacity(pools: []const *pool_.Pool) usize {
    var n: usize = 0;
    for (pools) |p| n += p.opt.lines;
    return n;
}

fn fake(base: u64, lines: usize, active: usize) pool_.Pool {
    var p: pool_.Pool = undefined;
    p.opt.lines = lines;
    p.active = active;
    p.lines.mem.ptr = base;
    p.lines.layout = cuda.exl3_pool.Layout.init(.{ 300, 700, 300 });
    return p;
}

test "exl3: ring slots cover only the active lines of each pool, at their addresses and rooms" {
    var a = fake(0x10_0000, 40, 12);
    var b = fake(0x90_0000, 40, 40);
    const pools = [_]*pool_.Pool{ &a, &b };
    try std.testing.expectEqual(@as(usize, 80), capacity(&pools));
    var addr: [80]u64 = undefined;
    var off: [80][3]u32 = undefined;
    var room: [80][3]u32 = undefined;
    const lay = a.lines.layout;
    try std.testing.expectEqual(@as(usize, 52), fill(&pools, &addr, &off, &room));
    try std.testing.expectEqual(@as(u64, 0x10_0000), addr[0]);
    try std.testing.expectEqual(@as(u64, 0x10_0000 + 11 * lay.line_bytes), addr[11]);
    try std.testing.expectEqual(@as(u64, 0x90_0000), addr[12]); // the next pool follows the last active line, not line 40
    try std.testing.expectEqual(@as(u64, 0x90_0000 + 39 * lay.line_bytes), addr[51]);
    try std.testing.expectEqual(@as(u32, @intCast(lay.off[1])), off[3][1]);
    try std.testing.expectEqual(@as(u32, @intCast(lay.line_bytes - lay.off[2])), room[3][2]);
    // the controller grew pool a: the same arrays now hold more of it, and pool b's slots move up
    a.active = 20;
    try std.testing.expectEqual(@as(usize, 60), fill(&pools, &addr, &off, &room));
    try std.testing.expectEqual(@as(u64, 0x90_0000), addr[20]);
    // at its minimum a pool still has its lines
    a.active = 10;
    try std.testing.expectEqual(@as(usize, 50), fill(&pools, &addr, &off, &room));
    // not elastic: active is every line, and the slots are what they were before elastic pools existed
    a.active = 40;
    try std.testing.expectEqual(@as(usize, 80), fill(&pools, &addr, &off, &room));
    try std.testing.expectEqual(@as(u64, 0x10_0000 + 39 * lay.line_bytes), addr[39]);
}
