//! A layer-split state's restore: layers [0, cut) of a landed Packed state into a live slot
const std = @import("std");
const metal = @import("metal.zig");
const prefix = @import("../kv_prefix.zig");

const Packed = metal.Packed;

pub const Error = error{ BadCut, SnapshotLayout } || prefix.Error;

const pool: prefix.Layout = .{ .ratios = &Packed.ratios, .strides = &.{ Packed.main_stride, Packed.index_stride } };

/// The owners whose layer is below `cut`; the cut must be an owner's layer (a group is never split).
pub fn ownersBelow(cut: u32) Error!usize {
    if (std.mem.indexOfScalar(u32, &Packed.owners, cut) == null) return error.BadCut;
    var n: usize = 0;
    for (Packed.owners) |o| n += @intFromBool(o < cut);
    return n;
}

/// Copy from `src` into `dst` (same layer count; capacities may differ): the window rings of layers [0, cut)
pub fn layers(dst: Packed, src: Packed, cut: u32, position: u32) Error!void {
    const owners = try ownersBelow(cut);
    if (dst.layers != src.layers or dst.live.len != src.live.len or dst.live.len < dst.liveBytes()) return error.SnapshotLayout;
    const ring = @as(usize, Packed.window) * Packed.ring_stride;
    @memcpy(dst.live[0 .. cut * ring], src.live[0 .. cut * ring]);
    for (0..owners) |id| {
        const carry: usize = 2 * @as(usize, Packed.ratios[id]) * 512 * 4;
        @memcpy(dst.live[dst.partialOff(id)..][0..carry], src.live[src.partialOff(id)..][0..carry]);
        @memcpy(dst.live[dst.captureOff(id)..][0..Packed.capture_bytes], src.live[src.captureOff(id)..][0..Packed.capture_bytes]);
    }
    try prefix.copyOwners(pool, owners, dst.storage, dst.capacity, src.storage, src.capacity, position);
}

fn slot(a: std.mem.Allocator, layer_count: u32, capacity: u32, fill: u8) !Packed {
    const probe: Packed = .{ .live = &.{}, .storage = &.{}, .layers = layer_count, .capacity = capacity };
    const p: Packed = .{ .live = try a.alloc(u8, probe.liveBytes()), .storage = try a.alloc(u8, Packed.storageBytes(capacity)), .layers = layer_count, .capacity = capacity };
    @memset(p.live, fill);
    @memset(p.storage, fill);
    return p;
}

test "a restore below the cut takes the source's rings, carries and pool rows, and leaves the rest of the slot alone" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    for ([_]struct { cut: u32, layer_count: u32, position: u32 }{ .{ .cut = 14, .layer_count = 40, .position = 301 }, .{ .cut = 8, .layer_count = 43, .position = 64 }, .{ .cut = 2, .layer_count = 40, .position = 5 } }) |cs| {
        // the slot (capacity 1024) as a prefill left it, and a landed state at its own, smaller capacity
        const dst = try slot(a, cs.layer_count, 1024, 0xd0);
        const src = try slot(a, cs.layer_count, @intCast(cs.position + cs.position % 2), 0x5c);
        const before_live = try a.dupe(u8, dst.live);
        const before_storage = try a.dupe(u8, dst.storage);
        try layers(dst, src, cs.cut, cs.position);
        const owners = try ownersBelow(cs.cut);
        const ring = @as(usize, Packed.window) * Packed.ring_stride;
        // live: rings below the cut, then per owner below the cut its carry and capture; everything else as before
        const want_live = try a.dupe(u8, before_live);
        @memset(want_live[0 .. cs.cut * ring], 0x5c);
        for (0..owners) |id| {
            @memset(want_live[dst.partialOff(id)..][0 .. 2 * @as(usize, Packed.ratios[id]) * 512 * 4], 0x5c);
            @memset(want_live[dst.captureOff(id)..][0..Packed.capture_bytes], 0x5c);
        }
        try std.testing.expectEqualSlices(u8, want_live, dst.live);
        // storage: the rows the position uses of each owner below the cut, both sub-arrays
        const want_storage = try a.dupe(u8, before_storage);
        for (0..owners) |id| {
            const rows = @min(prefix.usedRows(cs.position, Packed.ratios[id], src.capacity), prefix.usedRows(cs.position, Packed.ratios[id], dst.capacity));
            const rows_all = dst.capacity / Packed.ratios[id];
            @memset(want_storage[dst.mainOff(id)..][0 .. rows * Packed.main_stride], 0x5c);
            @memset(want_storage[dst.mainOff(id) + rows_all * Packed.main_stride ..][0 .. rows * Packed.index_stride], 0x5c);
        }
        try std.testing.expectEqualSlices(u8, want_storage, dst.storage);
    }
    // a cut inside a group, or one the state does not have, is refused and changes nothing
    const dst = try slot(a, 40, 64, 1);
    const src = try slot(a, 40, 64, 2);
    for ([_]u32{ 0, 1, 3, 9, 13, 15, 40 }) |bad| try std.testing.expectError(error.BadCut, layers(dst, src, bad, 10));
    for (dst.live) |b| try std.testing.expectEqual(@as(u8, 1), b);
    const wrong = try slot(a, 43, 64, 2);
    try std.testing.expectError(error.SnapshotLayout, layers(dst, wrong, 8, 10));
}
