//! Row-sized pool copies: a state kept at n rows restores into a pool of any capacity from n up, and only n's rows move.
const std = @import("std");
const prefix = @import("kv_prefix.zig");
const Packed = @import("handoff/metal.zig").Packed;
const t = std.testing;

const layout: prefix.Layout = .{ .ratios = &Packed.ratios, .strides = &.{ Packed.main_stride, Packed.index_stride } };

fn pattern(id: usize, sub: usize, row: usize, byte: usize) u8 {
    return @truncate(id * 131 + sub * 31 + row * 7 + byte * 3 + 1);
}

fn fill(buf: []u8, capacity: u32) void {
    for (Packed.ratios, 0..) |r, id| for (layout.strides, 0..) |stride, sub| for (0..capacity / r) |row| for (0..stride) |b| {
        buf[layout.subOffset(id, sub, capacity) + row * stride + b] = pattern(id, sub, row, b);
    };
}

/// Every row `position` uses holds the pattern; the rest still holds `rest`.
fn expectRows(buf: []const u8, capacity: u32, position: u32, rest: u8) !void {
    for (Packed.ratios, 0..) |r, id| for (layout.strides, 0..) |stride, sub| for (0..capacity / r) |row| {
        const used = row < prefix.usedRows(position, r, capacity);
        for (0..stride) |b| {
            const got = buf[layout.subOffset(id, sub, capacity) + row * stride + b];
            try t.expectEqual(if (used) pattern(id, sub, row, b) else rest, got);
        }
    };
}

test "kv prefix: the layout is the hand-off's packed layout, owner by owner" {
    for ([_]u32{ 2, 128, 1000, 4096, 262144, 1048576 }) |capacity| {
        try t.expectEqual(Packed.storageBytes(capacity), layout.storageBytes(capacity));
        const p: Packed = .{ .live = &.{}, .storage = &.{}, .layers = 40, .capacity = capacity };
        for (0..Packed.ratios.len) |id| {
            try t.expectEqual(p.mainOff(id), layout.blockOffset(id, capacity));
            try t.expectEqual(p.mainOff(id), layout.subOffset(id, 0, capacity));
            // the index rows follow the main rows, as the hand-off lands them
            try t.expectEqual(p.mainOff(id) + @as(usize, capacity / Packed.ratios[id]) * Packed.main_stride, layout.subOffset(id, 1, capacity));
        }
    }
    // 890 bytes a token, as the cache plan says
    try t.expectEqual(@as(usize, 1048576 * 890), layout.storageBytes(1048576));
}

test "kv prefix: a kept state is laid out for its position rounded up to a whole group, never past the slot" {
    try t.expectEqual(@as(u32, 0), prefix.capacityFor(0, 4096));
    try t.expectEqual(@as(u32, 2), prefix.capacityFor(1, 4096));
    try t.expectEqual(@as(u32, 128), prefix.capacityFor(128, 4096));
    try t.expectEqual(@as(u32, 130), prefix.capacityFor(129, 4096));
    try t.expectEqual(@as(u32, 4095), prefix.capacityFor(4095, 4095));
    try t.expectEqual(@as(u32, std.math.maxInt(u32)), prefix.capacityFor(std.math.maxInt(u32), std.math.maxInt(u32)));
    try t.expectEqual(@as(usize, 65), prefix.usedRows(129, 2, 4096));
    try t.expectEqual(@as(usize, 64), prefix.usedRows(128, 2, 4096));
    try t.expectEqual(@as(usize, 129), prefix.usedRows(129, 1, 4096));
    try t.expectEqual(@as(usize, 4), prefix.usedRows(9, 2, 9)); // an odd capacity holds no row for the open group
}

test "kv prefix: save at n then restore into a larger pool moves exactly n's rows and leaves the rest" {
    const slot: u32 = 512;
    const big: u32 = 2048;
    const slot_pool = try t.allocator.alloc(u8, layout.storageBytes(slot));
    defer t.allocator.free(slot_pool);
    const big_pool = try t.allocator.alloc(u8, layout.storageBytes(big));
    defer t.allocator.free(big_pool);
    for ([_]u32{ 1, 2, 3, 127, 128, 129, 300, 511, 512 }) |position| {
        fill(slot_pool, slot);
        const kept = prefix.capacityFor(position, slot);
        const saved = try t.allocator.alloc(u8, layout.storageBytes(kept));
        defer t.allocator.free(saved);
        @memset(saved, 0x55);
        try prefix.copy(layout, saved, kept, slot_pool, slot, position);
        try expectRows(saved, kept, position, 0x55);
        // a state saved at n is n-sized: 890 bytes a token
        try t.expect(saved.len <= @as(usize, position + 1) * 890);

        @memset(big_pool, 0xAA);
        try prefix.copy(layout, big_pool, big, saved, kept, position);
        try expectRows(big_pool, big, position, 0xAA);
        // and back into the slot it came from
        @memset(slot_pool, 0xBB);
        try prefix.copy(layout, slot_pool, slot, saved, kept, position);
        try expectRows(slot_pool, slot, position, 0xBB);
    }
}

test "kv prefix: a state restores into a smaller pool that still holds its position" {
    const big: u32 = 1024;
    const small: u32 = 200;
    const big_pool = try t.allocator.alloc(u8, layout.storageBytes(big));
    defer t.allocator.free(big_pool);
    const small_pool = try t.allocator.alloc(u8, layout.storageBytes(small));
    defer t.allocator.free(small_pool);
    fill(big_pool, big);
    @memset(small_pool, 0xCC);
    try prefix.copy(layout, small_pool, small, big_pool, big, 199);
    try expectRows(small_pool, small, 199, 0xCC);
    try t.expectError(error.SnapshotLayout, prefix.copy(layout, small_pool, small, big_pool, big, 201));
}

test "kv prefix: a pool of the wrong size or a position past either pool is refused before any byte moves" {
    var a: [8192]u8 = undefined;
    var b: [8192]u8 = undefined;
    const cap = 8;
    @memset(&b, 7);
    try t.expect(layout.storageBytes(cap) < a.len);
    try t.expectError(error.SnapshotLayout, prefix.copy(layout, a[0..10], cap, b[0..layout.storageBytes(cap)], cap, 4));
    try t.expectError(error.SnapshotLayout, prefix.copy(layout, a[0..layout.storageBytes(cap)], cap, b[0..10], cap, 4));
    try t.expectError(error.SnapshotLayout, prefix.copy(layout, a[0..layout.storageBytes(cap)], cap, b[0..layout.storageBytes(cap)], cap, 9));
    try t.expectEqual(@as(u8, 7), b[0]);
}
