//! One embedding row from the host table. Not a VRAM pin and not a device copy.
//! `stayOf` on the embed name is `host`. One row is `hidden * 2` bytes, and
//! that same row is what both ranks would receive. The other embed bytes stay
//! on the host: this function does not copy the table, and it does not call
//! cudaMemcpy. There is no device on this path.

const std = @import("std");
const shape = @import("exl3_shape.zig");
const place = @import("exl3_place.zig");

/// bf16 hidden width. 2560 * 2.
pub const row_bytes: usize = shape.hidden * 2;

/// Five embed tensors, measured 2026-10-16. None of this is a device allocation.
pub const embed_bytes: u64 = 1_274_685_444;

/// `vocab * hidden * 2`. The class is larger. The difference is not a row.
pub const table_bytes: u64 = @as(u64, shape.vocab) * row_bytes;

/// Bytes that sit with the table on the host and are not gathered.
pub const extra_host_bytes: u64 = embed_bytes - table_bytes;

comptime {
    std.debug.assert(row_bytes == 5120);
    std.debug.assert(table_bytes == 1_271_398_400);
    std.debug.assert(extra_host_bytes == 3_287_044);
    // Not an integer number of rows, so the tail cannot be indexed as one.
    std.debug.assert(extra_host_bytes % row_bytes != 0);
}

/// Copies token `token`'s row. The id is not reduced modulo vocab.
/// Negative and past-vocab ids fail before any index. A row this slice does
/// not hold fails too, including a tail shorter than one row. Only `row_bytes`
/// are copied.
pub fn gather(table: []const u8, token: i64) error{ TokenOutOfRange, ShortTable }![row_bytes]u8 {
    // A negative id cast to usize would wrap. Fail before that cast.
    if (token < 0) return error.TokenOutOfRange;
    const id: u64 = @intCast(token);
    if (id >= shape.vocab) return error.TokenOutOfRange;
    const start: usize = @intCast(id * row_bytes);
    if (table.len < start + row_bytes) return error.ShortTable;
    var row: [row_bytes]u8 = undefined;
    @memcpy(row[0..], table[start..][0..row_bytes]);
    return row;
}

test "stayOf embed is host, and the class is not a device copy" {
    const names = [_][]const u8{
        "embed",
        "model.embed_tokens.weight",
    };
    for (names) |name| {
        try std.testing.expectEqual(place.Stay.host, try place.stayOf(.{ .name = name, .rank = 2 }, 0, 2));
        try std.testing.expectEqual(place.Stay.host, try place.stayOf(.{ .name = name, .rank = 2 }, 1, 2));
    }
    try std.testing.expectEqual(@as(usize, 5120), shape.hidden * 2);
    try std.testing.expectEqual(@as(usize, 5120), row_bytes);
    try std.testing.expectEqual(@as(u64, 1_271_398_400), table_bytes);
    try std.testing.expectEqual(@as(u64, 3_287_044), extra_host_bytes);
    try std.testing.expectEqual(embed_bytes, table_bytes + extra_host_bytes);
    // The tail is not another gathered row, and it is not uploaded with one.
    try std.testing.expect(extra_host_bytes % row_bytes != 0);
    try std.testing.expect(extra_host_bytes != 0);
}

test "one token copies one row and leaves the rest of the table" {
    var table: [row_bytes * 2 + 4]u8 = undefined;
    @memset(table[0..row_bytes], 0x10);
    @memset(table[row_bytes .. row_bytes * 2], 0x20);
    @memset(table[row_bytes * 2 ..], 0xFF);
    // Second half of row 1 differs, so a rank cut of this row is visible.
    @memset(table[row_bytes + row_bytes / 2 .. row_bytes * 2], 0x21);

    const row0 = try gather(&table, 0);
    const row1 = try gather(&table, 1);
    try std.testing.expectEqual(@as(usize, 5120), row0.len);
    try std.testing.expectEqualSlices(u8, table[0..row_bytes], &row0);
    try std.testing.expectEqualSlices(u8, table[row_bytes .. row_bytes * 2], &row1);
    try std.testing.expect(row0[0] != row1[0]);
    // The result is one row, not the two-row table and not the 4-byte tail.
    try std.testing.expect(row0.len != table.len);
    try std.testing.expect(std.mem.indexOf(u8, &row0, table[row_bytes * 2 ..]) == null);
    try std.testing.expect(std.mem.indexOf(u8, &row1, table[row_bytes * 2 ..]) == null);

    // A single host row is enough. The other embed bytes are not read.
    var one: [row_bytes]u8 = undefined;
    @memset(&one, 0x3C);
    const only = try gather(&one, 0);
    try std.testing.expectEqualSlices(u8, &one, &only);
    try std.testing.expect(one.len < embed_bytes);

    // Past this slice, even inside vocab. The tail is shorter than a row.
    try std.testing.expectError(error.ShortTable, gather(table[0 .. row_bytes + 4], 1));
}

test "an out-of-range token fails and does not wrap" {
    var table: [row_bytes * 2]u8 = undefined;
    @memset(table[0..row_bytes], 0x10);
    @memset(table[row_bytes..], 0x20);

    // `% vocab` would turn these into row 0 or row 1. They must not.
    try std.testing.expectEqual(@as(usize, 0), @as(usize, shape.vocab) % shape.vocab);
    try std.testing.expectEqual(@as(usize, 1), (shape.vocab + 1) % shape.vocab);
    const row0 = try gather(&table, 0);
    const row1 = try gather(&table, 1);
    try std.testing.expect(row0[0] != row1[0]);

    const bad = [_]i64{
        -1,
        -@as(i64, shape.vocab),
        std.math.minInt(i64),
        shape.vocab,
        shape.vocab + 1,
        shape.vocab * 2,
        std.math.maxInt(i64),
    };
    for (bad) |token| {
        try std.testing.expectError(error.TokenOutOfRange, gather(&table, token));
    }
}

test "both ranks would receive the same 5120-byte row" {
    var table: [row_bytes]u8 = undefined;
    @memset(table[0 .. row_bytes / 2], 0x11);
    @memset(table[row_bytes / 2 ..], 0x22);

    const rank0 = try gather(&table, 0);
    const rank1 = try gather(&table, 0);
    try std.testing.expectEqualSlices(u8, &rank0, &rank1);
    try std.testing.expectEqual(@as(usize, 5120), rank0.len);
    // A cut would hand each rank 2560 bytes. This row still holds both halves.
    try std.testing.expectEqual(@as(u8, 0x11), rank0[0]);
    try std.testing.expectEqual(@as(u8, 0x22), rank0[row_bytes / 2]);
    try std.testing.expect(rank0.len != row_bytes / 2);
}
