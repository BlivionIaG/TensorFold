//! The slot protocol between the host and an external device: a 64-byte header whose first word is written last, then
//! the payload. Level A hands data over in whole command buffers or stream operations (the flag is stored after the
//! producer is complete). Level B stamps every 16 bytes with the step's sequence number so a reader can see for
//! itself which groups have arrived: a Metal kernel's flag can overtake its data (measured, 1 to 4 buffers of 3,000).
const std = @import("std");
const device = @import("device.zig");

pub const header_bytes = 64;
pub const group_bytes = 16;
pub const group_data = 12;

pub const Kind = enum(u16) { empty = 0, hidden = 1, token = 2, abort = 3 };

/// 0 means ok; any other value is an error the consumer must not mistake for data.
pub const Status = enum(u32) { ok = 0, aborted = 1, peer_lost = 2, bad_reply = 3 };

pub const Header = extern struct {
    /// Written last, with release ordering; sequence numbers start at 1, 0 is an empty slot.
    seq: u32,
    kind: u16,
    layer: u16,
    rows: u32,
    bytes: u32,
    status: u32,
    pad: [header_bytes - 20]u8 = @splat(0),
};

comptime {
    std.debug.assert(@sizeOf(Header) == header_bytes);
}

pub const Error = error{ Timeout, StaleReply, Aborted, Oversize };

pub const Slot = struct {
    base: [*]u8,
    /// Header plus payload.
    capacity: usize,

    pub fn init(base: [*]u8, capacity: usize) Slot {
        std.debug.assert(capacity > header_bytes and @intFromPtr(base) % 64 == 0);
        return .{ .base = base, .capacity = capacity };
    }

    pub fn header(s: Slot) *Header {
        return @ptrCast(@alignCast(s.base));
    }

    pub fn data(s: Slot) []u8 {
        return (s.base + header_bytes)[0 .. s.capacity - header_bytes];
    }

    pub fn clear(s: Slot) void {
        @memset(s.base[0..s.capacity], 0);
    }

    /// The word the sequence number lives in, as the device sees it (offset 0 of the slot).
    pub fn seqWord(s: Slot) *u32 {
        return &s.header().seq;
    }

    pub fn peek(s: Slot) u32 {
        return @atomicLoad(u32, &s.header().seq, .acquire);
    }

    /// Publish a payload the caller holds: copy it in, fill the header fields, then release the sequence word.
    pub fn publish(s: Slot, seq: u32, kind: Kind, layer: u16, rows: u32, payload: []const u8, status: Status) Error!void {
        if (payload.len > s.data().len) return error.Oversize;
        @memcpy(s.data()[0..payload.len], payload);
        s.publishHeader(seq, kind, layer, rows, @intCast(payload.len), status);
    }

    /// Publish a payload that is already in place (a device wrote it): header fields, then the sequence word.
    pub fn publishHeader(s: Slot, seq: u32, kind: Kind, layer: u16, rows: u32, bytes: u32, status: Status) void {
        const h = s.header();
        h.kind = @backingInt(kind);
        h.layer = layer;
        h.rows = rows;
        h.bytes = bytes;
        h.status = @backingInt(status);
        @atomicStore(u32, &h.seq, seq, .release);
    }

    /// Wait until the sequence word equals `seq`; returns the header read after the acquire. `timeout_ns` 0 is unbounded.
    pub fn wait(s: Slot, seq: u32, timeout_ns: u64) Error!Header {
        const start = device.nowNs();
        var spins: u32 = 0;
        while (s.peek() != seq) {
            spins +%= 1;
            if (spins < 2000) {
                std.atomic.spinLoopHint();
                continue;
            }
            if (timeout_ns != 0 and device.nowNs() - start >= timeout_ns) return error.Timeout;
            device.pause(20_000);
        }
        const h = s.header().*;
        if (h.status != 0) return error.Aborted;
        return h;
    }
};

/// Bytes a payload of `n` occupies once stamped.
pub fn stampedLen(n: usize) usize {
    return (n + group_data - 1) / group_data * group_bytes;
}

/// Bytes `rows` rows of `row_bytes` take once stamped, each row on its own groups (a row starts on a group boundary).
pub fn stampedRowsLen(rows: usize, row_bytes: usize) usize {
    return rows * stampedLen(row_bytes);
}

/// `packStamped` for `rows` consecutive rows of `row_bytes` in `src`.
pub fn packStampedRows(dst: []u8, src: []const u8, rows: usize, row_bytes: usize, stamp: u32) void {
    std.debug.assert(src.len >= rows * row_bytes and dst.len >= stampedRowsLen(rows, row_bytes));
    for (0..rows) |r| packStamped(dst[r * stampedLen(row_bytes) ..], src[r * row_bytes ..][0..row_bytes], stamp);
}

/// Pack `src` into `dst` as 16-byte groups of 12 data bytes and the 32-bit stamp (little endian); the last group is zero padded.
pub fn packStamped(dst: []u8, src: []const u8, stamp: u32) void {
    std.debug.assert(dst.len >= stampedLen(src.len));
    var g: usize = 0;
    var at: usize = 0;
    while (at < src.len) : (g += 1) {
        const n = @min(group_data, src.len - at);
        const out = dst[g * group_bytes ..][0..group_bytes];
        @memset(out[0..group_data], 0);
        @memcpy(out[0..n], src[at..][0..n]);
        std.mem.writeInt(u32, out[group_data..][0..4], stamp, .little);
        at += n;
    }
}

/// Read `out.len` payload bytes out of a stamped region. A group counts only when its stamp equals `stamp` before and
/// after its data is copied; a group that has not arrived is re-read, at most `retries` times each. Returns how many
/// re-reads were needed. Visibility of a whole 16-byte group as one unit is assumed, not yet verified on a card.
pub fn readStamped(out: []u8, region: []const u8, stamp: u32, retries: u32) Error!u32 {
    std.debug.assert(region.len >= stampedLen(out.len));
    var again: u32 = 0;
    var g: usize = 0;
    var at: usize = 0;
    while (at < out.len) : (g += 1) {
        const n = @min(group_data, out.len - at);
        const src = region[g * group_bytes ..][0..group_bytes];
        const stamp_ptr: *align(1) const u32 = @ptrCast(src[group_data..].ptr);
        var tries: u32 = 0;
        while (true) {
            const before = @atomicLoad(u32, @as(*align(4) const u32, @alignCast(stamp_ptr)), .acquire);
            if (before == stamp) {
                @memcpy(out[at..][0..n], src[0..n]);
                const after = @atomicLoad(u32, @as(*align(4) const u32, @alignCast(stamp_ptr)), .acquire);
                if (after == stamp) break;
            }
            tries += 1;
            again += 1;
            if (tries > retries) return error.StaleReply;
            if (tries > 64) device.pause(2_000) else std.atomic.spinLoopHint();
        }
        at += n;
    }
    return again;
}

/// `readStamped` for `rows` consecutive stamped rows; returns the re-reads in all.
pub fn readStampedRows(out: []u8, region: []const u8, rows: usize, row_bytes: usize, stamp: u32, retries: u32) Error!u32 {
    std.debug.assert(out.len >= rows * row_bytes and region.len >= stampedRowsLen(rows, row_bytes));
    var again: u32 = 0;
    for (0..rows) |r| again += try readStamped(out[r * row_bytes ..][0..row_bytes], region[r * stampedLen(row_bytes) ..], stamp, retries);
    return again;
}

test "stamped payloads round trip for every length across a group boundary" {
    var src: [100]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = @truncate(i * 7 + 3);
    var region: [stampedLen(100)]u8 align(16) = undefined;
    for (0..src.len + 1) |n| {
        packStamped(&region, src[0..n], 9);
        var out: [100]u8 = @splat(0xaa);
        const again = try readStamped(out[0..n], &region, 9, 0);
        try std.testing.expectEqual(@as(u32, 0), again);
        try std.testing.expectEqualSlices(u8, src[0..n], out[0..n]);
    }
    try std.testing.expectEqual(@as(usize, 16), stampedLen(1));
    try std.testing.expectEqual(@as(usize, 16), stampedLen(12));
    try std.testing.expectEqual(@as(usize, 32), stampedLen(13));
}

test "a stale stamp is never accepted as data" {
    var src: [48]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = @intCast(i);
    var region: [stampedLen(48)]u8 align(16) = undefined;
    packStamped(&region, &src, 4);
    var out: [48]u8 = undefined;
    // the previous step's stamp is still in the second group: the reader refuses after its retries
    std.mem.writeInt(u32, region[16 + group_data ..][0..4], 3, .little);
    try std.testing.expectError(error.StaleReply, readStamped(&out, &region, 4, 5));
    // a stamp from a future step is just as wrong
    packStamped(&region, &src, 4);
    std.mem.writeInt(u32, region[group_bytes + group_data ..][0..4], 5, .little);
    try std.testing.expectError(error.StaleReply, readStamped(&out, &region, 4, 5));
}

test "a late producer is waited for and the data is the producer's" {
    const gpa = std.testing.allocator;
    const n = 4096;
    const region = try gpa.alignedAlloc(u8, .@"16", stampedLen(n));
    defer gpa.free(region);
    @memset(region, 0);
    var src: [n]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = @truncate(i *% 31 +% 1);
    const Late = struct {
        fn go(dst: []u8, from: []const u8) void {
            // the groups land out of order and late, last group first
            const groups = stampedLen(from.len) / group_bytes;
            var packed_all: [stampedLen(n)]u8 = undefined;
            packStamped(&packed_all, from, 1);
            var g = groups;
            while (g > 0) {
                g -= 1;
                device.pause(5_000);
                @memcpy(dst[g * group_bytes ..][0..group_bytes], packed_all[g * group_bytes ..][0..group_bytes]);
            }
        }
    };
    const t = try std.Thread.spawn(.{}, Late.go, .{ region, &src });
    var out: [n]u8 = undefined;
    const again = try readStamped(&out, region, 1, 1_000_000);
    t.join();
    try std.testing.expect(again > 0);
    try std.testing.expectEqualSlices(u8, &src, &out);
}

test "a slot publishes data before its sequence word and refuses an aborted step" {
    var mem: [4096]u8 align(64) = undefined;
    const s: Slot = .init(&mem, mem.len);
    s.clear();
    try std.testing.expectEqual(@as(u32, 0), s.peek());
    try std.testing.expectError(error.Timeout, s.wait(1, 100_000));
    try s.publish(1, .hidden, 3, 2, "hello", .ok);
    const h = try s.wait(1, 1_000);
    try std.testing.expectEqual(@as(u32, 5), h.bytes);
    try std.testing.expectEqualSlices(u8, "hello", s.data()[0..5]);
    try s.publish(2, .abort, 0, 0, "", .peer_lost);
    try std.testing.expectError(error.Aborted, s.wait(2, 1_000));
    var big: [5000]u8 = undefined;
    try std.testing.expectError(error.Oversize, s.publish(3, .hidden, 0, 1, &big, .ok));
}

test "stamped rows start on a group boundary and round trip" {
    var src: [3 * 20]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = @truncate(i * 5 + 1);
    var region: [stampedRowsLen(3, 20)]u8 align(16) = undefined;
    packStampedRows(&region, &src, 3, 20, 7);
    try std.testing.expectEqual(@as(usize, 3 * 32), region.len);
    var out: [3 * 20]u8 = undefined;
    try std.testing.expectEqual(@as(u32, 0), try readStampedRows(&out, &region, 3, 20, 7, 0));
    try std.testing.expectEqualSlices(u8, &src, &out);
    try std.testing.expectError(error.StaleReply, readStampedRows(&out, &region, 3, 20, 8, 2));
}
