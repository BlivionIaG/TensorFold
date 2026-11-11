//! A kept state as the disk store's byte image: its words, window rings, taps, then each owner's rows by position.
const std = @import("std");
const prefix = @import("kv_prefix.zig");

/// Image parts ahead of the pool's: the words, the rings with the compressor carry, the taps ring.
pub const fixed_parts = 3;
pub const max_regions = fixed_parts + 2 * 8;

/// A kept state's buffers as the image reads them (`header` stands for the words that are not buffers).
pub const State = struct {
    header: []u8,
    live: []u8,
    taps: []u8,
    storage: []u8,
    capacity: u32,
    position: u32,
};

/// Rows of an owner of `ratio` a state at `at` holds: its whole groups and the one a carried row opens.
pub fn rowsAt(at: u32, ratio: u32) u64 {
    return std.math.divCeil(u32, at, ratio) catch unreachable;
}

const Region = struct { off: u64, bytes: []u8 };

fn regions(l: prefix.Layout, s: State, out: *[max_regions]Region) []Region {
    var off: u64 = 0;
    var n: usize = 0;
    for ([fixed_parts][]u8{ s.header, s.live, s.taps }) |b| {
        out[n] = .{ .off = off, .bytes = b };
        off += b.len;
        n += 1;
    }
    for (l.ratios, 0..) |ratio, id| for (l.strides, 0..) |stride, sub| {
        const len: usize = @intCast(rowsAt(s.position, ratio) * stride);
        out[n] = .{ .off = off, .bytes = s.storage[l.subOffset(id, sub, s.capacity)..][0..len] };
        off += len;
        n += 1;
    };
    return out[0..n];
}

/// Copy `ranges` (`.off`, `.len` into the image) between the state and `buf`, back to back; `into_state` lands `buf`.
pub fn move(l: prefix.Layout, s: State, ranges: anytype, buf: []u8, into_state: bool) void {
    var table: [max_regions]Region = undefined;
    const regs = regions(l, s, &table);
    var at: usize = 0;
    for (ranges) |g| {
        var off: u64 = g.off;
        var left: usize = @intCast(g.len);
        for (regs) |r| {
            if (left == 0) break;
            if (off >= r.off + r.bytes.len) continue;
            const from: usize = @intCast(off - r.off);
            const n = @min(left, r.bytes.len - from);
            if (into_state) @memcpy(r.bytes[from..][0..n], buf[at..][0..n]) else @memcpy(buf[at..][0..n], r.bytes[from..][0..n]);
            at += n;
            off += n;
            left -= n;
        }
    }
}
