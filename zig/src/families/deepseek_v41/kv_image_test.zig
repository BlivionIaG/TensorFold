//! The disk image of a kept state: ranges read it, and landing them in a blank state of another capacity gives it back.
const std = @import("std");
const prefix = @import("kv_prefix.zig");
const image = @import("kv_image.zig");
const Packed = @import("handoff/metal.zig").Packed;
const t = std.testing;

const layout: prefix.Layout = .{ .ratios = &Packed.ratios, .strides = &.{ Packed.main_stride, Packed.index_stride } };
const header_len = 36;
const live_len = 1000;
const taps_len = 333;
const Range = struct { off: u64, len: u64 };

/// A state at `position` in a pool laid out for `capacity`: a pattern in every byte it holds, and `rest` elsewhere.
const Fixture = struct {
    buf: []u8,
    state: image.State,

    fn init(position: u32, capacity: u32, seed: u8, rest: u8) !Fixture {
        const pool = layout.storageBytes(capacity);
        const buf = try t.allocator.alloc(u8, header_len + live_len + taps_len + pool);
        @memset(buf, rest);
        const s: image.State = .{
            .header = buf[0..header_len],
            .live = buf[header_len..][0..live_len],
            .taps = buf[header_len + live_len ..][0..taps_len],
            .storage = buf[header_len + live_len + taps_len ..],
            .capacity = capacity,
            .position = position,
        };
        if (seed != 0) {
            for (s.header, 0..) |*b, i| b.* = seed +% @as(u8, @truncate(i));
            for (s.live, 0..) |*b, i| b.* = seed *% 3 +% @as(u8, @truncate(i * 5));
            for (s.taps, 0..) |*b, i| b.* = seed *% 7 +% @as(u8, @truncate(i * 11));
            for (layout.ratios, 0..) |r, id| for (layout.strides, 0..) |stride, sub| {
                const n = image.rowsAt(position, r) * stride;
                for (0..n) |i| s.storage[layout.subOffset(id, sub, capacity) + i] = seed +% @as(u8, @truncate(id * 17 + sub * 29 + i * 13));
            };
        }
        return .{ .buf = buf, .state = s };
    }

    fn deinit(f: Fixture) void {
        t.allocator.free(f.buf);
    }

    fn size(f: Fixture) u64 {
        var n: u64 = header_len + live_len + taps_len;
        for (layout.ratios) |r| for (layout.strides) |stride| {
            n += image.rowsAt(f.state.position, r) * stride;
        };
        return n;
    }
};

fn readAll(f: Fixture, out: []u8) void {
    image.move(layout, f.state, &[_]Range{.{ .off = 0, .len = out.len }}, out, false);
}

test "kv image: a state's image holds its words, rings, taps, then each owner's used rows in order" {
    const f = try Fixture.init(300, 512, 9, 0xEE);
    defer f.deinit();
    const out = try t.allocator.alloc(u8, @intCast(f.size()));
    defer t.allocator.free(out);
    readAll(f, out);
    try t.expectEqualSlices(u8, f.state.header, out[0..header_len]);
    try t.expectEqualSlices(u8, f.state.live, out[header_len..][0..live_len]);
    try t.expectEqualSlices(u8, f.state.taps, out[header_len + live_len ..][0..taps_len]);
    var at: usize = header_len + live_len + taps_len;
    for (layout.ratios, 0..) |r, id| for (layout.strides, 0..) |stride, sub| {
        const n: usize = @intCast(image.rowsAt(300, r) * stride);
        try t.expectEqualSlices(u8, f.state.storage[layout.subOffset(id, sub, 512)..][0..n], out[at..][0..n]);
        at += n;
    };
    try t.expectEqual(out.len, at);
}

test "kv image: ranges in odd pieces read the same bytes as one range, and land them in another capacity" {
    for ([_]u32{ 1, 2, 129, 255, 256, 257, 1000, 1023 }) |position| {
        const src = try Fixture.init(position, prefix.capacityFor(position, 1024), 5, 0);
        defer src.deinit();
        const n: usize = @intCast(src.size());
        const whole = try t.allocator.alloc(u8, n);
        defer t.allocator.free(whole);
        readAll(src, whole);
        // the store's cuts: pieces that straddle parts, read back to back
        var pieces: [7]Range = undefined;
        var at: u64 = 0;
        for (&pieces, 0..) |*p, i| {
            const len = if (i == pieces.len - 1) n - at else @min(n - at, 1 + (n * (i + 3)) / 23);
            p.* = .{ .off = at, .len = len };
            at += len;
        }
        const joined = try t.allocator.alloc(u8, n);
        defer t.allocator.free(joined);
        image.move(layout, src.state, &pieces, joined, false);
        try t.expectEqualSlices(u8, whole, joined);
        // landed in a blank state laid out for a far larger capacity, in the same pieces
        const dst = try Fixture.init(position, 4096, 0, 0x55);
        defer dst.deinit();
        image.move(layout, dst.state, &pieces, joined, true);
        const back = try t.allocator.alloc(u8, n);
        defer t.allocator.free(back);
        readAll(dst, back);
        try t.expectEqualSlices(u8, whole, back);
        // and nothing past the used rows moved
        for (layout.ratios, 0..) |r, id| for (layout.strides, 0..) |stride, sub| {
            const used: usize = @intCast(image.rowsAt(position, r) * stride);
            const start = layout.subOffset(id, sub, 4096);
            const end = start + @as(usize, 4096 / r) * stride;
            for (dst.state.storage[start + used .. end]) |b| try t.expectEqual(@as(u8, 0x55), b);
        };
    }
}
