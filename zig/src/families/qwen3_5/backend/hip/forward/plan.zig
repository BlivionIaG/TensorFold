//! A lane round's plan on the device; launches depend only on its `Shape`, so one graph serves any streams and caches.

const std = @import("std");
const hip = @import("hip");
const state = @import("state.zig");

const round_shape = @import("round_shape.zig");

pub const buckets = round_shape.buckets;
pub const min_span = round_shape.min_span;
pub const bucketOf = round_shape.bucketOf;
pub const spanOf = round_shape.spanOf;
pub const strideOf = round_shape.strideOf;
pub const Shape = round_shape.Shape;

/// Where each array sits in the plan buffer, in 4-byte words (the 8-byte arrays on even words).
pub const Layout = struct {
    layers: usize,
    tokens: usize,
    pos: usize,
    slot: usize,
    first: usize,
    count: usize,
    keep: usize,
    desc: usize,
    snaps: usize,
    words: usize,
    rope: usize,
    rotary: usize,
    /// Per row, the candidates the top-k launch takes (0: none).
    ks: usize,
    /// The host address of the words the argmax launch also stores its picks to (8 bytes; 0: none).
    mirror: usize,

    pub fn of(rows: usize, slots: usize, layers: usize) Layout {
        var l: Layout = undefined;
        l.layers = layers;
        l.tokens = 0;
        l.pos = rows;
        l.slot = 2 * rows;
        l.first = 3 * rows;
        l.count = l.first + slots;
        l.keep = l.count + slots;
        l.desc = std.mem.alignForward(usize, l.keep + slots, 2);
        l.snaps = l.desc + 2 * slots;
        l.rope = l.snaps + 4 * layers;
        l.rotary = l.rope + 3 * rows;
        l.ks = l.rotary + rows;
        l.mirror = std.mem.alignForward(usize, l.ks + rows, 2);
        l.words = l.mirror + 2;
        return l;
    }
};

fn put64(w: []u32, v: u64) void {
    w[0] = @truncate(v);
    w[1] = @truncate(v >> 32);
}

/// One stream's window of a round: its descriptor, the slot its first row takes, its tokens.
pub const Window = struct { desc: u64, pos: usize, tokens: []const u32, shift: i32 = 0 };

/// Pinned plans a queued round is written in, one after the other: a round's copy reads its words when the GPU gets to it, so a
/// slot is not written again before the rounds that used it ran (two rounds are in flight at most).
pub const pipe_ring = 4;

/// What a round's draw launches read from the plan: each row's candidate count, and where the argmax also stores its picks.
pub const Draws = struct { ks: []const i32 = &.{}, mirror: u64 = 0 };

/// The plan buffer of one engine, as large as its biggest round, and the pinned words a round is written in.
pub const Buffer = struct {
    host: hip.HostBuffer,
    dev: hip.DeviceBuffer,
    /// Pages a head of a layer's pool holds: the kernels find a page's keys and values with it.
    pool_pages: u32,
    /// The plans of the rounds queued ahead (`pipe_ring` slots of `slot_bytes` each) and the next one to write.
    ring: hip.HostBuffer,
    slot_bytes: usize,
    ring_at: usize = 0,

    /// Device bytes a plan of `rows` rows over `layers` layers holds.
    pub fn deviceBytes(rows: usize, layers: usize) usize {
        return 4 * Layout.of(rows, rows + 1, layers).words;
    }

    pub fn init(d: *const hip.Runtime, rows: usize, layers: usize, pool_pages: usize) !Buffer {
        const bytes = deviceBytes(rows, layers);
        var host = try hip.HostBuffer.alloc(d, bytes);
        errdefer host.free();
        var ring = try hip.HostBuffer.alloc(d, pipe_ring * bytes);
        errdefer ring.free();
        return .{ .host = host, .dev = try hip.DeviceBuffer.alloc(d, bytes), .pool_pages = @intCast(pool_pages), .ring = ring, .slot_bytes = bytes };
    }

    pub fn deinit(b: *Buffer) void {
        b.dev.free();
        b.ring.free();
        b.host.free();
    }

    /// The device words of `shape` as launches read them.
    pub fn args(b: *const Buffer, l: Layout) hip.plan_ops.Args {
        const at = b.dev.base();
        return .{
            .pos = at + 4 * l.pos,
            .slot = at + 4 * l.slot,
            .first = at + 4 * l.first,
            .count = at + 4 * l.count,
            .desc = at + 4 * l.desc,
            .snaps = at + 4 * l.snaps,
            .pages = @intCast(state.tableWord(l.layers)),
            .pool = b.pool_pages,
        };
    }

    /// Writes the round into the pinned words: windows, the scratch slot's `pad` rows, each linear layer's snapshots, its draws.
    pub fn fill(b: *Buffer, l: Layout, shape: Shape, windows: []const Window, scratch: u64, snaps: []const [2]u64, draws: Draws) void {
        fillInto(b.host.slice(u32), l, shape, windows, scratch, snaps, draws);
    }

    /// The next ring slot's words, for `fillInto`; the slot after it is next.
    pub fn nextSlot(b: *Buffer) []u32 {
        const at = b.ring_at * b.slot_bytes;
        b.ring_at = (b.ring_at + 1) % pipe_ring;
        return std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(b.ring.bytes[at..][0..b.slot_bytes])));
    }

    /// The filled words up to the device in one copy (a graph holds the copy: it runs from the same pinned words).
    pub fn send(b: *const Buffer, stream: hip.abi.Stream, l: Layout) !void {
        try hip.raw.upload(b.dev, 0, b.host.bytes[0 .. 4 * l.words], stream);
    }

    /// A queued round's filled ring words up to the device, ahead of its graph. A one-row round whose token is already on
    /// the device (its first word: the pick of the round before) is sent from the second word on.
    pub fn sendQueued(b: *const Buffer, stream: hip.abi.Stream, l: Layout, words: []const u32, token_on_device: bool) !void {
        const from: usize = if (token_on_device) 1 else 0;
        if (from == 1 and l.pos != 1) return error.Invalid;
        try hip.raw.upload(b.dev, 4 * from, std.mem.sliceAsBytes(words[from..l.words]), stream);
    }
};

/// Writes a round's plan into `w`: windows, the scratch slot's `pad` rows, each linear layer's snapshots, and its draws.
pub fn fillInto(w: []u32, l: Layout, shape: Shape, windows: []const Window, scratch: u64, snaps: []const [2]u64, draws: Draws) void {
    const slots: usize = shape.slots;
    var at: usize = 0;
    for (windows, 0..) |win, s| {
        w[l.first + s] = @intCast(at);
        w[l.count + s] = @intCast(win.tokens.len);
        w[l.keep + s] = std.math.maxInt(u32);
        put64(w[l.desc + 2 * s ..], win.desc);
        for (win.tokens, 0..) |t, i| {
            w[l.tokens + at + i] = t;
            w[l.pos + at + i] = @intCast(win.pos + i);
            w[l.slot + at + i] = @intCast(s);
            const rp = @as(i32, @intCast(win.pos + i)) + win.shift;
            for (0..3) |axis| w[l.rope + 3 * (at + i) + axis] = @bitCast(rp);
            w[l.rotary + at + i] = @bitCast(rp);
        }
        at += win.tokens.len;
    }
    // slots past the streams' are empty and the last is the scratch slot, where padding rows run from its start
    for (windows.len..slots - 1) |s| {
        w[l.first + s] = 0;
        w[l.count + s] = 0;
        w[l.keep + s] = std.math.maxInt(u32);
        put64(w[l.desc + 2 * s ..], 0);
    }
    const pad = shape.rows - at;
    w[l.first + slots - 1] = @intCast(at);
    w[l.count + slots - 1] = @intCast(pad);
    w[l.keep + slots - 1] = std.math.maxInt(u32);
    put64(w[l.desc + 2 * (slots - 1) ..], scratch);
    for (0..pad) |i| {
        w[l.tokens + at + i] = 0;
        w[l.pos + at + i] = @intCast(i);
        w[l.slot + at + i] = @intCast(slots - 1);
        for (0..3) |axis| w[l.rope + 3 * (at + i) + axis] = @intCast(i);
        w[l.rotary + at + i] = @intCast(i);
    }
    for (snaps, 0..) |pair, i| {
        put64(w[l.snaps + 4 * i ..], pair[0]);
        put64(w[l.snaps + 4 * i + 2 ..], pair[1]);
    }
    // the padding rows, past the draws', take no candidates
    for (0..shape.rows) |i| w[l.ks + i] = if (i < draws.ks.len) @bitCast(draws.ks[i]) else 0;
    put64(w[l.mirror..], draws.mirror);
}

test "layout keeps the eight-byte arrays on even words" {
    const l = Layout.of(5, 4, 3);
    try std.testing.expect(l.desc % 2 == 0 and l.snaps % 2 == 0 and l.mirror % 2 == 0);
    try std.testing.expect(l.words >= l.snaps + 12);
    // the draw words close the plan: nothing lies past the mirror address
    try std.testing.expectEqual(l.mirror + 2, l.words);
    try std.testing.expect(l.ks >= l.rotary + 5 and l.mirror >= l.ks + 5);
}

test "round rotary metadata keeps each lane's shift separate from physical cache positions" {
    const l = Layout.of(5, 3, 0);
    var words: [128]u32 = undefined;
    const windows = [_]Window{
        .{ .desc = 0x1000, .pos = 10, .tokens = &.{ 5, 6 }, .shift = -3 },
        .{ .desc = 0x2000, .pos = 20, .tokens = &.{7}, .shift = -9 },
    };
    fillInto(words[0..l.words], l, .{ .rows = 5, .slots = 3, .span = 64 }, &windows, 0x3000, &.{}, .{});
    try std.testing.expectEqualSlices(u32, &.{ 10, 11, 20, 0, 1 }, words[l.pos..][0..5]);
    try std.testing.expectEqualSlices(u32, &.{ 7, 7, 7, 8, 8, 8, 11, 11, 11, 0, 0, 0, 1, 1, 1 }, words[l.rope..][0..15]);
    try std.testing.expectEqualSlices(u32, &.{ 7, 8, 11, 0, 1 }, words[l.rotary..][0..5]);
}

test "a round's draws: candidate counts a row, zero for padding rows, and the host address the picks also go to" {
    const l = Layout.of(4, 3, 0);
    var words: [128]u32 = @splat(0xdead);
    const windows = [_]Window{.{ .desc = 0x1000, .pos = 3, .tokens = &.{ 9, 8 } }};
    fillInto(words[0..l.words], l, .{ .rows = 4, .slots = 3, .span = 64, .stride = 32, .argmax = false }, &windows, 0x3000, &.{}, .{ .ks = &.{ 24, 0 }, .mirror = 0x1122_3344_5566_7788 });
    try std.testing.expectEqualSlices(u32, &.{ 24, 0, 0, 0 }, words[l.ks..][0..4]);
    try std.testing.expectEqual(@as(u32, 0x5566_7788), words[l.mirror]);
    try std.testing.expectEqual(@as(u32, 0x1122_3344), words[l.mirror + 1]);
    fillInto(words[0..l.words], l, .{ .rows = 4, .slots = 3, .span = 64 }, &windows, 0x3000, &.{}, .{});
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0, 0 }, words[l.ks..][0..4]);
    try std.testing.expectEqual(@as(u32, 0), words[l.mirror] | words[l.mirror + 1]);
}
