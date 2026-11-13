const std = @import("std");
const eh = @import("engram_host.zig");
const ea = @import("engram_ahead.zig");
const t = std.testing;

const dim: usize = 32;
const world: u32 = 2;

/// Rows that are a function of (layer, row, column); `hold` parks reads, `fail` fails them, `overlap` flags two in.
const Fake = struct {
    fail: std.atomic.Value(bool) = .init(false),
    hold: std.atomic.Value(bool) = .init(false),
    entered: std.atomic.Value(u32) = .init(0),
    active: std.atomic.Value(u32) = .init(0),
    overlap: std.atomic.Value(bool) = .init(false),

    fn source(x: *Fake) eh.Source {
        return .{ .ptr = x, .vtable = &.{ .issue = issue, .rows = rows } };
    }

    fn enter(x: *Fake) void {
        if (x.active.fetchAdd(1, .acq_rel) != 0) x.overlap.store(true, .release);
        _ = x.entered.fetchAdd(1, .acq_rel);
        for (0..20) |_| std.Thread.yield() catch {};
        while (x.hold.load(.acquire)) std.Thread.yield() catch {};
    }

    fn issue(p: *anyopaque, _: usize, _: []const i64) anyerror!void {
        const x: *Fake = @ptrCast(@alignCast(p));
        x.enter();
        _ = x.active.fetchSub(1, .acq_rel);
    }

    fn rows(p: *anyopaque, li: usize, idx: []const i64, d: usize, out: []u16) anyerror!void {
        const x: *Fake = @ptrCast(@alignCast(p));
        x.enter();
        defer _ = x.active.fetchSub(1, .acq_rel);
        if (x.fail.load(.acquire)) return error.EngramReadFailed;
        for (idx, 0..) |r, i| for (0..d) |j| {
            out[i * d + j] = @truncate(@as(u64, @bitCast(r)) *% 31 +% j +% li * 7);
        };
    }
};

const Fixture = struct {
    h: eh.Host,
    bytes: []align(4) u8,
    layer_buf: [eh.max_layers]u32 = undefined,

    fn init(a: std.mem.Allocator, fx: *Fixture) !void {
        const ref: []const u8 = @embedFile("fixtures/engram-hash-ref.bin");
        const L = std.mem.readInt(u32, ref[8..12], .little);
        const M = std.mem.readInt(u32, ref[12..16], .little);
        const V = std.mem.readInt(u32, ref[16..20], .little);
        const head = 24 + 4 * L + 8 * L * M;
        fx.bytes = try a.alignedAlloc(u8, .@"4", head + 4 * @as(usize, V));
        @memcpy(fx.bytes[0..head], ref[0..head]);
        @memset(fx.bytes[head..], 0);
        fx.h = try eh.Host.loadLegacyFixture(fx.bytes, 8, 2, 16_000_000, &fx.layer_buf);
    }

    fn cols(fx: *const Fixture) usize {
        return fx.h.headShard(0, world)[1] * dim;
    }
};

fn window(a: std.mem.Allocator, n: usize, seed: u32) ![]u32 {
    const ids = try a.alloc(u32, n);
    for (ids, 0..) |*v, i| v.* = @as(u32, @intCast(i)) *% 7 +% seed;
    return ids;
}

fn inlineRows(a: std.mem.Allocator, fx: *const Fixture, src: eh.Source, layer: u32, ids: []const u32, lb: []const u32) ![]u16 {
    const out = try a.alloc(u16, ids.len * fx.cols());
    errdefer a.free(out);
    try fx.h.windowRows(a, src, ids, lb, layer, 0, world, dim, out);
    return out;
}

fn waitEntered(fake: *Fake, n: u32) void {
    while (fake.entered.load(.acquire) < n) std.Thread.yield() catch {};
}

test "engram ahead: taken rows are the inline path's, byte for byte, on every layer" {
    const a = t.allocator;
    var fx: Fixture = undefined;
    try Fixture.init(a, &fx);
    defer a.free(fx.bytes);
    var fake: Fake = .{};
    const g = try ea.Ahead.open(a, &fx.h, fake.source(), 0, world, dim);
    defer g.close();
    const ids = try window(a, 300, 5);
    defer a.free(ids);
    const lb = [_]u32{ 11, 12, 13 };
    g.submit(ids, &lb);
    for (fx.h.layer_ids) |L| {
        const want = try inlineRows(a, &fx, fake.source(), L, ids, &lb);
        defer a.free(want);
        const got = try a.alloc(u16, want.len);
        defer a.free(got);
        try t.expect(g.take(L, ids, &lb, got));
        try t.expectEqualSlices(u16, want, got);
    }
    try t.expectEqual(@as(usize, 0), g.held());
    try t.expectEqual(@as(u64, fx.h.layer_ids.len), g.stats.took);
    // the buffers go back to the spares: a second window gathers into them and still matches
    const ids2 = try window(a, 300, 9);
    defer a.free(ids2);
    g.submit(ids2, &lb);
    const want = try inlineRows(a, &fx, fake.source(), fx.h.layer_ids[0], ids2, &lb);
    defer a.free(want);
    const got = try a.alloc(u16, want.len);
    defer a.free(got);
    try t.expect(g.take(fx.h.layer_ids[0], ids2, &lb, got));
    try t.expectEqualSlices(u16, want, got);
    g.release();
    try t.expectEqual(@as(usize, 0), g.held());
}

test "engram ahead: a request that is not the queued window is a miss, never other rows" {
    const a = t.allocator;
    var fx: Fixture = undefined;
    try Fixture.init(a, &fx);
    defer a.free(fx.bytes);
    var fake: Fake = .{};
    const g = try ea.Ahead.open(a, &fx.h, fake.source(), 0, world, dim);
    defer g.close();
    const ids = try window(a, 200, 3);
    defer a.free(ids);
    const lb = [_]u32{ 11, 12 };
    g.submit(ids, &lb);
    const L = fx.h.layer_ids[0];
    const out = try a.alloc(u16, ids.len * fx.cols());
    defer a.free(out);
    const other = try a.dupe(u32, ids);
    defer a.free(other);
    other[199] +%= 1;
    try t.expect(!g.take(L, other, &lb, out));
    try t.expect(!g.take(L, ids, lb[0..1], out));
    try t.expect(!g.take(L, ids, &.{}, out));
    try t.expect(!g.take(L +% 1000, ids, &lb, out));
    try t.expect(!g.take(L, ids[0..150], &lb, out[0 .. 150 * fx.cols()]));
    try t.expectEqual(@as(u64, 5), g.stats.missed);
    // the queued window is still there for its own request
    const want = try inlineRows(a, &fx, fake.source(), L, ids, &lb);
    defer a.free(want);
    try t.expect(g.take(L, ids, &lb, out));
    try t.expectEqualSlices(u16, want, out);
    // a buffer of the wrong size is refused and the job dropped: the caller gathers inline
    try t.expect(!g.take(fx.h.layer_ids[fx.h.layer_ids.len - 1], ids, &lb, out[0..10]));
    try t.expectEqual(@as(u64, 1), g.stats.failed);
    g.release();
}

test "engram ahead: decode-size windows never use the worker" {
    const a = t.allocator;
    var fx: Fixture = undefined;
    try Fixture.init(a, &fx);
    defer a.free(fx.bytes);
    var fake: Fake = .{};
    const g = try ea.Ahead.open(a, &fx.h, fake.source(), 0, world, dim);
    defer g.close();
    const ids = try window(a, ea.min_rows, 1);
    defer a.free(ids);
    g.submit(ids, &.{});
    try t.expectEqual(@as(usize, 0), g.held());
    const out = try a.alloc(u16, ids.len * fx.cols());
    defer a.free(out);
    try t.expect(!g.take(fx.h.layer_ids[0], ids, &.{}, out));
    try t.expectEqual(@as(u64, 0), g.stats.missed);
}

test "engram ahead: a failed gather is a miss, and the window can be queued again" {
    const a = t.allocator;
    var fx: Fixture = undefined;
    try Fixture.init(a, &fx);
    defer a.free(fx.bytes);
    var fake: Fake = .{};
    const g = try ea.Ahead.open(a, &fx.h, fake.source(), 0, world, dim);
    defer g.close();
    const ids = try window(a, 200, 3);
    defer a.free(ids);
    const lb = [_]u32{7};
    fake.fail.store(true, .release);
    g.submit(ids, &lb);
    const out = try a.alloc(u16, ids.len * fx.cols());
    defer a.free(out);
    for (fx.h.layer_ids) |L| try t.expect(!g.take(L, ids, &lb, out));
    try t.expectEqual(@as(u64, fx.h.layer_ids.len), g.stats.failed);
    try t.expectEqual(@as(usize, 0), g.held());
    fake.fail.store(false, .release);
    g.submit(ids, &lb);
    const L = fx.h.layer_ids[0];
    const want = try inlineRows(a, &fx, fake.source(), L, ids, &lb);
    defer a.free(want);
    try t.expect(g.take(L, ids, &lb, out));
    try t.expectEqualSlices(u16, want, out);
    g.release();
}

test "engram ahead: the worker's reads and the launch thread's never run inside the table together" {
    const a = t.allocator;
    var fx: Fixture = undefined;
    try Fixture.init(a, &fx);
    defer a.free(fx.bytes);
    var fake: Fake = .{};
    const g = try ea.Ahead.open(a, &fx.h, fake.source(), 0, world, dim);
    defer g.close();
    const ids = try window(a, 200, 3);
    defer a.free(ids);
    const mine = try window(a, 200, 4);
    defer a.free(mine);
    const lb = [_]u32{ 1, 2 };
    g.submit(ids, &lb);
    for (0..8) |_| {
        const rows = try inlineRows(a, &fx, g.source(), fx.h.layer_ids[0], mine, &lb);
        a.free(rows);
    }
    const out = try a.alloc(u16, ids.len * fx.cols());
    defer a.free(out);
    for (fx.h.layer_ids) |L| try t.expect(g.take(L, ids, &lb, out));
    try t.expect(!fake.overlap.load(.acquire));
    g.release();
}

test "engram ahead: held windows are bounded (oldest dropped), a window is queued once" {
    const a = t.allocator;
    var fx: Fixture = undefined;
    try Fixture.init(a, &fx);
    defer a.free(fx.bytes);
    var fake: Fake = .{};
    const g = try ea.Ahead.open(a, &fx.h, fake.source(), 0, world, dim);
    defer g.close();
    const per = fx.h.layer_ids.len + 1;
    const w1 = try window(a, 200, 1);
    defer a.free(w1);
    const w2 = try window(a, 200, 2);
    defer a.free(w2);
    const w3 = try window(a, 200, 3);
    defer a.free(w3);
    const lb = [_]u32{9};
    fake.hold.store(true, .release); // the worker parks inside the first window's first read
    g.submit(w1, &lb);
    waitEntered(&fake, 1);
    g.submit(w1, &lb);
    try t.expectEqual(per, g.held());
    g.submit(w2, &lb);
    try t.expectEqual(2 * per, g.held());
    g.submit(w3, &lb);
    try t.expectEqual(2 * per, g.held());
    const out = try a.alloc(u16, w1.len * fx.cols());
    defer a.free(out);
    try t.expect(!g.take(fx.h.layer_ids[0], w1, &lb, out)); // dropped for the newer windows
    fake.hold.store(false, .release);
    const L = fx.h.layer_ids[0];
    const want = try inlineRows(a, &fx, fake.source(), L, w3, &lb);
    defer a.free(want);
    try t.expect(g.take(L, w3, &lb, out));
    try t.expectEqualSlices(u16, want, out);
    g.release();
    try t.expectEqual(@as(usize, 0), g.held());
}

fn releaseThen(g: *ea.Ahead, done: *std.atomic.Value(bool)) void {
    g.release();
    done.store(true, .release);
}

test "engram ahead: release drops the queued windows, waits for the running one and frees everything" {
    const a = t.allocator;
    var fx: Fixture = undefined;
    try Fixture.init(a, &fx);
    defer a.free(fx.bytes);
    var fake: Fake = .{};
    const g = try ea.Ahead.open(a, &fx.h, fake.source(), 0, world, dim);
    defer g.close();
    const w1 = try window(a, 200, 1);
    defer a.free(w1);
    const w2 = try window(a, 200, 2);
    defer a.free(w2);
    const lb = [_]u32{9};
    fake.hold.store(true, .release);
    g.submit(w1, &lb);
    waitEntered(&fake, 1);
    g.submit(w2, &lb);
    var released: std.atomic.Value(bool) = .init(false);
    const th = try std.Thread.spawn(.{}, releaseThen, .{ g, &released });
    for (0..2000) |_| std.Thread.yield() catch {};
    try t.expect(!released.load(.acquire)); // the running read is not abandoned
    fake.hold.store(false, .release);
    th.join();
    try t.expect(released.load(.acquire));
    try t.expectEqual(@as(usize, 0), g.held());
}

test "engram ahead: close with windows queued and a read running joins and frees everything" {
    const a = t.allocator;
    var fx: Fixture = undefined;
    try Fixture.init(a, &fx);
    defer a.free(fx.bytes);
    var fake: Fake = .{};
    const g = try ea.Ahead.open(a, &fx.h, fake.source(), 0, world, dim);
    const w1 = try window(a, 200, 1);
    defer a.free(w1);
    const lb = [_]u32{9};
    fake.hold.store(true, .release);
    g.submit(w1, &lb);
    waitEntered(&fake, 1);
    fake.hold.store(false, .release);
    g.close();
}
