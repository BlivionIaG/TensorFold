//! The elastic K/V arena: chunks mapped on demand under a pool whose addresses never move, on a fake driver and range.
const std = @import("std");
const cuda = @import("cuda");
const pages = @import("cuda_kv_lanes.zig");
const kvel = @import("cuda_kv_elastic.zig");
const pg = @import("cuda_pages.zig");
const wt = @import("cuda_prompt_wt.zig");
const ranks = @import("cuda_rank_engine.zig");
const wire = @import("cuda_rank_wire.zig");
const E = @import("cuda_prompt.zig").Engine;
const D = cuda.paging_test_device.Device;
const t = std.testing;

/// A fake virtual range at its own base: which chunks are mapped, the calls made and the failures asked for.
pub fn Range(comptime tag: u64) type {
    return struct {
        const base: u64 = (tag + 1) << 44;
        var chunk: usize = 0;
        var mapped: [1024]bool = @splat(false);
        var maps: usize = 0;
        var unmaps: usize = 0;
        pub var calls: usize = 0;
        var opens: usize = 0;
        var closes: usize = 0;
        pub var fail_map_call: usize = 0; // the nth map call fails (0: none)
        var fail_unmap = false;
        var fail_unmap_call: usize = 0; // the nth unmap call fails (0: none)
        var unmap_calls: usize = 0;
        var reserved: usize = 0;
        var anchor: u8 = 0;

        fn reset(bytes: usize) void {
            chunk = bytes;
            mapped = @splat(false);
            maps = 0;
            unmaps = 0;
            calls = 0;
            opens = 0;
            closes = 0;
            fail_map_call = 0;
            fail_unmap = false;
            fail_unmap_call = 0;
            unmap_calls = 0;
            reserved = 0;
        }

        fn open(_: std.mem.Allocator, _: *const cuda.Driver, bytes: usize) anyerror!kvel.Maps {
            opens += 1;
            reserved = bytes;
            return .{ .ctx = &anchor, .base = base, .chunk = chunk, .map = map, .unmap = unmap, .close = close };
        }

        fn map(_: *anyopaque, at: u64, len: usize) bool {
            calls += 1;
            if (calls == fail_map_call) return false;
            const i: usize = @intCast((at - base) / chunk);
            std.debug.assert(len == chunk and (at - base) % chunk == 0 and !mapped[i]);
            mapped[i] = true;
            maps += 1;
            return true;
        }

        fn unmap(_: *anyopaque, at: u64, len: usize) bool {
            unmap_calls += 1;
            if (fail_unmap or unmap_calls == fail_unmap_call) return false;
            const i: usize = @intCast((at - base) / chunk);
            std.debug.assert(len == chunk and mapped[i]);
            mapped[i] = false;
            unmaps += 1;
            return true;
        }

        fn close(_: *anyopaque) void {
            closes += 1;
        }

        fn count() usize {
            var n: usize = 0;
            for (mapped) |m| n += @intFromBool(m);
            return n;
        }

        /// Mapped chunks are exactly the first `n`.
        fn prefix(n: usize) bool {
            for (mapped, 0..) |m, i| if (m != (i < n)) return false;
            return true;
        }
    };
}

/// The driver's fills, with the pool's published pages as each one ran.
const Rec = struct {
    const Fill = struct { at: u64, len: usize, published: usize };
    var fills: [1024]Fill = undefined;
    var n: usize = 0;
    var calls: usize = 0;
    var fail_at: usize = 0; // the nth fill fails (0: none)
    var watch: ?*pages.Pool = null;

    fn reset() void {
        n = 0;
        calls = 0;
        fail_at = 0;
        watch = null;
    }

    fn fill(dst: u64, _: u8, len: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        calls += 1;
        if (calls == fail_at) return cuda.abi.error_not_ready;
        if (n < fills.len) {
            fills[n] = .{ .at = dst, .len = len, .published = if (watch) |p| p.mappedPages() else 0 };
            n += 1;
        }
        return 0;
    }

    fn legacy(dst: u64, v: u8, len: usize) callconv(.c) c_int {
        return fill(dst, v, len, null);
    }
};

var queue: cuda.Stream = undefined;

pub fn driver() cuda.Driver {
    Rec.reset();
    var d = D.driver();
    d.api.cuMemsetD8Async = Rec.fill;
    d.api.cuMemsetD8_v2 = Rec.legacy;
    return d;
}

pub fn engine(d: *cuda.Driver) E {
    var e: E = undefined;
    e.driver = d;
    queue = .{ .d = d, .handle = D.stream };
    e.stream = &queue;
    e.eos = 0;
    e.round_graph = .{};
    return e;
}

const tokens = 16;
const stride = pg.pageBytesOf(tokens, wt.attn_n);
const chunk_bytes = stride + stride / 2; // chunks that cut pages: a page is usable only inside mapped bytes

fn chunksOf(pages_n: usize) usize {
    return (pages_n * stride + chunk_bytes - 1) / chunk_bytes;
}

/// The pool, its layout and its page count: the lifetime tests' shape, a lane's run and room for the rest.
pub fn made(e: *E, comptime R: type, streams: usize) !struct { pool: *pages.Pool, count: usize, run: usize } {
    R.reset(chunk_bytes);
    const l = pages.layout(64, tokens, null, 0);
    const count = 3 * l.runPages() + 24;
    const pool = try pages.Pool.initWith(t.allocator, e, l, 64, null, streams, count, .{ .open = R.open });
    return .{ .pool = pool, .count = count, .run = l.runPages() };
}

test "elastic K/V: the arena base, index base and every lane pointer keep the plain layout's offsets and never move" {
    const R = Range(0);
    var d = driver();
    var e = engine(&d);
    defer cuda.completion.drain(&d, e.stream.*, "the test driver must drain before it leaves");
    const m = try made(&e, R, 1);
    const p = m.pool;
    defer p.deinit();
    const l = pages.layout(64, tokens, null, 0);
    const plain = try pages.Pool.initWith(t.allocator, &e, l, 64, null, 1, m.count, null);
    defer plain.deinit();
    try t.expectEqual(R.base, p.arena.ptr);
    try t.expectEqual(plain.ibase - plain.arena.ptr, p.ibase - p.arena.ptr);
    const a = plain.lanes[0].seq;
    const b = p.lanes[0].seq;
    for (0..wt.attn_n) |i| {
        try t.expectEqual(a.kc[i].ptr - plain.arena.ptr, b.kc[i].ptr - p.arena.ptr);
        try t.expectEqual(a.vc[i].ptr - plain.arena.ptr, b.vc[i].ptr - p.arena.ptr);
        try t.expectEqual(a.ikc[i].ptr - plain.arena.ptr, b.ikc[i].ptr - p.arena.ptr);
        try t.expectEqual(a.pooled[i].ptr - plain.arena.ptr, b.pooled[i].ptr - p.arena.ptr);
    }
    const base = p.arena.ptr;
    const ibase = p.ibase;
    const table = p.lanes[0].seq.paged.?.table.ptr;
    const key = p.lanes[0].seq.kc[3].ptr;
    const min = m.run + 1;
    for ([_]usize{ min, m.count, m.count / 2, 0, m.count - 1, m.count }) |target| {
        try p.resizeMapped(target);
        try t.expectEqual(base, p.arena.ptr);
        try t.expectEqual(ibase, p.ibase);
        try t.expectEqual(base, p.lanes[0].seq.paged.?.base);
        try t.expectEqual(ibase, p.lanes[0].seq.paged.?.ibase);
        try t.expectEqual(table, p.lanes[0].seq.paged.?.table.ptr);
        try t.expectEqual(key, p.lanes[0].seq.kc[3].ptr);
        try t.expectEqual(@max(target, min), p.mappedPages());
    }
}

test "elastic K/V: every chunk is mapped and zero-filled before its pages are published, and the ledger counts the mapped bytes" {
    const R = Range(0);
    var d = driver();
    var e = engine(&d);
    defer cuda.completion.drain(&d, e.stream.*, "the test driver must drain before it leaves");
    const before = cuda.memory.usage(false);
    const m = try made(&e, R, 0);
    const p = m.pool;
    const chunks = chunksOf(m.count);
    try t.expectEqual(chunks, R.count());
    try t.expect(R.prefix(chunks));
    try t.expectEqual(m.count * stride, R.reserved);
    try t.expectEqual(chunks * chunk_bytes, p.mappedBytes());
    try t.expectEqual(before.mapped + chunks * chunk_bytes, cuda.memory.usage(false).mapped);
    try t.expect(cuda.memory.usage(false).device >= before.device + chunks * chunk_bytes);
    var seen: usize = 0;
    for (Rec.fills[0..Rec.n]) |f| if (f.at >= R.base and f.at < R.base + chunks * chunk_bytes) {
        try t.expectEqual(R.base + seen * chunk_bytes, f.at); // each chunk whole, in order, once
        try t.expectEqual(@as(usize, chunk_bytes), f.len);
        seen += 1;
    };
    try t.expectEqual(chunks, seen);
    try t.expectEqual(m.count, p.mappedPages());
    try t.expectEqual(@as(usize, 1), R.opens);
    p.deinit();
    try t.expectEqual(@as(usize, 0), R.count());
    try t.expectEqual(@as(usize, 1), R.closes);
    try t.expectEqual(before.mapped, cuda.memory.usage(false).mapped);
}

test "elastic K/V: shrink is refused while a lane holds pages past the target, then unmaps exactly the tail chunks" {
    const R = Range(0);
    var d = driver();
    var e = engine(&d);
    defer cuda.completion.drain(&d, e.stream.*, "the test driver must drain before it leaves");
    const m = try made(&e, R, 1);
    const p = m.pool;
    defer p.deinit();
    var seq = try p.open(20); // the run takes the lowest pages, the stream's pages the next
    const held = m.run + p.layout.pages(20); // lowest-first: pages [0, held) are the lane's
    for (p.lanes[0].tl.pages.ids.items) |id| try t.expect(id < held);
    try t.expect(p.lanes[0].run + m.run <= held);
    const chunks = chunksOf(m.count);
    const accounted = cuda.memory.usage(false).mapped;
    try t.expectError(error.Busy, p.resizeMapped(held - 1));
    try t.expectEqual(m.count, p.mappedPages());
    try t.expectEqual(@as(usize, 0), R.unmaps);
    try t.expect(R.prefix(chunks));
    try t.expectEqual(accounted, cuda.memory.usage(false).mapped);
    try p.resizeMapped(held); // what the lane holds, no more
    try t.expectEqual(held, p.mappedPages());
    try t.expect(R.prefix(chunksOf(held)));
    try t.expectEqual(chunksOf(held) * chunk_bytes, p.mappedBytes());
    try t.expect(p.mappedPages() * stride <= chunksOf(held) * chunk_bytes); // every published page lies inside mapped bytes
    try t.expect(!p.pool.fits(1)); // nothing past the target is ever counted
    try t.expect(p.close(&seq));
    try p.resizeMapped(0); // the least: a page and a run for the lane
    try t.expectEqual(m.run + 1, p.mappedPages());
    try t.expect(R.prefix(chunksOf(m.run + 1)));
    try t.expectEqual(accounted - (chunks - chunksOf(m.run + 1)) * chunk_bytes, cuda.memory.usage(false).mapped);
    // the tail grows back: only the chunks that were gone are mapped and zeroed, and their pages follow the fills
    Rec.reset();
    Rec.watch = p;
    const gone = chunksOf(m.count) - chunksOf(m.run + 1);
    try p.resizeMapped(m.count);
    try t.expectEqual(m.count, p.mappedPages());
    try t.expect(R.prefix(chunks));
    var fresh: usize = 0;
    for (Rec.fills[0..Rec.n]) |f| if (f.at >= R.base and f.at < R.base + chunks * chunk_bytes) {
        try t.expectEqual(@as(usize, chunk_bytes), f.len);
        try t.expectEqual(m.run + 1, f.published); // still the old pages when the fill ran
        fresh += 1;
    };
    try t.expectEqual(gone, fresh);
    seq = try p.open(20);
    try t.expect(p.close(&seq));
}

test "elastic K/V: a failed map, fill or unmap leaves the published pages and the ledger consistent, and a retry succeeds" {
    const R = Range(0);
    var d = driver();
    var e = engine(&d);
    defer cuda.completion.drain(&d, e.stream.*, "the test driver must drain before it leaves");
    const m = try made(&e, R, 0);
    const p = m.pool;
    defer p.deinit();
    try p.resizeMapped(m.run + 1);
    const have = p.mappedPages();
    const chunks = R.count();
    const accounted = cuda.memory.usage(false).mapped;
    R.calls = 0;
    R.fail_map_call = 2; // the second chunk of the grow fails to map
    try t.expectError(error.FlashNextKvMapFailed, p.resizeMapped(m.count));
    try t.expectEqual(have, p.mappedPages());
    try t.expect(R.prefix(chunks)); // the first new chunk was unmapped again
    try t.expectEqual(accounted, cuda.memory.usage(false).mapped);
    R.fail_map_call = 0;
    Rec.reset();
    Rec.fail_at = 2; // the second fill fails
    try t.expectError(error.NotReady, p.resizeMapped(m.count));
    try t.expectEqual(have, p.mappedPages());
    try t.expect(R.prefix(chunks));
    try t.expectEqual(accounted, cuda.memory.usage(false).mapped);
    Rec.reset();
    try p.resizeMapped(m.count); // nothing was poisoned
    try t.expectEqual(m.count, p.mappedPages());
    const full = R.count();
    const full_accounted = cuda.memory.usage(false).mapped;
    const maps_before = R.maps;
    const unmaps_before = R.unmaps;
    R.unmap_calls = 0;
    R.fail_unmap_call = 2; // the second chunk of the shrink will not unmap: the first has already gone
    try t.expectError(error.FlashNextKvMapFailed, p.resizeMapped(m.run + 1));
    try t.expectEqual(@as(usize, 1), R.unmaps - unmaps_before); // the control: the shrink really started
    try t.expectEqual(@as(usize, 1), R.maps - maps_before); // and its chunk came back
    try t.expectEqual(m.count, p.mappedPages()); // every page published again, none lost
    try t.expect(R.prefix(full));
    try t.expectEqual(full_accounted, cuda.memory.usage(false).mapped);
    R.fail_unmap_call = 0;
    try p.resizeMapped(m.run + 1); // nothing was poisoned
    try t.expectEqual(have, p.mappedPages());
    try t.expect(R.prefix(chunks));
    try t.expectEqual(accounted, cuda.memory.usage(false).mapped);
}

test "elastic K/V: the minimum is held, an impossible one is refused before a range is reserved, and the knobs parse" {
    const R = Range(0);
    var d = driver();
    var e = engine(&d);
    defer cuda.completion.drain(&d, e.stream.*, "the test driver must drain before it leaves");
    R.reset(chunk_bytes);
    const l = pages.layout(64, tokens, null, 0);
    const count = 3 * l.runPages() + 24;
    try t.expectError(error.BadFlashNextKvElastic, pages.Pool.initWith(t.allocator, &e, l, 64, null, 1, count, .{ .min_pages = count + 1, .open = R.open }));
    try t.expectEqual(@as(usize, 0), R.opens); // refused before a range was reserved
    const p = try pages.Pool.initWith(t.allocator, &e, l, 64, null, 1, count, .{ .min_pages = 7, .open = R.open });
    defer p.deinit();
    try p.resizeMapped(0);
    try t.expectEqual(@as(usize, 7), p.mappedPages());
    try p.resizeMapped(count * 2);
    try t.expectEqual(count, p.mappedPages());
    const plain = try pages.Pool.initWith(t.allocator, &e, l, 64, null, 1, count, null);
    defer plain.deinit();
    try t.expectError(error.BadFlashNextKvElastic, plain.resizeMapped(1));
    try t.expectEqual(plain.arena.len, plain.mappedBytes());
    try t.expect(!(try kvel.parseOn(null)) and !(try kvel.parseOn("")) and !(try kvel.parseOn(" 0 ")));
    try t.expect(try kvel.parseOn("1"));
    try t.expectError(error.BadFlashNextKvElastic, kvel.parseOn("yes"));
    try t.expectEqual(@as(?usize, null), try kvel.parseMin(null));
    try t.expectEqual(@as(?usize, 12), try kvel.parseMin(" 12"));
    try t.expectError(error.BadFlashNextKvElastic, kvel.parseMin("-1"));
}

test "elastic K/V: chunks round up, the last may reach past the arena, and a page is usable only inside mapped bytes" {
    const R = Range(0);
    var d = driver();
    const e = engine(&d);
    R.reset(1000);
    const zero: kvel.Zero = .{ .d = &d, .stream = e.stream.* };
    var s = try kvel.Span.open(t.allocator, &d, R.open, zero, 2500);
    defer s.close();
    try t.expectEqual([5]usize{ 0, 1, 1, 2, 3 }, [5]usize{ s.chunksFor(0), s.chunksFor(1), s.chunksFor(1000), s.chunksFor(1001), s.chunksFor(1 << 30) });
    try s.grow(3);
    try t.expectEqual(@as(usize, 3000), s.mappedBytes()); // the last chunk is whole,
    try t.expectEqual(@as(usize, 2500), s.usableBytes()); // the pages end where the arena does
    try t.expectEqual(@as(usize, 6), s.usableBytes() / 400); // 400-byte pages: six fit, the seventh would not
    try s.shrink(2);
    try t.expectEqual(@as(usize, 2000), s.usableBytes());
    try t.expectEqual(@as(usize, 5), s.usableBytes() / 400);
    try t.expect(R.prefix(2));
    try s.grow(1); // a smaller want maps nothing
    try t.expect(R.prefix(2));
}

test "elastic K/V: the leader resizes both ranks to the same pages, a refusal sends nothing and a split poisons" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // the peer rank is a Unix socket pair
    const R0 = Range(0);
    const R1 = Range(1);
    var d = driver();
    var e = engine(&d);
    defer cuda.completion.drain(&d, e.stream.*, "the test driver must drain before it leaves");
    const m0 = try made(&e, R0, 1);
    const a = m0.pool;
    defer a.deinit();
    const m1 = try made(&e, R1, 1);
    const b = m1.pool;
    defer b.deinit();
    try t.expectEqual(a.tally(), b.tally());
    var pair: [2]std.posix.socket_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
    for (pair) |fd| {
        const flags = std.c.fcntl(fd, std.c.F.GETFL);
        if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
    }
    const peer: wire.Link = .{ .fd = pair[1] };
    var rank: ranks.Session = .{ .gpa = t.allocator, .rank = 0, .link = .{ .fd = pair[0] }, .collective = undefined, .engine = &e, .pool = a };
    rank.collective.failed = false;
    rank.collective.comm = null;
    var lane = try a.open(20);
    try t.expectError(error.ElasticRefused, rank.resizeKv(1)); // Busy here: the peer hears nothing
    try t.expectError(error.RankTimeout, peer.receiveHeader(wire.deadline(20)));
    try t.expect(a.close(&lane) and !rank.stopped);
    const target = m0.count - 10;
    try b.resizeMapped(target); // what the peer's own resize makes of it
    try peer.sendHeader(.{ .kind = .ack, .sequence = 1, .arg = b.tally() }, wire.deadline(1000));
    try rank.resizeKv(target);
    const sent = try peer.receiveHeader(wire.deadline(1000));
    try t.expectEqual(wire.Kind.resize, sent.kind);
    try t.expectEqual(@as(u64, target), sent.arg);
    try t.expectEqual(@as(u64, 1), sent.sequence);
    try t.expectEqual(a.tally(), b.tally());
    try t.expectEqual(target, a.mappedPages());
    try t.expect(!rank.stopped);
    try b.resizeMapped(target + 3); // a peer that mapped something else
    try peer.sendHeader(.{ .kind = .ack, .sequence = 2, .arg = b.tally() }, wire.deadline(1000));
    try t.expectError(error.RemoteRank, rank.resizeKv(target - 2));
    try t.expect(rank.stopped);
    try t.expect(!rank.healthy());
}
