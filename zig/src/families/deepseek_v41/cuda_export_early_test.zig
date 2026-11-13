//! A layer-split export's producer side over synthetic state: the early entry, rows streamed while it computes
const std = @import("std");
const fabric = @import("fabric");
const plan = @import("handoff/plan.zig");
const ex = @import("handoff/export.zig");
const split = @import("handoff/split.zig");
const s = @import("handoff/synth.zig");
const cx = @import("cuda_export.zig");
const cxt = @import("cuda_export_test.zig");
const file = @import("handoff_file.zig");
const serve = @import("cuda_export_serve.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;
const model = "dsv41-early-test";
const upto: u64 = 700;
const cut: u32 = 14;

/// A source whose layer-split state is a synthetic world's, as the engine's capture hands it
const Fake = struct {
    arena: std.heap.ArenaAllocator,
    world: s.World,
    snap: cxt.Snap,
    released: u32 = 0,
    taken_cut: u32 = 0,

    fn init() !*Fake {
        const f = try gpa.create(Fake);
        errdefer gpa.destroy(f);
        f.* = .{ .arena = .init(gpa), .world = undefined, .snap = undefined };
        errdefer f.arena.deinit();
        f.world = try s.make(f.arena.allocator(), .{ .upto = upto, .ds_rows = 128, .seed = 77 });
        f.snap = try cxt.snapshotOf(f.arena.allocator(), f.world);
        return f;
    }

    fn deinit(f: *Fake) void {
        f.arena.deinit();
        gpa.destroy(f);
    }

    fn source(f: *Fake) cx.Source {
        return .{ .ptr = f, .take = take, .release = release, .take_cut = takeCut };
    }

    fn take(_: *anyopaque, _: std.mem.Allocator, _: u64) anyerror!cx.Captured {
        return error.NotASplitState;
    }

    fn takeCut(ptr: *anyopaque, _: std.mem.Allocator, at: u64, layer_cut: u32) anyerror!cx.Captured {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        const a = f.arena.allocator();
        f.taken_cut += 1;
        var pieces: std.ArrayList(cx.Piece) = .empty;
        for (f.snap.pieces) |p| if (std.mem.indexOf(u8, p.role, ".ring.") == null) try pieces.append(a, .{ .role = p.role, .bytes = try a.dupe(u8, p.bytes) });
        const pages = try a.dupe(cx.Pages, f.snap.pages);
        for (pages) |*pg| if (pg.layer >= layer_cut) {
            pg.comp = &.{};
            pg.index = &.{};
        };
        return .{ .ctx = f, .state = .{ .pos = at, .ds_rows = 0, .pieces = pieces.items, .pages = pages, .cut = layer_cut } };
    }

    fn release(ctx: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.released += 1;
    }
};

fn tokensOf(arena: std.mem.Allocator, n: u64) ![]u32 {
    const t = try arena.alloc(u32, n);
    for (t, 0..) |*x, i| x.* = @intCast(i * 13 + 1);
    return t;
}

fn rowOf(i: u64) [split.row_bytes]u8 {
    var row: [split.row_bytes]u8 = undefined;
    for (&row, 0..) |*b, k| b.* = @truncate(i * 29 + k);
    return row;
}

fn open(fake: *Fake, ttl_ns: u64, held_bytes: u64) !*serve.Produce {
    var problem: []const u8 = "";
    return serve.Produce.open(gpa, io, fake.source(), .{ .http = true, .model = model, .ttl_ns = ttl_ns, .held_bytes = held_bytes }, &problem);
}

/// What a reader of the early export got: every message it read, or why it stopped.
const Read = struct {
    p: *serve.Produce,
    id: [16]u8,
    bytes: std.ArrayList(u8) = .empty,
    err: ?anyerror = null,

    fn run(r: *Read) void {
        var why: []const u8 = "";
        const reading = r.p.read(r.id, &why) catch |e| {
            r.err = e;
            return;
        };
        defer reading.close();
        while (reading.next() catch |e| {
            r.err = e;
            return;
        }) |m| r.bytes.appendSlice(gpa, m) catch |e| {
            r.err = e;
            return;
        };
    }
};

/// The state parts a pull lands, checked against the library's own export; the boundary rows into `mirror`.
const Landed = struct {
    mirror: *fabric.row_stage.Stage,
    want: []const ex.Part,
    parts: usize = 0,
    ok: bool = true,
    boundary_first: bool = true,

    fn put(ptr: *anyopaque, layer: *const fabric.manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const l: *Landed = @ptrCast(@alignCast(ptr));
        if (split.cutOfLayer(layer.index) != null) {
            if (l.parts != 0) l.boundary_first = false;
            return l.mirror.put(row_start, bytes[0..@intCast(rows * split.row_bytes)]);
        }
        if (l.parts >= l.want.len or !std.mem.eql(u8, l.want[l.parts].bytes, bytes)) l.ok = false;
        l.parts += 1;
    }
};

/// The producer as the engine drives it: the chunks' rows published a little apart, then the state at the cut.
fn produce(p: *serve.Produce, run: *serve.Produce.Early, pause_ms: u32, fail_at: ?u64) !void {
    var chunks: split.Chunks = .{ .n = upto, .max = 128 };
    while (chunks.next()) |c| {
        if (pause_ms > 0) try std.Io.sleep(io, .fromMilliseconds(pause_ms), .awake);
        if (fail_at) |f| if (c.row0 >= f) return run.fail();
        const dst = run.rows(c.row0, c.rows);
        for (0..c.rows) |i| @memcpy(dst[i * split.row_bytes ..][0..split.row_bytes], &rowOf(c.row0 + i));
        try run.publish(c.row0 + c.rows);
    }
    run.land(try p.saveCut(upto, cut));
}

test "the early entry is held before any compute, counts the stage and the parts, and its layout is the plan's" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fake = try Fake.init();
    defer fake.deinit();
    const p = try open(fake, std.time.ns_per_hour, 1 << 30);
    defer p.stop();
    const tokens = try tokensOf(a, upto);
    const id: [16]u8 = @splat(0x61);
    var run = try p.openEarly(id, tokens, cut);
    defer run.close();
    // the table has it at once: a decoder's OPEN succeeds with nothing computed
    const entry = p.table.get(id).?;
    try std.testing.expect(entry == .ready);
    try std.testing.expectEqual(@as(u16, 2), entry.ready.protocol());
    try std.testing.expectEqual(@as(u64, 0), run.early.stage.available());
    // the layers are exactly `split.describe`: the boundary stream first, then the state of layers below the cut
    const want = try split.describe(a, upto, cut);
    try std.testing.expectEqual(want.len, entry.ready.layers.len);
    var state_bytes: u64 = 0;
    for (want[1..], entry.ready.layers[1..], entry.ready.sources[1..]) |w, l, src| {
        try std.testing.expectEqual(w.index, l.index);
        try std.testing.expectEqual(w.parts.len, src.parts.len);
        try std.testing.expect(l.index < cut);
        for (w.parts, src.parts) |wp, bytes| {
            try std.testing.expectEqual(wp.bytes(), bytes.len);
            state_bytes += bytes.len;
        }
        try std.testing.expect(src.latch != null);
    }
    try std.testing.expect(entry.ready.sources[0].stage != null);
    try std.testing.expectEqual(upto, entry.ready.layers[0].rows());
    // the same parts as the plan of `Shape{cut}`: every spec once, in order, none at or above the cut, no drafter ring
    const specs = try plan.specs(a, split.stateShape(upto, cut));
    var count: usize = 0;
    for (entry.ready.sources[1..]) |src| count += src.parts.len;
    try std.testing.expectEqual(specs.len, count);
    var plan_bytes: u64 = 0;
    for (specs) |sp| {
        try std.testing.expect(sp.layer < cut and sp.kind != .ring);
        plan_bytes += sp.bytes();
    }
    try std.testing.expectEqual(plan_bytes, state_bytes);
    // the held bytes are the stage's and the parts': what the table's cap counts
    try std.testing.expectEqual(upto * split.row_bytes + state_bytes, p.table.held_bytes);
    run.fail();
}

test "an export whose rows and state do not fit the cap is refused by name, and nothing is held" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fake = try Fake.init();
    defer fake.deinit();
    const p = try open(fake, std.time.ns_per_hour, upto * split.row_bytes); // the stage alone fills it: the parts do not fit
    defer p.stop();
    try std.testing.expectError(error.HeldExportsFull, p.openEarly(@splat(0x62), try tokensOf(a, upto), cut));
    try std.testing.expectEqual(@as(usize, 0), p.held.count());
    try std.testing.expectEqual(@as(u64, 0), p.table.held_bytes);
    try std.testing.expect(p.table.get(@splat(0x62)) == null);
    // a cut that is not an index source has no manifest
    const roomy = try open(fake, std.time.ns_per_hour, 1 << 30);
    defer roomy.stop();
    try std.testing.expectError(error.BadShape, roomy.openEarly(@splat(0x63), try tokensOf(a, upto), 11));
    try std.testing.expectEqual(@as(usize, 0), roomy.held.count());
}

test "rows are served while the producer computes, then the state: what a decoder lands is the cut's export" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fake = try Fake.init();
    defer fake.deinit();
    const p = try open(fake, std.time.ns_per_hour, 1 << 30);
    const tokens = try tokensOf(a, upto);
    const id: [16]u8 = @splat(0x64);
    var run = try p.openEarly(id, tokens, cut);
    var reader: Read = .{ .p = p, .id = id };
    defer reader.bytes.deinit(gpa);
    const t = try std.Thread.spawn(.{}, Read.run, .{&reader});
    try produce(p, &run, 3, null);
    t.join();
    try std.testing.expectEqual(@as(?anyerror, null), reader.err);
    try std.testing.expectEqual(@as(u32, 1), fake.taken_cut);

    var mirror = try fabric.row_stage.Stage.init(gpa, upto, split.row_bytes);
    defer mirror.deinit();
    // the library's own conversion of the same state: the parts a decoder must land, byte for byte
    const want = try ex.build(a, split.stateShape(upto, cut), fake.world.cuda);
    var landed: Landed = .{ .mirror = &mirror, .want = want };
    var reason: fabric.decoder.Reason = .{};
    const got = try file.pull(a, reader.bytes.items, .{ .model = model, .tokens = tokens, .protocol = 2 }, .{ .ptr = &landed, .put_fn = Landed.put }, &reason);
    try std.testing.expectEqual(@as(?u32, cut), split.cutOf(got.manifest));
    try std.testing.expectEqual(upto, mirror.available());
    try std.testing.expect(landed.boundary_first and landed.ok);
    try std.testing.expectEqual(want.len, landed.parts);
    for (0..upto) |i| try std.testing.expectEqualSlices(u8, &rowOf(i), mirror.slice(i, 1));
    // the export is the cut's: no layer at or above it, no drafter ring
    for (got.manifest.layers[1..]) |l| try std.testing.expect(l.index < cut);
    const specs = try plan.specs(a, split.stateShape(upto, cut));
    try std.testing.expectEqual(specs.len, want.len);
    for (specs, want) |sp, part| try std.testing.expectEqual(sp.bytes(), part.bytes.len);
    // the capture was handed back once the state had landed
    run.close();
    p.stop();
    try std.testing.expectEqual(@as(u32, 1), fake.released);
}

test "a producer that fails mid-split fails the decoder's pull, the failure is recorded, and everything is freed" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fake = try Fake.init();
    defer fake.deinit();
    const p = try open(fake, std.time.ns_per_hour, 1 << 30);
    const id: [16]u8 = @splat(0x65);
    var run = try p.openEarly(id, try tokensOf(a, upto), cut);
    var reader: Read = .{ .p = p, .id = id };
    defer reader.bytes.deinit(gpa);
    const t = try std.Thread.spawn(.{}, Read.run, .{&reader});
    try produce(p, &run, 3, 300);
    t.join();
    try std.testing.expectEqual(@as(?anyerror, error.ProducerFailed), reader.err);
    try std.testing.expect(run.early.stage.failed.load(.acquire) and run.early.latch.failed.load(.acquire));
    // the state never came: nothing was captured, and the stage holds the rows that did
    try std.testing.expectEqual(@as(u32, 0), fake.taken_cut);
    try std.testing.expect(run.early.stage.available() >= 256 and run.early.stage.available() < upto);
    // a reader that comes later gets the same failure (the entry stays until its time is up)
    var late: Read = .{ .p = p, .id = id };
    defer late.bytes.deinit(gpa);
    late.run();
    try std.testing.expectEqual(@as(?anyerror, error.ProducerFailed), late.err);
    run.close();
    p.stop();
}

test "a decoder that releases the export mid-split ends the run, and the producer frees it at the last lease" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fake = try Fake.init();
    defer fake.deinit();
    const p = try open(fake, std.time.ns_per_hour, 1 << 30);
    const id: [16]u8 = @splat(0x66);
    var run = try p.openEarly(id, try tokensOf(a, upto), cut);
    try std.testing.expect(!run.gone());
    try std.testing.expect(p.discard(id)); // the decoder's CLOSE, rows still being produced
    try std.testing.expect(run.gone());
    // the stage is still the run's to write: the lease keeps it
    const dst = run.rows(0, 4);
    @memset(dst, 1);
    try run.publish(4);
    var why: []const u8 = "";
    try std.testing.expectError(error.NotHeld, p.read(id, &why));
    run.fail();
    run.close(); // the last lease: the export is freed here (the leak check proves it)
    p.stop();
}

test "an entry is not expired while its split runs, and its clock starts when the state lands" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fake = try Fake.init();
    defer fake.deinit();
    const ttl = 150 * std.time.ns_per_ms;
    const p = try open(fake, ttl, 1 << 30);
    const id: [16]u8 = @splat(0x67);
    var run = try p.openEarly(id, try tokensOf(a, upto), cut);
    // chunks 60 ms apart for far longer than the TTL: each publish restarts the clock
    var chunks: split.Chunks = .{ .n = upto, .max = 128 };
    var elapsed: u64 = 0;
    while (chunks.next()) |c| {
        try std.Io.sleep(io, .fromMilliseconds(60), .awake);
        elapsed += 60 * std.time.ns_per_ms;
        const dst = run.rows(c.row0, c.rows);
        for (0..c.rows) |i| @memcpy(dst[i * split.row_bytes ..][0..split.row_bytes], &rowOf(c.row0 + i));
        try run.publish(c.row0 + c.rows);
        try p.table.expire();
        try std.testing.expect(p.table.get(id) != null);
    }
    try std.testing.expect(elapsed > 2 * ttl);
    run.land(try p.saveCut(upto, cut));
    try std.testing.expectEqual(fabric.row_stage.Stage.Wait.ready, run.early.latch.wait(1, 5 * std.time.ns_per_s, null));
    // the TTL runs from the landing: still held shortly after it, gone a TTL later
    try p.table.expire();
    try std.testing.expect(p.table.get(id) != null);
    try std.Io.sleep(io, .fromMilliseconds(260), .awake);
    var why: []const u8 = "";
    try std.testing.expectError(error.NotHeld, p.read(id, &why)); // the sweep freed it
    p.stop();
}

var fake_now: u64 = 0;

fn fakeClock() u64 {
    return fake_now;
}

test "the table's touch restarts one entry's clock and no other's" {
    var t: fabric.producer.Table = .{ .gpa = gpa, .ttl_ns = 100, .clock = fakeClock };
    defer t.deinit();
    const first: [16]u8 = @splat(1);
    const second: [16]u8 = @splat(2);
    fake_now = 0;
    try t.add(first, "a", .{ .failed = "x" });
    try t.add(second, "b", .{ .failed = "y" });
    fake_now = 80;
    t.touch(first);
    t.touch(@splat(3)); // not held: nothing
    fake_now = 150;
    try t.expire();
    try std.testing.expect(t.get(first) != null);
    try std.testing.expect(t.get(second) == null);
    fake_now = 260;
    try t.expire();
    try std.testing.expect(t.get(first) == null);
}
