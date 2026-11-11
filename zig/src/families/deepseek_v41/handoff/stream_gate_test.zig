//! A streamed export: PULL for a part still being built is answered `wait`, and the decoder takes ready parts first.
const std = @import("std");
const fabric = @import("fabric");
const plan = @import("plan.zig");
const ex = @import("export.zig");
const metal = @import("metal.zig");
const im = @import("import.zig");
const p2 = @import("p2.zig");
const s = @import("synth.zig");

const producer = fabric.producer;
const half = 4 << 20;
const model = "dsv41-test";
const chunk: u64 = 70 << 10;

/// Part readiness by flat part number; `first[position]` is the flat number of a layer's first part.
const Gate = struct {
    ready: []std.atomic.Value(bool),
    first: []usize,
    bad: std.atomic.Value(bool) = .init(false),

    fn state(ptr: *anyopaque, position: u32, part: u64) producer.PartState {
        const g: *Gate = @ptrCast(@alignCast(ptr));
        if (g.bad.load(.acquire)) return .failed;
        return if (g.ready[g.first[position] + part].load(.acquire)) .ready else .pending;
    }

    fn why(_: *anyopaque) []const u8 {
        return "the export failed";
    }

    fn gate(g: *Gate) producer.Gate {
        return .{ .ptr = g, .state_fn = state, .why_fn = why };
    }
};

/// Records the order parts arrive in (flat part numbers), then lands them in the import.
const Order = struct {
    inner: *p2.ShardSink,
    seen: std.ArrayList(usize) = .empty,
    first: []const usize,
    gpa: std.mem.Allocator,
    count: *std.atomic.Value(usize),

    fn put(ptr: *anyopaque, layer: *const fabric.manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const o: *Order = @ptrCast(@alignCast(ptr));
        try o.seen.append(o.gpa, o.first[layer.index] + @as(usize, @intCast(row_start)));
        _ = o.count.fetchAdd(1, .acq_rel);
        try o.inner.sink().put_fn(o.inner, layer, row_start, rows, bytes);
    }
};

fn target(arena: std.mem.Allocator, capacity: u32) !metal.Packed {
    const probe: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = 43, .capacity = capacity };
    return .{ .live = try arena.alloc(u8, probe.liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(capacity)), .layers = 43, .capacity = capacity };
}

/// Flips parts ready a little apart (`evens_first`: the even ones, then the odd ones once all evens are taken).
fn flipper(g: *Gate, live: *std.atomic.Value(bool), evens_first: bool, fail: bool, taken: *std.atomic.Value(usize)) void {
    const n = g.ready.len;
    const evens = (n + 1) / 2;
    for (0..n) |k| {
        fabric.words.pause(300 * std.time.ns_per_us);
        if (fail and k == n / 2) {
            g.bad.store(true, .release);
            break;
        }
        if (evens_first and k == evens) { // the decoder skipped every odd part: it holds all the even ones before any odd one is ready
            for (0..5000) |_| {
                if (taken.load(.acquire) >= evens) break;
                fabric.words.pause(std.time.ns_per_ms);
            }
        }
        g.ready[if (!evens_first) k else if (k < evens) 2 * k else 2 * (k - evens) + 1].store(true, .release);
    }
    live.store(false, .release);
}

fn pullStreamed(a: std.mem.Allocator, arena: std.mem.Allocator, upto: u64, evens_first: bool, fail: bool, never: bool, open_ms: u64, reason: *fabric.decoder.Reason, order: *std.ArrayList(usize)) !struct { pulled: anyerror!fabric.decoder.Pulled, via: metal.Packed, direct: metal.Packed } {
    const shape: plan.Shape = .{ .upto = upto, .chunk = chunk };
    const w = try s.make(arena, .{ .upto = upto, .seed = upto });
    const tokens = try arena.alloc(u32, upto);
    for (tokens, 0..) |*t, i| t.* = @intCast(i * 7 + 1);
    const parts = try ex.build(arena, shape, w.cuda);
    const built = try p2.layers(arena, parts);

    var g: Gate = .{ .ready = try arena.alloc(std.atomic.Value(bool), parts.len), .first = try arena.alloc(usize, built.layers.len) };
    for (g.ready) |*r| r.* = .init(false);
    var at: usize = 0;
    for (built.layers, g.first) |l, *f| {
        f.* = at;
        at += l.parts.len;
    }

    var link = try fabric.link_fake.Link.init(a, half, half, true, .immediate);
    defer link.deinit();
    link.up();
    const mac = link.box(0);
    const peer = link.box(1);
    var table: producer.Table = .{ .gpa = a, .ttl_ns = 60 * std.time.ns_per_s, .clock = fabric.words.nowNs };
    defer table.deinit();
    var serving: producer.Serving = undefined;
    serving.init(a, &peer, fabric.words.native, &table, model);
    defer serving.deinit();
    const id: [16]u8 = @splat(5);
    var exp: producer.Export = .{ .handoff = id, .tokens = tokens, .first_token = 0, .tokens_per_row = 1, .layers = built.layers, .sources = built.sources, .gate = g.gate() };
    defer a.free(exp.frames);
    try table.add(id, "test", .{ .ready = &exp });
    var stop: std.atomic.Value(bool) = .init(false);
    const daemons = try std.Thread.spawn(.{}, fabric.link_fake.Link.run, .{ &link, &stop });
    const service = try std.Thread.spawn(.{}, producer.Serving.run, .{&serving});
    var live: std.atomic.Value(bool) = .init(true);
    var taken: std.atomic.Value(usize) = .init(0);
    const flips = if (never) null else try std.Thread.spawn(.{}, flipper, .{ &g, &live, evens_first, fail, &taken });
    defer {
        if (flips) |t| t.join();
        stop.store(true, .release);
        serving.stop.store(true, .release);
        daemons.join();
        service.join();
    }
    var client = try fabric.mailbox.Client.init(&mac, fabric.words.native, 30 * std.time.ns_per_s);
    const via = try target(arena, 1024);
    var imp = try im.Import.init(arena, shape, .{ .@"packed" = via }, .{});
    var inner: p2.ShardSink = .{ .imp = &imp, .shard = 0 };
    var rec: Order = .{ .inner = &inner, .first = g.first, .gpa = a, .count = &taken };
    defer rec.seen.deinit(a);
    const sink: fabric.decoder.Sink = .{ .ptr = &rec, .put_fn = Order.put };
    const pulled = fabric.decoder.pull(arena, &client, id, .{ .model = model, .tokens = tokens, .protocol = 2, .streaming = if (never) null else &live, .open_timeout_ns = open_ms * std.time.ns_per_ms }, sink, reason);
    try order.appendSlice(a, rec.seen.items);
    if (pulled) |_| _ = try imp.complete() else |_| {}

    const direct = try target(arena, 1024);
    var imp2 = try im.Import.init(arena, shape, .{ .@"packed" = direct }, .{});
    var buf: [40]u8 = undefined;
    for (parts) |p| try imp2.put(0, p.spec.layer, p.spec.name(&buf), p.spec.dtype, p.spec.rows, p.spec.cols, p.bytes);
    _ = try imp2.complete();
    return .{ .pulled = pulled, .via = via, .direct = direct };
}

test "a streamed export is pulled as its parts become ready, the even ones first, into the same state" {
    const a = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    var reason: fabric.decoder.Reason = .{};
    var order: std.ArrayList(usize) = .empty;
    defer order.deinit(a);
    const r = try pullStreamed(a, arena_state.allocator(), 517, true, false, false, 30_000, &reason, &order);
    const got = try r.pulled;
    try std.testing.expect(got.frames > 8);
    try std.testing.expectEqualSlices(u8, r.direct.live, r.via.live);
    try std.testing.expectEqualSlices(u8, r.direct.storage, r.via.storage);
    try std.testing.expectEqual(got.frames, order.items.len);
    // the decoder did not wait for part 1: it took every ready even part, then the odd ones
    const evens = (order.items.len + 1) / 2;
    for (order.items, 0..) |x, i| try std.testing.expect((x % 2 == 0) == (i < evens));
}

test "a streamed export that breaks half way is refused with the producer's reason" {
    const a = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    var reason: fabric.decoder.Reason = .{};
    var order: std.ArrayList(usize) = .empty;
    defer order.deinit(a);
    const r = try pullStreamed(a, arena_state.allocator(), 517, false, true, false, 30_000, &reason, &order);
    try std.testing.expectError(error.Refused, r.pulled);
    try std.testing.expectEqualStrings("the export failed", reason.text());
}

test "a part that never becomes ready ends the pull as Expired once the producer is no longer prefilling" {
    const a = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    var reason: fabric.decoder.Reason = .{};
    var order: std.ArrayList(usize) = .empty;
    defer order.deinit(a);
    const r = try pullStreamed(a, arena_state.allocator(), 100, false, false, true, 200, &reason, &order);
    try std.testing.expectError(error.Expired, r.pulled);
}
