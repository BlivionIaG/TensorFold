//! TF_PREFILL_STREAM: a Remote pulls while its trigger's prefill runs; a failing trigger is the fill's error.
const std = @import("std");
const fabric = @import("fabric");
const plan = @import("plan.zig");
const ex = @import("export.zig");
const im = @import("import.zig");
const metal = @import("metal.zig");
const p2 = @import("p2.zig");
const landing = @import("landing.zig");
const s = @import("synth.zig");

const producer = fabric.producer;
const ps = fabric.prefill_source;
const half = 4 << 20;
const model = "dsv41-test";

fn target(arena: std.mem.Allocator, capacity: u32) !metal.Packed {
    const probe: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = 43, .capacity = capacity };
    return .{ .live = try arena.alloc(u8, probe.liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(capacity)), .layers = 43, .capacity = capacity };
}

/// Part readiness by flat part number.
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
        return "the producer's prefill failed";
    }
};

/// The producer's trigger: registers a gated export, readies its parts (evens first), and returns (or fails half way).
const Trigger = struct {
    table: *producer.Table,
    layers: []const fabric.manifest.Layer,
    sources: []const fabric.frames.Source,
    gate: Gate,
    exp: producer.Export = undefined,
    fail: bool,
    asked: u32 = 0,

    fn start(ptr: *anyopaque, id: [16]u8, tokens: []const u32) anyerror!void {
        const t: *Trigger = @ptrCast(@alignCast(ptr));
        t.asked += 1;
        fabric.words.pause(5 * std.time.ns_per_ms); // the decoder's OPEN finds nothing yet and waits
        t.exp = .{ .handoff = id, .tokens = tokens[0 .. tokens.len - 1], .first_token = 0, .tokens_per_row = 1, .layers = t.layers, .sources = t.sources, .gate = .{ .ptr = &t.gate, .state_fn = Gate.state, .why_fn = Gate.why } };
        try t.table.add(id, "test", .{ .ready = &t.exp });
        const n = t.gate.ready.len;
        const half_n = (n + 1) / 2;
        for (0..n) |k| {
            fabric.words.pause(100 * std.time.ns_per_us);
            if (t.fail and k == n / 2) {
                t.gate.bad.store(true, .release);
                return error.PrefillFailed;
            }
            t.gate.ready[if (k < half_n) 2 * k else 2 * (k - half_n) + 1].store(true, .release);
        }
    }
};

/// A link that drops under the pull: after every `every`-th landed part (so with no request in flight) it goes down,
/// and comes back as a new generation unless `stay`.
const Flapping = struct { every: u32, stay: bool = false };

/// The pull's sink, wrapped: parts go on to the landing, then the link flaps as `Flapping` says.
const Flapper = struct {
    inner: ps.StateSink,
    link: *fabric.link_fake.Link,
    how: Flapping,
    puts: u32 = 0,
    flaps: u32 = 0,

    fn state(f: *Flapper) ps.StateSink {
        return .{ .ptr = f, .put_fn = put, .commit_fn = commit, .abort_fn = abort };
    }

    fn put(ptr: *anyopaque, layer: *const fabric.manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const f: *Flapper = @ptrCast(@alignCast(ptr));
        try f.inner.put_fn(f.inner.ptr, layer, row_start, rows, bytes);
        f.puts += 1;
        if (f.puts % f.how.every != 0 or (f.how.stay and f.flaps > 0)) return;
        f.link.down();
        if (!f.how.stay) {
            f.link.up();
            fabric.words.pause(3 * std.time.ns_per_ms); // a real re-link takes longer than the serving loop's tick, which sees the zeroed request word
        }
        f.flaps += 1;
    }

    fn commit(ptr: *anyopaque, upto: u64, provenance: ps.Provenance) void {
        const f: *Flapper = @ptrCast(@alignCast(ptr));
        f.inner.commit_fn(f.inner.ptr, upto, provenance);
    }

    fn abort(ptr: *anyopaque) void {
        const f: *Flapper = @ptrCast(@alignCast(ptr));
        f.inner.abort_fn(f.inner.ptr);
    }
};

fn pullStreamed(upto: u64, fail: bool, flap: ?Flapping) !void {
    const a = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = upto, .seed = upto });
    const prompt = try arena.alloc(u32, upto + 1);
    for (prompt, 0..) |*t, i| t.* = @intCast(i * 7 + 1);
    const parts = try ex.build(arena, .{ .upto = upto }, w.cuda);
    const built = try p2.layers(arena, parts);

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
    var stop: std.atomic.Value(bool) = .init(false);
    const daemons = try std.Thread.spawn(.{}, fabric.link_fake.Link.run, .{ &link, &stop });
    const service = try std.Thread.spawn(.{}, producer.Serving.run, .{&serving});
    defer {
        stop.store(true, .release);
        serving.stop.store(true, .release);
        daemons.join();
        service.join();
    }

    var trig: Trigger = .{ .table = &table, .layers = built.layers, .sources = built.sources, .fail = fail, .gate = .{ .ready = try arena.alloc(std.atomic.Value(bool), parts.len), .first = try arena.alloc(usize, built.layers.len) } };
    for (trig.gate.ready) |*r| r.* = .init(false);
    var at: usize = 0;
    for (built.layers, trig.gate.first) |l, *f| {
        f.* = at;
        at += l.parts.len;
    }
    defer if (trig.exp.frames.len > 0) a.free(trig.exp.frames);

    var client = try fabric.mailbox.Client.init(&mac, fabric.words.native, 30 * std.time.ns_per_s);
    var remote: ps.Remote = .{ .gpa = a, .policy = .{ .enabled = true, .min_prefix = 0, .tail = 1 }, .client = &client, .trigger = .{ .ptr = &trig, .start_fn = Trigger.start }, .model = model, .protocol = 2, .stream = true };
    if (flap) |f| if (f.stay) {
        remote.link_wait_ns = 200 * std.time.ns_per_ms;
    };
    const offer = remote.source().offer(prompt, 0).?;
    try std.testing.expectEqual(upto, offer.upto);
    const via = try target(arena, 2048);
    var land: landing.Landing = undefined;
    try land.init(a, upto, .{ .@"packed" = via });
    defer land.deinit();
    var flapper: Flapper = .{ .inner = land.state(), .link = &link, .how = flap orelse .{ .every = 1 } };
    const sink = if (flap != null) flapper.state() else land.state();
    if (fail) {
        // the producer's failure is the fill's error: not the pull's Aborted or Refused
        try std.testing.expectError(error.PrefillFailed, remote.source().fill(prompt, offer, sink));
        return;
    }
    if (flap) |f| if (f.stay) {
        // the link never comes back: the pull's wait runs out and the fill ends with the link's error
        try std.testing.expectError(error.LinkDown, remote.source().fill(prompt, offer, sink));
        try std.testing.expectEqual(@as(u32, 1), flapper.flaps);
        return;
    };
    try remote.source().fill(prompt, offer, sink);
    try std.testing.expectEqual(@as(u32, 1), trig.asked);
    try std.testing.expect(remote.trigger_ns > 0 and remote.bytes > 0);
    if (flap != null) try std.testing.expect(flapper.flaps > 0);
    _ = try land.complete();

    const direct = try target(arena, 2048);
    var imp = try im.Import.init(arena, .{ .upto = upto }, .{ .@"packed" = direct }, .{ .drafter = false });
    var buf: [40]u8 = undefined;
    for (parts) |p| try imp.put(0, p.spec.layer, p.spec.name(&buf), p.spec.dtype, p.spec.rows, p.spec.cols, p.bytes);
    _ = try imp.complete();
    try std.testing.expectEqualSlices(u8, direct.live, via.live);
    try std.testing.expectEqualSlices(u8, direct.storage, via.storage);
}

test "a Remote with stream on pulls while the producer's prefill runs and lands the direct import's bytes" {
    try pullStreamed(300, false, null);
}

test "a producer whose prefill fails half way is the fill's error" {
    try pullStreamed(300, true, null);
}

test "a link that re-links under the pull is waited for, and every part still lands once, even when the last one's CLOSE is lost" {
    try pullStreamed(300, false, .{ .every = 1 }); // a new generation after every part, the last included
    try pullStreamed(1301, false, .{ .every = 7 });
}

test "a link that drops under the pull and does not come back ends the fill with LinkDown after the wait" {
    try pullStreamed(300, false, .{ .every = 5, .stay = true });
}
