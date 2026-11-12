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

fn pullStreamed(upto: u64, fail: bool) !void {
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
    const offer = remote.source().offer(prompt, 0).?;
    try std.testing.expectEqual(upto, offer.upto);
    const via = try target(arena, 2048);
    var land: landing.Landing = undefined;
    try land.init(a, upto, .{ .@"packed" = via });
    defer land.deinit();
    if (fail) {
        // the producer's failure is the fill's error: not the pull's Aborted or Refused
        try std.testing.expectError(error.PrefillFailed, remote.source().fill(prompt, offer, land.state()));
        return;
    }
    try remote.source().fill(prompt, offer, land.state());
    try std.testing.expectEqual(@as(u32, 1), trig.asked);
    try std.testing.expect(remote.trigger_ns > 0 and remote.bytes > 0);
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
    try pullStreamed(300, false);
}

test "a producer whose prefill fails half way is the fill's error" {
    try pullStreamed(300, true);
}
