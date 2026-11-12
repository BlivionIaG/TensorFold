//! The producer's real split stream decoded the way the M5 does: boundary rows to a stage, the state to a Landing.
const std = @import("std");
const fabric = @import("fabric");
const split = @import("handoff/split.zig");
const metal = @import("handoff/metal.zig");
const landing = @import("handoff/landing.zig");
const file = @import("handoff_file.zig");

const gpa = std.testing.allocator;
const model = "dsv41-land-test";
const cut: u32 = 14;
/// a long prompt's tokens before the cut's last (17,100 tokens), on a slot of 32,767 rows
const upto: u64 = 17099;
const capacity: u32 = 32767;

fn rowOf(i: u64) [split.row_bytes]u8 {
    var row: [split.row_bytes]u8 = undefined;
    for (&row, 0..) |*b, k| b.* = @truncate(i * 31 + k);
    return row;
}

/// The producer's export as `cuda_export.early` lays it out: the boundary layer from a stage, then the state's parts.
const Export = struct {
    arena: std.heap.ArenaAllocator,
    exp: fabric.producer.Export,
    buffers: [][]u8,
    tokens: []u32,

    fn make(stage: *fabric.row_stage.Stage, latch: *fabric.row_stage.Stage) !*Export {
        const x = try gpa.create(Export);
        errdefer gpa.destroy(x);
        x.arena = .init(gpa);
        errdefer x.arena.deinit();
        const a = x.arena.allocator();
        const layers = try split.describe(a, upto, cut);
        const sources = try a.alloc(fabric.frames.Source, layers.len);
        const wait_ns = 10 * std.time.ns_per_s;
        sources[0] = .{ .bytes = &.{}, .shape = &.{}, .block_dim = 0, .elem = 2, .rows = &.{}, .stage = stage, .wait_ns = wait_ns };
        var parts: usize = 0;
        for (layers[1..]) |l| parts += l.parts.len;
        x.buffers = try a.alloc([]u8, parts);
        var at: usize = 0;
        for (layers[1..], sources[1..]) |l, *s| {
            const pieces = try a.alloc([]const u8, l.parts.len);
            for (l.parts, pieces) |p, *piece| {
                x.buffers[at] = try a.alloc(u8, @intCast(p.bytes()));
                piece.* = x.buffers[at];
                at += 1;
            }
            s.* = .{ .bytes = &.{}, .shape = &.{}, .block_dim = 0, .elem = 1, .rows = &.{}, .parts = pieces, .latch = latch, .wait_ns = wait_ns };
        }
        x.tokens = try a.alloc(u32, upto);
        for (x.tokens, 0..) |*t, i| t.* = @intCast(i + 5);
        x.exp = .{ .handoff = @splat(0x5b), .tokens = x.tokens, .first_token = 0, .tokens_per_row = 1, .layers = layers, .sources = sources };
        return x;
    }

    fn deinit(x: *Export) void {
        if (x.exp.frames.len > 0) gpa.free(x.exp.frames);
        x.arena.deinit();
        gpa.destroy(x);
    }
};

/// The chunks of the producer's schedule published a little apart, then the state (zeros are a valid state's bytes).
fn produce(stage: *fabric.row_stage.Stage, latch: *fabric.row_stage.Stage, x: *Export) void {
    var chunks: split.Chunks = .{ .n = upto, .max = 2048 };
    while (chunks.next()) |c| {
        fabric.words.pause(std.time.ns_per_ms);
        for (0..c.rows) |i| stage.put(c.row0 + i, &rowOf(c.row0 + i)) catch return stage.fail();
    }
    for (x.buffers) |b| @memset(b, 0);
    latch.publish(1) catch latch.fail();
}

/// The M5 session's pull sink (native Session.put): boundary rows to the stage, every other layer to the landing.
const Session = struct {
    landing: *landing.Landing,
    stage: *fabric.row_stage.Stage,

    fn put(ptr: *anyopaque, layer: *const fabric.manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const s: *Session = @ptrCast(@alignCast(ptr));
        if (split.cutOfLayer(layer.index) == null) return s.landing.decoderSink().put_fn(s.landing.decoderSink().ptr, layer, row_start, rows, bytes);
        if (split.cutOfLayer(layer.index).? != cut or layer.rows() != upto) return error.Mismatch;
        try s.stage.put(row_start, bytes);
    }
};

test "a long prompt's split stream pulls without a protocol error and lands the state complete" {
    var stage = try fabric.row_stage.Stage.init(gpa, upto, split.row_bytes);
    defer stage.deinit();
    var latch = try fabric.row_stage.Stage.init(gpa, 1, 0);
    defer latch.deinit();
    const x = try Export.make(&stage, &latch);
    defer x.deinit();
    const thread = try std.Thread.spawn(.{}, produce, .{ &stage, &latch, x });
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    const written = file.write(gpa, &out.writer, &x.exp, model, 0, 1);
    thread.join();
    try written;

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const probe: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = 40, .capacity = capacity };
    const via: metal.Packed = .{ .live = try arena.alloc(u8, probe.liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(capacity)), .layers = 40, .capacity = capacity };
    @memset(via.storage, 0);
    var l: landing.Landing = undefined;
    try l.initCut(arena, upto, cut, .{ .@"packed" = via });
    defer l.deinit();
    var mirror = try fabric.row_stage.Stage.init(gpa, upto, split.row_bytes);
    defer mirror.deinit();
    var session: Session = .{ .landing = &l, .stage = &mirror };
    var reason: fabric.decoder.Reason = .{};
    const got = try file.pull(arena, out.written(), .{ .model = model, .tokens = x.tokens, .protocol = 2 }, .{ .ptr = &session, .put_fn = Session.put }, &reason);
    try std.testing.expectEqual(@as(?u32, cut), split.cutOf(got.manifest));
    try std.testing.expectEqual(upto, mirror.available());
    for ([_]u64{ 0, 127, 128, 2048, upto - 1 }) |i| try std.testing.expectEqualSlices(u8, &rowOf(i), mirror.slice(i, 1));
    const res = try l.complete();
    try std.testing.expectEqual(upto, res.upto);
}
