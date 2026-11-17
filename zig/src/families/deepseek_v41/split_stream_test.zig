//! A layer-split export streams: the boundary rows are served as the producer publishes them, the state after them.
const std = @import("std");
const fabric = @import("fabric");
const plan = @import("handoff/plan.zig");
const p2 = @import("handoff/p2.zig");
const split = @import("handoff/split.zig");
const file = @import("handoff_file.zig");

const gpa = std.testing.allocator;
const model = "dsv41-split-test";
const upto: u64 = 700;
const cut: u32 = 14;

/// A split export over `stage` and `latch`: the boundary layer's source, then every state part's buffer.
const Export = struct {
    arena: std.heap.ArenaAllocator,
    exp: fabric.producer.Export,
    buffers: [][]u8,
    tokens: []u32,

    fn make(stage: *fabric.row_stage.Stage, latch: *fabric.row_stage.Stage, wait_ns: u64) !*Export {
        const x = try gpa.create(Export);
        errdefer gpa.destroy(x);
        x.arena = .init(gpa);
        errdefer x.arena.deinit();
        const a = x.arena.allocator();
        const layers = try split.describe(a, upto, cut);
        const sources = try a.alloc(fabric.frames.Source, layers.len);
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
        x.exp = .{ .handoff = @splat(0x5a), .tokens = x.tokens, .first_token = 0, .tokens_per_row = 1, .layers = layers, .sources = sources };
        return x;
    }

    fn deinit(x: *Export) void {
        if (x.exp.frames.len > 0) gpa.free(x.exp.frames);
        x.arena.deinit();
        gpa.destroy(x);
    }
};

fn rowOf(i: u64) [split.row_bytes]u8 {
    var row: [split.row_bytes]u8 = undefined;
    for (&row, 0..) |*b, k| b.* = @truncate(i * 31 + k);
    return row;
}

/// The producer's side on its own thread: chunks of the schedule published a little apart, then the state.
const Producer = struct {
    stage: *fabric.row_stage.Stage,
    latch: *fabric.row_stage.Stage,
    x: *Export,
    fail_at: ?u64 = null,

    fn run(p: *Producer) void {
        var chunks: split.Chunks = .{ .n = upto, .max = 256 };
        while (chunks.next()) |c| {
            fabric.words.pause(2 * std.time.ns_per_ms);
            if (p.fail_at) |f| if (c.row0 >= f) return p.stage.fail();
            for (0..c.rows) |i| p.stage.putWait(c.row0 + i, &rowOf(c.row0 + i), 5 * std.time.ns_per_s, null) catch return p.stage.fail();
        }
        for (p.x.buffers, 0..) |b, i| @memset(b, @truncate(i + 1));
        p.latch.publish(1) catch p.latch.fail();
    }
};

/// The producer's replies as the HTTP route writes them, read whole while the producer still computes.
fn stream(x: *Export) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    try file.write(gpa, &out.writer, &x.exp, model, 0, 1);
    return out.toOwnedSlice();
}

const Landed = struct {
    stage: *fabric.row_stage.Stage,
    state_parts: usize = 0,
    order_ok: bool = true,

    fn put(ptr: *anyopaque, layer: *const fabric.manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const l: *Landed = @ptrCast(@alignCast(ptr));
        if (split.cutOfLayer(layer.index) != null) {
            // boundary frames arrive before any state frame, and in row order
            if (l.state_parts != 0) l.order_ok = false;
            return l.stage.put(row_start, bytes[0..@intCast(rows * split.row_bytes)]);
        }
        l.state_parts += 1;
    }
};

test "a split export is served while the producer computes, boundary rows first and the state after" {
    var stage = try fabric.row_stage.Stage.init(gpa, upto, split.row_bytes);
    defer stage.deinit();
    var latch = try fabric.row_stage.Stage.init(gpa, 1, 0);
    defer latch.deinit();
    const x = try Export.make(&stage, &latch, 5 * std.time.ns_per_s);
    defer x.deinit();
    var p: Producer = .{ .stage = &stage, .latch = &latch, .x = x };
    const t = try std.Thread.spawn(.{}, Producer.run, .{&p});
    const bytes = try stream(x);
    t.join();
    defer gpa.free(bytes);

    var mirror = try fabric.row_stage.Stage.init(gpa, upto, split.row_bytes);
    defer mirror.deinit();
    var landed: Landed = .{ .stage = &mirror };
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var reason: fabric.decoder.Reason = .{};
    const got = try file.pull(arena.allocator(), bytes, .{ .model = model, .tokens = x.tokens, .protocol = 2 }, .{ .ptr = &landed, .put_fn = Landed.put }, &reason);
    try std.testing.expectEqual(@as(?u32, cut), split.cutOf(got.manifest));
    try std.testing.expectEqual(upto, mirror.available());
    try std.testing.expect(landed.order_ok);
    for (0..upto) |i| try std.testing.expectEqualSlices(u8, &rowOf(i), mirror.slice(i, 1));
    const specs = try plan.specs(gpa, split.stateShape(upto, cut));
    defer gpa.free(specs);
    try std.testing.expectEqual(specs.len, landed.state_parts);
    try std.testing.expectEqual(upto, got.manifest.layers[0].rows());
    // the state layers are the plan's: the importer's own check takes the manifest minus the boundary layer
    var rest = got.manifest;
    rest.layers = got.manifest.layers[1..];
    try p2.check(arena.allocator(), split.stateShape(upto, cut), rest);
}

test "a producer that fails mid-stream fails the frame it was serving, and so the decoder's pull" {
    var stage = try fabric.row_stage.Stage.init(gpa, upto, split.row_bytes);
    defer stage.deinit();
    var latch = try fabric.row_stage.Stage.init(gpa, 1, 0);
    defer latch.deinit();
    const x = try Export.make(&stage, &latch, 5 * std.time.ns_per_s);
    defer x.deinit();
    var p: Producer = .{ .stage = &stage, .latch = &latch, .x = x, .fail_at = 300 };
    const t = try std.Thread.spawn(.{}, Producer.run, .{&p});
    defer t.join();
    try std.testing.expectError(error.ProducerFailed, stream(x));
}

test "a state that never lands is a stall, not a hang" {
    var stage = try fabric.row_stage.Stage.init(gpa, upto, split.row_bytes);
    defer stage.deinit();
    var latch = try fabric.row_stage.Stage.init(gpa, 1, 0);
    defer latch.deinit();
    const x = try Export.make(&stage, &latch, 20 * std.time.ns_per_ms);
    defer x.deinit();
    for (0..upto) |i| try stage.put(i, &rowOf(i));
    try std.testing.expectError(error.ProducerStalled, stream(x));
}

test "a producer with a window of two chunks streams a prompt of three times its size, and the reader frees the window as it goes" {
    // 512 rows held of 700: the producer writes row 512 only once the frames over the first rows were served
    var stage = try fabric.row_stage.Stage.initRing(gpa, upto, split.row_bytes, 512, 0);
    defer stage.deinit();
    try std.testing.expect(stage.isRing());
    var latch = try fabric.row_stage.Stage.init(gpa, 1, 0);
    defer latch.deinit();
    const x = try Export.make(&stage, &latch, 5 * std.time.ns_per_s);
    defer x.deinit();
    var p: Producer = .{ .stage = &stage, .latch = &latch, .x = x };
    const t = try std.Thread.spawn(.{}, Producer.run, .{&p});
    const bytes = try stream(x);
    t.join();
    defer gpa.free(bytes);
    try std.testing.expect(!stage.failed.load(.acquire));
    try std.testing.expectEqual(upto, stage.available());
    try std.testing.expectEqual(upto, stage.released.load(.acquire)); // every frame went out, the window is all free
    var mirror = try fabric.row_stage.Stage.init(gpa, upto, split.row_bytes);
    defer mirror.deinit();
    var landed: Landed = .{ .stage = &mirror };
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var reason: fabric.decoder.Reason = .{};
    _ = try file.pull(arena.allocator(), bytes, .{ .model = model, .tokens = x.tokens, .protocol = 2 }, .{ .ptr = &landed, .put_fn = Landed.put }, &reason);
    try std.testing.expectEqual(upto, mirror.available());
    for (0..upto) |i| try std.testing.expectEqualSlices(u8, &rowOf(i), mirror.slice(i, 1));
}

test "a window nobody reads stalls the producer instead of growing" {
    var stage = try fabric.row_stage.Stage.initRing(gpa, upto, split.row_bytes, 256, 0);
    defer stage.deinit();
    for (0..256) |i| try stage.putWait(i, &rowOf(i), 0, null);
    try std.testing.expectError(error.WindowFull, stage.putWait(256, &rowOf(256), 5 * std.time.ns_per_ms, null));
    try std.testing.expect(stage.blocked_ns.load(.monotonic) > 0);
}
