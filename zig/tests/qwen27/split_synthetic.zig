//! The split gate on generated weights (hidden 512, five layers: linear, linear, linear, attention, linear), no files:
//! every cut, order and level over the mock window is token-identical to the unsplit run, with decode rows, with the
//! prompt in chunks and with two sequences at once, and a stuck or cut link is a named error with a clean next run.
const std = @import("std");
const tf = @import("tensorfold");
const h = @import("split_harness.zig");
const graph = @import("synthetic_graph.zig");
const q = tf.qwen27;

const layers = 5;

// the generated graph keeps its tensor names for the process's life: those come from the arena the context holds
fn load(ctx: *const anyopaque, _: std.mem.Allocator, range: q.range.Range) anyerror!*q.model.Model {
    const arena: *const std.mem.Allocator = @ptrCast(@alignCast(ctx));
    const m = try graph.createLayers(arena.*, layers);
    m.range = range;
    return m;
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const arena = init.arena.allocator();
    const src = h.Source{ .ctx = &arena, .load = load, .layers = layers };
    const first = [_]u32{ 5, 17, 3, 90, 11, 42, 8, 60, 29, 71, 14, 33 };
    const second = [_]u32{ 61, 2, 77, 9, 40, 23, 88 };
    var report = h.Report{};
    const m = try load(&arena, a, .{ .first = 0, .last = layers });
    defer m.deinit();
    const c0 = try h.reference(a, m, &first, 32, &report);
    defer h.freeCase(a, c0);
    const c1 = try h.reference(a, m, &second, 32, null);
    defer h.freeCase(a, c1);
    var distinct: usize = 0;
    for (c0.single, 0..) |t, i| distinct += @intFromBool(std.mem.indexOfScalar(u32, c0.single[0..i], t) == null);
    std.debug.print("qwen27 synthetic split: unsplit reference has {d} distinct tokens in {d}\n", .{ distinct, c0.single.len });
    try h.cuts(a, src, &.{ 1, 2, 3, 4 }, .{ c0, c1 }, &report);
    std.debug.print("qwen27 synthetic split: {d} decode, {d} chunked and {d} two-sequence runs identical; {d:.2} ms/position split against {d:.2} unsplit\n", .{ report.runs, report.chunked_runs, report.stream_runs, @as(f64, @floatFromInt(report.split_ns / report.split_steps)) / 1e6, @as(f64, @floatFromInt(report.unsplit_ns)) / 1e6 });
}
