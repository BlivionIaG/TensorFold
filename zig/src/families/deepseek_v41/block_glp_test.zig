//! Host IR contract: paired fold/writer placement and steered final, with no GLP emission when disabled.

const std = @import("std");
const t = std.testing;
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const calls = @import("block_calls.zig");
const Config = @import("config.zig").Config;
const glp_plan = @import("glp_plan.zig");

const ns = "tf_dsv41_glp_v1.";
/// steered blocks: before a plain site, before Engram block 14, before DSpark target 37, and the last (the head)
const steered = [_]u32{ 1, 13, 36, 39 };

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try @import("widths_listing.zig").parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try @import("widths_listing.zig").parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    for (0..43) |layer| try w.gm.put(a, @intCast(layer), .{ (1 << 4) | (1 << 6) | (1 << 8), (1 << 4) | (1 << 10) });
    for (0..43) |layer| for (4..8) |g| {
        const first = try std.fmt.allocPrint(a, "L{d}.attn.wo_a.0", .{layer});
        try w.dense.put(a, try std.fmt.allocPrint(a, "L{d}.attn.wo_a.{d}", .{ layer, g }), try w.k2(first));
    };
    return w;
}

const Fake = struct {
    const Layer = struct { id: u32, dir: []const f32 };
    const Vector = struct { mode: enum { add, project }, width: usize, layers: []const Layer, file_sha256: [32]u8 };
    alpha: f32,
    vector: Vector,
};

fn planOf(a: std.mem.Allocator, ids: []const u32, mode: @FieldType(Fake.Vector, "mode")) !glp_plan.Plan {
    const dir = try a.alloc(f32, 5120);
    @memset(dir, 0.25);
    const layers = try a.alloc(Fake.Layer, ids.len);
    for (layers, ids) |*l, id| l.* = .{ .id = id, .dir = dir };
    return (try glp_plan.Plan.init(a, Fake{ .alpha = 0.5, .vector = .{ .mode = mode, .width = 5120, .layers = layers, .file_sha256 = @splat(3) } })).?;
}

fn count(cs: []const calls.Call, name: []const u8) usize {
    var n: usize = 0;
    for (cs) |c| n += @intFromBool(std.mem.eql(u8, c.name, name));
    return n;
}

fn nth(cs: []const calls.Call, name: []const u8, k: usize) ?usize {
    var seen: usize = 0;
    for (cs, 0..) |c, i| if (std.mem.eql(u8, c.name, name)) {
        if (seen == k) return i;
        seen += 1;
    };
    return null;
}

fn role(x: calls.Arg) []const u8 {
    return switch (x.t.role) {
        .buf => |b| b,
        .weight => |w| w,
        .empty => "",
    };
}

fn names(a: std.mem.Allocator, cs: []const calls.Call) ![]const []const u8 {
    const out = try a.alloc([]const u8, cs.len);
    for (cs, out) |c, *n| n.* = c.name;
    return out;
}

test "GLP: a decode window's fold sits between its post and the next site, the writer after the attention exchange" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const plan = try planOf(a, &steered, .project);
    for ([_]i64{ 1, 16, 17, 64 }) |n| {
        const cs = try block.emit(a, &cfg, &w, .{ .glp = &plan, .taps = true }, &backbone, n, 4096 - 64, true);
        try t.expectEqual(@as(usize, 4), count(cs, ns ++ "stream"));
        try t.expectEqual(@as(usize, 4), count(cs, ns ++ "writer"));
        try t.expectEqual(@as(usize, 1), count(cs, ns ++ "final"));
        for (0..4) |k| {
            const f = nth(cs, ns ++ "stream", k).?;
            // the post alone, in place, right before the edit
            try t.expectEqualStrings("_site", cs[f - 1].name);
            const x = cs[f].args[0].arg;
            try t.expect(std.mem.startsWith(u8, role(x), "w.x"));
            // the direction: this layer's row of the table, fp32 [width] at its byte offset
            const d = cs[f].args[2].arg.t;
            try t.expectEqualStrings(glp_plan.dirs_role, role(cs[f].args[2].arg));
            try t.expectEqual(@as(i64, @intCast(k)) * 5120 * 4, d.offset);
            try t.expectEqual(@as(f64, 0.5), cs[f].args[3].arg.f);
            try t.expectEqual(@as(i64, 1), cs[f].args[4].arg.i);
            try t.expectEqual(n, cs[f].args[5].arg.i);
            try t.expectEqual(@as(i64, 5120), cs[f].args[6].arg.i);
            // the DSpark tap of target 37 rides on block 36's fold; the others write none
            const tap = cs[f].args[1].arg.t;
            if (steered[k] == 36) try t.expectEqualStrings("w.taps", role(cs[f].args[1].arg)) else try t.expect(tap.role == .empty);
            if (n <= 16) {
                // block 13 feeds an Engram block; the others the site2 collapse; block 39 the final and the head
                const next = cs[f + 1].name;
                if (steered[k] == 13) try t.expectEqualStrings("glue.engram_rows", next);
                if (steered[k] == 1 or steered[k] == 36) try t.expectEqualStrings("tf_dsv41_mhc_cuda_v1.run", next);
                if (steered[k] == 39) try t.expectEqualStrings(ns ++ "final", next);
            }
            const wr = nth(cs, ns ++ "writer", k).?;
            try t.expectEqualStrings("glue.exchange", cs[wr - 1].name);
            try t.expectEqualStrings("w.recv.attn", role(cs[wr].args[0].arg));
            try t.expectEqual(@as(usize, 3), cs[wr].args[0].arg.t.shape.len);
            try t.expectEqual(n, cs[wr].args[0].arg.t.shape[1]);
        }
        const fin = nth(cs, ns ++ "final", 0).?;
        try t.expectEqualStrings(ns ++ "stream", cs[fin - 1].name);
        try t.expectEqualStrings("w.out", role(cs[fin].args[3].arg));
        try t.expectEqualStrings("glue.kit_logits", cs[cs.len - 1].name);
        // each steered block's writer comes before its own fold (the attention precedes the FFN write it folds)
        for (0..4) |k| try t.expect(nth(cs, ns ++ "writer", k).? < nth(cs, ns ++ "stream", k).?);
    }
}

test "GLP: no plan, a strength-less run or a vector for no emitted block emits not one GLP call and the same launches" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    for ([_]i64{ 1, 16, 17 }) |n| {
        const off = try block.emit(a, &cfg, &w, .{}, &backbone, n, 4096 - 64, true);
        for (off) |c| try t.expect(!std.mem.startsWith(u8, c.name, "tf_dsv41_glp_v1"));
        // a vector whose only layer is outside the emitted layers steers nothing: the same call names
        const far = try planOf(a, &.{30}, .add);
        const part = try block.emit(a, &cfg, &w, .{ .glp = &far }, backbone[0..10], n, 4096 - 64, false);
        const plain = try block.emit(a, &cfg, &w, .{}, backbone[0..10], n, 4096 - 64, false);
        try t.expectEqualDeep(try names(a, plain), try names(a, part));
    }
}

test "GLP: the plan's direction table is its own role, sized to the whole table" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const plan = try planOf(a, &steered, .add);
    var bplan: buffers.Plan = .{ .a = a };
    try bplan.add(try block.emit(a, &cfg, &w, .{ .glp = &plan }, &backbone, 16, 4096 - 16, true));
    try t.expect(bplan.sizes.get(glp_plan.dirs_role).? <= plan.tableBytes());
    try plan.reserve(&bplan);
    try t.expectEqual(plan.tableBytes(), bplan.sizes.get(glp_plan.dirs_role).?);
    try t.expectEqual(buffers.Scope.persistent, buffers.scopeOf(glp_plan.dirs_role));
}

test "GLP: prefill segments steer the same places, short and long, and the head after the last block" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const plan = try planOf(a, &steered, .add);
    for ([_]i64{ 16, 300, 2048 }) |n| {
        const cs = try bp.emitPrefill(a, &cfg, &w, .{ .glp = &plan }, &backbone, n, 0, true);
        try t.expectEqual(@as(usize, 4), count(cs, ns ++ "stream"));
        try t.expectEqual(@as(usize, 4), count(cs, ns ++ "writer"));
        try t.expectEqual(@as(usize, 1), count(cs, ns ++ "final"));
        for (0..4) |k| {
            const f = nth(cs, ns ++ "stream", k).?;
            try t.expectEqualStrings("_site", cs[f - 1].name);
            try t.expectEqual(@as(i64, 0), cs[f].args[4].arg.i);
            const wr = nth(cs, ns ++ "writer", k).?;
            try t.expect(std.mem.endsWith(u8, role(cs[wr].args[0].arg), "recv.attn"));
            try t.expect(wr < f);
        }
        try t.expectEqualStrings("glue.kit_logits", cs[cs.len - 1].name);
        // a long segment's final writes "w.out" itself; no second final norm follows it
        const fin = nth(cs, ns ++ "final", 0).?;
        try t.expectEqualStrings("w.out", role(cs[fin].args[3].arg));
        for (cs[fin..]) |c| try t.expect(!std.mem.eql(u8, c.name, "_finish_k"));
        const off = try bp.emitPrefill(a, &cfg, &w, .{}, &backbone, n, 0, true);
        for (off) |c| try t.expect(!std.mem.startsWith(u8, c.name, "tf_dsv41_glp_v1"));
    }
}

test "GLP: a run that would show or skip an unsteered fold is refused, never silently unsteered" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const plan = try planOf(a, &steered, .add);
    try t.expectError(error.InvalidBlock, block.emit(a, &cfg, &w, .{ .glp = &plan, .trace = true }, &backbone, 16, 0, true));
    try t.expectError(error.InvalidBlock, block.emit(a, &cfg, &w, .{ .glp = &plan }, backbone[3..8], 16, 0, false));
    try t.expectError(error.InvalidBlock, bp.emitPrefill(a, &cfg, &w, .{ .glp = &plan, .trace = true }, &backbone, 300, 0, true));
    try t.expectError(error.InvalidBlock, bp.emitReplay(a, &cfg, &w, .{ .glp = &plan }, backbone[0..21], 300, 0, .encoder));
}
