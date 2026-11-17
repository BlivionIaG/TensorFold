//! Host tests of CED replay's programs (TF_DSV41_PREFILL=replay): encoder pass + decoder pass == the full segment.
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const check = @import("calls_check.zig");
const ced = @import("ced.zig");
const Config = @import("config.zig").Config;
const dspark_emit = @import("dspark_emit.zig");

test {
    _ = ced;
}

const limit: i64 = 4096;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    for (0..40) |L| try w.gm.put(a, @intCast(L), .{ (1 << 3) | (1 << 4), (1 << 3) | (1 << 4) });
    return w;
}

fn poolOf(split: bool) block.Pool {
    const pages = @divExact(limit, block.Pool.page);
    return .{ .comp_pages = if (split) @divExact(pages, 2) + 2 else pages + 1, .ik_pages = pages + 1, .pts = pages, .split = split };
}

fn roleOf(x: calls.Arg) []const u8 {
    return switch (x) {
        .t => |t| switch (t.role) {
            .buf => |b| b,
            .weight => |w| w,
            .empty => "",
        },
        else => "",
    };
}

fn named(c: calls.Call, name: []const u8) ?calls.Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    return null;
}

/// The same launch on the same roles (names and every tensor's role).
fn same(x: calls.Call, y: calls.Call) bool {
    if (!std.mem.eql(u8, x.name, y.name) or x.args.len != y.args.len) return false;
    for (x.args, y.args) |p, q| if (!std.mem.eql(u8, roleOf(p.arg), roleOf(q.arg))) return false;
    return true;
}

/// The first index at or after `from` where `needle` occurs in `hay`.
fn find(hay: []const calls.Call, from: usize, needle: []const calls.Call) ?usize {
    var i = from;
    outer: while (i + needle.len <= hay.len) : (i += 1) {
        for (needle, 0..) |c, j| if (!same(c, hay[i + j])) continue :outer;
        return i;
    }
    return null;
}

test "CED: the encoder pass + the decoder pass are the whole segment's calls, layer 20's compressor at the encoder's end" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    const d = try ced.decoderStart(&cfg);
    // the fixture's layers (0-24): the encoder 0 .. 20, the decoder 20 .. 24
    var layers: [25]u32 = undefined;
    for (&layers, 0..) |*l, i| l.* = @intCast(i);
    const enc_layers = layers[0 .. d + 1];
    const dec_layers = layers[d..];
    // short segments, split boundaries, top-k, replay rows; contiguous, pooled and split KV; R0 = 0 and past it
    for ([_]?block.Pool{ null, poolOf(false), poolOf(true) }) |pool| for ([_]i64{ 12, 30, 64, 100, ced.keep }) |n| for ([_]i64{ 0, 1000 }) |start| {
        var o: block.Options = .{ .limit = limit, .rope_rows = limit + 2048 };
        o.pool = pool;
        const whole = try bp.emitPrefill(a, &cfg, &w, o, &layers, n, start, false);
        const enc = try bp.emitReplay(a, &cfg, &w, o, enc_layers, n, start, .encoder);
        const dec = try bp.emitReplay(a, &cfg, &w, o, dec_layers, n, start, .decoder);
        // the encoder: the whole segment's calls through layer 20's attention site, then its compressor, the stash
        const stash = enc[enc.len - 1];
        try testing.expectEqualStrings("glue.ced_stash", stash.name);
        // the common prefix (layer 20's site included), the compressor's projection, then _compress's calls
        var c0: usize = 0;
        while (c0 < enc.len and c0 < whole.len and same(enc[c0], whole[c0])) c0 += 1;
        const proj_end = for (enc[c0..], c0..) |c, i| {
            if (std.mem.eql(u8, c.name, "glue.proj")) break i + 1;
        } else return error.TestNoProjection;
        const prefix = enc[0..c0];
        const comp_proj = enc[c0..proj_end];
        const comp = enc[proj_end .. enc.len - 1];
        try testing.expect(comp_proj.len >= 3 and comp.len >= 4);
        try testing.expect(std.mem.indexOf(u8, roleOf(comp_proj[0].args[1].arg), "L20.attn.comp_wkv") != null);
        // the whole segment without the compressor's two runs is the decoder pass after its two glue steps
        const rest = whole[prefix.len..];
        const ip = find(rest, 0, comp_proj) orelse return error.TestNoProjection;
        const ic = find(rest, ip + comp_proj.len, comp) orelse return error.TestNoCompressor;
        var left: std.ArrayList(calls.Call) = .empty;
        try left.appendSlice(a, rest[0..ip]);
        try left.appendSlice(a, rest[ip + comp_proj.len .. ic]);
        try left.appendSlice(a, rest[ic + comp.len ..]);
        try testing.expectEqualStrings("glue.positions", dec[0].name);
        try testing.expectEqualStrings("glue.ced_load", dec[1].name);
        try testing.expectEqual(left.items.len, dec.len - 2);
        for (left.items, dec[2..]) |x, y| try testing.expectEqualStrings(x.name, y.name);
        // windows from R0 = the pass's start
        try testing.expectEqual(@as(usize, 4), dec[0].args.len);
        try testing.expectEqual(start, dec[0].args[3].arg.i);
        // the stash: the encoder's streams / normed input / current coefficients into the ring, the decoder's set 0
        try testing.expectEqualStrings("w.out", roleOf(stash.args[1].arg));
        for (ced.roles, 5..) |role, j| {
            try testing.expectEqualStrings(role, roleOf(stash.args[j].arg));
            try testing.expectEqual(@as(i64, ced.ring), stash.args[j].arg.t.shape[0]);
            try testing.expectEqualStrings(role, roleOf(dec[1].args[j].arg));
        }
        try testing.expectEqualStrings("w.x0", roleOf(dec[1].args[0].arg));
        try testing.expectEqualStrings("w.c0.pre", roleOf(dec[1].args[2].arg));
        for (stash.args[0..5], dec[1].args[0..5]) |x, y| {
            try testing.expectEqual(n, x.arg.t.shape[0]);
            try testing.expectEqual(n, y.arg.t.shape[0]);
        }
        // no comp_wkv in the decoder (the encoder stored layer 20's rows); every boundary in place
        var in_place: usize = 0;
        for (dec) |c| {
            for (c.args) |x| if (x.arg == .t and x.arg.t.role == .weight) try testing.expect(std.mem.indexOf(u8, x.arg.t.role.weight, "comp_wkv") == null);
            if (std.mem.eql(u8, c.name, "_site") and named(c, "POST_ON").?.b) {
                try testing.expectEqualStrings(roleOf(named(c, "X").?), roleOf(named(c, "XOUT").?));
                try testing.expect(!named(c, "SPLIT").?.b);
                in_place += 1;
            }
            if (std.mem.eql(u8, c.name, "tf_dsv41_mhc_cuda_v1.run") and c.args[2].arg.t.role != .empty) {
                try testing.expectEqualStrings(roleOf(c.args[0].arg), roleOf(c.args[1].arg));
                in_place += 1;
            }
            if (std.mem.eql(u8, c.name, "_fused")) try testing.expectEqualStrings("w.lo", roleOf(named(c, "LO").?));
        }
        try testing.expect(in_place >= 2 * dec_layers.len - 1);
    };
}

/// The layer number of a role that belongs to one ("L13.attn.wq_a.suh", "w.L13.proj", "s.L13.swa.v"), else null.
fn layerOfRole(role: []const u8) ?u32 {
    var rest = role;
    for ([_][]const u8{ "w.", "s." }) |p| if (std.mem.startsWith(u8, rest, p)) {
        rest = rest[p.len..];
    };
    if (rest.len < 2 or rest[0] != 'L' or !std.ascii.isDigit(rest[1])) return null;
    var end: usize = 1;
    while (end < rest.len and std.ascii.isDigit(rest[end])) end += 1;
    return std.fmt.parseInt(u32, rest[1..end], 10) catch null;
}

test "layer split: a cut's program runs the layers below it as the whole segment does, then ends on the boundary rows" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    var layers: [25]u32 = undefined;
    for (&layers, 0..) |*l, i| l.* = @intCast(i);
    for ([_]?block.Pool{ null, poolOf(false) }) |pool| for ([_]u32{ 2, 8, 14 }) |cut| for ([_]i64{ 33, 100, 300 }) |n| for ([_]i64{ 0, 1008 }) |start| {
        var o: block.Options = .{ .limit = limit, .rope_rows = limit + 2048 };
        o.pool = pool;
        const whole = try bp.emitPrefill(a, &cfg, &w, o, &layers, n, start, false);
        const cs = try bp.emitReplay(a, &cfg, &w, o, layers[0..cut], n, start, .split);
        // it ends on the boundary glue: the residual (post applied, 4 x 5120 a row), the pre, the layer
        const last = cs[cs.len - 1];
        try testing.expectEqualStrings("glue.split_rows", last.name);
        try testing.expectEqual(@as(usize, 3), last.args.len);
        try testing.expectEqualStrings("w.trace", roleOf(last.args[0].arg));
        try testing.expectEqual(n, last.args[0].arg.t.shape[0]);
        try testing.expectEqual(@as(i64, 4 * 5120), last.args[0].arg.t.shape[1]);
        try testing.expectEqual(n, last.args[1].arg.t.shape[0]);
        try testing.expectEqual(@as(i64, 4), last.args[1].arg.t.shape[1]);
        try testing.expectEqual(@as(i64, cut - 1), last.args[2].arg.i);
        // the whole segment's calls up to the cut layer's last, then only the post and the boundary glue differ
        var common: usize = 0;
        while (common < cs.len and common < whole.len and same(cs[common], whole[common])) common += 1;
        try testing.expect(common + 5 >= cs.len and whole.len > cs.len);
        // no layer from the cut on is touched: no weight, buffer or state role of it, no head, no final norm, no stash
        var splits: usize = 0;
        var holds: usize = 0;
        for (cs) |c| {
            if (std.mem.eql(u8, c.name, "glue.split_rows")) splits += 1;
            if (std.mem.eql(u8, c.name, "glue.split_hold")) holds += 1;
            try testing.expect(!std.mem.startsWith(u8, c.name, "glue.ced_") and !std.mem.startsWith(u8, c.name, "glue.cone_"));
            for (c.args) |x| {
                if (x.arg != .t) continue;
                const role = roleOf(x.arg);
                if (layerOfRole(role)) |L| try testing.expect(L < cut);
                try testing.expect(!std.mem.eql(u8, role, "norm") and !std.mem.startsWith(u8, role, "head") and !std.mem.eql(u8, role, "w.logits"));
            }
        }
        try testing.expectEqual(@as(usize, 1), splits);
        try testing.expectEqual(@as(usize, 1), holds); // the previous chunk's copy is awaited before the boundary is rewritten
        // a prefix that is not run whole (a layer past the list) has no cut to end on
        try testing.expectError(error.Unsupported, bp.emitReplay(a, &cfg, &w, o, &.{}, n, start, .split));
    };
}

test "CED: the decoder pass needs the decoder's first layer first, the encoder ends on a kv source" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    const o: block.Options = .{ .limit = limit, .rope_rows = limit + 2048 };
    // an encoder pass ending on a layer that is not a kv source has no compressor to run
    try testing.expectError(error.Unsupported, bp.emitReplay(a, &cfg, &w, o, &.{ 0, 1, 2, 3 }, 40, 0, .encoder));
    try testing.expectError(error.Unsupported, bp.emitReplay(a, &cfg, &w, o, &.{}, 40, 0, .decoder));
}

test "CED: the decoder pass emits for every tail shape up to the served 1M limit (pooled and split KV)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const cfg: Config = .{};
    var w = try widths(arena.allocator());
    const big: i64 = 1 << 20;
    const pages = @divExact(big, block.Pool.page);
    var layers: [40]u32 = undefined;
    for (&layers, 0..) |*l, i| l.* = @intCast(i);
    const dec = layers[try ced.decoderStart(&cfg)..];
    for ([_]bool{ false, true }) |split| {
        var o: block.Options = .{ .limit = big, .rope_rows = big + 2048, .index_budget = 64 << 20 };
        o.pool = .{ .comp_pages = if (split) @divExact(pages, 2) + 2 else pages + 1, .ik_pages = pages + 1, .pts = pages, .split = split };
        // a prompt of n <= 128 tokens: its n - 1 rows from 0; a longer one: 127 rows ending at n - 1 < the limit
        var m: i64 = 1;
        while (m <= ced.keep) : (m += 1) {
            var pa = std.heap.ArenaAllocator.init(testing.allocator);
            defer pa.deinit();
            _ = try bp.emitReplay(pa.allocator(), &cfg, &w, o, dec, m, 0, .decoder);
        }
        for ([_]i64{ 1, 4096 - 127, 65536, 524288 - 127, 524288, big - 1 - ced.keep }) |r0| {
            var pa = std.heap.ArenaAllocator.init(testing.allocator);
            defer pa.deinit();
            _ = try bp.emitReplay(pa.allocator(), &cfg, &w, o, dec, ced.keep, r0, .decoder);
        }
    }
}

test "CED: the decoder pass writes the tail's DSpark taps (layers 37-39, rows 0 .. m), the encoder none; one ingest takes them" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    var layers: [40]u32 = undefined;
    for (&layers, 0..) |*l, i| l.* = @intCast(i);
    const d = try ced.decoderStart(&cfg);
    const targets = cfg.dspark_targets.items();
    const D: i64 = cfg.hidden;
    const T: i64 = @as(i64, @intCast(targets.len)) * D;
    for ([_]i64{ 12, 44, ced.keep }) |m| {
        const o: block.Options = .{ .limit = limit, .rope_rows = limit + 2048, .taps = true };
        const dec = try bp.emitReplay(a, &cfg, &w, o, layers[d..], m, 1000, .decoder);
        const enc = try bp.emitReplay(a, &cfg, &w, o, layers[0 .. d + 1], 2048, 0, .encoder);
        // each target layer's column block of "w.taps", the pass's m rows, written once
        var seen: [8]usize = @splat(0);
        for (dec) |c| for (c.args) |x| if (x.arg == .t and std.mem.eql(u8, roleOf(x.arg), "w.taps")) {
            try testing.expectEqual(m, x.arg.t.shape[0]);
            try testing.expectEqual(T, x.arg.t.stride[0]);
            const j: usize = @intCast(@divExact(x.arg.t.offset, 2 * D));
            try testing.expect(j < targets.len);
            seen[j] += 1;
        };
        for (seen[0..targets.len]) |k| try testing.expectEqual(@as(usize, 1), k);
        for (enc) |c| for (c.args) |x| try testing.expect(!std.mem.eql(u8, roleOf(x.arg), "w.taps"));
        // the hand-off: all m rows from taps row 0, at R0; the pass's ingest reads exactly them
        const h = ced.handoff(1000, @intCast(m), cfg.window).?;
        // the ingest's main_proj (not in the backbone fixture): a width of the DSpark blocks' own, as slot_drafts_test
        if (w.dense.get("dspark.main_proj") == null) try w.dense.put(a, "dspark.main_proj", try w.k2("L40.attn.wkv"));
        try testing.expectEqual(ced.Handoff{ .start = 1000, .row0 = 0, .n = @intCast(m) }, h);
        const ing = try dspark_emit.emitIngest(a, &cfg, &w, o, h.n, h.row0);
        var reads: usize = 0;
        for (ing) |c| for (c.args) |x| if (x.arg == .t and std.mem.eql(u8, roleOf(x.arg), "w.taps")) {
            try testing.expectEqual(m, x.arg.t.shape[0]);
            try testing.expectEqual(T, x.arg.t.shape[1]);
            try testing.expectEqual(@as(i64, 0), x.arg.t.offset);
            reads += 1;
        };
        try testing.expect(reads >= 1);
    }
}
