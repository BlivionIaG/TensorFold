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

test "CED: the exact replay's programs: the encoder fills the cone ring, a program hands the next its rows, candidates shift" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    const d = try ced.decoderStart(&cfg);
    var layers: [25]u32 = undefined;
    for (&layers, 0..) |*l, i| l.* = @intCast(i);
    const dec = layers[d..];
    const o: block.Options = .{ .limit = limit, .rope_rows = limit + 2048 };
    // the encoder pass with the cone ring: the plain pass, then its stash again into the cone's ring
    const plain = try bp.emitReplay(a, &cfg, &w, o, layers[0 .. d + 1], 300, 1000, .encoder);
    const enc = try bp.emitReplay(a, &cfg, &w, o, layers[0 .. d + 1], 300, 1000, .encoder_cone);
    try testing.expectEqual(plain.len + 1, enc.len);
    try testing.expectEqualStrings("glue.ced_stash", enc[enc.len - 2].name);
    const stash = enc[enc.len - 1];
    try testing.expectEqualStrings("glue.cone_stash", stash.name);
    for (ced.cone_roles, 5..) |role, j| {
        try testing.expectEqualStrings(role, roleOf(stash.args[j].arg));
        try testing.expectEqual(@as(i64, ced.cone_keep), stash.args[j].arg.t.shape[0]);
    }
    // the programs: [20, 24) with the next one's first site, then [24, 25) from the ring, candidates 40 rows in
    var buf: [8]ced.Group = undefined;
    const gs = ced.groups(&cfg, dec, 4, &buf);
    try testing.expectEqual(@as(usize, 2), gs.len);
    const mid = try bp.emitCone(a, &cfg, &w, o, dec[gs[0].first .. gs[0].first + gs[0].count + 1], 300, 1000, .cone_mid, null);
    try testing.expectEqualStrings("glue.positions", mid[0].name);
    try testing.expectEqualStrings("glue.cone_load", mid[1].name);
    try testing.expectEqualStrings("glue.cone_stash", mid[mid.len - 1].name);
    try testing.expectEqual(@as(i64, 1000), mid[0].args[3].arg.i);
    for (ced.cone_roles, 5..) |role, j| try testing.expectEqualStrings(role, roleOf(mid[1].args[j].arg));
    const last = try bp.emitCone(a, &cfg, &w, o, dec[gs[1].first .. gs[1].first + gs[1].count], 200, 1100, .cone_last, 40);
    try testing.expectEqualStrings("glue.cone_load", last[1].name);
    for (last) |c| try testing.expect(!std.mem.eql(u8, c.name, "glue.cone_stash") and !std.mem.eql(u8, c.name, "glue.ced_stash"));
    // layer 24 (an index source) reads layer 20's candidates from the program's first row: 40 rows into them
    var shifted: usize = 0;
    const cb: i64 = cfg.candidate_blocks;
    for (last) |c| for (c.args) |x| if (x.arg == .t and std.mem.eql(u8, roleOf(x.arg), "w.cand")) {
        try testing.expect(x.arg.t.offset >= 40 * cb * 4);
        shifted += 1;
    };
    try testing.expect(shifted > 0);
    // the first program computes them (nothing shifted), and the decoder replay's own part is unchanged by the new ones
    for (mid) |c| for (c.args) |x| if (x.arg == .t and std.mem.eql(u8, roleOf(x.arg), "w.cand")) try testing.expect(x.arg.t.offset < 40 * cb * 4);
    try testing.expectError(error.Unsupported, bp.emitCone(a, &cfg, &w, o, &.{}, 300, 1000, .cone_last, null));
}

fn programNames(a: std.mem.Allocator, cs: []const calls.Call, out: *std.ArrayList([]const u8)) !void {
    for (cs) |c| try out.append(a, c.name);
}

test "CED: the exact replay's programs, chained, run the decoder pass's calls in its order; every shape emits to the 1M limit" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    var layers: [40]u32 = undefined;
    for (&layers, 0..) |*l, i| l.* = @intCast(i);
    const dec = layers[try ced.decoderStart(&cfg)..];
    const big: i64 = 1 << 20;
    const pages = @divExact(big, block.Pool.page);
    for ([_]u32{ 1, 4, 5, 20 }) |group| for ([_]bool{ false, true }) |k4| for ([_]i64{ 300, 2033, 2541 }) |n| for ([_]i64{ 0, 4096, 524288 - 1 }) |start| {
        var o: block.Options = .{ .limit = big, .rope_rows = big + 2048, .index_budget = 64 << 20, .taps = true };
        o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages };
        if (k4) {
            o.prefill_rows = 4096;
            o.pf4k = true;
        } else if (n > o.prefill_rows) continue;
        var buf: [20]ced.Group = undefined;
        const gs = ced.groups(&cfg, dec, group, &buf);
        const whole = try bp.emitReplay(a, &cfg, &w, o, dec, n, start, .decoder);
        var want: std.ArrayList([]const u8) = .empty;
        try programNames(a, whole[2..], &want);
        var got: std.ArrayList([]const u8) = .empty;
        for (gs, 0..) |g, gi| {
            const last = gi + 1 == gs.len;
            const lay = dec[g.first .. g.first + g.count + @intFromBool(!last)];
            const cs = try bp.emitCone(a, &cfg, &w, o, lay, n, start, if (last) .cone_last else .cone_mid, if (gi == 0) null else 0);
            try testing.expectEqualStrings("glue.cone_load", cs[1].name);
            try programNames(a, cs[2 .. cs.len - @intFromBool(!last)], &got);
            if (!last) try testing.expectEqualStrings("glue.cone_stash", cs[cs.len - 1].name);
        }
        try testing.expectEqual(want.items.len, got.items.len);
        for (want.items, got.items) |x, y| try testing.expectEqualStrings(x, y);
        // the drafter's taps: written once each, by the last program, whose first layer is below every target
        var taps: usize = 0;
        for (gs, 0..) |g, gi| {
            const last = gi + 1 == gs.len;
            const cs = try bp.emitCone(a, &cfg, &w, o, dec[g.first .. g.first + g.count + @intFromBool(!last)], n, start, if (last) .cone_last else .cone_mid, if (gi == 0) null else 0);
            for (cs) |c| for (c.args) |x| if (x.arg == .t and std.mem.eql(u8, roleOf(x.arg), "w.taps")) {
                taps += 1;
                try testing.expect(last);
            };
        }
        var whole_taps: usize = 0;
        for (whole) |c| for (c.args) |x| if (x.arg == .t and std.mem.eql(u8, roleOf(x.arg), "w.taps")) {
            whole_taps += 1;
        };
        try testing.expectEqual(whole_taps, taps);
    };
}

const prompt_lengths = [_]u64{ 128, 328, 1024, 1810, 2047, 2048, 2600, 4096, 10572 };

test "CED: the exact replay's rows per layer equal the cone rule's, for prompts from the window past the cone" {
    const cfg: Config = .{};
    const d = try ced.decoderStart(&cfg);
    const nd = cfg.layers - d;
    var layers: [64]u32 = undefined;
    for (layers[0..cfg.layers], 0..) |*l, i| l.* = @intCast(i);
    const dec = layers[d..cfg.layers];
    const W = cfg.window;
    try testing.expectEqual(ced.cone_keep, ced.groupRows(W, nd, 0, std.math.maxInt(u64)));
    for (prompt_lengths) |n| for ([_]u32{ 1, 4, 5, 20 }) |group| {
        const end = n - 1;
        var buf: [32]ced.Group = undefined;
        const gs = ced.groups(&cfg, dec, group, &buf);
        var prev: u32 = std.math.maxInt(u32);
        var seen: u32 = 0;
        for (gs) |g| {
            const rows = ced.groupRows(W, nd, g.first, end);
            // the program starts on its first layer's cone and a later program never runs more rows than an earlier one
            try testing.expectEqual(@min(end, try ced.layerRows(&cfg, ced.replay_rows, d + g.first)), rows);
            try testing.expect(rows <= prev and rows <= ced.cone_keep and rows <= end);
            prev = rows;
            for (g.first..g.first + g.count) |k| {
                const rule = try ced.layerRowsOf(&cfg, ced.replay_rows, d + @as(u32, @intCast(k)), @intCast(end));
                // every layer of the program replays its cone's rows (a group of one: exactly them)
                try testing.expect(rows >= rule);
                if (g.count == 1) try testing.expectEqual(rule, rows);
                seen += 1;
            }
        }
        try testing.expectEqual(nd, seen);
        // the first program starts at the prompt's start when the cone is not shorter, else on cone rows of the stash
        try testing.expectEqual(@min(end, ced.cone_keep), ced.groupRows(W, nd, gs[0].first, end));
        // the last program: the window's rows (or the whole prompt below it)
        try testing.expectEqual(@min(end, W), ced.groupRows(W, nd, gs[gs.len - 1].first + gs[gs.len - 1].count - 1, end));
        // the stash a prompt run from 0 leaves holds every program's rows
        try testing.expect(ced.coneHeld(W, nd, end, end -| ced.cone_keep));
    };
}

test "CED: a prompt resumed from a snapshot keeps the exact replay only when the stash after it holds the cone" {
    const cfg: Config = .{};
    const T = struct { n: u64, at: u64, exact: bool };
    const cases = [_]T{
        .{ .n = 128, .at = 64, .exact = true }, // 127 prefilled rows: the bounded replay is already exact
        .{ .n = 129, .at = 0, .exact = true },
        .{ .n = 129, .at = 64, .exact = false },
        .{ .n = 329, .at = 320, .exact = false }, // the 328-token prompt's snapshot under a longer one
        .{ .n = 1811, .at = 0, .exact = true },
        .{ .n = 1811, .at = 320, .exact = false }, // the 1,810-token export resumed at the 328-token one's snapshot
        .{ .n = 2048, .at = 16, .exact = false },
        .{ .n = 2049, .at = 16, .exact = false },
        .{ .n = 2601, .at = 59, .exact = true }, // end 2,600: the cone is 2,541 rows, 2,541 remain behind 59
        .{ .n = 2601, .at = 64, .exact = false },
        .{ .n = 4097, .at = 1024, .exact = true },
        .{ .n = 4097, .at = 1600, .exact = false },
        .{ .n = 10573, .at = 1792, .exact = true }, // the 10,572-token export resumed at the 1,810-token one's snapshot
        .{ .n = 10573, .at = 8031, .exact = true },
        .{ .n = 10573, .at = 8032, .exact = false },
    };
    for (cases) |c| try testing.expectEqual(c.exact, ced.resumeExact(&cfg, c.n - 1, c.at));
    // a resume at 0 is a cold prompt: exact for every length
    for (prompt_lengths) |n| try testing.expect(ced.resumeExact(&cfg, n - 1, 0));
}

test "warm resume: the cone is usable from exact rings at its first row, any number of rows after them" {
    const W: u32 = 128;
    const nd: u32 = 20;
    // a fresh prompt (stash from 0) and a held cone behave as before
    try testing.expect(ced.coneUsable(W, nd, 300, 0, null));
    try testing.expect(ced.coneUsable(W, nd, 9000, 9000 - ced.cone_keep, null));
    // a short run behind a resumed entry is usable only when the rings are exact at the entry
    try testing.expect(!ced.coneUsable(W, nd, 8200, 8192, null));
    try testing.expect(ced.coneUsable(W, nd, 8200, 8192, 8192));
    try testing.expect(!ced.coneUsable(W, nd, 8200, 8192, 8000));
    // no new rows: nothing to replay
    try testing.expect(!ced.coneUsable(W, nd, 8192, 8192, 8192));
    // the replay's rows are those since the entry, capped by each program's cone
    try testing.expectEqual(@as(u32, 200), ced.groupRows(W, nd, 0, 200));
    try testing.expectEqual(@as(u32, 200), ced.groupRows(W, nd, 16, 200));
    try testing.expectEqual(ced.cone_keep, ced.groupRows(W, nd, 0, 5000));
    try testing.expectEqual(@as(u32, 2541 - 127 * 16), ced.groupRows(W, nd, 16, 5000));
    // warm entries carry their own tag: neither a bounded replay's nor a full-mode entry resumes as one
    try testing.expect(ced.tag_warm != ced.tag and ced.tag_warm != ced.tag_fast and ced.tag_warm & ced.tag == ced.tag);
}
