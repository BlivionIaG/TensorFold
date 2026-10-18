//! Stream D of the EXL3 step: broader correctness of the assembled forward, long context, and the prefix cache.
//!   tf-cuda-test exl3-ctx <dir> tf <layers csv> <deep> <ids>...        multi-prompt teacher forcing vs the f64 oracle; prompts
//!                                                                      beyond the first `deep` are only run for configs <= 12 layers
//!   tf-cuda-test exl3-ctx <dir> long <layers> <ids> <oracle 0|1> <gen> <expect csv|->
//!                                                                      a ~2300 token prompt through the sparse indexer
//!   tf-cuda-test exl3-ctx <dir> prefix <layers> <A> <B> <C> <gen>      snapshot after A inside A+B, restore for A+C vs from scratch
//!   tf-cuda-test exl3-ctx <dir> needle <layers> <gen> <doc> <suffix>...  one long document; suffix 0 live, then restored suffixes
//! `ids` are comma separated token id lists (tools/exl3_ctx_prompts.py writes them).

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const real = @import("exl3_real_tests.zig");
const fwd = @import("exl3_fwd.zig");
const ft = @import("exl3_fwd_tests.zig");
const cpu = @import("exl3_fwd_cpu.zig");
const fple = @import("exl3_fwd_ple.zig");
const gdn_cpu = @import("exl3_gdn_cpu.zig");
const attn_cpu = @import("exl3_attn_cpu.zig");
const snap = @import("exl3_snap.zig");
const prefix = @import("exl3_prefix.zig");
const cache_rules = @import("exl3_cache.zig");
const window = @import("exl3_window.zig");
const st = @import("safetensors.zig");

const Gpu = check.Gpu;
const hidden = fwd.hidden;
const wide = fwd.wide;
const vocab = fwd.vocab;
const nowMs = ft.nowMs;

fn ngramPath(buf: []u8, dir: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/ngram_embedding.safetensors", .{dir});
}

/// Feeds `tokens` in chunks of at most `chunk` rows. `h_all` keeps every row's streams; `logits` every row's logits.
fn feed(e: *fwd.Engine, tokens: []const u32, chunk: usize, hbuf: []f32, h_all: ?[]f32, logits: ?[]f32) !void {
    var off: usize = 0;
    while (off < tokens.len) {
        const n: usize = @min(chunk, tokens.len - off);
        try e.forward(tokens[off..][0..n], hbuf[0 .. n * wide]);
        if (h_all) |ha| @memcpy(ha[off * wide ..][0 .. n * wide], hbuf[0 .. n * wide]);
        if (logits) |lg| try e.logitsOf(hbuf[0 .. n * wide], n, lg[off * vocab ..][0 .. n * vocab]);
        off += n;
    }
}

fn top2(xs: anytype) struct { a: usize, b: usize } {
    var a: usize = 0;
    var b: usize = 1;
    if (xs[1] > xs[0]) {
        a = 1;
        b = 0;
    }
    for (xs, 0..) |v, i| {
        if (i < 2) continue;
        if (v > xs[a]) {
            b = a;
            a = i;
        } else if (v > xs[b]) b = i;
    }
    return .{ .a = a, .b = b };
}

// -- mode: tf ---------------------------------------------------------------------------------------------------

const HeadJob = struct {
    m: *const cpu.QMat,
    x: []const f64,
    out: []f64,
    gpa: std.mem.Allocator,
    fn row(j: *@This(), t: usize) void {
        const tmp = j.gpa.alloc(f64, hidden) catch return;
        defer j.gpa.free(tmp);
        j.m.apply(j.x[t * hidden ..][0..hidden], j.out[t * vocab ..][0..vocab], tmp);
    }
};

fn resetOState(gpa: std.mem.Allocator, o: *ft.OLayer, cap: usize) !void {
    switch (o.state) {
        .gdn => |*s| {
            s.deinit(gpa);
            o.state = .{ .gdn = try gdn_cpu.State.init(gpa, gdn_cpu.qwen) };
        },
        .attn => |*c| {
            c.deinit();
            o.state = .{ .attn = try attn_cpu.Cache.init(gpa, cap) };
        },
    }
}

fn parseCsv(gpa: std.mem.Allocator, text: []const u8) ![]usize {
    var list: std.ArrayList(usize) = .empty;
    errdefer list.deinit(gpa);
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |tok| {
        if (tok.len == 0) continue;
        try list.append(gpa, try std.fmt.parseInt(usize, tok, 10));
    }
    return list.toOwnedSlice(gpa);
}

fn teacherForcing(gpu: Gpu, pack: *const real.Pack, dir: []const u8, cfgs: []const usize, deep: usize, prompts: []const []const u32) !void {
    const gpa = gpu.gpa;
    var pbuf: [512]u8 = undefined;
    const ngram = try ngramPath(&pbuf, dir);
    var maxlen: usize = 0;
    for (prompts) |p| maxlen = @max(maxlen, p.len);
    const P = prompts.len;
    const shallow_limit: usize = 12;
    var shallow_max: usize = 0;
    for (cfgs) |n| {
        if (n <= shallow_limit) shallow_max = @max(shallow_max, n);
    }
    const count = struct {
        fn of(n: usize, p: usize, d: usize) usize {
            return if (n <= shallow_limit) p else @min(d, p);
        }
    }.of;

    // ---- device logits for every config / prompt
    const dev = try gpa.alloc([][]f32, cfgs.len);
    defer gpa.free(dev);
    var offset: f32 = 0;
    for (cfgs, 0..) |n, ci| {
        const cnt = count(n, P, deep);
        dev[ci] = try gpa.alloc([]f32, cnt);
        var t0 = nowMs(gpu);
        var eng = try fwd.Engine.init(gpu, pack, .{ .first_layer = 0, .layers = n, .max_rows = maxlen, .cap = maxlen + 8, .head = true, .ngram_path = if (n > fwd.ple_layer) ngram else null });
        defer eng.deinit();
        offset = eng.offset;
        std.debug.print("== device, {d} layers, {d} prompts (PLE {s}): loaded in {d:.0} ms\n", .{ n, cnt, if (eng.ple != null) "on" else "off", nowMs(gpu) - t0 });
        const hbuf = try gpa.alloc(f32, maxlen * wide);
        defer gpa.free(hbuf);
        for (0..cnt) |pi| {
            dev[ci][pi] = try gpa.alloc(f32, prompts[pi].len * vocab);
            try eng.reset();
            t0 = nowMs(gpu);
            try feed(&eng, prompts[pi], maxlen, hbuf, null, dev[ci][pi]);
            std.debug.print("  prompt {d} ({d} tokens): {d:.0} ms\n", .{ pi, prompts[pi].len, nowMs(gpu) - t0 });
        }
    }

    // ---- oracle, layer by layer, with taps at the config depths
    const t0 = nowMs(gpu);
    const orc = try gpa.alloc([][]f64, cfgs.len);
    defer gpa.free(orc);
    for (cfgs, 0..) |n, ci| orc[ci] = try gpa.alloc([]f64, count(n, P, deep));
    const hs = try gpa.alloc([]f64, P);
    defer gpa.free(hs);
    const embed_table = (pack.find("model.language_model.embed_tokens.weight") orelse return error.MissingTensor).bytes;
    for (hs, prompts) |*h, p| h.* = try ft.embedStreams(gpa, embed_table, p);
    var ng_file: ?st.File = null;
    defer if (ng_file) |*f| f.close(gpu.io);
    var ple_o: ?fple.Ple = null;
    defer if (ple_o) |*p| p.deinit();
    var max_n: usize = 0;
    for (cfgs) |n| max_n = @max(max_n, n);
    if (max_n > fwd.ple_layer) {
        ng_file = try st.File.open(gpa, gpu.io, ngram);
        ple_o = try fwd.loadPle(gpu, pack, &ng_file.?);
    }
    var head = try ft.qmat(gpa, pack, "lm_head", 48);
    defer head.deinit(gpa);
    const mixer_hc = try fwd.HcW.load(gpa, pack, "model.language_model.hyper_connection_mixer", .{}, false);
    defer mixer_hc.free(gpa);
    for (0..max_n) |l| {
        const lt = nowMs(gpu);
        const active = if (l < shallow_max) P else @min(deep, P);
        if (ple_o != null and l == fwd.ple_layer) {
            for (0..active) |pi| {
                ple_o.?.reset();
                const out = try gpa.alloc(f64, prompts[pi].len * wide);
                defer gpa.free(out);
                try ple_o.?.rows(prompts[pi], hs[pi], out);
                for (hs[pi], out) |*d, s| d.* += s;
            }
        }
        var ol = try ft.OLayer.create(gpa, pack, l, offset, maxlen + 8);
        defer ol.destroy();
        for (0..active) |pi| {
            try resetOState(gpa, ol, maxlen + 8);
            try cpu.layerRows(gpa, ol.weights(), &ol.state, 0, offset, hs[pi], prompts[pi].len, null);
        }
        for (cfgs, 0..) |n, ci| {
            if (n != l + 1) continue;
            for (0..orc[ci].len) |pi| {
                const rows = prompts[pi].len;
                const mixed = try gpa.alloc(f64, rows * hidden);
                defer gpa.free(mixed);
                try cpu.hcPreRows(gpa, mixer_hc.weights(), offset, hs[pi], rows, mixed, null);
                orc[ci][pi] = try gpa.alloc(f64, rows * vocab);
                var hj: HeadJob = .{ .m = &head, .x = mixed, .out = orc[ci][pi], .gpa = gpa };
                cpu.parallelFor(rows, 48, &hj, HeadJob.row);
            }
            std.debug.print("  oracle logits tapped after {d} layers\n", .{n});
        }
        if (l < 4 or l % 8 == 7) std.debug.print("  oracle layer {d} ({d} prompts) {d:.0} ms\n", .{ l, active, nowMs(gpu) - lt });
    }
    std.debug.print("  oracle total {d:.0} ms\n", .{nowMs(gpu) - t0});

    // ---- compare
    var all_ok = true;
    for (cfgs, 0..) |n, ci| {
        var agree: usize = 0;
        var ties: usize = 0;
        var hard: usize = 0;
        var total: usize = 0;
        var num: f64 = 0;
        var den: f64 = 0;
        var lmax: f64 = 0;
        for (dev[ci], orc[ci], 0..) |dl, ol, pi| {
            const rows = prompts[pi].len;
            var pmax: f64 = 0;
            var pnum: f64 = 0;
            var pden: f64 = 0;
            var pagree: usize = 0;
            for (dl, ol) |a, b| {
                const d = @as(f64, a) - b;
                pmax = @max(pmax, @abs(d));
                pnum += d * d;
                pden += b * b;
            }
            for (0..rows) |t| {
                const g = dl[t * vocab ..][0..vocab];
                const o = ol[t * vocab ..][0..vocab];
                const gi = ft.argmax(g);
                const t2 = top2(o);
                const oi = t2.a;
                total += 1;
                if (gi == oi) {
                    agree += 1;
                    pagree += 1;
                    continue;
                }
                const gap = o[oi] - o[gi];
                const margin = o[oi] - o[t2.b];
                const tie = gap <= 2 * pmax;
                if (tie) ties += 1 else hard += 1;
                std.debug.print("  FLIP {d} layers prompt {d} pos {d} (input {d}): device {d} (oracle logit {d:.4}) vs oracle {d} ({d:.4}); oracle gap {d:.4}, oracle top1-top2 margin {d:.4} => {s}\n", .{ n, pi, t, prompts[pi][t], gi, o[gi], oi, o[oi], gap, margin, if (tie) "near-tie" else "HARD FLIP" });
            }
            num += pnum;
            den += pden;
            lmax = @max(lmax, pmax);
            std.debug.print("  {d} layers prompt {d} ({d} tokens): top-1 {d}/{d}, logits max_abs {d:.4}, rms abs {d:.5}, rms rel {e:.2}\n", .{ n, pi, rows, pagree, rows, pmax, @sqrt(pnum / @as(f64, @floatFromInt(rows * vocab))), @sqrt(pnum / pden) });
        }
        std.debug.print("RESULT {d} layers: {d} prompts, {d} positions: top-1 exact {d}, near-tie flips {d}, hard flips {d}; logits max_abs {d:.4}, rms rel {e:.2}\n", .{ n, dev[ci].len, total, agree, ties, hard, lmax, @sqrt(num / den) });
        if (hard != 0) all_ok = false;
    }
    try check.expect(all_ok, "teacher forcing: a top-1 flip beyond a near-tie (see FLIP lines)", .{});
    check.pass("multi-prompt teacher forcing vs the f64 oracle at {any} layers", .{cfgs});
}

// -- mode: long -------------------------------------------------------------------------------------------------

fn longContext(gpu: Gpu, pack: *const real.Pack, dir: []const u8, n_layers: usize, ids: []const u32, with_oracle: bool, gen: usize, expect: []const u32, chunk: usize) !void {
    const gpa = gpu.gpa;
    var pbuf: [512]u8 = undefined;
    const ngram = try ngramPath(&pbuf, dir);
    const n = ids.len;
    const cap = (n + gen + 8 + 3) / 4 * 4;
    std.debug.print("== long context: {d} prompt tokens, {d} layers, cap {d}, oracle {s}\n", .{ n, n_layers, cap, if (with_oracle) "on" else "off" });
    var t0 = nowMs(gpu);
    var eng = try fwd.Engine.init(gpu, pack, .{ .first_layer = 0, .layers = n_layers, .max_rows = chunk, .cap = cap, .head = true, .ngram_path = if (n_layers > fwd.ple_layer) ngram else null });
    defer eng.deinit();
    std.debug.print("  loaded in {d:.0} ms\n", .{nowMs(gpu) - t0});
    const hbuf = try gpa.alloc(f32, chunk * wide);
    defer gpa.free(hbuf);
    const h_all: ?[]f32 = if (with_oracle) try gpa.alloc(f32, n * wide) else null;
    defer if (h_all) |h| gpa.free(h);
    const lg = try gpa.alloc(f32, chunk * vocab);
    defer gpa.free(lg);
    const last_rows = if (n % chunk == 0) chunk else n % chunk;

    // prefill in chunks, logits only on the last chunk
    t0 = nowMs(gpu);
    var off: usize = 0;
    var nonfinite: usize = 0;
    while (off < n) {
        const m: usize = @min(chunk, n - off);
        try eng.forward(ids[off..][0..m], hbuf[0 .. m * wide]);
        for (hbuf[0 .. m * wide]) |v| nonfinite += @intFromBool(!std.math.isFinite(v));
        if (h_all) |ha| @memcpy(ha[off * wide ..][0 .. m * wide], hbuf[0 .. m * wide]);
        off += m;
        if ((off / chunk) % 8 == 0 or off == n) std.debug.print("  prefilled {d}/{d} in {d:.0} s (non-finite stream values so far {d})\n", .{ off, n, (nowMs(gpu) - t0) / 1000, nonfinite });
        if (off == n) try eng.logitsOf(hbuf[0 .. m * wide], m, lg[0 .. m * vocab]);
    }
    const prefill_s = (nowMs(gpu) - t0) / 1000;
    var finite_logits: usize = 0;
    for (lg[0 .. last_rows * vocab]) |v| finite_logits += @intFromBool(std.math.isFinite(v));
    const last = lg[(last_rows - 1) * vocab ..][0..vocab];
    const t2 = top2(last);
    std.debug.print("  prefill {d} tokens in {d:.1} s ({d:.2} s/token, a measurement of this test not a throughput claim); logits finite {d}/{d}\n", .{ n, prefill_s, prefill_s / @as(f64, @floatFromInt(n)), finite_logits, last_rows * vocab });
    var five: [5]usize = undefined;
    for (0..5) |k| {
        var best: usize = 0;
        var bv: f32 = -std.math.inf(f32);
        for (last, 0..) |v, i| {
            var skip = false;
            for (five[0..k]) |o| skip = skip or o == i;
            if (!skip and v > bv) {
                bv = v;
                best = i;
            }
        }
        five[k] = best;
    }
    std.debug.print("LAST top5 ids:", .{});
    for (five) |i| std.debug.print(" {d}({d:.3})", .{ i, last[i] });
    std.debug.print("  top1-top2 margin {d:.3}\n", .{last[t2.a] - last[t2.b]});

    // greedy continuation
    var next: u32 = @intCast(t2.a);
    var gen_ids: std.ArrayList(u32) = .empty;
    defer gen_ids.deinit(gpa);
    var g: usize = 0;
    while (g < gen) : (g += 1) {
        try gen_ids.append(gpa, next);
        if (g + 1 == gen) break;
        var one = [_]u32{next};
        t0 = nowMs(gpu);
        try eng.forward(&one, hbuf[0..wide]);
        try eng.logitsOf(hbuf[0..wide], 1, lg[0..vocab]);
        next = @intCast(ft.argmax(lg[0..vocab]));
    }
    std.debug.print("GEN ids=", .{});
    for (gen_ids.items, 0..) |v, i| std.debug.print("{d}{s}", .{ v, if (i + 1 < gen_ids.items.len) "," else "\n" });
    if (expect.len > 0) {
        const k = @min(expect.len, gen_ids.items.len);
        const same = k > 0 and std.mem.eql(u32, expect[0..k], gen_ids.items[0..k]);
        std.debug.print("NEEDLE expected ids {any}, generated {any}: {s}\n", .{ expect, gen_ids.items, if (same) "MATCH" else "MISMATCH" });
    }
    std.debug.print("  non-finite stream values: {d}\n", .{nonfinite});
    try check.expect(nonfinite == 0 and finite_logits == last_rows * vocab, "long prompt produced non-finite values", .{});

    if (!with_oracle) {
        check.pass("long prompt {d} tokens x {d} layers: runs, finite logits (no oracle at this depth)", .{ n, n_layers });
        return;
    }

    // ---- oracle on the same layers
    t0 = nowMs(gpu);
    const embed_table = (pack.find("model.language_model.embed_tokens.weight") orelse return error.MissingTensor).bytes;
    const h = try ft.embedStreams(gpa, embed_table, ids);
    defer gpa.free(h);
    var ng_file: ?st.File = null;
    defer if (ng_file) |*f| f.close(gpu.io);
    var ple_o: ?fple.Ple = null;
    defer if (ple_o) |*p| p.deinit();
    if (n_layers > fwd.ple_layer) {
        ng_file = try st.File.open(gpa, gpu.io, ngram);
        ple_o = try fwd.loadPle(gpu, pack, &ng_file.?);
    }
    for (0..n_layers) |l| {
        const lt = nowMs(gpu);
        if (ple_o != null and l == fwd.ple_layer) {
            // the PLE add-back sits at the start of decoder layer 1, on the streams layer 0 left
            const out = try gpa.alloc(f64, n * wide);
            defer gpa.free(out);
            try ple_o.?.rows(ids, h, out);
            for (h, out) |*d, s| d.* += s;
            std.debug.print("  oracle PLE done {d:.0} ms\n", .{nowMs(gpu) - lt});
        }
        var ol = try ft.OLayer.create(gpa, pack, l, eng.offset, cap);
        defer ol.destroy();
        try cpu.layerRows(gpa, ol.weights(), &ol.state, 0, eng.offset, h, n, null);
        std.debug.print("  oracle layer {d} ({s}) {d:.0} s\n", .{ l, if (fwd.Layer.isAttn(l)) "attention" else "GDN", (nowMs(gpu) - lt) / 1000 });
    }
    // stream error per region: dense (< 2051 keys) and sparse (>= 2051)
    const split: usize = 2051;
    var dense = ft.Err{ .max_abs = 0, .peak = 0, .rel = 0 };
    var sparse = ft.Err{ .max_abs = 0, .peak = 0, .rel = 0 };
    for (0..n) |t| {
        const e = ft.errOf(h_all.?[t * wide ..][0..wide], h[t * wide ..][0..wide]);
        const slot_ = if (t + 1 < split) &dense else &sparse;
        slot_.max_abs = @max(slot_.max_abs, e.max_abs);
        slot_.peak = @max(slot_.peak, e.peak);
        // report the worst per-row relative error, the same measure as the layer test
        slot_.rel = @max(slot_.rel, e.rel);
    }
    std.debug.print("RESULT streams after {d} layers vs oracle: rows with < 2051 keys: worst per-row rel {d:.5} (max_abs {d:.3}, peak {d:.1}); rows with >= 2051 keys (sparse 512-block selection): worst per-row rel {d:.5} (max_abs {d:.3}, peak {d:.1})\n", .{ n_layers, dense.rel, dense.max_abs, dense.peak, sparse.rel, sparse.max_abs, sparse.peak });
    // where does it diverge: worst per-row rel per block of 64 rows
    {
        var b0: usize = 0;
        while (b0 < n) : (b0 += 64) {
            var worst: f64 = 0;
            var worst_t: usize = b0;
            for (b0..@min(b0 + 64, n)) |t| {
                const e = ft.errOf(h_all.?[t * wide ..][0..wide], h[t * wide ..][0..wide]);
                if (e.rel > worst) {
                    worst = e.rel;
                    worst_t = t;
                }
            }
            if (b0 < 512 or b0 % 512 == 0 or worst > 0.1) std.debug.print("  rows {d}..{d}: worst per-row rel {d:.5} at row {d} (token {d})\n", .{ b0, @min(b0 + 64, n) - 1, worst, worst_t, ids[worst_t] });
        }
    }
    // mean-over-rows rel in the sparse region as well (worst rows can sit on a different expert pick)
    var srel: f64 = 0;
    var srows: usize = 0;
    for (@min(split - 1, n)..n) |t| {
        const e = ft.errOf(h_all.?[t * wide ..][0..wide], h[t * wide ..][0..wide]);
        srel += e.rel;
        srows += 1;
    }
    if (srows > 0) std.debug.print("  sparse region mean per-row rel {d:.5} over {d} rows\n", .{ srel / @as(f64, @floatFromInt(srows)), srows });

    // logits at the last 8 positions
    const k8: usize = @min(8, last_rows);
    const mixed = try gpa.alloc(f64, k8 * hidden);
    defer gpa.free(mixed);
    const hw = try fwd.HcW.load(gpa, pack, "model.language_model.hyper_connection_mixer", .{}, false);
    defer hw.free(gpa);
    try cpu.hcPreRows(gpa, hw.weights(), eng.offset, h[(n - k8) * wide ..], k8, mixed, null);
    var head = try ft.qmat(gpa, pack, "lm_head", 48);
    defer head.deinit(gpa);
    const lo = try gpa.alloc(f64, k8 * vocab);
    defer gpa.free(lo);
    var hj: HeadJob = .{ .m = &head, .x = mixed, .out = lo, .gpa = gpa };
    cpu.parallelFor(k8, 48, &hj, HeadJob.row);
    var flips: usize = 0;
    var lmax: f64 = 0;
    for (0..k8) |r| {
        const dl = lg[(last_rows - k8 + r) * vocab ..][0..vocab];
        const ol = lo[r * vocab ..][0..vocab];
        for (dl, ol) |a, b| lmax = @max(lmax, @abs(@as(f64, a) - b));
        const gi = ft.argmax(dl);
        const o2 = top2(ol);
        if (gi != o2.a) {
            flips += 1;
            std.debug.print("  FLIP at position {d}: device {d}, oracle {d}; oracle gap {d:.4}, oracle margin {d:.4}\n", .{ n - k8 + r, gi, o2.a, ol[o2.a] - ol[gi], ol[o2.a] - ol[o2.b] });
        }
    }
    std.debug.print("RESULT logits over the last {d} positions: top-1 equal {d}/{d}, max_abs {d:.4}\n", .{ k8, k8 - flips, k8, lmax });
    std.debug.print("  oracle total {d:.0} s\n", .{(nowMs(gpu) - t0) / 1000});
    try check.expect(sparse.rel <= 0.03 and dense.rel <= 0.03, "long prompt: streams differ from the oracle (dense {d:.4}, sparse {d:.4})", .{ dense.rel, sparse.rel });
    check.pass("long prompt {d} tokens x {d} layers: streams match the f64 oracle (dense rows {d:.5}, sparse-selection rows {d:.5}), last-8 top-1 equal {d}/{d}", .{ n, n_layers, dense.rel, sparse.rel, k8 - flips, k8 });
}

// -- mode: prefix -----------------------------------------------------------------------------------------------

fn bitsEqual(a: []const f32, b: []const f32) struct { equal: bool, max_abs: f64, differing: usize } {
    var diff: usize = 0;
    var mx: f64 = 0;
    for (a, b) |x, y| {
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) {
            diff += 1;
            mx = @max(mx, @abs(@as(f64, x) - y));
        }
    }
    return .{ .equal = diff == 0, .max_abs = mx, .differing = diff };
}

fn greedy(e: *fwd.Engine, first_logits: []const f32, count: usize, out: *std.ArrayList(u32), hbuf: []f32, lg: []f32) !void {
    const gpa = e.gpu.gpa;
    var next: u32 = @intCast(ft.argmax(first_logits));
    var g: usize = 0;
    while (g < count) : (g += 1) {
        try out.append(gpa, next);
        if (g + 1 == count) break;
        var one = [_]u32{next};
        try e.forward(&one, hbuf[0..wide]);
        try e.logitsOf(hbuf[0..wide], 1, lg[0..vocab]);
        next = @intCast(ft.argmax(lg[0..vocab]));
    }
}

fn prefixCache(gpu: Gpu, pack: *const real.Pack, dir: []const u8, n_layers: usize, a: []const u32, b: []const u32, c: []const u32, gen: usize) !void {
    const gpa = gpu.gpa;
    var pbuf: [512]u8 = undefined;
    const ngram = try ngramPath(&pbuf, dir);
    const chunk: usize = 64;
    const total_max = a.len + @max(b.len, c.len);
    const cap = (total_max + gen + 8 + 3) / 4 * 4;
    std.debug.print("== prefix cache, {d} layers: |A| {d}, |B| {d}, |C| {d}, cap {d}\n", .{ n_layers, a.len, b.len, c.len, cap });
    var eng = try fwd.Engine.init(gpu, pack, .{ .first_layer = 0, .layers = n_layers, .max_rows = chunk, .cap = cap, .head = true, .ngram_path = if (n_layers > fwd.ple_layer) ngram else null });
    defer eng.deinit();
    const hbuf = try gpa.alloc(f32, chunk * wide);
    defer gpa.free(hbuf);
    const lg1 = try gpa.alloc(f32, chunk * vocab);
    defer gpa.free(lg1);
    const lg_s = try gpa.alloc(f32, c.len * vocab); // scratch run, logits at the C rows
    defer gpa.free(lg_s);
    const lg_r = try gpa.alloc(f32, c.len * vocab); // restored run
    defer gpa.free(lg_r);
    const lg_s2 = try gpa.alloc(f32, c.len * vocab); // second scratch run (determinism control)
    defer gpa.free(lg_s2);
    const dec = try gpa.alloc(f32, vocab);
    defer gpa.free(dec);

    var store = prefix.PrefixStore(snap.Snapshot).init(gpa, 1 << 30);
    defer store.deinit();

    // P1 = A + B: A, take the snapshot at the A boundary, then B
    var t0 = nowMs(gpu);
    try eng.reset();
    try feed(&eng, a, chunk, hbuf, null, null);
    const t_a = nowMs(gpu) - t0;
    t0 = nowMs(gpu);
    var s = try snap.capture(&eng);
    const t_cap = nowMs(gpu) - t0;
    const s_bytes = s.bytes();
    store.put(a, s, s_bytes) catch |err| {
        s.deinit(gpa);
        return err;
    };
    try feed(&eng, b, chunk, hbuf, null, null);
    std.debug.print("  P1 = A+B run; A prefill {d:.0} ms, snapshot capture {d:.0} ms\n", .{ t_a, t_cap });
    std.debug.print("  snapshot at |A| = {d} tokens: {d} bytes ({d:.1} MiB): GDN conv+rec {d}, KV {d}, index keys {d}, pooled {d}, PLE {d}\n", .{ a.len, s_bytes, @as(f64, @floatFromInt(s_bytes)) / 1048576.0, store.items.items[0].snap.conv.len + store.items.items[0].snap.rec.len, store.items.items[0].snap.kv.len, store.items.items[0].snap.ik.len, store.items.items[0].snap.pooled.len, (if (store.items.items[0].snap.tail) |t| t.len * 8 + 16 else 0) + (if (store.items.items[0].snap.rows) |r| r.tail.len * 4 + 16 else 0) });

    // P2 = A + C from scratch (A in one call, C in the same chunking the restored run uses)
    const p2 = try gpa.alloc(u32, a.len + c.len);
    defer gpa.free(p2);
    @memcpy(p2[0..a.len], a);
    @memcpy(p2[a.len..], c);
    try eng.reset();
    t0 = nowMs(gpu);
    try feed(&eng, a, chunk, hbuf, null, null);
    const t_a2 = nowMs(gpu) - t0;
    t0 = nowMs(gpu);
    try feed(&eng, c, chunk, hbuf, null, lg_s);
    const t_c = nowMs(gpu) - t0;
    var gen_s: std.ArrayList(u32) = .empty;
    defer gen_s.deinit(gpa);
    try greedy(&eng, lg_s[(c.len - 1) * vocab ..][0..vocab], gen, &gen_s, hbuf, dec);

    // determinism control: the same scratch run again
    try eng.reset();
    try feed(&eng, a, chunk, hbuf, null, null);
    try feed(&eng, c, chunk, hbuf, null, lg_s2);
    const ctrl = bitsEqual(lg_s, lg_s2);

    // P2 restored
    const hit = store.lookup(p2) orelse {
        try check.expect(false, "prefix store missed a stored prefix of P2", .{});
        return;
    };
    try check.expect(hit.len == a.len, "hit length {d} != |A| {d}", .{ hit.len, a.len });
    const cached_ctx = hit.len;
    t0 = nowMs(gpu);
    try snap.restore(&eng, hit.snap);
    const t_rest = nowMs(gpu) - t0;
    try check.expect(eng.pos == cached_ctx, "restored pos {d}", .{eng.pos});
    t0 = nowMs(gpu);
    try feed(&eng, p2[cached_ctx..], chunk, hbuf, null, lg_r);
    const t_c2 = nowMs(gpu) - t0;
    var gen_r: std.ArrayList(u32) = .empty;
    defer gen_r.deinit(gpa);
    try greedy(&eng, lg_r[(c.len - 1) * vocab ..][0..vocab], gen, &gen_r, hbuf, dec);

    const cmp = bitsEqual(lg_s, lg_r);
    std.debug.print("RESULT prefix: restored-vs-scratch logits over {d} C rows: {s} ({d} of {d} values differ, max_abs {e:.2}); scratch-vs-scratch control: {s} ({d} differ)\n", .{ c.len, if (cmp.equal) "BYTE-IDENTICAL" else "NOT identical", cmp.differing, c.len * vocab, cmp.max_abs, if (ctrl.equal) "byte-identical" else "not identical", ctrl.differing });
    std.debug.print("RESULT prefix: greedy {d} tokens scratch {any} restored {any}: {s}\n", .{ gen, gen_s.items, gen_r.items, if (std.mem.eql(u32, gen_s.items, gen_r.items)) "IDENTICAL" else "DIFFERENT" });
    std.debug.print("RESULT prefix cost: scratch P2 = A {d:.0} ms + C {d:.0} ms; restored = restore {d:.0} ms + C {d:.0} ms; computing A avoided: saved {d:.0} ms (A {d:.0} ms vs restore {d:.0} ms), {d} of {d} prompt tokens skipped\n", .{ t_a2, t_c, t_rest, t_c2, t_a2 - t_rest, t_a2, t_rest, cached_ctx, p2.len });

    // refusal rules
    try check.expect(store.lookup(a) == null, "a prompt equal to the stored prefix must not hit (resumeAt keeps its last token)", .{});
    try check.expect(store.lookup(c) == null, "unrelated tokens must not hit", .{});
    std.debug.print("  lookup(A alone) -> miss, lookup(C) -> miss; resumeAt({d}) = {d}\n", .{ a.len, cache_rules.resumeAt(@intCast(a.len)) });

    // negative controls: the same restore with one piece of state lost must change the logits, so the identity above is not vacuous
    const lg_x = try gpa.alloc(f32, c.len * vocab);
    defer gpa.free(lg_x);
    const losses = [_][]const u8{ "GDN recurrent state zeroed", "attention KV zeroed", "PLE conv tail zeroed", "index keys and pooled blocks zeroed" };
    var all_sensitive = true;
    for (losses, 0..) |what, which| {
        if (which == 2 and eng.ple == null and eng.ple_rows == null) continue;
        if (which == 3 and eng.opt.first_layer + eng.layers.len < 4) continue; // no attention layer loaded
        // dense attention reads no index keys: they only matter once a row has more than 2051 keys
        if (which == 3 and a.len + c.len <= 2051) continue;
        try snap.restore(&eng, hit.snap);
        switch (which) {
            0 => try eng.table.rec.fill8(0, null),
            1 => try eng.b.kv_all.fill8(0, null),
            2 => if (eng.ple) |*p| {
                @memset(p.tail, 0);
            } else {
                // the `rows` path: the id history and conv tail back to a fresh sequence's zeros
                var z = try eng.pleState();
                defer z.deinit(gpa);
                try eng.pleRestore(z);
            },
            else => {
                try eng.b.ik_all.fill8(0, null);
                try eng.b.pooled_all.fill8(0, null);
            },
        }
        try feed(&eng, p2[cached_ctx..], chunk, hbuf, null, lg_x);
        const d = bitsEqual(lg_s, lg_x);
        std.debug.print("  control ({s}): {d} of {d} logits differ, max_abs {d:.4} => {s}\n", .{ what, d.differing, c.len * vocab, d.max_abs, if (d.equal) "NOT SENSITIVE" else "sensitive" });
        if (d.equal) all_sensitive = false;
    }
    try check.expect(all_sensitive, "a negative control did not change the logits", .{});
    try check.expect(cmp.equal or (cmp.max_abs < 1e-3 and std.mem.eql(u32, gen_s.items, gen_r.items)), "restored P2 differs from scratch P2", .{});
    try check.expect(std.mem.eql(u32, gen_s.items, gen_r.items), "greedy continuation differs", .{});
    check.pass("prefix cache {d} layers: P2 = A+C from a restored A equals P2 from scratch ({s}), greedy {d} tokens identical", .{ n_layers, if (cmp.equal) "byte-identical logits" else "logits within tolerance", gen });
    _ = window;
}

// -- mode: needle -----------------------------------------------------------------------------------------------

fn top5Print(last: []const f32) void {
    var five: [5]usize = undefined;
    for (0..5) |k| {
        var best: usize = 0;
        var bv: f32 = -std.math.inf(f32);
        for (last, 0..) |v, i| {
            var skip = false;
            for (five[0..k]) |o| skip = skip or o == i;
            if (!skip and v > bv) {
                bv = v;
                best = i;
            }
        }
        five[k] = best;
    }
    std.debug.print("  top5:", .{});
    for (five) |i| std.debug.print(" {d}({d:.3})", .{ i, last[i] });
    std.debug.print("\n", .{});
}

/// One long shared document, then several suffixes: the first runs live, the document state is then restored for each suffix
/// (the first suffix again, to compare the live and restored continuations), greedy `gen` tokens each.
fn needleRun(gpu: Gpu, pack: *const real.Pack, dir: []const u8, n_layers: usize, gen: usize, doc: []const u32, suffixes: []const []const u32) !void {
    const gpa = gpu.gpa;
    var pbuf: [512]u8 = undefined;
    const ngram = try ngramPath(&pbuf, dir);
    const chunk: usize = 64;
    var max_suffix: usize = 0;
    for (suffixes) |s| max_suffix = @max(max_suffix, s.len);
    const cap = (doc.len + max_suffix + gen + 8 + 3) / 4 * 4;
    std.debug.print("== needle: document {d} tokens, {d} suffixes (max {d} tokens), {d} layers, cap {d}\n", .{ doc.len, suffixes.len, max_suffix, n_layers, cap });
    var eng = try fwd.Engine.init(gpu, pack, .{ .first_layer = 0, .layers = n_layers, .max_rows = chunk, .cap = cap, .head = true, .ngram_path = if (n_layers > fwd.ple_layer) ngram else null });
    defer eng.deinit();
    const hbuf = try gpa.alloc(f32, chunk * wide);
    defer gpa.free(hbuf);
    const lg = try gpa.alloc(f32, max_suffix * vocab);
    defer gpa.free(lg);
    const dec = try gpa.alloc(f32, vocab);
    defer gpa.free(dec);

    var t0 = nowMs(gpu);
    try eng.reset();
    var off: usize = 0;
    while (off < doc.len) {
        const m: usize = @min(chunk, doc.len - off);
        try eng.forward(doc[off..][0..m], hbuf[0 .. m * wide]);
        off += m;
        if ((off / chunk) % 8 == 0 or off == doc.len) std.debug.print("  document {d}/{d} in {d:.0} s\n", .{ off, doc.len, (nowMs(gpu) - t0) / 1000 });
    }
    const t_doc = nowMs(gpu) - t0;
    t0 = nowMs(gpu);
    var s = try snap.capture(&eng);
    const t_cap = nowMs(gpu) - t0;
    const s_bytes = s.bytes();
    var store = prefix.PrefixStore(snap.Snapshot).init(gpa, 1 << 30);
    defer store.deinit();
    store.put(doc, s, s_bytes) catch |err| {
        s.deinit(gpa);
        return err;
    };
    std.debug.print("  snapshot after the document: {d} bytes ({d:.1} MiB) captured in {d:.0} ms; the document took {d:.0} s\n", .{ s_bytes, @as(f64, @floatFromInt(s_bytes)) / 1048576.0, t_cap, t_doc / 1000 });

    var live_gen: std.ArrayList(u32) = .empty;
    defer live_gen.deinit(gpa);
    // runs: suffix 0 live, then restore + suffix 0 again, then restore + each remaining suffix
    var run_i: usize = 0;
    while (run_i < suffixes.len + 1) : (run_i += 1) {
        const si: usize = if (run_i == 0) 0 else run_i - 1;
        const suf = suffixes[si];
        if (run_i > 0) {
            const full = try gpa.alloc(u32, doc.len + suf.len);
            defer gpa.free(full);
            @memcpy(full[0..doc.len], doc);
            @memcpy(full[doc.len..], suf);
            const hit = store.lookup(full) orelse {
                try check.expect(false, "prefix store missed the document", .{});
                return;
            };
            t0 = nowMs(gpu);
            try snap.restore(&eng, hit.snap);
            std.debug.print("  restored the document ({d} tokens) in {d:.0} ms\n", .{ hit.len, nowMs(gpu) - t0 });
        }
        t0 = nowMs(gpu);
        try feed(&eng, suf, chunk, hbuf, null, lg[0 .. suf.len * vocab]);
        const last = lg[(suf.len - 1) * vocab ..][0..vocab];
        std.debug.print("SUFFIX {d} ({s}, {d} tokens, {d:.0} s)\n", .{ si, if (run_i == 0) "live" else "restored", suf.len, (nowMs(gpu) - t0) / 1000 });
        top5Print(last);
        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(gpa);
        try greedy(&eng, last, gen, &out, hbuf, dec);
        std.debug.print("SUFFIX {d} GEN ids=", .{si});
        for (out.items, 0..) |v, i| std.debug.print("{d}{s}", .{ v, if (i + 1 < out.items.len) "," else "\n" });
        if (run_i == 0) {
            try live_gen.appendSlice(gpa, out.items);
        } else if (si == 0) {
            const same = std.mem.eql(u32, live_gen.items, out.items);
            std.debug.print("RESULT needle: suffix 0 live vs restored continuation: {s}\n", .{if (same) "IDENTICAL" else "DIFFERENT"});
            try check.expect(same, "live and restored continuation differ", .{});
        }
    }
    check.pass("needle run: {d}-token document x {d} layers, {d} suffixes finished (outputs above, decode ids with tools/exl3_tok.py)", .{ doc.len, n_layers, suffixes.len });
}

// -- entry ---------------------------------------------------------------------------------------------------------

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 3) return error.MissingArgument;
    const gpa = gpu.gpa;
    const dir = rest[0];
    const mode = rest[1];
    var pack: real.Pack = .{ .files = undefined };
    defer pack.close(gpu.io);
    try pack.openAll(gpu, dir);
    if (std.mem.eql(u8, mode, "tf")) {
        const cfgs = try parseCsv(gpa, rest[2]);
        defer gpa.free(cfgs);
        const deep = try std.fmt.parseInt(usize, rest[3], 10);
        var prompts: std.ArrayList([]u32) = .empty;
        defer {
            for (prompts.items) |p| gpa.free(p);
            prompts.deinit(gpa);
        }
        for (rest[4..]) |arg| try prompts.append(gpa, try ft.parseIds(gpa, arg));
        try teacherForcing(gpu, &pack, dir, cfgs, deep, @ptrCast(prompts.items));
    } else if (std.mem.eql(u8, mode, "long")) {
        const n = try std.fmt.parseInt(usize, rest[2], 10);
        const ids = try ft.parseIds(gpa, rest[3]);
        defer gpa.free(ids);
        const with_oracle = std.mem.eql(u8, rest[4], "1");
        const gen = try std.fmt.parseInt(usize, rest[5], 10);
        const expect: []u32 = if (rest.len > 6 and !std.mem.eql(u8, rest[6], "-")) try ft.parseIds(gpa, rest[6]) else try gpa.alloc(u32, 0);
        defer gpa.free(expect);
        const chunk: usize = if (rest.len > 7) try std.fmt.parseInt(usize, rest[7], 10) else 64;
        try longContext(gpu, &pack, dir, n, ids, with_oracle, gen, expect, chunk);
    } else if (std.mem.eql(u8, mode, "prefix")) {
        const n = try std.fmt.parseInt(usize, rest[2], 10);
        const a = try ft.parseIds(gpa, rest[3]);
        defer gpa.free(a);
        const b = try ft.parseIds(gpa, rest[4]);
        defer gpa.free(b);
        const c = try ft.parseIds(gpa, rest[5]);
        defer gpa.free(c);
        const gen = try std.fmt.parseInt(usize, rest[6], 10);
        try prefixCache(gpu, &pack, dir, n, a, b, c, gen);
    } else if (std.mem.eql(u8, mode, "needle")) {
        const n = try std.fmt.parseInt(usize, rest[2], 10);
        const gen = try std.fmt.parseInt(usize, rest[3], 10);
        const doc = try ft.parseIds(gpa, rest[4]);
        defer gpa.free(doc);
        var suffixes: std.ArrayList([]u32) = .empty;
        defer {
            for (suffixes.items) |x| gpa.free(x);
            suffixes.deinit(gpa);
        }
        for (rest[5..]) |arg| try suffixes.append(gpa, try ft.parseIds(gpa, arg));
        try needleRun(gpu, &pack, dir, n, gen, doc, @ptrCast(suffixes.items));
    } else return error.UnknownMode;
}
