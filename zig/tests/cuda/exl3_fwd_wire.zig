//! Engine wiring checks for exl3_fwd.zig: the expert pool, the device hyper-connection / router path, and the PLE row path, each
//! against the existing engine (host hyper-connection, on-demand expert lines, S5 PLE) on the same 16-token prompt and the same
//! 16-token continuation.
//!
//! `wire <cfg> [<cfg> ...]`: the first cfg is the baseline. It prefills the 16-token prompt, then decodes greedily (15 decode steps
//! feed continuation tokens 0..14, so 16 tokens are generated). Every later cfg prefills the same prompt and is TEACHER-FORCED
//! through the baseline's continuation, so each of the 31 positions (16 prompt rows + 15 decode rows) has a top-1 and a logit
//! vector in both runs. A cfg is a '+' list: `base` | `hc` (device hyper-connection + router dots) | `pN` (N pool lines per
//! layer) | `s6b` (PLE from the S6b row path) | `ssd` (S6b rows by pread behind the 4096-row hot cache).
//! Top-1 must be equal at all 31 positions (then the greedy continuations are equal too: they are the argmaxes).

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const real = @import("exl3_real_tests.zig");
const st = @import("safetensors.zig");
const fwd = @import("exl3_fwd.zig");
const fple = @import("exl3_fwd_ple.zig");
const plerows = @import("exl3_fwd_plerows.zig");
const ids_mod = @import("exl3_ple_ids.zig");

const Gpu = check.Gpu;
const hidden = fwd.hidden;
const wide = fwd.wide;
const vocab = fwd.vocab;

const prompt_ids = [_]u32{ 760, 6511, 314, 9338, 369, 11751, 13, 1049, 369, 3750, 364, 279, 242476, 300, 21262, 13 };
const decode_steps = 15;
const positions = prompt_ids.len + decode_steps;

fn nowMs(gpu: Gpu) f64 {
    return @as(f64, @floatFromInt(check.now(gpu.io))) / 1e6;
}

fn mib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1048576.0;
}

const Cfg = struct {
    name: []const u8,
    hc: bool = false,
    pool: usize = 0,
    ple: fwd.PleImpl = .f64_reference,
    ssd: bool = false,
    prof: bool = false,

    fn parse(text: []const u8) !Cfg {
        var c: Cfg = .{ .name = text };
        var it = std.mem.splitScalar(u8, text, '+');
        while (it.next()) |tok| {
            if (std.mem.eql(u8, tok, "base")) continue;
            if (std.mem.eql(u8, tok, "hc")) {
                c.hc = true;
            } else if (std.mem.eql(u8, tok, "s6b")) {
                c.ple = .rows;
            } else if (std.mem.eql(u8, tok, "prof")) {
                c.prof = true;
            } else if (std.mem.eql(u8, tok, "ssd")) {
                c.ple = .rows;
                c.ssd = true;
            } else if (tok.len > 1 and tok[0] == 'p') {
                c.pool = try std.fmt.parseInt(usize, tok[1..], 10);
            } else return error.BadConfig;
        }
        return c;
    }
};

const Result = struct {
    top1: [positions]u32,
    logits: []f32, // [positions][vocab]
    cont: [decode_steps + 1]u32, // generated tokens (baseline: its own greedy; others: the teacher-forced ones)
    prefill_ms: f64,
    step_ms: [decode_steps]f64,
    stats: fwd.Stats,
    prof: [6]u64,
    picks: []u16,
    gaps: []f32,
    pick_n: usize,
    layer_rows: usize,
    init_ms: f64,
    pool_mib: f64,
    free_after_mib: f64,
};

fn argmax(xs: []const f32) u32 {
    var best: usize = 0;
    for (xs, 0..) |v, i| if (v > xs[best]) {
        best = i;
    };
    return @intCast(best);
}

fn runCfg(gpu: Gpu, pack: *const real.Pack, dir: []const u8, cfg: Cfg, forced: ?[]const u32) !Result {
    const gpa = gpu.gpa;
    var path_buf: [512]u8 = undefined;
    const ngram_path = try std.fmt.bufPrint(&path_buf, "{s}/ngram_embedding.safetensors", .{dir});
    var t0 = nowMs(gpu);
    var eng = try fwd.Engine.init(gpu, pack, .{
        .max_rows = prompt_ids.len,
        .cap = 128,
        .ngram_path = ngram_path,
        .pool_lines = cfg.pool,
        .device_hc = cfg.hc,
        .ple_impl = cfg.ple,
        .ple_mode = if (cfg.ssd) .ssd else .page_cache,
        .profile = cfg.prof,
    });
    defer eng.deinit();
    var r: Result = undefined;
    r.init_ms = nowMs(gpu) - t0;
    r.pool_mib = mib(eng.pool_bytes);
    r.free_after_mib = mib((try gpu.ctx.memInfo()).free);
    r.logits = try gpa.alloc(f32, positions * vocab);
    errdefer gpa.free(r.logits);
    const h = try gpa.alloc(f32, prompt_ids.len * wide);
    defer gpa.free(h);

    // 16 prefill rows + 15 decode rows, 48 layers, 10 picks
    const pick_cap = positions * fwd.layers_total * fwd.topk;
    r.picks = try gpa.alloc(u16, pick_cap);
    errdefer gpa.free(r.picks);
    eng.pick_buf = r.picks;
    r.gaps = try gpa.alloc(f32, positions * fwd.layers_total);
    errdefer gpa.free(r.gaps);
    eng.gap_buf = r.gaps;
    try eng.reset();
    t0 = nowMs(gpu);
    try eng.forward(&prompt_ids, h);
    try eng.logitsOf(h, prompt_ids.len, r.logits[0 .. prompt_ids.len * vocab]);
    r.prefill_ms = nowMs(gpu) - t0;
    for (0..prompt_ids.len) |t| r.top1[t] = argmax(r.logits[t * vocab ..][0..vocab]);
    r.cont[0] = if (forced) |f| f[0] else r.top1[prompt_ids.len - 1];
    for (0..decode_steps) |i| {
        var one = [_]u32{r.cont[i]};
        t0 = nowMs(gpu);
        try eng.forward(&one, h[0..wide]);
        const row = r.logits[(prompt_ids.len + i) * vocab ..][0..vocab];
        try eng.logitsOf(h[0..wide], 1, row);
        r.step_ms[i] = nowMs(gpu) - t0;
        r.top1[prompt_ids.len + i] = argmax(row);
        r.cont[i + 1] = if (forced) |f| f[i + 1] else r.top1[prompt_ids.len + i];
    }
    r.stats = eng.stats;
    r.prof = eng.prof;
    r.pick_n = eng.pick_n;
    r.layer_rows = fwd.layers_total;
    return r;
}

fn mean(xs: []const f64) f64 {
    var s: f64 = 0;
    for (xs) |x| s += x;
    return s / @as(f64, @floatFromInt(xs.len));
}

fn minOf(xs: []const f64) f64 {
    var m = xs[0];
    for (xs) |x| m = @min(m, x);
    return m;
}

fn describe(c: Cfg, r: Result) void {
    std.debug.print("  [{s}] init {d:.0} ms; pool {d:.0} MiB; device free after init {d:.0} MiB; prefill(16 rows) {d:.0} ms; decode step mean {d:.0} ms, min {d:.0} ms (forward + logits + 248k-wide copy, a test measurement)\n", .{
        c.name, r.init_ms, r.pool_mib, r.free_after_mib, r.prefill_ms, mean(&r.step_ms), minOf(&r.step_ms),
    });
    if (c.prof) {
        const names = [_][]const u8{ "hc+mixer", "post_attn", "hc+gate", "route_wait", "moe", "post_moe" };
        std.debug.print("  [{s}] profile, total over prefill + 15 decode steps (ms):", .{c.name});
        for (names, r.prof) |n, v| std.debug.print(" {s} {d:.0}", .{ n, @as(f64, @floatFromInt(v)) / 1e6 });
        std.debug.print("\n", .{});
    }
    if (r.stats.rows != 0) {
        const total = r.stats.hits + r.stats.misses;
        std.debug.print("  [{s}] pool: {d} routed picks, {d} hits, {d} misses staged by H2D ({d:.1}% hit), staging fill {d:.0} ms total\n", .{
            c.name, total, r.stats.hits, r.stats.misses, 100.0 * @as(f64, @floatFromInt(r.stats.hits)) / @as(f64, @floatFromInt(total)), @as(f64, @floatFromInt(r.stats.fill_ns)) / 1e6,
        });
    }
}

pub fn run(gpu: Gpu, pack: *const real.Pack, dir: []const u8, args: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    if (args.len < 1) return error.MissingArgument;
    const base_cfg = try Cfg.parse(args[0]);
    std.debug.print("== wire: baseline [{s}], {d} prompt tokens + {d} decode steps = {d} positions\n", .{ base_cfg.name, prompt_ids.len, decode_steps, positions });
    const base = try runCfg(gpu, pack, dir, base_cfg, null);
    defer gpa.free(base.logits);
    defer gpa.free(base.picks);
    defer gpa.free(base.gaps);
    describe(base_cfg, base);
    std.debug.print("  baseline greedy continuation (16 tokens): ", .{});
    for (base.cont, 0..) |t, i| std.debug.print("{d}{s}", .{ t, if (i + 1 < base.cont.len) "," else "\n" });

    var failed = false;
    for (args[1..]) |a| {
        const cfg = try Cfg.parse(a);
        const r = runCfg(gpu, pack, dir, cfg, &base.cont) catch |err| {
            std.debug.print("FAIL [{s}] {s}\n", .{ cfg.name, @errorName(err) });
            failed = true;
            continue;
        };
        defer gpa.free(r.logits);
        defer gpa.free(r.picks);
        defer gpa.free(r.gaps);
        describe(cfg, r);
        // router picks, row by row in call / layer / row order
        {
            const n = @min(base.pick_n, r.pick_n);
            var diff_rows: usize = 0;
            var first: ?usize = null;
            for (0..n) |i| {
                if (!std.mem.eql(u16, base.picks[i * fwd.topk ..][0..fwd.topk], r.picks[i * fwd.topk ..][0..fwd.topk])) {
                    diff_rows += 1;
                    if (first == null) first = i;
                }
            }
            std.debug.print("  [{s}] router picks: {d} routed rows compared, {d} rows pick a different expert set", .{ cfg.name, n, diff_rows });
            if (first) |f| {
                std.debug.print("; router gap(10th pick - best unpicked) there: base {e:.2}, this {e:.2}", .{ base.gaps[f], r.gaps[f] });
                // calls: prefill (16 rows layer-major: 48 layers x 16 rows), then one row x 48 layers a decode step
                const pre = fwd.layers_total * prompt_ids.len;
                if (f < pre) std.debug.print("; first at prefill layer {d} row {d}", .{ f / prompt_ids.len, f % prompt_ids.len }) else std.debug.print("; first at decode step {d} layer {d}", .{ (f - pre) / fwd.layers_total, (f - pre) % fwd.layers_total });
            }
            std.debug.print("\n", .{});
        }
        var agree: usize = 0;
        var lmax: f64 = 0;
        var worst_pos: usize = 0;
        var num: f64 = 0;
        var den: f64 = 0;
        var worst_gap: f64 = 0;
        var worst_gap_pos: usize = 0;
        for (0..positions) |p| {
            const a_row = base.logits[p * vocab ..][0..vocab];
            const b_row = r.logits[p * vocab ..][0..vocab];
            var pos_max: f64 = 0;
            for (a_row, b_row) |x, y| {
                const d = @as(f64, y) - x;
                pos_max = @max(pos_max, @abs(d));
                num += d * d;
                den += @as(f64, x) * x;
            }
            if (pos_max > lmax) {
                lmax = pos_max;
                worst_pos = p;
            }
            if (base.top1[p] == r.top1[p]) {
                agree += 1;
            } else {
                // how far the baseline itself separates the two picks
                const gap = @as(f64, a_row[base.top1[p]]) - a_row[r.top1[p]];
                if (gap > worst_gap) {
                    worst_gap = gap;
                    worst_gap_pos = p;
                }
                std.debug.print("  [{s}] position {d}: top-1 {d} vs baseline {d} (baseline gap {d:.4})\n", .{ cfg.name, p, r.top1[p], base.top1[p], gap });
            }
        }
        // own greedy continuation: the argmaxes of the decode rows, which equal the baseline's when top-1 agrees at 15..30
        var greedy_equal = true;
        for (0..decode_steps) |i| {
            if (r.top1[prompt_ids.len - 1 + i] != base.cont[i]) greedy_equal = false;
        }
        const rel = @sqrt(num / den);
        std.debug.print("  [{s}] vs [{s}]: top-1 equal at {d}/{d} positions (greedy continuation {s}); logits max |diff| {d:.5} at position {d}, rms rel {e:.2}\n", .{
            cfg.name, base_cfg.name, agree, positions, if (greedy_equal) "identical" else "DIFFERS", lmax, worst_pos, rel,
        });
        if (agree == positions) {
            check.pass("[{s}] equals [{s}]: top-1 at all {d} positions (16-token prompt, 15 decode rows, greedy continuation identical); logits max |diff| {d:.5}, rms rel {e:.2}", .{ cfg.name, base_cfg.name, positions, lmax, rel });
        } else {
            std.debug.print("FAIL [{s}]: top-1 differs at {d} positions (worst baseline gap {d:.4} at {d})\n", .{ cfg.name, positions - agree, worst_gap, worst_gap_pos });
            failed = true;
        }
    }
    if (failed) return error.TestFailed;
}

// -- every routed expert has the shape of expert 0 ---------------------------------------------------------------

pub fn halvesScan(gpu: Gpu, pack: *const real.Pack) !void {
    _ = gpu;
    const projs = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
    var bad: usize = 0;
    var line_bytes: usize = 0;
    var seen_halves: [3][17]bool = @splat(@splat(false));
    for (0..fwd.layers_total) |l| {
        var ref: [3][3]usize = undefined;
        for (projs, 0..) |p, i| {
            const t = try fwd.tensor(pack, "model.language_model.layers.{d}.mlp.experts.0.{s}.trellis", .{ l, p });
            ref[i] = .{ t.dim(0), t.dim(1), t.dim(2) };
            seen_halves[i][t.dim(2) / 8] = true;
        }
        for (1..fwd.experts) |e| {
            for (projs, 0..) |p, i| {
                const t = try fwd.tensor(pack, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.trellis", .{ l, e, p });
                if (t.dim(0) != ref[i][0] or t.dim(1) != ref[i][1] or t.dim(2) != ref[i][2]) bad += 1;
            }
        }
        const lb = (ref[0][0] * ref[0][1] * ref[0][2] + ref[1][0] * ref[1][1] * ref[1][2] + ref[2][0] * ref[2][1] * ref[2][2]) * 2 + (hidden * 2 + fwd.inter * 2) * 2 + (fwd.inter * 2 + hidden * 2);
        line_bytes = @max(line_bytes, lb);
        if (l % 8 == 0 or l == fwd.layers_total - 1) std.debug.print("  layer {d}: halves gate/up/down = {d}/{d}/{d}, line {d} bytes\n", .{ l, ref[0][2] / 8, ref[1][2] / 8, ref[2][2] / 8, lb });
    }
    try check.expect(bad == 0, "{d} routed experts differ in shape from expert 0 of their layer", .{bad});
    check.pass("all 48 x 512 routed experts have the trellis shape of expert 0 of their layer (one pool layout per layer); largest line {d} bytes ({d:.2} MiB)", .{ line_bytes, mib(line_bytes) });
}

// -- PLE: S5 f64 reference vs the S6b row path ----------------------------------------------------------------------

pub fn pleCheck(gpu: Gpu, pack: *const real.Pack, dir: []const u8) !void {
    const gpa = gpu.gpa;
    var path_buf: [512]u8 = undefined;
    const ngram_path = try std.fmt.bufPrint(&path_buf, "{s}/ngram_embedding.safetensors", .{dir});
    const tokens = prompt_ids[0..];
    const rows = tokens.len;

    // The real layer-1 input: the streams after decoder layer 0 (device, host hyper-connection, no PLE reached).
    var eng = try fwd.Engine.init(gpu, pack, .{ .first_layer = 0, .layers = 1, .max_rows = rows, .cap = 128, .head = false });
    defer eng.deinit();
    const h = try gpa.alloc(f32, rows * wide);
    defer gpa.free(h);
    try eng.forward(tokens, h);
    var peak_h: f32 = 0;
    for (h) |v| peak_h = @max(peak_h, @abs(v));
    std.debug.print("== PLE check on the real layer-1 input (streams after layer 0, {d} rows, peak |h| {d:.3})\n", .{ rows, peak_h });

    var ng = try st.File.open(gpa, gpu.io, ngram_path);
    defer ng.close(gpu.io);
    // S5: f64 reference
    var s5 = try fwd.loadPleOffset(gpu, pack, &ng, eng.offset);
    defer s5.deinit();
    const h64 = try gpa.alloc(f64, rows * wide);
    defer gpa.free(h64);
    for (h64, h) |*d, s| d.* = s;
    const out5 = try gpa.alloc(f64, rows * wide);
    defer gpa.free(out5);
    try s5.rows(tokens, h64, out5);
    // S6b: row path, both reader modes
    const w = try fwd.loadPleWeights(gpa, pack, eng.offset);
    defer w.free(gpa);
    const modes = [_]plerows.Mode{ .page_cache, .ssd };
    var worst_all: f64 = 0;
    var s6_first: ?[]f32 = null;
    defer if (s6_first) |b| gpa.free(b);
    for (modes) |mode| {
        const r6 = try plerows.Rows.init(gpa, &ng, mode, w.weights());
        defer r6.deinit();
        const out6 = try gpa.alloc(f32, rows * wide);
        defer gpa.free(out6);
        // ids: S5 hash vs S6b derived layout on the same history (eos, eos)
        var seq: [2 + prompt_ids.len]i64 = undefined;
        seq[0] = fwd.eos_token;
        seq[1] = fwd.eos_token;
        for (tokens, 0..) |t, i| seq[2 + i] = t;
        const rid = try gpa.alloc([ids_mod.heads]u64, rows);
        defer gpa.free(rid);
        ids_mod.ids(r6.layout, r6.history, tokens, rid);
        var ids_equal = true;
        for (0..rows) |i| {
            var a: [16]u64 = undefined;
            s5.hash.idsAt(&seq, 2 + i, &a);
            if (!std.mem.eql(u64, &a, &rid[i])) ids_equal = false;
        }
        try r6.rows(tokens, h, out6);
        var max_abs: f64 = 0;
        var peak: f64 = 0;
        var num: f64 = 0;
        var den: f64 = 0;
        for (out5, out6) |a, b| {
            const d = @as(f64, b) - a;
            max_abs = @max(max_abs, @abs(d));
            peak = @max(peak, @abs(a));
            num += d * d;
            den += a * a;
        }
        const hs = r6.hotStats();
        std.debug.print("  S6b [{s}] vs S5: ids {s} on {d} tokens x 16 heads; PLE add-back max |diff| {d:.5}, peak |S5| {d:.4} ({d:.2}% of peak), rms rel {e:.2}{s}\n", .{
            @tagName(mode), if (ids_equal) "equal" else "DIFFER", rows, max_abs, peak, 100.0 * max_abs / peak, @sqrt(num / den),
            if (mode == .ssd) "; hot cache lookups/misses printed below" else "",
        });
        if (mode == .ssd) std.debug.print("  S6b [ssd] hot cache (4096 rows): {d} lookups, {d} misses\n", .{ hs.lookups, hs.misses });
        try check.expect(ids_equal, "S5 and S6b n-gram ids differ", .{});
        worst_all = @max(worst_all, max_abs / peak);
        if (s6_first == null) {
            s6_first = try gpa.dupe(f32, out6);
        } else {
            // page-cache and ssd rows are byte-identical, so the add-backs must be bit-equal
            try check.expect(std.mem.eql(u8, std.mem.sliceAsBytes(s6_first.?), std.mem.sliceAsBytes(out6)), "S6b page-cache and ssd add-backs differ", .{});
        }
    }
    // the S5 add-back is f64 with no intermediate rounding; S6b rounds to bf16 where the engine does (8 bits: 2^-9 each, a chain of ~8)
    try check.expect(worst_all <= 0.03, "S5 and S6b PLE add-backs disagree beyond bf16 rounding ({d:.4} of peak)", .{worst_all});
    check.pass("PLE: S5 (f64) and S6b (bf16-rounded rows path) agree on the real layer-1 input: ids equal, add-back max diff {d:.2}% of peak (<= 3%), page-cache and ssd readers bit-equal", .{100.0 * worst_all});
}
