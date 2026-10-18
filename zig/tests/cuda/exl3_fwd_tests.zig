//! The assembled EXL3 forward (exl3_fwd.zig) against the f64 CPU oracle (exl3_fwd_cpu.zig).
//! Usage: tf-cuda-test exl3-forward <pack dir> layer [layers, default 0,3]   one decoder layer, real weights, prefill 8 + 4 decode rows
//!        tf-cuda-test exl3-forward <pack dir> tf <N layers> [ids]            teacher-forced logits after N layers vs the oracle
//!        tf-cuda-test exl3-forward <pack dir> gen <N layers> <count> [ids]   greedy continuation on the device (no oracle)
//! `ids` is a comma separated token id list (default: "The capital of France is Paris. It is known for the Eiffel Tower.").

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const real = @import("exl3_real_tests.zig");
const fwd = @import("exl3_fwd.zig");
const cpu = @import("exl3_fwd_cpu.zig");
const fple = @import("exl3_fwd_ple.zig");
const hc = @import("exl3_hc.zig");
const attn_cpu = @import("exl3_attn_cpu.zig");
const gdn_cpu = @import("exl3_gdn_cpu.zig");
const embed = @import("exl3_embed.zig");

const Gpu = check.Gpu;
const hidden = fwd.hidden;
const wide = fwd.wide;
const topk = fwd.topk;
const vocab = fwd.vocab;

const context_plus = 2;
const default_ids = [_]u32{ 760, 6511, 314, 9338, 369, 11751, 13, 1049, 369, 3750, 364, 279, 242476, 300, 21262, 13 };

pub fn parseIds(gpa: std.mem.Allocator, text: []const u8) ![]u32 {
    var list: std.ArrayList(u32) = .empty;
    errdefer list.deinit(gpa);
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |tok| {
        if (tok.len == 0) continue;
        try list.append(gpa, try std.fmt.parseInt(u32, tok, 10));
    }
    return list.toOwnedSlice(gpa);
}

pub fn nowMs(gpu: Gpu) f64 {
    return @as(f64, @floatFromInt(check.now(gpu.io))) / 1e6;
}

// -- oracle weights --------------------------------------------------------------------------------------------

pub fn qmat(gpa: std.mem.Allocator, pack: *const real.Pack, base: []const u8, threads: usize) !cpu.QMat {
    const m = try real.mat(pack, gpa, base);
    defer {
        gpa.free(m.trellis);
        gpa.free(m.suh);
        gpa.free(m.svh);
    }
    return cpu.QMat.init(gpa, m.halves, m.trellis, m.tiles_k, m.tiles_n, m.suh, m.svh, threads);
}

const ExpertCtx = struct {
    gpa: std.mem.Allocator,
    pack: *const real.Pack,
    layer: usize,

    fn load(ctx: *anyopaque, e: usize, out: *[3]cpu.QMat) anyerror!void {
        const self: *ExpertCtx = @ptrCast(@alignCast(ctx));
        const names = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
        var made: usize = 0;
        errdefer for (0..made) |i| out[i].deinit(self.gpa);
        while (made < 3) : (made += 1) {
            var nb: [200]u8 = undefined;
            out[made] = try qmat(self.gpa, self.pack, try std.fmt.bufPrint(&nb, "model.language_model.layers.{d}.mlp.experts.{d}.{s}", .{ self.layer, e, names[made] }), 1);
        }
    }
};

pub const OLayer = struct {
    gpa: std.mem.Allocator,
    index: usize,
    attn_hc: fwd.HcW,
    mlp_hc: fwd.HcW,
    gdn: ?cpu.GdnW = null,
    attn: ?cpu.AttnW = null,
    moe: cpu.MoeW,
    ctx: ExpertCtx,
    state: cpu.LayerState,

    pub fn create(gpa: std.mem.Allocator, pack: *const real.Pack, l: usize, offset: f32, cap: usize) !*OLayer {
        const o = try gpa.create(OLayer);
        errdefer gpa.destroy(o);
        o.gpa = gpa;
        o.index = l;
        o.gdn = null;
        o.attn = null;
        o.attn_hc = try fwd.HcW.load(gpa, pack, "model.language_model.layers.{d}.attn_hyper_connection", .{l}, true);
        o.mlp_hc = try fwd.HcW.load(gpa, pack, "model.language_model.layers.{d}.mlp_hyper_connection", .{l}, true);
        var nb: [8][200]u8 = undefined;
        const mstem = try std.fmt.bufPrint(&nb[7], "model.language_model.layers.{d}.mlp.shared_expert", .{l});
        o.ctx = .{ .gpa = gpa, .pack = pack, .layer = l };
        o.moe = .{
            .gate16 = try fwd.f16Of(gpa, try fwd.tensor(pack, "model.language_model.layers.{d}.mlp.gate.weight", .{l})),
            .sgate16 = try fwd.f16Of(gpa, try fwd.tensor(pack, "model.language_model.layers.{d}.mlp.shared_expert_gate.weight", .{l})),
            .shared = .{
                try qmat(gpa, pack, try std.fmt.bufPrint(&nb[0], "{s}.gate_proj", .{mstem}), 48),
                try qmat(gpa, pack, try std.fmt.bufPrint(&nb[1], "{s}.up_proj", .{mstem}), 48),
                try qmat(gpa, pack, try std.fmt.bufPrint(&nb[2], "{s}.down_proj", .{mstem}), 48),
            },
            .source = .{ .ctx = &o.ctx, .load = ExpertCtx.load },
        };
        if (fwd.Layer.isAttn(l)) {
            const stem = try std.fmt.bufPrint(&nb[7], "model.language_model.layers.{d}.self_attn", .{l});
            var s: [5][200]u8 = undefined;
            var a: cpu.AttnW = undefined;
            a.q = try qmat(gpa, pack, try std.fmt.bufPrint(&s[0], "{s}.q_proj", .{stem}), 48);
            a.k = try qmat(gpa, pack, try std.fmt.bufPrint(&s[1], "{s}.k_proj", .{stem}), 48);
            a.v = try qmat(gpa, pack, try std.fmt.bufPrint(&s[2], "{s}.v_proj", .{stem}), 48);
            a.o = try qmat(gpa, pack, try std.fmt.bufPrint(&s[3], "{s}.o_proj", .{stem}), 48);
            a.idx = try qmat(gpa, pack, try std.fmt.bufPrint(&s[4], "{s}.indexer.index_qk_proj", .{stem}), 48);
            const names = [4][]const u8{ "q_norm.weight", "k_norm.weight", "indexer.q_layernorm.weight", "indexer.k_layernorm.weight" };
            var sc: [4][]f32 = undefined;
            for (names, 0..) |nm, i| {
                const t = try fwd.tensor(pack, "model.language_model.layers.{d}.self_attn.{s}", .{ l, nm });
                sc[i] = try fwd.f32Of(gpa, t);
                for (sc[i]) |*v| v.* += offset;
            }
            a.norms = .{ .q = sc[0], .k = sc[1], .iq = sc[2], .ik = sc[3], .inv = attn_cpu.invFreq() };
            o.attn = a;
            o.state = .{ .attn = try attn_cpu.Cache.init(gpa, cap) };
        } else {
            const stem = try std.fmt.bufPrint(&nb[7], "model.language_model.layers.{d}.linear_attn", .{l});
            var s: [3][200]u8 = undefined;
            var g: cpu.GdnW = undefined;
            g.qkv = try qmat(gpa, pack, try std.fmt.bufPrint(&s[0], "{s}.in_proj_qkv", .{stem}), 48);
            g.z = try qmat(gpa, pack, try std.fmt.bufPrint(&s[1], "{s}.in_proj_z", .{stem}), 48);
            g.out = try qmat(gpa, pack, try std.fmt.bufPrint(&s[2], "{s}.out_proj", .{stem}), 48);
            g.a_w = try fwd.f64Of(gpa, try fwd.tensor(pack, "model.language_model.layers.{d}.linear_attn.in_proj_a.weight", .{l}));
            g.b_w = try fwd.f64Of(gpa, try fwd.tensor(pack, "model.language_model.layers.{d}.linear_attn.in_proj_b.weight", .{l}));
            g.params = .{
                .conv_w = try fwd.f64Of(gpa, try fwd.tensor(pack, "model.language_model.layers.{d}.linear_attn.conv1d.weight", .{l})),
                .a_log = try fwd.f64Of(gpa, try fwd.tensor(pack, "model.language_model.layers.{d}.linear_attn.A_log", .{l})),
                .dt_bias = try fwd.f64Of(gpa, try fwd.tensor(pack, "model.language_model.layers.{d}.linear_attn.dt_bias", .{l})),
                .norm_w = try fwd.f64Of(gpa, try fwd.tensor(pack, "model.language_model.layers.{d}.linear_attn.norm.weight", .{l})),
            };
            o.gdn = g;
            o.state = .{ .gdn = try gdn_cpu.State.init(gpa, gdn_cpu.qwen) };
        }
        return o;
    }

    pub fn destroy(o: *OLayer) void {
        const gpa = o.gpa;
        o.attn_hc.free(gpa);
        o.mlp_hc.free(gpa);
        gpa.free(o.moe.gate16);
        gpa.free(o.moe.sgate16);
        o.moe.deinit(gpa);
        if (o.gdn) |*g| g.deinit(gpa);
        if (o.attn) |*a| a.deinit(gpa);
        switch (o.state) {
            .gdn => |*s| s.deinit(gpa),
            .attn => |*c| c.deinit(),
        }
        gpa.destroy(o);
    }

    pub fn weights(o: *OLayer) cpu.LayerW {
        return .{
            .attn_hc = o.attn_hc.weights(),
            .mlp_hc = o.mlp_hc.weights(),
            .mixer = if (o.gdn != null) .{ .gdn = &o.gdn.? } else .{ .attn = &o.attn.? },
            .moe = &o.moe,
        };
    }
};

/// Embedding rows expanded x4, f64, [rows][10240].
pub fn embedStreams(gpa: std.mem.Allocator, table: []const u8, tokens: []const u32) ![]f64 {
    const h = try gpa.alloc(f64, tokens.len * wide);
    for (tokens, 0..) |tk, t| {
        const row = try embed.gather(table, tk);
        for (0..hidden) |i| {
            const v: f64 = fwd.bf16(std.mem.readInt(u16, row[i * 2 ..][0..2], .little));
            for (0..4) |s| h[t * wide + s * hidden + i] = v;
        }
    }
    return h;
}

pub const Err = struct {
    max_abs: f64,
    peak: f64,
    rel: f64,
};

pub fn errOf(got: anytype, want: []const f64) Err {
    var max: f64 = 0;
    var peak: f64 = 0;
    for (want, 0..) |w, i| {
        const g: f64 = @floatCast(got[i]);
        max = @max(max, @abs(g - w));
        peak = @max(peak, @abs(w));
    }
    return .{ .max_abs = max, .peak = peak, .rel = if (peak > 0) max / peak else 0 };
}

fn toF64(gpa: std.mem.Allocator, xs: anytype) ![]f64 {
    const out = try gpa.alloc(f64, xs.len);
    for (out, xs) |*d, s| d.* = @floatCast(s);
    return out;
}

// -- mode: layer -------------------------------------------------------------------------------------------------

fn layerCheck(gpu: Gpu, pack: *const real.Pack, l: usize, tokens: []const u32) !void {
    const gpa = gpu.gpa;
    const prefill_n = 8;
    const cap = 64;
    std.debug.print("== layer {d} ({s}): prefill {d} rows, then {d} single-row decode steps\n", .{ l, if (fwd.Layer.isAttn(l)) "full attention + MoE" else "GDN + MoE", prefill_n, tokens.len - prefill_n });
    var t0 = nowMs(gpu);
    var eng = try fwd.Engine.init(gpu, pack, .{ .first_layer = l, .layers = 1, .max_rows = 8, .cap = cap, .head = false });
    defer eng.deinit();
    std.debug.print("  device layer loaded in {d:.0} ms\n", .{nowMs(gpu) - t0});
    t0 = nowMs(gpu);
    const ol = try OLayer.create(gpa, pack, l, eng.offset, cap);
    defer ol.destroy();
    std.debug.print("  oracle layer decoded in {d:.0} ms (offset {d})\n", .{ nowMs(gpu) - t0, eng.offset });

    const R = 8;
    const tr: fwd.Trace = .{
        .attn_branch = try gpa.alloc(f64, R * hidden),
        .mlp_branch = try gpa.alloc(f64, R * hidden),
        .picks = try gpa.alloc(u16, R * topk),
        .h_mid = try gpa.alloc(f64, R * wide),
    };
    defer {
        gpa.free(tr.attn_branch);
        gpa.free(tr.mlp_branch);
        gpa.free(tr.picks);
        gpa.free(tr.h_mid);
    }
    const otr: fwd.Trace = .{
        .attn_branch = try gpa.alloc(f64, R * hidden),
        .mlp_branch = try gpa.alloc(f64, R * hidden),
        .picks = try gpa.alloc(u16, R * topk),
        .h_mid = try gpa.alloc(f64, R * wide),
    };
    defer {
        gpa.free(otr.attn_branch);
        gpa.free(otr.mlp_branch);
        gpa.free(otr.picks);
        gpa.free(otr.h_mid);
    }
    eng.trace = tr;

    const calls = [_][2]usize{ .{ 0, prefill_n }, .{ 8, 1 }, .{ 9, 1 }, .{ 10, 1 }, .{ 11, 1 } };
    var worst_attn: f64 = 0;
    var worst_mid: f64 = 0;
    var worst_mlp: f64 = 0;
    var worst_h: f64 = 0;
    var rows_same: usize = 0;
    var rows_total: usize = 0;
    var gpu_ms: f64 = 0;
    for (calls) |c| {
        const toks = tokens[c[0] .. c[0] + c[1]];
        const n = c[1];
        const hg = try gpa.alloc(f32, n * wide);
        defer gpa.free(hg);
        t0 = nowMs(gpu);
        try eng.forward(toks, hg);
        gpu_ms += nowMs(gpu) - t0;

        const ho = try embedStreams(gpa, eng.embed_table, toks);
        defer gpa.free(ho);
        try cpu.layerRows(gpa, ol.weights(), &ol.state, c[0], eng.offset, ho, n, otr);

        const e_attn = errOf(tr.attn_branch[0 .. n * hidden], otr.attn_branch[0 .. n * hidden]);
        const e_mid = errOf(tr.h_mid[0 .. n * wide], otr.h_mid[0 .. n * wide]);
        var same: usize = 0;
        for (0..n) |t| {
            var a: [topk]u16 = undefined;
            var b: [topk]u16 = undefined;
            @memcpy(&a, tr.picks[t * topk ..][0..topk]);
            @memcpy(&b, otr.picks[t * topk ..][0..topk]);
            std.mem.sort(u16, &a, {}, std.sort.asc(u16));
            std.mem.sort(u16, &b, {}, std.sort.asc(u16));
            same += @intFromBool(std.mem.eql(u16, &a, &b));
        }
        rows_same += same;
        rows_total += n;
        const e_mlp = errOf(tr.mlp_branch[0 .. n * hidden], otr.mlp_branch[0 .. n * hidden]);
        const e_h = errOf(hg, ho);
        std.debug.print("  rows {d}..{d}: mixer branch rel {d:.5} (peak {d:.3}); streams after mixer rel {d:.5}; MoE picks same set {d}/{d}, MoE branch rel {d:.5} (peak {d:.3}); layer output streams rel {d:.5} (peak {d:.2}, max_abs {e:.2})\n", .{
            c[0], c[0] + n - 1, e_attn.rel, e_attn.peak, e_mid.rel, same, n, e_mlp.rel, e_mlp.peak, e_h.rel, e_h.peak, e_h.max_abs,
        });
        worst_attn = @max(worst_attn, e_attn.rel);
        worst_mid = @max(worst_mid, e_mid.rel);
        if (same == n) worst_mlp = @max(worst_mlp, e_mlp.rel);
        if (same == n) worst_h = @max(worst_h, e_h.rel);
    }
    std.debug.print("  device time for the 5 calls: {d:.0} ms\n", .{gpu_ms});
    // fp16 rounds the inputs and outputs of every projection; GDN is gated at 4e-3 of peak and attention at 3% in their own stages
    const attn_tol: f64 = if (fwd.Layer.isAttn(l)) 0.03 else 0.01;
    try check.expect(worst_attn <= attn_tol, "layer {d}: mixer branch differs from the oracle ({d:.4} > {d})", .{ l, worst_attn, attn_tol });
    try check.expect(worst_mid <= 0.01, "layer {d}: streams after the mixer differ ({d:.4})", .{ l, worst_mid });
    try check.expect(worst_mlp <= 0.02, "layer {d}: MoE branch differs from the oracle ({d:.4})", .{ l, worst_mlp });
    try check.expect(worst_h <= 0.01, "layer {d}: layer output streams differ ({d:.4})", .{ l, worst_h });
    check.pass("layer {d} ({s}): full layer (HC pre, norm, mixer, HC post, MoE, HC post) on the device matches the f64 oracle: mixer branch rel {d:.5}, streams rel {d:.5}, MoE branch rel {d:.5}, output rel {d:.5}; MoE picks equal on {d}/{d} rows", .{
        l, if (fwd.Layer.isAttn(l)) "attention" else "GDN", worst_attn, worst_mid, worst_mlp, worst_h, rows_same, rows_total,
    });
}

// -- mode: tf ----------------------------------------------------------------------------------------------------

pub fn argmax(xs: []const f32) usize {
    var best: usize = 0;
    for (xs, 0..) |v, i| if (v > xs[best]) {
        best = i;
    };
    return best;
}

fn top5(xs: []const f32, out: *[5]usize) void {
    var taken: [5]bool = @splat(false);
    _ = &taken;
    for (0..5) |k| {
        var best: usize = 0;
        var bv: f32 = -std.math.inf(f32);
        for (xs, 0..) |v, i| {
            var skip = false;
            for (out[0..k]) |o| skip = skip or o == i;
            if (!skip and v > bv) {
                bv = v;
                best = i;
            }
        }
        out[k] = best;
    }
}

fn teacherForced(gpu: Gpu, pack: *const real.Pack, dir: []const u8, n_layers: usize, tokens: []const u32) !void {
    const gpa = gpu.gpa;
    const rows = tokens.len;
    var path_buf: [512]u8 = undefined;
    const ngram_path = try std.fmt.bufPrint(&path_buf, "{s}/ngram_embedding.safetensors", .{dir});
    std.debug.print("== teacher-forced, {d} layers, {d} tokens\n", .{ n_layers, rows });

    // device
    var t0 = nowMs(gpu);
    var eng = try fwd.Engine.init(gpu, pack, .{ .first_layer = 0, .layers = n_layers, .max_rows = rows, .cap = 128, .head = true, .ngram_path = if (n_layers > fwd.ple_layer) ngram_path else null });
    defer eng.deinit();
    std.debug.print("  device model loaded in {d:.0} ms (PLE {s})\n", .{ nowMs(gpu) - t0, if (eng.ple != null) "on" else "not reached by these layers" });
    t0 = nowMs(gpu);
    const hg = try gpa.alloc(f32, rows * wide);
    defer gpa.free(hg);
    try eng.forward(tokens, hg);
    const t_fwd = nowMs(gpu) - t0;
    const lg = try gpa.alloc(f32, rows * vocab);
    defer gpa.free(lg);
    try eng.logitsOf(hg, rows, lg);
    std.debug.print("  device forward {d:.0} ms (prefill {d} rows, includes on-demand expert uploads), + head {d:.0} ms\n", .{ t_fwd, rows, nowMs(gpu) - t0 - t_fwd });

    // oracle
    t0 = nowMs(gpu);
    const h = try embedStreams(gpa, eng.embed_table, tokens);
    defer gpa.free(h);
    var ple_o: ?fple.Ple = null;
    var ng_file: ?@import("safetensors.zig").File = null;
    defer if (ng_file) |*f| f.close(gpu.io);
    if (eng.ple != null) {
        // independent instance: the same host path, own state
        ng_file = try @import("safetensors.zig").File.open(gpa, gpu.io, ngram_path);
        ple_o = try fwd.loadPle(gpu, pack, &ng_file.?);
    }
    defer if (ple_o) |*p| p.deinit();
    for (0..n_layers) |l| {
        const lt = nowMs(gpu);
        if (ple_o != null and l == fwd.ple_layer) {
            const out = try gpa.alloc(f64, rows * wide);
            defer gpa.free(out);
            try ple_o.?.rows(tokens, h, out);
            for (h, out) |*d, s| d.* += s;
        }
        const ol = try OLayer.create(gpa, pack, l, eng.offset, 128);
        defer ol.destroy();
        try cpu.layerRows(gpa, ol.weights(), &ol.state, 0, eng.offset, h, rows, null);
        if (l < 4 or l % 8 == 7) std.debug.print("  oracle layer {d} done in {d:.0} ms\n", .{ l, nowMs(gpu) - lt });
    }
    // collapse + head
    const mixed = try gpa.alloc(f64, rows * hidden);
    defer gpa.free(mixed);
    try cpu.hcPreRows(gpa, eng.mixer.weights(), eng.offset, h, rows, mixed, null);
    var head = try qmat(gpa, pack, "lm_head", 48);
    defer head.deinit(gpa);
    const lo = try gpa.alloc(f64, rows * vocab);
    defer gpa.free(lo);
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
    var hj: HeadJob = .{ .m = &head, .x = mixed, .out = lo, .gpa = gpa };
    cpu.parallelFor(rows, 48, &hj, HeadJob.row);
    std.debug.print("  oracle forward + head {d:.0} ms\n", .{nowMs(gpu) - t0});

    // compare
    var agree: usize = 0;
    var ties: usize = 0;
    var lmax: f64 = 0;
    var lrms_num: f64 = 0;
    var lrms_den: f64 = 0;
    var worst_gap: f64 = 0;
    for (0..rows) |t| {
        const g = lg[t * vocab ..][0..vocab];
        const o = lo[t * vocab ..][0..vocab];
        for (g, o) |a, b| {
            const d = @as(f64, a) - b;
            lmax = @max(lmax, @abs(d));
            lrms_num += d * d;
            lrms_den += b * b;
        }
    }
    std.debug.print("  pos  input   gpu_top1 (logit)      oracle_top1 (logit)   match  oracle gap of gpu's pick  gpu_top5\n", .{});
    for (0..rows) |t| {
        const g = lg[t * vocab ..][0..vocab];
        const o = lo[t * vocab ..][0..vocab];
        const gi = argmax(g);
        var oi: usize = 0;
        for (o, 0..) |v, i| if (v > o[oi]) {
            oi = i;
        };
        var five: [5]usize = undefined;
        top5(g, &five);
        const gap = o[oi] - o[gi];
        const same = gi == oi;
        // a flip is a near-tie when the oracle itself separates the two candidates by no more than twice the run's worst logit error
        const tie = !same and gap <= 2 * lmax;
        agree += @intFromBool(same);
        ties += @intFromBool(tie);
        if (!same) worst_gap = @max(worst_gap, gap);
        std.debug.print("  {d:>3}  {d:>6}  {d:>6} ({d:>8.3})   {d:>6} ({d:>8.3})   {s}   {d:.4}  {d},{d},{d},{d},{d}\n", .{ t, tokens[t], gi, g[gi], oi, o[oi], if (same) "yes" else if (tie) "tie" else "NO ", gap, five[0], five[1], five[2], five[3], five[4] });
    }
    std.debug.print("  top-1: exact {d}/{d}, near-tie flips {d} (oracle gap <= 2 x max logit error, worst gap {d:.4}); logits vs oracle: max_abs {d:.4}, rms rel {e:.2}\n", .{ agree, rows, ties, worst_gap, lmax, @sqrt(lrms_num / lrms_den) });
    try check.expect(agree + ties == rows, "{d} layers: top-1 differs from the oracle beyond a near-tie at {d} of {d} positions", .{ n_layers, rows - agree - ties, rows });
    check.pass("{d} layers, {d} teacher-forced positions: device top-1 equals the f64 oracle's at {d} positions, {d} near-tie flips (logits max_abs {d:.4}, rms rel {e:.2})", .{ n_layers, rows, agree, ties, lmax, @sqrt(lrms_num / lrms_den) });
}

// -- mode: gen ---------------------------------------------------------------------------------------------------

fn generate(gpu: Gpu, pack: *const real.Pack, dir: []const u8, n_layers: usize, count: usize, prompt: []const u32, use_ple: bool) !void {
    const gpa = gpu.gpa;
    var path_buf: [512]u8 = undefined;
    const ngram_path = try std.fmt.bufPrint(&path_buf, "{s}/ngram_embedding.safetensors", .{dir});
    var t0 = nowMs(gpu);
    var eng = try fwd.Engine.init(gpu, pack, .{ .first_layer = 0, .layers = n_layers, .max_rows = @max(prompt.len, 1), .cap = 128, .head = true, .ngram_path = if (n_layers > fwd.ple_layer and use_ple) ngram_path else null });
    defer eng.deinit();
    std.debug.print("== greedy generation, {d} layers: model loaded in {d:.0} ms (PLE {s})\n", .{ n_layers, nowMs(gpu) - t0, if (eng.ple != null) "on" else "off" });
    const lg = try gpa.alloc(f32, prompt.len * vocab);
    defer gpa.free(lg);
    const hg = try gpa.alloc(f32, prompt.len * wide);
    defer gpa.free(hg);
    t0 = nowMs(gpu);
    try eng.forward(prompt, hg);
    try eng.logitsOf(hg, prompt.len, lg);
    std.debug.print("  prefill {d} tokens: {d:.0} ms\n", .{ prompt.len, nowMs(gpu) - t0 });
    var next: u32 = @intCast(argmax(lg[(prompt.len - 1) * vocab ..][0..vocab]));
    std.debug.print("GEN prompt={any}\nGEN ids=", .{prompt});
    var step: usize = 0;
    while (step < count) : (step += 1) {
        std.debug.print("{d}{s}", .{ next, if (step + 1 < count) "," else "\n" });
        if (step + 1 == count) break;
        var one = [_]u32{next};
        t0 = nowMs(gpu);
        try eng.forward(&one, hg[0..wide]);
        try eng.logitsOf(hg[0..wide], 1, lg[0..vocab]);
        next = @intCast(argmax(lg[0..vocab]));
    }
    std.debug.print("  last decode step {d:.0} ms (host-heavy: HC on the CPU, experts uploaded per layer; not a throughput number)\n", .{nowMs(gpu) - t0});
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    const dir = rest[0];
    const mode = rest[1];
    var pack: real.Pack = .{ .files = undefined };
    defer pack.close(gpu.io);
    try pack.openAll(gpu, dir);

    if (std.mem.eql(u8, mode, "layer")) {
        const spec: []const u8 = if (rest.len > 2) rest[2] else "0,3";
        const toks: []const u32 = default_ids[0..12];
        var it = std.mem.splitScalar(u8, spec, ',');
        while (it.next()) |tok| try layerCheck(gpu, &pack, try std.fmt.parseInt(usize, tok, 10), toks);
    } else if (std.mem.eql(u8, mode, "tf")) {
        const n = try std.fmt.parseInt(usize, if (rest.len > 2) rest[2] else "2", 10);
        const ids = if (rest.len > 3) try parseIds(gpu.gpa, rest[3]) else try gpu.gpa.dupe(u32, &default_ids);
        defer gpu.gpa.free(ids);
        try teacherForced(gpu, &pack, dir, n, ids);
    } else if (std.mem.eql(u8, mode, "ple-ids")) {
        // prints the 16 n-gram row ids per token for an independent numpy cross-check (tools/exl3_ple_ids.py)
        var path_buf: [512]u8 = undefined;
        const ngram_path = try std.fmt.bufPrint(&path_buf, "{s}/ngram_embedding.safetensors", .{dir});
        var ng = try @import("safetensors.zig").File.open(gpu.gpa, gpu.io, ngram_path);
        defer ng.close(gpu.io);
        var p = try fwd.loadPle(gpu, &pack, &ng);
        defer p.deinit();
        var seq: [context_plus + default_ids.len]i64 = undefined;
        seq[0] = fwd.eos_token;
        seq[1] = fwd.eos_token;
        for (default_ids, 0..) |t, i| seq[2 + i] = t;
        for (0..default_ids.len) |i| {
            var ids: [16]u64 = undefined;
            p.hash.idsAt(&seq, 2 + i, &ids);
            std.debug.print("IDS {d}:", .{default_ids[i]});
            for (ids) |v| std.debug.print(" {d}", .{v});
            std.debug.print("\n", .{});
        }
    } else if (std.mem.eql(u8, mode, "gen")) {
        const n = try std.fmt.parseInt(usize, if (rest.len > 2) rest[2] else "48", 10);
        const count = try std.fmt.parseInt(usize, if (rest.len > 3) rest[3] else "16", 10);
        const ids = if (rest.len > 4) try parseIds(gpu.gpa, rest[4]) else try gpu.gpa.dupe(u32, default_ids[0..5]);
        defer gpu.gpa.free(ids);
        try generate(gpu, &pack, dir, n, count, ids, !(rest.len > 5 and std.mem.eql(u8, rest[5], "nople")));
    } else return error.UnknownMode;
}
