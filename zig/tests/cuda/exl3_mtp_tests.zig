//! The MTP head (exl3_mtp.zig) on the real pack.
//! Usage: tf-cuda-test exl3-mtp <pack dir> oracle            head step on the device (real engine streams in) vs the f64 oracle, 2 calls
//!        tf-cuda-test exl3-mtp <pack dir> spec [count] [ids ...]   serial vs single-draft speculative greedy, committed streams compared
//! `ids` are comma separated token-id lists (default: the two sample prompts of the brief, pre-tokenised with the pack's tokenizer.json).

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const real = @import("exl3_real_tests.zig");
const fwd = @import("exl3_fwd.zig");
const cpu = @import("exl3_fwd_cpu.zig");
const hc = @import("exl3_hc.zig");
const attn_cpu = @import("exl3_attn_cpu.zig");
const embed = @import("exl3_embed.zig");
const spec = @import("exl3_mtp_spec.zig");
const mtp = @import("exl3_mtp.zig");

const Gpu = check.Gpu;
const hidden = fwd.hidden;
const wide = fwd.wide;
const vocab = fwd.vocab;

// "The capital of France is" / "Once upon a time, in a small village by the sea" (tokenizers 0.22.2 on the pack's tokenizer.json)
const prompt_france = [_]u32{ 760, 6511, 314, 9338, 369 };
const prompt_village = [_]u32{ 12162, 5028, 264, 854, 11, 303, 264, 2526, 13721, 539, 279, 9117 };
const oracle_ids = [_]u32{ 760, 6511, 314, 9338, 369, 11751, 13, 1049, 369 };

/// Engine config from the mode suffix: `spec+hc+p64` (hc = device hyper-connection, required by the head; pN = per-layer expert pool lines).
const Cfg = struct {
    hc: bool = true,
    pool: usize = 0,

    fn parse(text: []const u8) !Cfg {
        var c: Cfg = .{};
        var it = std.mem.splitScalar(u8, text, '+');
        _ = it.next();
        while (it.next()) |tok| {
            if (std.mem.eql(u8, tok, "hc")) {
                c.hc = true;
            } else if (tok.len > 1 and tok[0] == 'p') {
                c.pool = try std.fmt.parseInt(usize, tok[1..], 10);
            } else return error.BadConfig;
        }
        return c;
    }
};

fn nowMs(gpu: Gpu) f64 {
    return @as(f64, @floatFromInt(check.now(gpu.io))) / 1e6;
}

fn parseIds(gpa: std.mem.Allocator, text: []const u8) ![]u32 {
    var list: std.ArrayList(u32) = .empty;
    errdefer list.deinit(gpa);
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |tok| {
        if (tok.len == 0) continue;
        try list.append(gpa, try std.fmt.parseInt(u32, tok, 10));
    }
    return list.toOwnedSlice(gpa);
}

// -- oracle ------------------------------------------------------------------------------------------------------

fn qmat(gpa: std.mem.Allocator, pack: *const real.Pack, base: []const u8, threads: usize) !cpu.QMat {
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

    fn load(ctx: *anyopaque, e: usize, out: *[3]cpu.QMat) anyerror!void {
        const self: *ExpertCtx = @ptrCast(@alignCast(ctx));
        const names = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
        var made: usize = 0;
        errdefer for (0..made) |i| out[i].deinit(self.gpa);
        while (made < 3) : (made += 1) {
            var nb: [200]u8 = undefined;
            out[made] = try qmat(self.gpa, self.pack, try std.fmt.bufPrint(&nb, mtp.layer_stem ++ ".mlp.experts.{d}.{s}", .{ e, names[made] }), 1);
        }
    }
};

const Oracle = struct {
    gpa: std.mem.Allocator,
    fc_e: cpu.QMat,
    fc_h: cpu.QMat,
    norm_e: []f32,
    norm_h: []f32,
    attn_hc: fwd.HcW,
    mlp_hc: fwd.HcW,
    attn: cpu.AttnW,
    moe: cpu.MoeW,
    ctx: ExpertCtx,
    state: cpu.LayerState,
    mixer: hc.Weights,
    offset: f64,

    fn init(gpa: std.mem.Allocator, pack: *const real.Pack, m: *const mtp.Mtp, offset: f32, cap: usize) !*Oracle {
        const o = try gpa.create(Oracle);
        errdefer gpa.destroy(o);
        o.gpa = gpa;
        o.offset = offset;
        o.fc_e = try qmat(gpa, pack, "mtp.fc_embedding", 48);
        o.fc_h = try qmat(gpa, pack, "mtp.fc_hidden", 48);
        o.norm_e = try gpa.dupe(f32, m.norm_e);
        o.norm_h = try gpa.dupe(f32, m.norm_h);
        o.attn_hc = try fwd.HcW.load(gpa, pack, mtp.layer_stem ++ ".attn_hyper_connection", .{}, true);
        o.mlp_hc = try fwd.HcW.load(gpa, pack, mtp.layer_stem ++ ".mlp_hyper_connection", .{}, true);
        o.ctx = .{ .gpa = gpa, .pack = pack };
        const ms = mtp.layer_stem ++ ".mlp.shared_expert";
        o.moe = .{
            .gate16 = try fwd.f16Of(gpa, try fwd.tensor(pack, mtp.layer_stem ++ ".mlp.gate.weight", .{})),
            .sgate16 = try fwd.f16Of(gpa, try fwd.tensor(pack, mtp.layer_stem ++ ".mlp.shared_expert_gate.weight", .{})),
            .shared = .{
                try qmat(gpa, pack, ms ++ ".gate_proj", 48),
                try qmat(gpa, pack, ms ++ ".up_proj", 48),
                try qmat(gpa, pack, ms ++ ".down_proj", 48),
            },
            .source = .{ .ctx = &o.ctx, .load = ExpertCtx.load },
        };
        const stem = mtp.layer_stem ++ ".self_attn";
        var a: cpu.AttnW = undefined;
        a.q = try qmat(gpa, pack, stem ++ ".q_proj", 48);
        a.k = try qmat(gpa, pack, stem ++ ".k_proj", 48);
        a.v = try qmat(gpa, pack, stem ++ ".v_proj", 48);
        a.o = try qmat(gpa, pack, stem ++ ".o_proj", 48);
        a.idx = try qmat(gpa, pack, stem ++ ".indexer.index_qk_proj", 48);
        const names = [4][]const u8{ "q_norm.weight", "k_norm.weight", "indexer.q_layernorm.weight", "indexer.k_layernorm.weight" };
        var sc: [4][]f32 = undefined;
        inline for (names, 0..) |nm, i| {
            sc[i] = try fwd.f32Of(gpa, try fwd.tensor(pack, stem ++ "." ++ nm, .{}));
            for (sc[i]) |*v| v.* += offset;
        }
        a.norms = .{ .q = sc[0], .k = sc[1], .iq = sc[2], .ik = sc[3], .inv = attn_cpu.invFreq() };
        o.attn = a;
        o.state = .{ .attn = try attn_cpu.Cache.init(gpa, cap) };
        o.mixer = m.mixer.weights();
        return o;
    }

    fn deinit(o: *Oracle) void {
        const gpa = o.gpa;
        o.fc_e.deinit(gpa);
        o.fc_h.deinit(gpa);
        gpa.free(o.norm_e);
        gpa.free(o.norm_h);
        o.attn_hc.free(gpa);
        o.mlp_hc.free(gpa);
        gpa.free(o.moe.gate16);
        gpa.free(o.moe.sgate16);
        o.moe.deinit(gpa);
        o.attn.deinit(gpa);
        o.state.attn.deinit();
        gpa.destroy(o);
    }

    /// The head on `rows` main-stream rows `h` (f32 from the engine) and next tokens at head position `pos0`.
    /// Returns the layer output streams in `x` [rows][wide] and the collapsed rows in `mixed` [rows][hidden].
    fn step(o: *Oracle, table: []const u8, h: []const f32, tokens: []const u32, pos0: usize, x: []f64, mixed: []f64) !void {
        const gpa = o.gpa;
        const rows = tokens.len;
        const e64 = try gpa.alloc(f64, hidden);
        defer gpa.free(e64);
        const en = try gpa.alloc(f64, hidden);
        defer gpa.free(en);
        const eo = try gpa.alloc(f64, hidden);
        defer gpa.free(eo);
        const hn = try gpa.alloc(f64, wide);
        defer gpa.free(hn);
        const h64 = try gpa.alloc(f64, wide);
        defer gpa.free(h64);
        const hs = try gpa.alloc(f64, hidden);
        defer gpa.free(hs);
        const tmp = try gpa.alloc(f64, hidden);
        defer gpa.free(tmp);
        for (0..rows) |t| {
            const row = try embed.gather(table, tokens[t]);
            for (e64, 0..) |*d, i| d.* = fwd.bf16(std.mem.readInt(u16, row[i * 2 ..][0..2], .little));
            spec.centredNormF64(e64, o.norm_e, en);
            o.fc_e.apply(en, eo, tmp);
            for (h64, h[t * wide ..][0..wide]) |*d, v| d.* = v;
            spec.centredNormF64(h64, o.norm_h, hn);
            for (0..4) |q| {
                o.fc_h.apply(hn[q * hidden ..][0..hidden], hs, tmp);
                for (0..hidden) |d| x[t * wide + q * hidden + d] = eo[d] + hs[d];
            }
        }
        const lw: cpu.LayerW = .{ .attn_hc = o.attn_hc.weights(), .mlp_hc = o.mlp_hc.weights(), .mixer = .{ .attn = &o.attn }, .moe = &o.moe };
        try cpu.layerRows(gpa, lw, &o.state, pos0, o.offset, x, rows, null);
        try cpu.hcPreRows(gpa, o.mixer, o.offset, x, rows, mixed, null);
    }
};

const Err = struct { max_abs: f64, peak: f64, rel: f64 };

fn errOf(got: anytype, want: []const f64) Err {
    var max: f64 = 0;
    var peak: f64 = 0;
    for (want, 0..) |w, i| {
        const g: f64 = @floatCast(got[i]);
        max = @max(max, @abs(g - w));
        peak = @max(peak, @abs(w));
    }
    return .{ .max_abs = max, .peak = peak, .rel = if (peak > 0) max / peak else 0 };
}

fn oracleCheck(gpu: Gpu, pack: *const real.Pack, dir: []const u8, cfg: Cfg) !void {
    const gpa = gpu.gpa;
    var path_buf: [512]u8 = undefined;
    const ngram_path = try std.fmt.bufPrint(&path_buf, "{s}/ngram_embedding.safetensors", .{dir});
    const P = 8;
    const prompt = oracle_ids[0..P];
    const next = oracle_ids[1 .. P + 1]; // teacher-forced: the head pairs row t with the real token t + 1
    std.debug.print("== MTP head vs f64 oracle: {d} prompt rows from the 48-layer engine, head calls of 6 + 2 rows\n", .{P});
    var t0 = nowMs(gpu);
    var eng = try fwd.Engine.init(gpu, pack, .{ .first_layer = 0, .layers = fwd.layers_total, .max_rows = 16, .cap = 128, .head = true, .ngram_path = ngram_path, .extra_kv = 1, .device_hc = cfg.hc, .pool_lines = cfg.pool });
    defer eng.deinit();
    var head = try mtp.Mtp.load(&eng, dir);
    defer head.deinit(&eng);
    std.debug.print("  engine + head loaded in {d:.0} ms\n", .{nowMs(gpu) - t0});
    var msum: f64 = 0;
    for (head.norm_e) |v| msum += v - eng.offset;
    var hsum: f64 = 0;
    for (head.norm_h) |v| hsum += v - eng.offset;
    var mixer_sum: f64 = 0;
    for (head.mixer.norm) |v| mixer_sum += v;
    std.debug.print("  stored-weight means: pre_fc_norm_embedding {d:.4}, pre_fc_norm_hidden {d:.4}, mtp mixer hc_norm {d:.4} (engine offset {d})\n", .{
        msum / hidden, hsum / wide, mixer_sum / wide, eng.offset,
    });

    const hmain = try gpa.alloc(f32, P * wide);
    defer gpa.free(hmain);
    t0 = nowMs(gpu);
    try eng.forward(prompt, hmain);
    std.debug.print("  engine prefill {d} rows: {d:.0} ms\n", .{ P, nowMs(gpu) - t0 });

    t0 = nowMs(gpu);
    const o = try Oracle.init(gpa, pack, &head, eng.offset, 128);
    defer o.deinit();
    std.debug.print("  oracle head weights decoded in {d:.0} ms\n", .{nowMs(gpu) - t0});

    const calls = [_][2]usize{ .{ 0, 6 }, .{ 6, 2 } };
    var worst_x: f64 = 0;
    var worst_m: f64 = 0;
    var gpu_ms: f64 = 0;
    for (calls) |c| {
        const n = c[1];
        t0 = nowMs(gpu);
        try head.step(&eng, hmain[c[0] * wide ..][0 .. n * wide], next[c[0]..][0..n]);
        gpu_ms += nowMs(gpu) - t0;
        const x = try gpa.alloc(f64, n * wide);
        defer gpa.free(x);
        const mixed = try gpa.alloc(f64, n * hidden);
        defer gpa.free(mixed);
        try o.step(eng.embed_table, hmain[c[0] * wide ..][0 .. n * wide], next[c[0]..][0..n], c[0], x, mixed);
        const ex = errOf(head.last_streams[0 .. n * wide], x);
        const em = errOf(head.last_mixed[0 .. n * hidden], mixed);
        std.debug.print("  head rows {d}..{d}: layer output streams rel {d:.5} (peak {d:.2}, max_abs {e:.2}); collapsed rel {d:.5} (peak {d:.2}, max_abs {e:.2})\n", .{
            c[0], c[0] + n - 1, ex.rel, ex.peak, ex.max_abs, em.rel, em.peak, em.max_abs,
        });
        worst_x = @max(worst_x, ex.rel);
        worst_m = @max(worst_m, em.rel);
        if (c[0] + n == P) {
            // the draft: GPU head logits of the last row vs the oracle's (same collapsed row, f64 head matvec)
            var hq = try qmat(gpa, pack, "lm_head", 48);
            defer hq.deinit(gpa);
            const lo = try gpa.alloc(f64, vocab);
            defer gpa.free(lo);
            const tmp = try gpa.alloc(f64, hidden);
            defer gpa.free(tmp);
            hq.apply(mixed[(n - 1) * hidden ..][0..hidden], lo, tmp);
            const lg = try gpa.alloc(f32, vocab);
            defer gpa.free(lg);
            try head.draftLogits(&eng, lg);
            var oi: usize = 0;
            for (lo, 0..) |v, i| if (v > lo[oi]) {
                oi = i;
            };
            const gi = mtp.argmax(lg);
            var lmax: f64 = 0;
            for (lg, lo) |a, b| lmax = @max(lmax, @abs(@as(f64, a) - b));
            const gap = lo[oi] - lo[gi];
            std.debug.print("  draft after the last row: device top-1 {d} (logit {d:.3}), oracle top-1 {d} (logit {d:.3}), oracle gap of the device pick {d:.4}, logits max_abs {d:.4}\n", .{ gi, lg[gi], oi, lo[oi], gap, lmax });
            try check.expect(gi == oi or gap <= 2 * lmax, "draft top-1 differs from the oracle beyond a near-tie ({d} vs {d}, gap {d:.4})", .{ gi, oi, gap });
        }
    }
    std.debug.print("  device head time for the 2 calls: {d:.0} ms\n", .{gpu_ms});
    // fp16 rounds the inputs and outputs of every projection (as the layer test): 1% of peak for the layer, 3% attention-bearing
    try check.expect(worst_x <= 0.03, "MTP layer output streams differ from the oracle ({d:.4})", .{worst_x});
    try check.expect(worst_m <= 0.03, "MTP collapsed rows differ from the oracle ({d:.4})", .{worst_m});
    check.pass("MTP head (fc combine, sparse-attention layer with its own KV over two calls, MoE, collapse, lm_head draft) on the device matches the f64 oracle: layer streams rel {d:.5}, collapsed rel {d:.5}", .{ worst_x, worst_m });
}

// -- speculative vs serial ---------------------------------------------------------------------------------------

fn specRun(gpu: Gpu, eng: *fwd.Engine, head: *mtp.Mtp, snap: *mtp.Snapshot, prompt: []const u32, count: usize, label: []const u8, corrupt_every: usize) !void {
    const gpa = gpu.gpa;
    const h = try gpa.alloc(f32, eng.opt.max_rows * wide);
    defer gpa.free(h);
    const lg = try gpa.alloc(f32, eng.opt.max_rows * vocab);
    defer gpa.free(lg);
    var model: mtp.Model = .{ .e = eng, .mtp = head, .snap = snap, .h = h, .logits = lg, .io = gpu.io, .corrupt_every = corrupt_every };
    const want = try gpa.alloc(u32, count);
    defer gpa.free(want);
    const got = try gpa.alloc(u32, count);
    defer gpa.free(got);

    std.debug.print("== {s}: prompt {any}, {d} tokens\n", .{ label, prompt, count });
    try eng.reset();
    head.reset();
    var t0 = nowMs(gpu);
    try spec.serial(&model, prompt, want);
    const t_serial = nowMs(gpu) - t0;
    std.debug.print("  serial ids={any}\n  serial: {d:.0} ms total (forward {d:.0} ms + head {d:.0} ms)\n", .{ want, t_serial, model.ms_forward, model.ms_head });

    try eng.reset();
    head.reset();
    model.ms_forward = 0;
    model.ms_head = 0;
    model.ms_mtp = 0;
    model.drafts = 0;
    var stats: spec.Stats = .{};
    t0 = nowMs(gpu);
    try spec.speculative(&model, prompt, got, &stats);
    const t_spec = nowMs(gpu) - t0;
    std.debug.print("  spec   ids={any}\n  spec:   {d:.0} ms total (forward {d:.0} ms + head {d:.0} ms + mtp {d:.0} ms)\n", .{ got, t_spec, model.ms_forward, model.ms_head, model.ms_mtp });
    std.debug.print("  draft rounds {d}, accepted {d}, acceptance rate {d:.3}, single-row reruns {d}, decode rows run {d} for {d} tokens\n", .{ stats.rounds, stats.accepted, stats.rate(), stats.reruns, stats.decode_rows, count - 1 });
    var first_diff: ?usize = null;
    for (want, got, 0..) |a, b, i| if (a != b and first_diff == null) {
        first_diff = i;
    };
    if (first_diff) |i| {
        std.debug.print("  FIRST DIFFERENCE at token {d}: serial {d} vs spec {d}\n", .{ i, want[i], got[i] });
    }
    try check.expect(first_diff == null, "{s}: speculative stream differs from the serial stream", .{label});
    check.pass("{s}: committed stream of {d} tokens is IDENTICAL, draft acceptance {d}/{d} rounds ({d:.1}%), {d} rejected drafts rolled back (GDN state, position, PLE history + tail)", .{
        label, count, stats.accepted, stats.rounds, 100.0 * stats.rate(), stats.reruns,
    });
}

fn specCheck(gpu: Gpu, pack: *const real.Pack, dir: []const u8, cfg: Cfg, count: usize, prompts: []const []const u32, corrupt_every: usize) !void {
    var path_buf: [512]u8 = undefined;
    const ngram_path = try std.fmt.bufPrint(&path_buf, "{s}/ngram_embedding.safetensors", .{dir});
    var t0 = nowMs(gpu);
    var eng = try fwd.Engine.init(gpu, pack, .{ .first_layer = 0, .layers = fwd.layers_total, .max_rows = 16, .cap = 128, .head = true, .ngram_path = ngram_path, .extra_kv = 1, .device_hc = cfg.hc, .pool_lines = cfg.pool });
    defer eng.deinit();
    var head = try mtp.Mtp.load(&eng, dir);
    defer head.deinit(&eng);
    var snap = try mtp.Snapshot.init(&eng);
    defer snap.deinit(&eng);
    std.debug.print("  engine + MTP head + snapshot loaded in {d:.0} ms (PLE {s})\n", .{ nowMs(gpu) - t0, if (eng.hasPle()) "on" else "off" });
    t0 = nowMs(gpu);
    for (prompts, 0..) |p, i| {
        var lb: [32]u8 = undefined;
        try specRun(gpu, &eng, &head, &snap, p, count, try std.fmt.bufPrint(&lb, "prompt {d}", .{i}), corrupt_every);
    }
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    const dir = rest[0];
    const cfg = try Cfg.parse(rest[1]);
    var mode_it = std.mem.splitScalar(u8, rest[1], '+');
    const mode = mode_it.next().?;
    var pack: real.Pack = .{ .files = undefined };
    defer pack.close(gpu.io);
    try pack.openAll(gpu, dir);
    if (std.mem.eql(u8, mode, "oracle")) {
        try oracleCheck(gpu, &pack, dir, cfg);
    } else if (std.mem.eql(u8, mode, "spec") or std.mem.eql(u8, mode, "spec-reject")) {
        // spec-reject: every 2nd draft is corrupted, so half the rounds exercise the rollback
        const corrupt: usize = if (std.mem.eql(u8, mode, "spec-reject")) 2 else 0;
        const count = try std.fmt.parseInt(usize, if (rest.len > 2) rest[2] else "24", 10);
        if (rest.len > 3) {
            var lists: std.ArrayList([]u32) = .empty;
            defer {
                for (lists.items) |l| gpu.gpa.free(l);
                lists.deinit(gpu.gpa);
            }
            for (rest[3..]) |t| try lists.append(gpu.gpa, try parseIds(gpu.gpa, t));
            const view: []const []const u32 = @ptrCast(lists.items);
            try specCheck(gpu, &pack, dir, cfg, count, view, corrupt);
        } else {
            const both = [_][]const u32{ &prompt_france, &prompt_village };
            try specCheck(gpu, &pack, dir, cfg, count, &both, corrupt);
        }
    } else return error.UnknownMode;
}
