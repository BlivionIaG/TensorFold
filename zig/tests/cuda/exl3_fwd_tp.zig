//! The assembled EXL3 forward at tp=2 on the two RTX 5060 Ti (no peer access). Same host flow as exl3_fwd.zig's Engine (embedding,
//! host hyper-connection, layer-major rows), but every layer's heavy work is split over two ranks, each with its own CUDA context:
//!   GDN        value heads 24r.., key heads 8r.. (exl3_tp.GdnTp), out_proj K-split
//!   attention  12 q heads + 1 KV head per rank (exl3_attn.Attn with q_heads 12 / kv_heads 1), the indexer recomputed per rank,
//!              o_proj K-split
//!   MoE        expert parallel: rank 0 owns routed ids 0..255 and the shared expert, rank 1 ids 256..511; the router runs on the host
//!              once per row (both ranks see the same picks and weights)
//!   head       lm_head replicated on rank 0 only
//! The ranks' partial branch outputs are downloaded and added on the host in fp32, rank 0 then rank 1 (exl3_gdn.reduceOut order),
//! before the hyper-connection post step; that sum is the only difference from tp=1 besides fp accumulation order inside the MoE.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const st = @import("safetensors.zig");
const real = @import("exl3_real_tests.zig");
const strips = @import("exl3_strips.zig");
const router = @import("exl3_router.zig");
const scan = @import("exl3_scan.zig");
const hc = @import("exl3_hc.zig");
const embed = @import("exl3_embed.zig");
const fple = @import("exl3_fwd_ple.zig");
const fwd = @import("exl3_fwd.zig");
const tpt = @import("exl3_tp_tests.zig");

const Gpu = check.Gpu;
const hidden = fwd.hidden;
const wide = fwd.wide;
const inter = fwd.inter;
const experts = fwd.experts;
const topk = fwd.topk;
const slots = fwd.slots;
const vocab = fwd.vocab;
const eps = fwd.eps;
const tpm = cuda.exl3_tp;
const attn_ = cuda.exl3_attn;
const experts_per_rank = experts / 2;

const projs = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };

/// Routed expert lines on one rank (copy of exl3_fwd.zig's private ExpertLines: it is not pub and that file is shared).
const ExpertLines = struct {
    tr: [3]cuda.DeviceBuffer,
    su: [3]cuda.DeviceBuffer,
    sv: [3]cuda.DeviceBuffer,
    k: [3]usize = .{ hidden, hidden, inter },
    n: [3]usize = .{ inter, inter, hidden },
    stage_in: []i16,
    stage_out: []i16,
    halves: [3]u8 = .{ 0, 0, 0 },

    const max_halves = 16;

    fn init(gpu: Gpu) !ExpertLines {
        var r: ExpertLines = undefined;
        r.k = .{ hidden, hidden, inter };
        r.n = .{ inter, inter, hidden };
        r.halves = .{ 0, 0, 0 };
        var made: usize = 0;
        errdefer for (0..made) |i| {
            r.tr[i].free();
            r.su[i].free();
            r.sv[i].free();
        };
        while (made < 3) : (made += 1) {
            const span = try cuda.exl3.tileSpan(r.k[made], r.n[made], max_halves);
            r.tr[made] = try cuda.DeviceBuffer.alloc(gpu.d, topk * span * 2);
            r.su[made] = try cuda.DeviceBuffer.alloc(gpu.d, topk * r.k[made] * 2);
            r.sv[made] = try cuda.DeviceBuffer.alloc(gpu.d, topk * r.n[made] * 2);
        }
        const big = try cuda.exl3.tileSpan(hidden, inter, max_halves);
        r.stage_in = try gpu.gpa.alloc(i16, big);
        r.stage_out = try gpu.gpa.alloc(i16, big);
        return r;
    }

    fn deinit(r: *ExpertLines, gpa: std.mem.Allocator) void {
        for (0..3) |i| {
            r.tr[i].free();
            r.su[i].free();
            r.sv[i].free();
        }
        gpa.free(r.stage_in);
        gpa.free(r.stage_out);
    }

    fn put(r: *ExpertLines, pack: *const real.Pack, layer: usize, e: usize, line: usize) !void {
        for (0..3) |p| {
            const tr = try fwd.tensor(pack, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.trellis", .{ layer, e, projs[p] });
            const su = try fwd.tensor(pack, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.suh", .{ layer, e, projs[p] });
            const sv = try fwd.tensor(pack, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.svh", .{ layer, e, projs[p] });
            const tiles_k = r.k[p] / 16;
            const tiles_n = r.n[p] / 16;
            if (tr.rank != 3 or tr.dim(0) != tiles_k or tr.dim(1) != tiles_n) return error.BadShape;
            const words = tr.dim(2);
            const halves: u8 = @intCast(words / 8);
            if (r.halves[p] == 0) r.halves[p] = halves else if (r.halves[p] != halves) return error.NonUniformExperts;
            const span = tiles_k * tiles_n * words;
            const dst_in = r.stage_in[0..span];
            @memcpy(std.mem.sliceAsBytes(dst_in), tr.bytes);
            try strips.reorder(dst_in, tiles_k, tiles_n, words, r.stage_out[0..span]);
            try r.tr[p].upload(line * span * 2, std.mem.sliceAsBytes(r.stage_out[0..span]));
            try r.su[p].upload(line * r.k[p] * 2, su.bytes);
            try r.sv[p].upload(line * r.n[p] * 2, sv.bytes);
        }
    }

    fn packed_(r: *const ExpertLines, p: usize) !cuda.exl3.Packed {
        return .{
            .trellis = r.tr[p].ptr,
            .suh = r.su[p].ptr,
            .svh = r.sv[p].ptr,
            .trellis_stride = try cuda.exl3.tileSpan(r.k[p], r.n[p], r.halves[p]),
            .suh_stride = r.k[p],
            .svh_stride = r.n[p],
            .halves = r.halves[p],
            .codebook = 2,
        };
    }
};

const TpLayer = struct {
    index: usize,
    attn_hc: fwd.HcW,
    mlp_hc: fwd.HcW,
    gate16: []f16,
    sgate16: []f16,
    shared: [3]fwd.Lin, // rank 0 only
    gdn: ?[2]tpt.GdnW = null,
    attn: ?[2]tpt.AttnSide = null,
};

/// Per-rank buffers.
const RankBufs = struct {
    table: tpm.Table,
    gdn_scratch: tpm.Scratch,
    kv_all: cuda.DeviceBuffer,
    ik_all: cuda.DeviceBuffer,
    pooled_all: cuda.DeviceBuffer,
    pos0: cuda.DeviceBuffer,
    x: cuda.DeviceBuffer,
    br: cuda.DeviceBuffer,
    qp: cuda.DeviceBuffer,
    kp: cuda.DeviceBuffer,
    vp: cuda.DeviceBuffer,
    ip: cuda.DeviceBuffer,
    q_out: cuda.DeviceBuffer,
    iq_out: cuda.DeviceBuffer,
    scores: cuda.DeviceBuffer,
    ids_sel: cuda.DeviceBuffer,
    nk: cuda.DeviceBuffer,
    attn_out: cuda.DeviceBuffer,
    ids: cuda.DeviceBuffer,
    lines: ExpertLines,
    y_gate: cuda.DeviceBuffer,
    y_up: cuda.DeviceBuffer,
    y_glu: cuda.DeviceBuffer,
    y_down: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    s_gate: cuda.DeviceBuffer,
    s_up: cuda.DeviceBuffer,
    s_glu: cuda.DeviceBuffer,
};

pub const TpEngine = struct {
    gpu: Gpu,
    pack: *const real.Pack,
    opt: fwd.Options,
    rs: tpt.Ranks,
    gdn: [2]tpm.GdnTp,
    at: [2]attn_.Attn,
    offset: f32,
    layers: []TpLayer,
    mixer: fwd.HcW,
    embed_table: []const u8,
    head: ?fwd.Lin,
    logits: cuda.DeviceBuffer,
    ple: ?fple.Ple,
    ngram_file: ?st.File,
    pos: usize = 0,
    bufs: [2]RankBufs,
    modules: usize = 0,
    /// device bytes each rank lost (cuMemGetInfo) between the start of init and its end
    resident: [2]usize = .{ 0, 0 },

    pub fn init(gpu: Gpu, pack: *const real.Pack, opt: fwd.Options) !*TpEngine {
        const gpa = gpu.gpa;
        const e = try gpa.create(TpEngine);
        errdefer gpa.destroy(e);
        e.gpu = gpu;
        e.pack = pack;
        e.opt = opt;
        e.pos = 0;
        e.ple = null;
        e.ngram_file = null;
        e.head = null;
        e.modules = 0;
        e.rs = try tpt.Ranks.open(gpu);
        const free0 = [2]usize{ try e.rs.r[0].freeNow(), try e.rs.r[1].freeNow() };
        for (0..2) |i| {
            try e.rs.r[i].enter();
            e.gdn[i] = try tpm.GdnTp.load(gpu.d);
            e.at[i] = try attn_.Attn.load(gpu.d);
            e.modules += 1;
        }

        var means: [fwd.layers_total]f32 = undefined;
        for (&means, 0..) |*m, l| {
            const w = try fwd.f32Of(gpa, try fwd.tensor(pack, "model.language_model.layers.{d}.attn_hyper_connection.hc_norm.weight", .{l}));
            defer gpa.free(w);
            var s: f64 = 0;
            for (w) |v| s += v;
            m.* = @floatCast(s / @as(f64, @floatFromInt(w.len)));
        }
        e.offset = try scan.centredOffset(&means);

        const n = opt.layers;
        e.layers = try gpa.alloc(TpLayer, n);
        for (0..n) |i| {
            const l = opt.first_layer + i;
            const lay = &e.layers[i];
            lay.index = l;
            lay.gdn = null;
            lay.attn = null;
            lay.attn_hc = try fwd.HcW.load(gpa, pack, "model.language_model.layers.{d}.attn_hyper_connection", .{l}, true);
            lay.mlp_hc = try fwd.HcW.load(gpa, pack, "model.language_model.layers.{d}.mlp_hyper_connection", .{l}, true);
            lay.gate16 = try fwd.f16Of(gpa, try fwd.tensor(pack, "model.language_model.layers.{d}.mlp.gate.weight", .{l}));
            lay.sgate16 = try fwd.f16Of(gpa, try fwd.tensor(pack, "model.language_model.layers.{d}.mlp.shared_expert_gate.weight", .{l}));
            if (lay.gate16.len != experts * hidden or lay.sgate16.len != hidden) return error.BadShape;
            try e.rs.r[0].enter();
            var nbuf: [3][160]u8 = undefined;
            const stem = try std.fmt.bufPrint(&nbuf[2], "model.language_model.layers.{d}.mlp.shared_expert", .{l});
            var nb2: [3][200]u8 = undefined;
            lay.shared[0] = try fwd.Lin.load(gpu, pack, try std.fmt.bufPrint(&nb2[0], "{s}.gate_proj", .{stem}));
            lay.shared[1] = try fwd.Lin.load(gpu, pack, try std.fmt.bufPrint(&nb2[1], "{s}.up_proj", .{stem}));
            lay.shared[2] = try fwd.Lin.load(gpu, pack, try std.fmt.bufPrint(&nb2[2], "{s}.down_proj", .{stem}));
            if (fwd.Layer.isAttn(l)) {
                const h = try tpt.AttnHost.load(gpu, pack, l);
                defer h.free(gpa);
                var sides: [2]tpt.AttnSide = undefined;
                for (0..2) |r| sides[r] = try tpt.AttnSide.build(e.rs.r[r], &e.at[r], gpu, h, r);
                lay.attn = sides;
            } else {
                const h = try tpt.GdnHost.load(gpu, pack, l);
                defer h.free(gpa);
                var ws: [2]tpt.GdnW = undefined;
                for (0..2) |r| ws[r] = try tpt.GdnW.load(e.rs.r[r], gpu, h, 2, r);
                lay.gdn = ws;
            }
            if (opt.verbose) std.debug.print("  loaded layer {d}\n", .{l});
        }
        e.mixer = try fwd.HcW.load(gpa, pack, "model.language_model.hyper_connection_mixer", .{}, false);
        e.embed_table = (pack.find("model.language_model.embed_tokens.weight") orelse return error.MissingTensor).bytes;
        try e.rs.r[0].enter();
        if (opt.head) e.head = try fwd.Lin.load(gpu, pack, "lm_head");

        const R = opt.max_rows;
        const cap = opt.cap;
        const nb = cap / 4;
        const slot = attn_.slot_bytes;
        for (0..2) |i| {
            try e.rs.r[i].enter();
            const b = &e.bufs[i];
            b.table = try tpm.Table.alloc(gpu.d, 36, 1);
            b.gdn_scratch = try tpm.Scratch.alloc(gpu.d, R);
            b.kv_all = try fwd.dalloc(gpu, 12 * cap * 1 * slot);
            b.ik_all = try fwd.dalloc(gpu, 12 * cap * 128 * 2);
            b.pooled_all = try fwd.dalloc(gpu, 12 * (nb + 2) * 128 * 2);
            b.pos0 = try fwd.dalloc(gpu, 4);
            b.x = try fwd.dalloc(gpu, R * hidden * 2);
            b.br = try fwd.dalloc(gpu, R * hidden * 2);
            b.qp = try fwd.dalloc(gpu, R * 6144 * 2);
            b.kp = try fwd.dalloc(gpu, R * 256 * 2);
            b.vp = try fwd.dalloc(gpu, R * 256 * 2);
            b.ip = try fwd.dalloc(gpu, R * 640 * 2);
            b.q_out = try fwd.dalloc(gpu, R * 3072 * 2);
            b.iq_out = try fwd.dalloc(gpu, R * 512 * 2);
            b.scores = try fwd.dalloc(gpu, R * (nb + 1) * 4);
            b.ids_sel = try fwd.dalloc(gpu, R * attn_.ids_stride * 4);
            b.nk = try fwd.dalloc(gpu, R * 4);
            b.attn_out = try fwd.dalloc(gpu, R * 3072 * 2);
            var ids_host: [topk]i32 = undefined;
            for (&ids_host, 0..) |*d, j| d.* = @intCast(j);
            b.ids = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&ids_host));
            b.lines = try ExpertLines.init(gpu);
            b.y_gate = try fwd.dalloc(gpu, topk * inter * 2);
            b.y_up = try fwd.dalloc(gpu, topk * inter * 2);
            b.y_glu = try fwd.dalloc(gpu, topk * inter * 2);
            b.y_down = try fwd.dalloc(gpu, slots * hidden * 2);
            b.wts = try fwd.dalloc(gpu, slots * 4);
            b.s_gate = try fwd.dalloc(gpu, inter * 2);
            b.s_up = try fwd.dalloc(gpu, inter * 2);
            b.s_glu = try fwd.dalloc(gpu, inter * 2);
            try e.rs.r[i].ctx.synchronize(); // the zero fills are async w.r.t. the host and unordered against the rank's stream
        }
        try e.rs.r[0].enter();
        e.logits = try fwd.dalloc(gpu, if (opt.head) R * vocab * 2 else 16);
        try e.rs.r[0].ctx.synchronize();
        if (opt.ngram_path) |p| {
            e.ngram_file = try st.File.open(gpa, gpu.io, p);
            e.ple = try fwd.loadPle(gpu, pack, &e.ngram_file.?);
        }
        for (0..2) |i| e.resident[i] = free0[i] -| try e.rs.r[i].freeNow();
        return e;
    }

    pub fn deinit(e: *TpEngine) void {
        const gpa = e.gpu.gpa;
        for (e.layers) |*l| {
            l.attn_hc.free(gpa);
            l.mlp_hc.free(gpa);
            gpa.free(l.gate16);
            gpa.free(l.sgate16);
            e.rs.r[0].enter() catch {};
            for (&l.shared) |*s| s.free();
            if (l.gdn) |*g| for (g) |*w| w.free();
            if (l.attn) |*a| for (a) |*s| s.free();
        }
        gpa.free(e.layers);
        e.mixer.free(gpa);
        e.rs.r[0].enter() catch {};
        if (e.head) |*h| h.free();
        e.logits.free();
        if (e.ple) |*p| p.deinit();
        if (e.ngram_file) |*f| f.close(e.gpu.io);
        for (0..2) |i| {
            e.rs.r[i].enter() catch {};
            const b = &e.bufs[i];
            b.table.free();
            b.gdn_scratch.free();
            inline for (.{ "kv_all", "ik_all", "pooled_all", "pos0", "x", "br", "qp", "kp", "vp", "ip", "q_out", "iq_out", "scores", "ids_sel", "nk", "attn_out", "ids", "y_gate", "y_up", "y_glu", "y_down", "wts", "s_gate", "s_up", "s_glu" }) |name| @field(b, name).free();
            b.lines.deinit(gpa);
        }
        for (0..e.modules) |i| {
            e.rs.r[i].enter() catch {};
            e.at[i].unload();
            e.gdn[i].unload();
        }
        e.rs.close(e.gpu);
        gpa.destroy(e);
    }

    pub fn reset(e: *TpEngine) !void {
        for (0..2) |i| {
            try e.rs.r[i].enter();
            try e.bufs[i].table.reset(e.rs.r[i].stream, 0);
            try e.rs.r[i].stream.synchronize();
        }
        e.pos = 0;
        if (e.ple) |*p| p.reset();
    }

    fn applyPle(e: *TpEngine, tokens: []const u32, h: []f32) !void {
        const gpa = e.gpu.gpa;
        const rows = tokens.len;
        const h64 = try gpa.alloc(f64, rows * wide);
        defer gpa.free(h64);
        const out = try gpa.alloc(f64, rows * wide);
        defer gpa.free(out);
        for (h64, h[0 .. rows * wide]) |*d, s| d.* = s;
        try e.ple.?.rows(tokens, h64, out);
        for (h[0 .. rows * wide], out) |*d, s| d.* += @floatCast(s);
    }

    /// Enqueue rank `rk`'s share of an attention layer (it reads `bufs.x`, leaves the o_proj partial in `bufs.br`).
    fn attnRun(e: *TpEngine, rk: usize, side: *const tpt.AttnSide, kv_index: usize, rows: usize) !void {
        const b = &e.bufs[rk];
        const cap = e.opt.cap;
        const nb = cap / 4;
        const slot = attn_.slot_bytes;
        const p0: i32 = @intCast(e.pos);
        try b.pos0.upload(0, std.mem.asBytes(&p0));
        try e.rs.r[rk].ctx.synchronize();
        const dims: attn_.Dims = .{ .q_heads = 12, .kv_heads = 1, .tokens = cap };
        const bufs = tpt.Bufs{
            .x = b.x.ptr,
            .qp = b.qp.ptr,
            .kp = b.kp.ptr,
            .vp = b.vp.ptr,
            .ip = b.ip.ptr,
            .q_out = b.q_out.ptr,
            .iq_out = b.iq_out.ptr,
            .scores = b.scores.ptr,
            .ids = b.ids_sel.ptr,
            .nk = b.nk.ptr,
            .attn = b.attn_out.ptr,
            .y = b.br.ptr,
            .kv = b.kv_all.ptr + kv_index * cap * slot,
            .ik = b.ik_all.ptr + kv_index * cap * 128 * 2,
            .pooled = b.pooled_all.ptr + kv_index * (nb + 2) * 128 * 2,
            .pos0 = b.pos0.ptr,
        };
        try tpt.pipeline(side, bufs, dims, rows);
    }

    /// One row of one rank's MoE share: its routed picks (ids of its half) and, on rank 0, the shared expert.
    fn moeRank(e: *TpEngine, rk: usize, lay: *TpLayer, row: usize, picks: []const u16, weights: []const f32) !void {
        const r = e.rs.r[rk];
        try r.enter();
        const b = &e.bufs[rk];
        const x_ptr = b.x.ptr + row * hidden * 2;
        const out_ptr = b.br.ptr + row * hidden * 2;
        var wr: [slots]f32 = undefined;
        var n: usize = 0;
        b.lines.halves = .{ 0, 0, 0 };
        for (picks[0..topk], 0..) |p, j| {
            if (p / experts_per_rank != rk) continue;
            try b.lines.put(e.pack, lay.index, p, n);
            wr[n] = weights[j];
            n += 1;
        }
        const shared = rk == 0;
        const total = n + @intFromBool(shared);
        if (total == 0) return; // br row stays zero
        const s = r.stream;
        if (shared) wr[n] = weights[topk];
        try b.wts.upload(0, std.mem.sliceAsBytes(wr[0..total]));
        try r.ctx.synchronize(); // the expert lines, weights and x row must have landed before the launches
        if (n > 0) try r.ex.hit(s, x_ptr, b.ids.ptr, try b.lines.packed_(0), try b.lines.packed_(1), try b.lines.packed_(2), b.y_gate.ptr, b.y_up.ptr, b.y_glu.ptr, b.y_down.ptr, 1, n, n);
        if (shared) {
            try lay.shared[0].run(&r.ex, s, x_ptr, b.s_gate.ptr, 1);
            try lay.shared[1].run(&r.ex, s, x_ptr, b.s_up.ptr, 1);
            try r.ex.swiglu(s, b.s_gate.ptr, b.s_up.ptr, b.s_glu.ptr, 1, inter);
            try lay.shared[2].run(&r.ex, s, b.s_glu.ptr, b.y_down.ptr + n * hidden * 2, 1);
        }
        try r.ex.combine(s, b.y_down.ptr, b.wts.ptr, out_ptr, 1, hidden, total);
    }

    /// Streams after the last loaded layer for `tokens` (rows consecutive from position `e.pos`), into `h` [rows][10240] f32.
    pub fn forward(e: *TpEngine, tokens: []const u32, h: []f32) !void {
        const rows = tokens.len;
        const gpa = e.gpu.gpa;
        if (rows == 0 or rows > e.opt.max_rows or e.pos + rows > e.opt.cap or h.len != rows * wide) return error.Shape;
        const x16 = try gpa.alloc(f16, rows * hidden);
        defer gpa.free(x16);
        var br16: [2][]f16 = undefined;
        br16[0] = try gpa.alloc(f16, rows * hidden);
        defer gpa.free(br16[0]);
        br16[1] = try gpa.alloc(f16, rows * hidden);
        defer gpa.free(br16[1]);
        const mixed = try gpa.alloc(f32, hidden);
        defer gpa.free(mixed);
        const branch = try gpa.alloc(f32, hidden);
        defer gpa.free(branch);
        const inject = try gpa.alloc(f32, rows * 4);
        defer gpa.free(inject);
        const scratch = try gpa.alloc(f32, hc.scratchLen(hc.real));
        defer gpa.free(scratch);

        for (tokens, 0..) |tk, t| {
            const row = try embed.gather(e.embed_table, tk);
            var r32: [hidden]f32 = undefined;
            for (&r32, 0..) |*d, i| d.* = fwd.bf16(std.mem.readInt(u16, row[i * 2 ..][0..2], .little));
            try hc.expand(hc.real, &r32, h[t * wide ..][0..wide]);
        }

        for (e.layers) |*lay| {
            const l = lay.index;
            if (e.ple != null and l == fwd.ple_layer) try e.applyPle(tokens, h);

            // attention / GDN block: both ranks read the same normed rows
            for (0..rows) |t| {
                try hc.pre(hc.real, lay.attn_hc.weights(), e.offset, eps, h[t * wide ..][0..wide], mixed, inject[t * 4 .. t * 4 + 4], scratch);
                for (x16[t * hidden ..][0..hidden], mixed) |*d, s| d.* = @floatCast(s);
            }
            for (0..2) |rk| {
                const r = e.rs.r[rk];
                try r.enter();
                const b = &e.bufs[rk];
                try b.x.upload(0, std.mem.sliceAsBytes(x16));
                try r.ctx.synchronize(); // pageable H2D is not ordered against the rank's non-blocking stream
                if (lay.gdn) |*g| {
                    const idx = fwd.Layer.gdnIndex(l);
                    try e.gdn[rk].forward(&r.ex, r.stream, g[rk].layer, try b.table.convPtr(0, idx), try b.table.recPtr(0, idx), b.x.ptr, rows, b.gdn_scratch, b.br.ptr);
                } else {
                    try e.attnRun(rk, &lay.attn.?[rk], fwd.Layer.kvIndex(l), rows);
                }
            }
            for (0..2) |rk| {
                try e.rs.r[rk].enter();
                try e.rs.r[rk].stream.synchronize();
                try e.bufs[rk].br.download(0, std.mem.sliceAsBytes(br16[rk]));
            }
            for (0..rows) |t| {
                for (branch, br16[0][t * hidden ..][0..hidden], br16[1][t * hidden ..][0..hidden]) |*d, a, c| d.* = @as(f32, @floatCast(a)) + @as(f32, @floatCast(c));
                const ht = h[t * wide ..][0..wide];
                try hc.post(hc.real, ht, branch, inject[t * 4 .. t * 4 + 4], ht);
            }

            // MoE block: expert parallel
            for (0..rows) |t| {
                try hc.pre(hc.real, lay.mlp_hc.weights(), e.offset, eps, h[t * wide ..][0..wide], mixed, inject[t * 4 .. t * 4 + 4], scratch);
                for (x16[t * hidden ..][0..hidden], mixed) |*d, s| d.* = @floatCast(s);
            }
            for (0..2) |rk| {
                try e.rs.r[rk].enter();
                try e.bufs[rk].x.upload(0, std.mem.sliceAsBytes(x16));
                try e.bufs[rk].br.fill8(0, null); // a rank with no pick for a row contributes zeros
                try e.rs.r[rk].ctx.synchronize();
            }
            for (0..rows) |t| {
                var x32: [hidden]f32 = undefined;
                for (&x32, x16[t * hidden ..][0..hidden]) |*d, s| d.* = @floatCast(s);
                var logits: [experts + 1]f32 = undefined;
                var weights: [slots]f32 = undefined;
                var rscratch: [experts]f32 = undefined;
                var picks: [slots]u16 = undefined;
                try router.launchDtype(.f16, 0, std.mem.sliceAsBytes(lay.gate16), &x32, std.mem.sliceAsBytes(lay.sgate16), &x32, topk, &logits, &picks, &weights, &rscratch);
                for (0..2) |rk| try e.moeRank(rk, lay, t, &picks, &weights);
                for (0..2) |rk| {
                    try e.rs.r[rk].enter();
                    try e.rs.r[rk].stream.synchronize(); // the next row overwrites the expert lines
                }
            }
            for (0..2) |rk| {
                try e.rs.r[rk].enter();
                try e.bufs[rk].br.download(0, std.mem.sliceAsBytes(br16[rk]));
            }
            for (0..rows) |t| {
                for (branch, br16[0][t * hidden ..][0..hidden], br16[1][t * hidden ..][0..hidden]) |*d, a, c| d.* = @as(f32, @floatCast(a)) + @as(f32, @floatCast(c));
                const ht = h[t * wide ..][0..wide];
                try hc.post(hc.real, ht, branch, inject[t * 4 .. t * 4 + 4], ht);
            }
            if (e.opt.verbose) std.debug.print("  layer {d} done\n", .{l});
        }
        e.pos += rows;
    }

    /// Final collapse and the lm_head (rank 0) for `rows` stream rows. f32 logits [rows][vocab].
    pub fn logitsOf(e: *TpEngine, h: []const f32, rows: usize, out: []f32) !void {
        const gpa = e.gpu.gpa;
        const head = e.head orelse return error.NoHead;
        if (h.len != rows * wide or out.len != rows * vocab or rows > e.opt.max_rows) return error.Shape;
        const x16 = try gpa.alloc(f16, rows * hidden);
        defer gpa.free(x16);
        const mixed = try gpa.alloc(f32, hidden);
        defer gpa.free(mixed);
        const scratch = try gpa.alloc(f32, hc.scratchLen(hc.real));
        defer gpa.free(scratch);
        for (0..rows) |t| {
            try hc.pre(hc.real, e.mixer.weights(), e.offset, eps, h[t * wide ..][0..wide], mixed, null, scratch);
            for (x16[t * hidden ..][0..hidden], mixed) |*d, s| d.* = @floatCast(s);
        }
        const r = e.rs.r[0];
        try r.enter();
        try e.bufs[0].x.upload(0, std.mem.sliceAsBytes(x16));
        try r.ctx.synchronize();
        try head.run(&r.ex, r.stream, e.bufs[0].x.ptr, e.logits.ptr, rows);
        try r.stream.synchronize();
        const l16 = try gpa.alloc(f16, rows * vocab);
        defer gpa.free(l16);
        try e.logits.download(0, std.mem.sliceAsBytes(l16));
        for (out, l16) |*d, s| d.* = @floatCast(s);
    }
};

// -- test ------------------------------------------------------------------------------------------------------

const default_ids = [_]u32{ 760, 6511, 314, 9338, 369, 11751, 13, 1049, 369, 3750, 364, 279, 242476, 300, 21262, 13 };

fn argmax(xs: []const f32) usize {
    var best: usize = 0;
    for (xs, 0..) |v, i| if (v > xs[best]) {
        best = i;
    };
    return best;
}

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

/// tf-cuda-test exl3-forward-tp <pack> [N layers, default 48] [ids]
/// Teacher-forced logits of the 16-token prompt after N layers: tp=1 Engine (one device) against the tp=2 engine (two devices).
pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const gpa = gpu.gpa;
    const dir = rest[0];
    const n_layers = try std.fmt.parseInt(usize, if (rest.len > 1) rest[1] else "48", 10);
    const tokens = if (rest.len > 2) try parseIds(gpa, rest[2]) else try gpa.dupe(u32, &default_ids);
    defer gpa.free(tokens);
    const rows = tokens.len;
    var pack = try tpt.openPack(gpu, dir);
    defer pack.close(gpu.io);
    var path_buf: [512]u8 = undefined;
    const ngram_path = try std.fmt.bufPrint(&path_buf, "{s}/ngram_embedding.safetensors", .{dir});
    const opt: fwd.Options = .{ .first_layer = 0, .layers = n_layers, .max_rows = rows, .cap = 128, .head = true, .ngram_path = if (n_layers > fwd.ple_layer) ngram_path else null };
    std.debug.print("== tp=2 vs tp=1, teacher-forced, {d} layers, {d} tokens\n", .{ n_layers, rows });

    const lg1 = try gpa.alloc(f32, rows * vocab);
    defer gpa.free(lg1);
    const lg2 = try gpa.alloc(f32, rows * vocab);
    defer gpa.free(lg2);
    const hg = try gpa.alloc(f32, rows * wide);
    defer gpa.free(hg);
    const hg2 = try gpa.alloc(f32, rows * wide);
    defer gpa.free(hg2);

    {
        var t0 = nowMs(gpu);
        var eng = try fwd.Engine.init(gpu, &pack, opt);
        defer eng.deinit();
        std.debug.print("  tp=1: loaded in {d:.0} ms (device 0 holds {d:.0} MiB of this process's buffers)\n", .{ nowMs(gpu) - t0, tpt.mib(@intCast(cuda.usage(false).device)) });
        t0 = nowMs(gpu);
        try eng.forward(tokens, hg);
        try eng.logitsOf(hg, rows, lg1);
        std.debug.print("  tp=1: forward + head {d:.0} ms\n", .{nowMs(gpu) - t0});
    }
    try gpu.ctx.makeCurrent();

    var t0 = nowMs(gpu);
    const eng2 = try TpEngine.init(gpu, &pack, opt);
    defer eng2.deinit();
    std.debug.print("  tp=2: loaded in {d:.0} ms; device bytes this process holds {d:.0} MiB in total; cuMemGetInfo free dropped by {d:.0} MiB on rank 0 and {d:.0} MiB on rank 1 during init (shared GPUs: other processes may add noise)\n", .{ nowMs(gpu) - t0, tpt.mib(@intCast(cuda.usage(false).device)), tpt.mib(eng2.resident[0]), tpt.mib(eng2.resident[1]) });
    t0 = nowMs(gpu);
    try eng2.forward(tokens, hg2);
    try eng2.logitsOf(hg2, rows, lg2);
    std.debug.print("  tp=2: forward + head {d:.0} ms (host-heavy; two ranks run their MoE rows back to back)\n", .{nowMs(gpu) - t0});

    // compare
    var stream_max: f64 = 0;
    var stream_peak: f64 = 0;
    for (hg, hg2) |a, b| {
        stream_max = @max(stream_max, @abs(@as(f64, a) - b));
        stream_peak = @max(stream_peak, @abs(@as(f64, a)));
    }
    var lmax: f64 = 0;
    var num: f64 = 0;
    var den: f64 = 0;
    for (lg1, lg2) |a, b| {
        const d = @as(f64, a) - b;
        lmax = @max(lmax, @abs(d));
        num += d * d;
        den += @as(f64, a) * a;
    }
    var agree: usize = 0;
    var ties: usize = 0;
    std.debug.print("  pos  input   tp1_top1 (logit)      tp2_top1 (logit)      match  tp1 gap of tp2's pick\n", .{});
    for (0..rows) |t| {
        const a = lg1[t * vocab ..][0..vocab];
        const b = lg2[t * vocab ..][0..vocab];
        const ai = argmax(a);
        const bi = argmax(b);
        const gap = a[ai] - a[bi];
        const same = ai == bi;
        const tie = !same and gap <= 2 * lmax;
        agree += @intFromBool(same);
        ties += @intFromBool(tie);
        std.debug.print("  {d:>3}  {d:>6}  {d:>6} ({d:>8.3})   {d:>6} ({d:>8.3})   {s}   {d:.4}\n", .{ t, tokens[t], ai, a[ai], bi, b[bi], if (same) "yes" else if (tie) "tie" else "NO ", gap });
    }
    std.debug.print("  streams after {d} layers: max_abs {e:.3} (peak {e:.3}); logits vs tp=1: max_abs {d:.4}, rms rel {e:.2}; top-1 exact {d}/{d}, near-tie flips {d}\n", .{ n_layers, stream_max, stream_peak, lmax, @sqrt(num / den), agree, rows, ties });
    try check.expect(agree + ties == rows, "{d} layers: tp=2 top-1 differs from tp=1 beyond a near-tie at {d} of {d} positions", .{ n_layers, rows - agree - ties, rows });
    check.pass("{d} layers, {d} teacher-forced positions: tp=2 top-1 equals tp=1's at {d} positions, {d} near-tie flips (logits max_abs {d:.4}, rms rel {e:.2})", .{ n_layers, rows, agree, ties, lmax, @sqrt(num / den) });
}
