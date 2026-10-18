//! The assembled EXL3 forward for Qwen3.8-Flash-Next at tp=1 on one device: embedding rows, hyper-connection streams, the
//! decoder layers (GDN or sparse attention, then the MoE block), the final stream collapse and the 5-bit lm_head.
//!
//! Where things live (tp=1, one 16 GB card):
//!   device, resident: every layer's GDN / attention projections, the shared expert, dense a/b, conv, norms, the lm_head,
//!                     GDN state, attention KV, indexer keys and pooled blocks
//!   host:             the 512 routed experts of every layer (the mmapped pack); the 10 selected experts' lines are
//!                     strips-reordered and uploaded synchronously into 10 device lines per projection, per row per layer
//!                     (no hot set, no LRU: that is the offload stage's job)
//!                     the embedding table, the hyper-connection weights (host f32, as `exl3_hc` designs it), the router
//!                     gates (host, `exl3_router`), the PLE block (`exl3_fwd_ple`)
//! A call to `forward` takes up to `max_rows` token rows through the layers layer-major (all rows through layer l before l + 1),
//! the rows of a layer's GDN / attention as one batch, the MoE row by row. Prefill and decode are the same call.

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
const attn_cpu = @import("exl3_attn_cpu.zig");
const fwd = @import("exl3_fwd_cpu.zig");
const fple = @import("exl3_fwd_ple.zig");

const Gpu = check.Gpu;
pub const hidden = 2560;
pub const wide = 4 * hidden;
pub const inter = 640;
pub const experts = 512;
pub const topk = 10;
pub const slots = topk + 1;
pub const vocab = 248320;
pub const layers_total = 48;
pub const eps: f32 = 1e-6;
pub const ple_layer = 1;
pub const eos_token: i64 = 248044;

// -- host tensor helpers ---------------------------------------------------------------------------------

pub fn tensor(pack: *const real.Pack, comptime fmt: []const u8, args: anytype) !st.Tensor {
    var buf: [200]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, args);
    return pack.find(name) orelse {
        std.debug.print("missing tensor {s}\n", .{name});
        return error.MissingTensor;
    };
}

pub fn bf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// F16 or BF16 tensor as owned f32 (exact).
pub fn f32Of(gpa: std.mem.Allocator, t: st.Tensor) ![]f32 {
    const n = t.bytes.len / 2;
    const out = try gpa.alloc(f32, n);
    for (out, 0..) |*d, i| {
        const w = std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little);
        d.* = switch (t.dtype) {
            .f16 => @floatCast(@as(f16, @bitCast(w))),
            .bf16 => bf16(w),
            else => return error.WrongDtype,
        };
    }
    return out;
}

pub fn f16Of(gpa: std.mem.Allocator, t: st.Tensor) ![]f16 {
    if (t.dtype != .f16) return error.WrongDtype;
    const out = try gpa.alloc(f16, t.bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

pub fn f64Of(gpa: std.mem.Allocator, t: st.Tensor) ![]f64 {
    const f = try f32Of(gpa, t);
    defer gpa.free(f);
    const out = try gpa.alloc(f64, f.len);
    for (out, f) |*d, s| d.* = s;
    return out;
}

pub fn i16Of(gpa: std.mem.Allocator, bytes: []const u8) ![]i16 {
    const out = try gpa.alloc(i16, bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

pub const HcW = struct {
    norm: []f32,
    down: []f32,
    up: []f32,
    inject: ?[]f32,

    pub fn load(gpa: std.mem.Allocator, pack: *const real.Pack, comptime base: []const u8, args: anytype, with_inject: bool) !HcW {
        const n = try f32Of(gpa, try tensor(pack, base ++ ".hc_norm.weight", args));
        const d = try f32Of(gpa, try tensor(pack, base ++ ".input_mix_weight_down.weight", args));
        const u = try f32Of(gpa, try tensor(pack, base ++ ".input_mix_weight_up.weight", args));
        const j: ?[]f32 = if (with_inject) try f32Of(gpa, try tensor(pack, base ++ ".block_inject_weight.weight", args)) else null;
        return .{ .norm = n, .down = d, .up = u, .inject = j };
    }

    pub fn weights(h: HcW) hc.Weights {
        return .{ .norm = h.norm, .down = h.down, .up = h.up, .inject = h.inject };
    }

    pub fn free(h: HcW, gpa: std.mem.Allocator) void {
        gpa.free(h.norm);
        gpa.free(h.down);
        gpa.free(h.up);
        if (h.inject) |j| gpa.free(j);
    }
};

// -- device weights ---------------------------------------------------------------------------------------

pub const Lin = struct {
    d: real.Dev,
    k: usize,
    n: usize,
    halves: u8,

    pub fn load(gpu: Gpu, pack: *const real.Pack, base: []const u8) !Lin {
        const m = try real.mat(pack, gpu.gpa, base);
        defer {
            gpu.gpa.free(m.trellis);
            gpu.gpa.free(m.suh);
            gpu.gpa.free(m.svh);
        }
        const d = try real.upload(gpu, m);
        return .{ .d = d, .k = m.tiles_k * 16, .n = m.tiles_n * 16, .halves = m.halves };
    }

    pub fn free(l: *Lin) void {
        l.d.free();
    }

    pub fn run(l: *const Lin, ex: *const cuda.exl3.Exl3, s: cuda.Stream, x: u64, y: u64, rows: usize) !void {
        try ex.linear(s, x, l.d.trellis.ptr, l.d.suh.ptr, l.d.svh.ptr, y, rows, l.k, l.n, l.halves, 2);
    }

    pub fn proj(l: *const Lin) cuda.exl3_gdn.Proj {
        return .{ .trellis = l.d.trellis.ptr, .suh = l.d.suh.ptr, .svh = l.d.svh.ptr, .k = l.k, .n = l.n, .halves = l.halves };
    }
};

fn devBytes(gpu: Gpu, t: st.Tensor) !cuda.DeviceBuffer {
    return cuda.DeviceBuffer.fromHost(gpu.d, t.bytes);
}

pub const GdnDev = struct {
    qkv: Lin,
    z: Lin,
    out: Lin,
    a: cuda.DeviceBuffer,
    b: cuda.DeviceBuffer,
    conv: cuda.DeviceBuffer,
    alog: cuda.DeviceBuffer,
    dt: cuda.DeviceBuffer,
    norm: cuda.DeviceBuffer,

    fn load(gpu: Gpu, pack: *const real.Pack, l: usize) !GdnDev {
        var nb: [4][128]u8 = undefined;
        const stem = try std.fmt.bufPrint(&nb[3], "model.language_model.layers.{d}.linear_attn", .{l});
        var qkv = try Lin.load(gpu, pack, try std.fmt.bufPrint(&nb[0], "{s}.in_proj_qkv", .{stem}));
        errdefer qkv.free();
        var z = try Lin.load(gpu, pack, try std.fmt.bufPrint(&nb[1], "{s}.in_proj_z", .{stem}));
        errdefer z.free();
        var out = try Lin.load(gpu, pack, try std.fmt.bufPrint(&nb[2], "{s}.out_proj", .{stem}));
        errdefer out.free();
        try check.expect(qkv.k == hidden and qkv.n == 10240 and z.n == 6144 and out.k == 6144 and out.n == hidden, "layer {d} GDN shapes", .{l});
        const ta = try tensor(pack, "model.language_model.layers.{d}.linear_attn.in_proj_a.weight", .{l});
        const tb = try tensor(pack, "model.language_model.layers.{d}.linear_attn.in_proj_b.weight", .{l});
        const tc = try tensor(pack, "model.language_model.layers.{d}.linear_attn.conv1d.weight", .{l});
        const tl = try tensor(pack, "model.language_model.layers.{d}.linear_attn.A_log", .{l});
        const td = try tensor(pack, "model.language_model.layers.{d}.linear_attn.dt_bias", .{l});
        const tn = try tensor(pack, "model.language_model.layers.{d}.linear_attn.norm.weight", .{l});
        try check.expect(ta.dtype == .f16 and tb.dtype == .f16 and tc.dtype == .bf16 and tl.dtype == .bf16 and td.dtype == .bf16 and tn.dtype == .bf16, "layer {d} GDN dense dtypes", .{l});
        return .{
            .qkv = qkv,
            .z = z,
            .out = out,
            .a = try devBytes(gpu, ta),
            .b = try devBytes(gpu, tb),
            .conv = try devBytes(gpu, tc),
            .alog = try devBytes(gpu, tl),
            .dt = try devBytes(gpu, td),
            .norm = try devBytes(gpu, tn),
        };
    }

    fn free(g: *GdnDev) void {
        g.qkv.free();
        g.z.free();
        g.out.free();
        g.a.free();
        g.b.free();
        g.conv.free();
        g.alog.free();
        g.dt.free();
        g.norm.free();
    }

    fn layer(g: *const GdnDev) cuda.exl3_gdn.Layer {
        return .{
            .qkv = g.qkv.proj(),
            .z = g.z.proj(),
            .out = g.out.proj(),
            .a_w = g.a.ptr,
            .b_w = g.b.ptr,
            .conv_w = g.conv.ptr,
            .a_log = g.alog.ptr,
            .dt_bias = g.dt.ptr,
            .norm_w = g.norm.ptr,
        };
    }
};

pub const AttnDev = struct {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    idx: Lin,
    q_scale: cuda.DeviceBuffer,
    k_scale: cuda.DeviceBuffer,
    iq_scale: cuda.DeviceBuffer,
    ik_scale: cuda.DeviceBuffer,

    fn scale(gpa: std.mem.Allocator, pack: *const real.Pack, l: usize, name: []const u8, offset: f32, mean_out: *f32) ![]f32 {
        var buf: [160]u8 = undefined;
        const t = pack.find(try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.self_attn.{s}", .{ l, name })) orelse return error.MissingTensor;
        const w = try f32Of(gpa, t);
        var s: f64 = 0;
        for (w) |v| s += v;
        mean_out.* = @floatCast(s / @as(f64, @floatFromInt(w.len)));
        for (w) |*v| v.* += offset;
        return w;
    }

    /// Returns the weights and, for the oracle, the four host scale arrays (owned by the caller).
    fn load(gpu: Gpu, pack: *const real.Pack, l: usize, offset: f32, host: *[4][]f32) !AttnDev {
        const gpa = gpu.gpa;
        var nb: [6][128]u8 = undefined;
        const stem = try std.fmt.bufPrint(&nb[5], "model.language_model.layers.{d}.self_attn", .{l});
        var q = try Lin.load(gpu, pack, try std.fmt.bufPrint(&nb[0], "{s}.q_proj", .{stem}));
        errdefer q.free();
        var k = try Lin.load(gpu, pack, try std.fmt.bufPrint(&nb[1], "{s}.k_proj", .{stem}));
        errdefer k.free();
        var v = try Lin.load(gpu, pack, try std.fmt.bufPrint(&nb[2], "{s}.v_proj", .{stem}));
        errdefer v.free();
        var o = try Lin.load(gpu, pack, try std.fmt.bufPrint(&nb[3], "{s}.o_proj", .{stem}));
        errdefer o.free();
        var idx = try Lin.load(gpu, pack, try std.fmt.bufPrint(&nb[4], "{s}.indexer.index_qk_proj", .{stem}));
        errdefer idx.free();
        try check.expect(q.k == hidden and q.n == 12288 and k.n == 512 and v.n == 512 and o.k == 6144 and o.n == hidden and idx.n == 640, "layer {d} attention shapes", .{l});
        var means: [4]f32 = undefined;
        host[0] = try scale(gpa, pack, l, "q_norm.weight", offset, &means[0]);
        host[1] = try scale(gpa, pack, l, "k_norm.weight", offset, &means[1]);
        host[2] = try scale(gpa, pack, l, "indexer.q_layernorm.weight", offset, &means[2]);
        host[3] = try scale(gpa, pack, l, "indexer.k_layernorm.weight", offset, &means[3]);
        // Shim for exl3_scan.centredOffset (NormAmbiguous: q/k means run 0.27 at layer 3 up to 0.59 at layer 31, indexer norms sit at -0.02..-0.08): the model defines all four as CenteredRMSNorm.
        try check.expect(means[0] < 0.75 and means[1] < 0.75 and @abs(means[2]) < 0.25 and @abs(means[3]) < 0.25, "layer {d}: attention norms are not gamma-1", .{l});
        return .{
            .q = q,
            .k = k,
            .v = v,
            .o = o,
            .idx = idx,
            .q_scale = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host[0])),
            .k_scale = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host[1])),
            .iq_scale = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host[2])),
            .ik_scale = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host[3])),
        };
    }

    fn free(a: *AttnDev) void {
        a.q.free();
        a.k.free();
        a.v.free();
        a.o.free();
        a.idx.free();
        a.q_scale.free();
        a.k_scale.free();
        a.iq_scale.free();
        a.ik_scale.free();
    }
};

pub const Layer = struct {
    index: usize,
    attn_hc: HcW,
    mlp_hc: HcW,
    gate16: []f16,
    sgate16: []f16,
    shared: [3]Lin,
    gdn: ?GdnDev = null,
    attn: ?AttnDev = null,

    pub fn isAttn(index: usize) bool {
        return index % 4 == 3;
    }

    /// Index of a GDN layer among the 36 GDN layers; index of an attention layer among the 12.
    pub fn gdnIndex(index: usize) usize {
        return index - index / 4;
    }

    pub fn kvIndex(index: usize) usize {
        return (index + 1) / 4 - 1;
    }

    fn deinit(l: *Layer, gpa: std.mem.Allocator) void {
        l.attn_hc.free(gpa);
        l.mlp_hc.free(gpa);
        gpa.free(l.gate16);
        gpa.free(l.sgate16);
        for (&l.shared) |*s| s.free();
        if (l.gdn) |*g| g.free();
        if (l.attn) |*a| a.free();
    }
};

// -- routed expert lines on the device ---------------------------------------------------------------------

const projs = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };

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

    /// Strips-reorders routed expert `e` of `layer` and uploads it as line `line` of each projection.
    fn put(r: *ExpertLines, pack: *const real.Pack, layer: usize, e: usize, line: usize) !void {
        for (0..3) |p| {
            const tr = try tensor(pack, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.trellis", .{ layer, e, projs[p] });
            const su = try tensor(pack, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.suh", .{ layer, e, projs[p] });
            const sv = try tensor(pack, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.svh", .{ layer, e, projs[p] });
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

// -- engine --------------------------------------------------------------------------------------------------

pub const Options = struct {
    first_layer: usize = 0,
    layers: usize = layers_total,
    max_rows: usize = 16,
    cap: usize = 128,
    head: bool = true,
    /// Path of ngram_embedding.safetensors; null leaves the PLE block out (reported by the caller).
    ngram_path: ?[]const u8 = null,
    verbose: bool = false,
};

pub const Trace = fwd.Trace;

pub const Engine = struct {
    gpu: Gpu,
    pack: *const real.Pack,
    opt: Options,
    ex: cuda.exl3.Exl3,
    gdn: cuda.exl3_gdn.Gdn,
    at: cuda.exl3_attn.Attn,
    stream: cuda.Stream,
    offset: f32,
    layers: []Layer,
    attn_host_scales: [][4][]f32,
    mixer: HcW,
    embed_table: []const u8,
    head: ?Lin,
    ple: ?fple.Ple,
    ngram_file: ?st.File,
    pos: usize = 0,
    trace: ?Trace = null,

    // device state and scratch
    table: cuda.exl3_gdn.Table,
    gdn_scratch: cuda.exl3_gdn.Scratch,
    kv_all: cuda.DeviceBuffer,
    ik_all: cuda.DeviceBuffer,
    pooled_all: cuda.DeviceBuffer,
    pos0: cuda.DeviceBuffer,
    x: cuda.DeviceBuffer, // fp16 [R][2560]
    br: cuda.DeviceBuffer, // fp16 [R][2560]
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
    ids: cuda.DeviceBuffer, // int32 0..9
    lines: ExpertLines,
    y_gate: cuda.DeviceBuffer,
    y_up: cuda.DeviceBuffer,
    y_glu: cuda.DeviceBuffer,
    y_down: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    s_gate: cuda.DeviceBuffer,
    s_up: cuda.DeviceBuffer,
    s_glu: cuda.DeviceBuffer,
    logits: cuda.DeviceBuffer,

    pub fn init(gpu: Gpu, pack: *const real.Pack, opt: Options) !Engine {
        const gpa = gpu.gpa;
        const R = opt.max_rows;
        const cap = opt.cap;
        const nb = cap / 4;
        var e: Engine = undefined;
        e.gpu = gpu;
        e.pack = pack;
        e.opt = opt;
        e.pos = 0;
        e.trace = null;
        e.ple = null;
        e.ngram_file = null;
        e.head = null;
        e.ex = try cuda.exl3.Exl3.load(gpu.d);
        e.gdn = try cuda.exl3_gdn.Gdn.load(gpu.d);
        e.at = try cuda.exl3_attn.Attn.load(gpu.d);
        e.stream = try cuda.Stream.init(gpu.d, true);

        // centred-norm offset: the locator the loader uses, over the 48 attention hc_norm means
        var means: [layers_total]f32 = undefined;
        for (&means, 0..) |*m, l| {
            const w = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.attn_hyper_connection.hc_norm.weight", .{l}));
            defer gpa.free(w);
            var s: f64 = 0;
            for (w) |v| s += v;
            m.* = @floatCast(s / @as(f64, @floatFromInt(w.len)));
        }
        e.offset = try scan.centredOffset(&means);

        const n = opt.layers;
        e.layers = try gpa.alloc(Layer, n);
        e.attn_host_scales = try gpa.alloc([4][]f32, n);
        for (e.attn_host_scales) |*s| s.* = .{ &.{}, &.{}, &.{}, &.{} };
        for (0..n) |i| {
            const l = opt.first_layer + i;
            const lay = &e.layers[i];
            lay.index = l;
            lay.gdn = null;
            lay.attn = null;
            lay.attn_hc = try HcW.load(gpa, pack, "model.language_model.layers.{d}.attn_hyper_connection", .{l}, true);
            lay.mlp_hc = try HcW.load(gpa, pack, "model.language_model.layers.{d}.mlp_hyper_connection", .{l}, true);
            lay.gate16 = try f16Of(gpa, try tensor(pack, "model.language_model.layers.{d}.mlp.gate.weight", .{l}));
            lay.sgate16 = try f16Of(gpa, try tensor(pack, "model.language_model.layers.{d}.mlp.shared_expert_gate.weight", .{l}));
            if (lay.gate16.len != experts * hidden or lay.sgate16.len != hidden) return error.BadShape;
            var nbuf: [3][160]u8 = undefined;
            const stem = try std.fmt.bufPrint(&nbuf[2], "model.language_model.layers.{d}.mlp.shared_expert", .{l});
            var nb2: [3][200]u8 = undefined;
            lay.shared[0] = try Lin.load(gpu, pack, try std.fmt.bufPrint(&nb2[0], "{s}.gate_proj", .{stem}));
            lay.shared[1] = try Lin.load(gpu, pack, try std.fmt.bufPrint(&nb2[1], "{s}.up_proj", .{stem}));
            lay.shared[2] = try Lin.load(gpu, pack, try std.fmt.bufPrint(&nb2[2], "{s}.down_proj", .{stem}));
            if (Layer.isAttn(l)) {
                lay.attn = try AttnDev.load(gpu, pack, l, e.offset, &e.attn_host_scales[i]);
            } else {
                lay.gdn = try GdnDev.load(gpu, pack, l);
            }
            if (opt.verbose) std.debug.print("  loaded layer {d}\n", .{l});
        }
        e.mixer = try HcW.load(gpa, pack, "model.language_model.hyper_connection_mixer", .{}, false);
        e.embed_table = (pack.find("model.language_model.embed_tokens.weight") orelse return error.MissingTensor).bytes;
        if (opt.head) e.head = try Lin.load(gpu, pack, "lm_head");

        e.table = try cuda.exl3_gdn.Table.alloc(gpu.d, 36, 1);
        e.gdn_scratch = try cuda.exl3_gdn.Scratch.alloc(gpu.d, R);
        const slot = cuda.exl3_attn.slot_bytes;
        e.kv_all = try dalloc(gpu, 12 * cap * 2 * slot);
        e.ik_all = try dalloc(gpu, 12 * cap * 128 * 2);
        e.pooled_all = try dalloc(gpu, 12 * (nb + 2) * 128 * 2);
        e.pos0 = try dalloc(gpu, 4);
        e.x = try dalloc(gpu, R * hidden * 2);
        e.br = try dalloc(gpu, R * hidden * 2);
        e.qp = try dalloc(gpu, R * 12288 * 2);
        e.kp = try dalloc(gpu, R * 512 * 2);
        e.vp = try dalloc(gpu, R * 512 * 2);
        e.ip = try dalloc(gpu, R * 640 * 2);
        e.q_out = try dalloc(gpu, R * 6144 * 2);
        e.iq_out = try dalloc(gpu, R * 512 * 2);
        e.scores = try dalloc(gpu, R * (nb + 1) * 4);
        e.ids_sel = try dalloc(gpu, R * cuda.exl3_attn.ids_stride * 4);
        e.nk = try dalloc(gpu, R * 4);
        e.attn_out = try dalloc(gpu, R * 6144 * 2);
        var ids_host: [topk]i32 = undefined;
        for (&ids_host, 0..) |*d, i| d.* = @intCast(i);
        e.ids = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&ids_host));
        e.lines = try ExpertLines.init(gpu);
        e.y_gate = try dalloc(gpu, topk * inter * 2);
        e.y_up = try dalloc(gpu, topk * inter * 2);
        e.y_glu = try dalloc(gpu, topk * inter * 2);
        e.y_down = try dalloc(gpu, slots * hidden * 2);
        e.wts = try dalloc(gpu, slots * 4);
        e.s_gate = try dalloc(gpu, inter * 2);
        e.s_up = try dalloc(gpu, inter * 2);
        e.s_glu = try dalloc(gpu, inter * 2);
        e.logits = try dalloc(gpu, if (opt.head) R * vocab * 2 else 16);

        if (opt.ngram_path) |p| {
            e.ngram_file = try st.File.open(gpa, gpu.io, p);
            e.ple = try loadPle(gpu, pack, &e.ngram_file.?);
        }
        return e;
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.gpu.gpa;
        for (e.layers) |*l| l.deinit(gpa);
        gpa.free(e.layers);
        for (e.attn_host_scales) |s| for (s) |a| gpa.free(a);
        gpa.free(e.attn_host_scales);
        e.mixer.free(gpa);
        if (e.head) |*h| h.free();
        if (e.ple) |*p| p.deinit();
        if (e.ngram_file) |*f| f.close(e.gpu.io);
        e.table.free();
        e.gdn_scratch.free();
        inline for (.{ "kv_all", "ik_all", "pooled_all", "pos0", "x", "br", "qp", "kp", "vp", "ip", "q_out", "iq_out", "scores", "ids_sel", "nk", "attn_out", "ids", "y_gate", "y_up", "y_glu", "y_down", "wts", "s_gate", "s_up", "s_glu", "logits" }) |name| @field(e, name).free();
        e.lines.deinit(gpa);
        e.stream.deinit();
        e.at.unload();
        e.gdn.unload();
        e.ex.unload();
    }

    /// New sequence: zero the GDN state and the position. KV needs no clearing (positions are written before they are read).
    pub fn reset(e: *Engine) !void {
        try e.table.reset(e.stream, 0);
        try e.stream.synchronize();
        e.pos = 0;
        if (e.ple) |*p| p.reset();
    }

    fn layerAt(e: *Engine, i: usize) *Layer {
        return &e.layers[i];
    }

    // -- one call --------------------------------------------------------------------------------------

    /// Streams after the last loaded layer for `tokens` (rows consecutive from position `e.pos`), into `h` [rows][10240] f32.
    pub fn forward(e: *Engine, tokens: []const u32, h: []f32) !void {
        const rows = tokens.len;
        const gpa = e.gpu.gpa;
        if (rows == 0 or rows > e.opt.max_rows or e.pos + rows > e.opt.cap or h.len != rows * wide) return error.Shape;
        const x16 = try gpa.alloc(f16, rows * hidden);
        defer gpa.free(x16);
        const br16 = try gpa.alloc(f16, rows * hidden);
        defer gpa.free(br16);
        const mixed = try gpa.alloc(f32, hidden);
        defer gpa.free(mixed);
        const branch = try gpa.alloc(f32, hidden);
        defer gpa.free(branch);
        const inject = try gpa.alloc(f32, rows * 4);
        defer gpa.free(inject);
        const scratch = try gpa.alloc(f32, hc.scratchLen(hc.real));
        defer gpa.free(scratch);

        // embedding rows -> expand x4
        for (tokens, 0..) |tk, t| {
            const row = try embed.gather(e.embed_table, tk);
            var r32: [hidden]f32 = undefined;
            for (&r32, 0..) |*d, i| d.* = bf16(std.mem.readInt(u16, row[i * 2 ..][0..2], .little));
            try hc.expand(hc.real, &r32, h[t * wide ..][0..wide]);
        }

        for (e.layers) |*lay| {
            const l = lay.index;
            if (e.ple != null and l == ple_layer) try e.applyPle(tokens, h);

            // attention / GDN block
            for (0..rows) |t| {
                try hc.pre(hc.real, lay.attn_hc.weights(), e.offset, eps, h[t * wide ..][0..wide], mixed, inject[t * 4 .. t * 4 + 4], scratch);
                for (x16[t * hidden ..][0..hidden], mixed) |*d, s| d.* = @floatCast(s);
            }
            try e.x.upload(0, std.mem.sliceAsBytes(x16));
            if (lay.gdn) |*g| {
                const idx = Layer.gdnIndex(l);
                try e.gdn.forward(&e.ex, e.stream, g.layer(), try e.table.convPtr(0, idx), try e.table.recPtr(0, idx), e.x.ptr, rows, e.gdn_scratch, e.br.ptr);
            } else {
                try e.attnRun(&lay.attn.?, Layer.kvIndex(l), rows);
            }
            try e.stream.synchronize();
            try e.br.download(0, std.mem.sliceAsBytes(br16));
            if (e.trace) |tr| {
                for (tr.attn_branch[0 .. rows * hidden], br16) |*d, s| d.* = @floatCast(s);
            }
            for (0..rows) |t| {
                for (branch, br16[t * hidden ..][0..hidden]) |*d, s| d.* = @floatCast(s);
                const ht = h[t * wide ..][0..wide];
                try hc.post(hc.real, ht, branch, inject[t * 4 .. t * 4 + 4], ht);
            }
            if (e.trace) |tr| for (tr.h_mid[0 .. rows * wide], h[0 .. rows * wide]) |*d, s| {
                d.* = s;
            };

            // MoE block
            for (0..rows) |t| {
                try hc.pre(hc.real, lay.mlp_hc.weights(), e.offset, eps, h[t * wide ..][0..wide], mixed, inject[t * 4 .. t * 4 + 4], scratch);
                for (x16[t * hidden ..][0..hidden], mixed) |*d, s| d.* = @floatCast(s);
            }
            try e.x.upload(0, std.mem.sliceAsBytes(x16));
            for (0..rows) |t| {
                var picks: [slots]u16 = undefined;
                try e.moeRow(lay, x16[t * hidden ..][0..hidden], e.x.ptr + t * hidden * 2, e.br.ptr + t * hidden * 2, &picks);
                if (e.trace) |tr| @memcpy(tr.picks[t * topk ..][0..topk], picks[0..topk]);
            }
            try e.stream.synchronize();
            try e.br.download(0, std.mem.sliceAsBytes(br16));
            if (e.trace) |tr| {
                for (tr.mlp_branch[0 .. rows * hidden], br16) |*d, s| d.* = @floatCast(s);
            }
            for (0..rows) |t| {
                for (branch, br16[t * hidden ..][0..hidden]) |*d, s| d.* = @floatCast(s);
                const ht = h[t * wide ..][0..wide];
                try hc.post(hc.real, ht, branch, inject[t * 4 .. t * 4 + 4], ht);
            }
            if (e.opt.verbose) std.debug.print("  layer {d} done\n", .{l});
        }
        e.pos += rows;
    }

    fn applyPle(e: *Engine, tokens: []const u32, h: []f32) !void {
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

    fn attnRun(e: *Engine, a: *const AttnDev, kv_index: usize, rows: usize) !void {
        const cap = e.opt.cap;
        const nb = cap / 4;
        const dims: cuda.exl3_attn.Dims = .{ .tokens = cap };
        const s = e.stream;
        const slot = cuda.exl3_attn.slot_bytes;
        const kv_ptr = e.kv_all.ptr + kv_index * cap * 2 * slot;
        const ik_ptr = e.ik_all.ptr + kv_index * cap * 128 * 2;
        const pooled_ptr = e.pooled_all.ptr + kv_index * (nb + 2) * 128 * 2;
        const p0: i32 = @intCast(e.pos);
        try e.pos0.upload(0, std.mem.asBytes(&p0));
        const inv = attn_cpu.invFreq();
        var inv_dev = try cuda.DeviceBuffer.fromHost(e.gpu.d, std.mem.sliceAsBytes(&inv));
        defer inv_dev.free();
        try a.q.run(&e.ex, s, e.x.ptr, e.qp.ptr, rows);
        try a.k.run(&e.ex, s, e.x.ptr, e.kp.ptr, rows);
        try a.v.run(&e.ex, s, e.x.ptr, e.vp.ptr, rows);
        try a.idx.run(&e.ex, s, e.x.ptr, e.ip.ptr, rows);
        try e.at.prep(s, dims, rows, .{
            .qp = e.qp.ptr,
            .kp = e.kp.ptr,
            .vp = e.vp.ptr,
            .ip = e.ip.ptr,
            .q_scale = a.q_scale.ptr,
            .k_scale = a.k_scale.ptr,
            .i_scale = a.iq_scale.ptr,
            .inv_freq = inv_dev.ptr,
            .pos0 = e.pos0.ptr,
            .q_out = e.q_out.ptr,
            .kv = kv_ptr,
            .iq_out = e.iq_out.ptr,
            .ik = ik_ptr,
        });
        try e.at.pool(s, dims, rows, ik_ptr, pooled_ptr, a.ik_scale.ptr, inv_dev.ptr, e.pos0.ptr);
        try e.at.scores(s, rows, e.iq_out.ptr, pooled_ptr, e.scores.ptr, e.pos0.ptr, nb, nb);
        try e.at.select(s, rows, e.scores.ptr, e.ids_sel.ptr, e.nk.ptr, e.pos0.ptr, nb);
        try e.at.decode(s, dims, rows, e.q_out.ptr, e.qp.ptr, kv_ptr, e.ids_sel.ptr, e.attn_out.ptr, e.pos0.ptr);
        try a.o.run(&e.ex, s, e.attn_out.ptr, e.br.ptr, rows);
        try s.synchronize(); // inv_dev is freed on return
    }

    /// One row of the MoE block: host router, the 10 routed experts' lines uploaded, device hit + shared expert + combine.
    fn moeRow(e: *Engine, lay: *Layer, x16: []const f16, x_ptr: u64, out_ptr: u64, picks: *[slots]u16) !void {
        var x32: [hidden]f32 = undefined;
        for (&x32, x16) |*d, s| d.* = @floatCast(s);
        var logits: [experts + 1]f32 = undefined;
        var weights: [slots]f32 = undefined;
        var scratch: [experts]f32 = undefined;
        try router.launchDtype(.f16, 0, std.mem.sliceAsBytes(lay.gate16), &x32, std.mem.sliceAsBytes(lay.sgate16), &x32, topk, &logits, picks, &weights, &scratch);
        e.lines.halves = .{ 0, 0, 0 };
        for (picks[0..topk], 0..) |p, line| try e.lines.put(e.pack, lay.index, p, line);
        try e.wts.upload(0, std.mem.sliceAsBytes(&weights));
        const s = e.stream;
        try e.ex.hit(s, x_ptr, e.ids.ptr, try e.lines.packed_(0), try e.lines.packed_(1), try e.lines.packed_(2), e.y_gate.ptr, e.y_up.ptr, e.y_glu.ptr, e.y_down.ptr, 1, topk, topk);
        const sg = &lay.shared[0];
        const su = &lay.shared[1];
        const sd = &lay.shared[2];
        try sg.run(&e.ex, s, x_ptr, e.s_gate.ptr, 1);
        try su.run(&e.ex, s, x_ptr, e.s_up.ptr, 1);
        try e.ex.swiglu(s, e.s_gate.ptr, e.s_up.ptr, e.s_glu.ptr, 1, inter);
        try sd.run(&e.ex, s, e.s_glu.ptr, e.y_down.ptr + topk * hidden * 2, 1);
        try e.ex.combine(s, e.y_down.ptr, e.wts.ptr, out_ptr, 1, hidden, slots);
        try s.synchronize(); // the next row overwrites the expert lines
    }

    /// Final collapse (hyper_connection_mixer, no inject) and the lm_head for `rows` stream rows. Returns f32 logits [rows][vocab].
    pub fn logitsOf(e: *Engine, h: []const f32, rows: usize, out: []f32) !void {
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
        try e.x.upload(0, std.mem.sliceAsBytes(x16));
        try head.run(&e.ex, e.stream, e.x.ptr, e.logits.ptr, rows);
        try e.stream.synchronize();
        const l16 = try gpa.alloc(f16, rows * vocab);
        defer gpa.free(l16);
        try e.logits.download(0, std.mem.sliceAsBytes(l16));
        for (out, l16) |*d, s| d.* = @floatCast(s);
    }
};

pub fn dalloc(gpu: Gpu, bytes: usize) !cuda.DeviceBuffer {
    var b = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    errdefer b.free();
    try b.fill8(0, null);
    return b;
}

// -- PLE loader ----------------------------------------------------------------------------------------------

pub fn loadPle(gpu: Gpu, pack: *const real.Pack, ng: *st.File) !fple.Ple {
    const gpa = gpu.gpa;
    var p: fple.Ple = undefined;
    p.gpa = gpa;
    const l = ple_layer;
    p.key_proj = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.key_proj.weight", .{l}));
    p.value_proj = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.value_proj.weight", .{l}));
    p.scale_key = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.norm_key.weight", .{l}));
    p.scale_query = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.norm_query.weight", .{l}));
    p.scale_conv = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.norm_conv.weight", .{l}));
    for ([_][]f32{ p.scale_key, p.scale_query, p.scale_conv }) |s| for (s) |*v| {
        v.* += 1.0;
    };
    p.conv_w = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.conv1d.weight", .{l}));
    if (p.key_proj.len != wide * hidden or p.value_proj.len != hidden * hidden or p.conv_w.len != wide * fple.kernel) return error.BadShape;

    const base = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.";
    const bias = ng.get(base ++ "head_bias") orelse return error.MissingTensor;
    p.head_bias = try f16Of(gpa, bias);
    const offs = ng.get(base ++ "head_offsets") orelse return error.MissingTensor;
    const sizes = ng.get(base ++ "head_vocab_sizes") orelse return error.MissingTensor;
    const mults = ng.get(base ++ "layer_multipliers") orelse return error.MissingTensor;
    if (offs.dtype != .i64 or sizes.dtype != .i64 or mults.dtype != .i64 or offs.numel() != fple.heads or mults.numel() != 3) return error.BadShape;
    for (0..fple.heads) |i| {
        p.hash.head_offsets[i] = std.mem.readInt(i64, offs.bytes[i * 8 ..][0..8], .little);
        p.hash.head_sizes[i] = std.mem.readInt(i64, sizes.bytes[i * 8 ..][0..8], .little);
    }
    for (0..3) |i| p.hash.multipliers[i] = std.mem.readInt(i64, mults.bytes[i * 8 ..][0..8], .little);
    p.hash.eos = eos_token;
    const shards = try gpa.alloc([]const u8, 128);
    errdefer gpa.free(shards);
    var rows_per: usize = 0;
    for (shards, 0..) |*s, i| {
        var nb: [160]u8 = undefined;
        const t = ng.get(try std.fmt.bufPrint(&nb, base ++ "shard_{d}.trellis", .{i})) orelse return error.MissingTensor;
        if (t.rank != 2 or t.dim(1) != fple.words_per_row or t.dtype != .i16) return error.BadShape;
        if (i == 0) rows_per = t.dim(0) else if (t.dim(0) != rows_per) return error.NonUniformShards;
        s.* = t.bytes;
    }
    p.shards = shards;
    p.shard_rows = rows_per;
    p.tail = try gpa.alloc(f64, fple.tail_rows * wide);
    p.reset();
    return p;
}
