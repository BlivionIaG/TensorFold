//! The assembled EXL3 forward for Qwen3.8-Flash-Next at tp=1 on one device: embedding rows, hyper-connection streams, the
//! decoder layers (GDN or sparse attention, then the MoE block), the final stream collapse and the 5-bit lm_head.
//!
//! Where things live (tp=1, one 16 GB card):
//!   device, resident: every layer's GDN / attention projections, the shared expert, dense a/b, conv, norms, the lm_head,
//!                     GDN state, attention KV, indexer keys and pooled blocks; with `device_hc` also the hyper-connection
//!                     matrices (F16, ~13 MB a layer pair x 48), the router gates and the stream buffer
//!   host:             the 512 routed experts of every layer (the mmapped pack), the embedding table, the PLE block
//!                     (`exl3_fwd_ple` f64 reference, or the `exl3_fwd_plerows` row path), and with `device_hc == false`
//!                     the hyper-connection weights (host f32, `exl3_hc`) and the router gates (`exl3_router`)
//! Routed experts, two ways (`Options.pool_lines`):
//!   0           on demand: the 10 selected experts' lines are strips-reordered and uploaded synchronously per row per layer
//!   N >= 10     a per-layer device pool of N lines (`cuda.exl3_offload.Pool`), hot set by `exl3_offload` (lifetime count, then
//!               LRU). A miss is a synchronous staged H2D into the evicted line on the pool's copy stream, then the GPU runs the
//!               row: a miss never goes to the CPU GEMV.
//! `device_hc` moves expand, hyper-connection pre/post, the collapse and the router dots to the GPU (`exl3_hc.cu`). The router
//! top-10 stays on the host (513 logits down per row of a layer, a 513-element select): the pool policy needs the ids on the
//! host anyway. Per layer that is one stream sync; nothing else crosses PCIe except missed expert lines.
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
const plerows = @import("exl3_fwd_plerows.zig");
const policy = @import("exl3_offload.zig");
const route = @import("exl3_route.zig");
const gr = @import("exl3_fwd_graph.zig");

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

    /// No host copy (the device holds the weights).
    pub const empty: HcW = .{ .norm = &.{}, .down = &.{}, .up = &.{}, .inject = null };
};

/// A hyper-connection's weights on the device, F16 exactly as shipped.
pub const HcDev = struct {
    norm: cuda.DeviceBuffer,
    down: cuda.DeviceBuffer,
    up: cuda.DeviceBuffer,
    inject: ?cuda.DeviceBuffer,

    pub fn load(gpu: Gpu, pack: *const real.Pack, comptime base: []const u8, args: anytype, with_inject: bool) !HcDev {
        const tn = try tensor(pack, base ++ ".hc_norm.weight", args);
        const td = try tensor(pack, base ++ ".input_mix_weight_down.weight", args);
        const tu = try tensor(pack, base ++ ".input_mix_weight_up.weight", args);
        if (tn.dtype != .f16 or td.dtype != .f16 or tu.dtype != .f16 or tn.bytes.len != wide * 2 or td.bytes.len != 320 * wide * 2 or tu.bytes.len != wide * 320 * 2) return error.BadShape;
        var norm = try devBytes(gpu, tn);
        errdefer norm.free();
        var down = try devBytes(gpu, td);
        errdefer down.free();
        var up = try devBytes(gpu, tu);
        errdefer up.free();
        var inject: ?cuda.DeviceBuffer = null;
        if (with_inject) {
            const tj = try tensor(pack, base ++ ".block_inject_weight.weight", args);
            if (tj.dtype != .f16 or tj.bytes.len != 4 * wide * 2) return error.BadShape;
            inject = try devBytes(gpu, tj);
        }
        return .{ .norm = norm, .down = down, .up = up, .inject = inject };
    }

    pub fn free(h: *HcDev) void {
        h.norm.free();
        h.down.free();
        h.up.free();
        if (h.inject) |*j| j.free();
    }

    pub fn weights(h: *const HcDev) cuda.exl3_hc.Weights {
        return .{ .norm = h.norm.ptr, .down = h.down.ptr, .up = h.up.ptr, .inject = if (h.inject) |j| j.ptr else null };
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
    return dupload(gpu, t.bytes);
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
        var a = try devBytes(gpu, ta);
        errdefer a.free();
        var b = try devBytes(gpu, tb);
        errdefer b.free();
        var conv = try devBytes(gpu, tc);
        errdefer conv.free();
        var alog = try devBytes(gpu, tl);
        errdefer alog.free();
        var dt = try devBytes(gpu, td);
        errdefer dt.free();
        const norm = try devBytes(gpu, tn);
        return .{ .qkv = qkv, .z = z, .out = out, .a = a, .b = b, .conv = conv, .alog = alog, .dt = dt, .norm = norm };
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


/// The attention norms (`q_norm`, `k_norm`, `indexer.q_layernorm`, `indexer.k_layernorm`) are all `CenteredRMSNorm` in the model
/// (qwen4_exp model_layers.py: `Attention.q_norm/k_norm` and `Indexer.q_layernorm/k_layernorm`). The reference loader
/// (model.py `load`) decides the storage convention ONCE, from the 48 attention hc_norm means (`norms_stored_around_one`), and
/// applies it to every CenteredRMSNorm. `AttnDev.load` therefore adds the engine's `offset` (that one decision, 1 for the
/// gamma-1 storage of this pack) to all four; there is no per-tensor range test. `checkIndexerConvention` is the independent
/// cross-check on the indexer norms' own means.
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

    fn scale(gpa: std.mem.Allocator, pack: *const real.Pack, stem: []const u8, name: []const u8, offset: f32) ![]f32 {
        var buf: [160]u8 = undefined;
        const t = pack.find(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, name })) orelse return error.MissingTensor;
        const w = try f32Of(gpa, t);
        for (w) |*v| v.* += offset;
        return w;
    }

    /// Returns the weights and, for the oracle, the four host scale arrays (owned by the caller; freed here on error).
    fn load(gpu: Gpu, pack: *const real.Pack, l: usize, offset: f32, host: *[4][]f32) !AttnDev {
        var sb: [128]u8 = undefined;
        return loadStem(gpu, pack, try std.fmt.bufPrint(&sb, "model.language_model.layers.{d}.self_attn", .{l}), offset, host);
    }

    /// The same for the attention block whose tensors are `<stem>.q_proj` ... (the MTP head's is `mtp.layers.0.self_attn`).
    pub fn loadStem(gpu: Gpu, pack: *const real.Pack, stem: []const u8, offset: f32, host: *[4][]f32) !AttnDev {
        const gpa = gpu.gpa;
        var nb: [5][128]u8 = undefined;
        const l = stem;
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
        try check.expect(q.k == hidden and q.n == 12288 and k.n == 512 and v.n == 512 and o.k == 6144 and o.n == hidden and idx.n == 640, "{s} attention shapes", .{l});
        const names = [4][]const u8{ "q_norm.weight", "k_norm.weight", "indexer.q_layernorm.weight", "indexer.k_layernorm.weight" };
        var made: usize = 0;
        errdefer for (host[0..made]) |a| gpa.free(a);
        for (names, 0..) |nm, i| {
            host[i] = try scale(gpa, pack, stem, nm, offset);
            made += 1;
        }
        var sc: [4]cuda.DeviceBuffer = undefined;
        var up: usize = 0;
        errdefer for (sc[0..up]) |*b| b.free();
        for (host, &sc) |h, *d| {
            d.* = try dupload(gpu, std.mem.sliceAsBytes(h));
            up += 1;
        }
        return .{ .q = q, .k = k, .v = v, .o = o, .idx = idx, .q_scale = sc[0], .k_scale = sc[1], .iq_scale = sc[2], .ik_scale = sc[3] };
    }

    pub fn free(a: *AttnDev) void {
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

/// Cross-check of the one storage decision on the indexer norms' own means (24 values over the 12 attention layers):
/// `exl3_scan.centredOffset` over them must give the engine's offset. Not applied per tensor.
fn checkIndexerConvention(gpa: std.mem.Allocator, pack: *const real.Pack, first: usize, n: usize, offset: f32) !void {
    var means: [64]f32 = undefined;
    var cnt: usize = 0;
    for (first..first + n) |l| {
        if (!Layer.isAttn(l)) continue;
        for ([_][]const u8{ "indexer.q_layernorm.weight", "indexer.k_layernorm.weight" }) |nm| {
            var buf: [160]u8 = undefined;
            const t = pack.find(try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.self_attn.{s}", .{ l, nm })) orelse return error.MissingTensor;
            const w = try f32Of(gpa, t);
            defer gpa.free(w);
            var s: f64 = 0;
            for (w) |v| s += v;
            means[cnt] = @floatCast(s / @as(f64, @floatFromInt(w.len)));
            cnt += 1;
        }
    }
    if (cnt < 8) return; // too few layers loaded for the scan to mean anything (single-layer test engines)
    const got = scan.centredOffset(means[0..cnt]) catch return error.NormConventionUnclear;
    if (got != offset) return error.NormConventionMismatch;
}

pub const Layer = struct {
    index: usize,
    attn_hc: HcW,
    mlp_hc: HcW,
    hc_dev: ?[2]HcDev = null,
    gate_dev: ?cuda.DeviceBuffer = null,
    sgate_dev: ?cuda.DeviceBuffer = null,
    gate16: []f16,
    sgate16: []f16,
    shared: [3]Lin,
    gdn: ?GdnDev = null,
    attn: ?AttnDev = null,
    ep: ?ExpertPool = null,
    /// Tensor-name stem of a layer outside `model.language_model.layers.N` (the MTP head: `mtp.layers.0`); its routed experts are
    /// `<stem>.mlp.experts.E.*`. Null for the 48 main layers.
    stem: ?[]const u8 = null,
    /// A layer that is not one of the 48 keeps its own expert heat table (policy layer tag 0) instead of `Engine.counts`.
    own_heat: ?*[layers_total][experts]u32 = null,

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

    fn load(gpu: Gpu, pack: *const real.Pack, l: usize, offset: f32, device_hc: bool, host_scales: *[4][]f32) !Layer {
        const gpa = gpu.gpa;
        var lay: Layer = undefined;
        lay.index = l;
        lay.gdn = null;
        lay.attn = null;
        lay.ep = null;
        lay.stem = null;
        lay.own_heat = null;
        lay.hc_dev = null;
        lay.gate_dev = null;
        lay.sgate_dev = null;
        if (device_hc) {
            lay.attn_hc = HcW.empty;
            lay.mlp_hc = HcW.empty;
        } else {
            lay.attn_hc = try HcW.load(gpa, pack, "model.language_model.layers.{d}.attn_hyper_connection", .{l}, true);
            errdefer lay.attn_hc.free(gpa);
            lay.mlp_hc = try HcW.load(gpa, pack, "model.language_model.layers.{d}.mlp_hyper_connection", .{l}, true);
        }
        errdefer if (!device_hc) {
            lay.attn_hc.free(gpa);
            lay.mlp_hc.free(gpa);
        };
        lay.gate16 = try f16Of(gpa, try tensor(pack, "model.language_model.layers.{d}.mlp.gate.weight", .{l}));
        errdefer gpa.free(lay.gate16);
        lay.sgate16 = try f16Of(gpa, try tensor(pack, "model.language_model.layers.{d}.mlp.shared_expert_gate.weight", .{l}));
        errdefer gpa.free(lay.sgate16);
        if (lay.gate16.len != experts * hidden or lay.sgate16.len != hidden) return error.BadShape;
        var stem_buf: [160]u8 = undefined;
        const stem = try std.fmt.bufPrint(&stem_buf, "model.language_model.layers.{d}.mlp.shared_expert", .{l});
        var nb2: [3][200]u8 = undefined;
        var made: usize = 0;
        errdefer for (lay.shared[0..made]) |*s| s.free();
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |pn, i| {
            lay.shared[i] = try Lin.load(gpu, pack, try std.fmt.bufPrint(&nb2[i], "{s}." ++ pn, .{stem}));
            made += 1;
        }
        if (device_hc) {
            var hcs: [2]HcDev = undefined;
            hcs[0] = try HcDev.load(gpu, pack, "model.language_model.layers.{d}.attn_hyper_connection", .{l}, true);
            errdefer hcs[0].free();
            hcs[1] = try HcDev.load(gpu, pack, "model.language_model.layers.{d}.mlp_hyper_connection", .{l}, true);
            errdefer hcs[1].free();
            lay.hc_dev = hcs;
            const gt = try tensor(pack, "model.language_model.layers.{d}.mlp.gate.weight", .{l});
            const sg = try tensor(pack, "model.language_model.layers.{d}.mlp.shared_expert_gate.weight", .{l});
            if (gt.dtype != .f16 or sg.dtype != .f16) return error.WrongDtype;
            var g = try devBytes(gpu, gt);
            errdefer g.free();
            lay.gate_dev = g;
            lay.sgate_dev = try devBytes(gpu, sg);
        }
        errdefer if (lay.hc_dev) |*h| {
            h[0].free();
            h[1].free();
            if (lay.gate_dev) |*g| g.free();
            if (lay.sgate_dev) |*g| g.free();
        };
        if (Layer.isAttn(l)) {
            lay.attn = try AttnDev.load(gpu, pack, l, offset, host_scales);
        } else {
            lay.gdn = try GdnDev.load(gpu, pack, l);
        }
        return lay;
    }

    pub fn deinit(l: *Layer, gpa: std.mem.Allocator) void {
        if (l.hc_dev) |*h| {
            h[0].free();
            h[1].free();
        } else {
            l.attn_hc.free(gpa);
            l.mlp_hc.free(gpa);
        }
        if (l.gate_dev) |*g| g.free();
        if (l.sgate_dev) |*g| g.free();
        gpa.free(l.gate16);
        gpa.free(l.sgate16);
        for (&l.shared) |*s| s.free();
        if (l.gdn) |*g| g.free();
        if (l.attn) |*a| a.free();
        if (l.ep) |*p| p.deinit(gpa);
        if (l.own_heat) |c| gpa.destroy(c);
    }
};

// -- routed expert lines on the device ---------------------------------------------------------------------

const projs = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };

/// Tensor `<field>` of routed expert `id`'s projection `proj` in `layer` (or in the layer outside the 48 named by `stem`).
fn expertTensor(pack: *const real.Pack, layer: usize, stem: ?[]const u8, id: usize, proj: []const u8, field: []const u8) !st.Tensor {
    var buf: [200]u8 = undefined;
    const name = if (stem) |sm|
        try std.fmt.bufPrint(&buf, "{s}.mlp.experts.{d}.{s}.{s}", .{ sm, id, proj, field })
    else
        try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.{s}", .{ layer, id, proj, field });
    return pack.find(name) orelse {
        std.debug.print("missing tensor {s}\n", .{name});
        return error.MissingTensor;
    };
}

/// On-demand lines: 10 device lines per projection, refilled synchronously for every row (`pool_lines == 0`).
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
            errdefer r.tr[made].free();
            r.su[made] = try cuda.DeviceBuffer.alloc(gpu.d, topk * r.k[made] * 2);
            errdefer r.su[made].free();
            r.sv[made] = try cuda.DeviceBuffer.alloc(gpu.d, topk * r.n[made] * 2);
        }
        errdefer for (0..3) |i| {
            r.tr[i].free();
            r.su[i].free();
            r.sv[i].free();
        };
        const big = try cuda.exl3.tileSpan(hidden, inter, max_halves);
        r.stage_in = try gpu.gpa.alloc(i16, big);
        errdefer gpu.gpa.free(r.stage_in);
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
        return r.putStem(pack, layer, null, e, line);
    }

    fn putStem(r: *ExpertLines, pack: *const real.Pack, layer: usize, stem: ?[]const u8, e: usize, line: usize) !void {
        for (0..3) |p| {
            const tr = try expertTensor(pack, layer, stem, e, projs[p], "trellis");
            const su = try expertTensor(pack, layer, stem, e, projs[p], "suh");
            const sv = try expertTensor(pack, layer, stem, e, projs[p], "svh");
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

/// One layer's hot set: `lines` device lines (`cuda.exl3_offload.Pool`, one pinned staging line each) and the host policy tables
/// (`exl3_offload.Vram`) that say which (layer, expert) each line holds. All 512 experts stay in the mapped pack.
const ExpertPool = struct {
    pool: cuda.exl3_offload.Pool,
    slots: []u16,
    line: []policy.Line,
    stamp: []u32,
    tag: []u8,
    vram: policy.Vram,
    /// 2-byte-aligned copy of one trellis (the mapping is only byte aligned).
    aligned: []i16,

    const Shapes = struct { spec: cuda.exl3_offload.Spec, biggest: usize };

    /// Expert 0's shapes fix the layer's line layout; `fill` refuses an expert that differs.
    fn shapesOf(pack: *const real.Pack, layer: usize, stem: ?[]const u8) !Shapes {
        var mats: [3]cuda.exl3_offload.Mat = undefined;
        var biggest: usize = 0;
        for (projs, &mats) |p, *m| {
            const tr = try expertTensor(pack, layer, stem, 0, p, "trellis");
            if (tr.rank != 3) return error.BadShape;
            m.* = .{ .k = tr.dim(0) * 16, .n = tr.dim(1) * 16, .halves = @intCast(tr.dim(2) / 8) };
            biggest = @max(biggest, tr.bytes.len / 2);
        }
        return .{ .spec = .{ .gate = mats[0], .up = mats[1], .down = mats[2] }, .biggest = biggest };
    }

    fn init(gpu: Gpu, pack: *const real.Pack, layer: usize, stem: ?[]const u8, lines: usize) !ExpertPool {
        const gpa = gpu.gpa;
        const sh = try shapesOf(pack, layer, stem);
        var pool = try cuda.exl3_offload.Pool.init(gpa, gpu.d, sh.spec, lines);
        errdefer pool.deinit();
        const slot_tab = try gpa.alloc(u16, lines);
        errdefer gpa.free(slot_tab);
        const line = try gpa.alloc(policy.Line, lines);
        errdefer gpa.free(line);
        const stamp = try gpa.alloc(u32, lines);
        errdefer gpa.free(stamp);
        const tag = try gpa.alloc(u8, lines);
        errdefer gpa.free(tag);
        const aligned = try gpa.alloc(i16, sh.biggest);
        return .{ .pool = pool, .slots = slot_tab, .line = line, .stamp = stamp, .tag = tag, .vram = policy.Vram.init(slot_tab, line, stamp, tag), .aligned = aligned };
    }

    fn deinit(p: *ExpertPool, gpa: std.mem.Allocator) void {
        p.pool.deinit();
        gpa.free(p.slots);
        gpa.free(p.line);
        gpa.free(p.stamp);
        gpa.free(p.tag);
        gpa.free(p.aligned);
    }

    /// Expert `id` of `layer` into one staged line (strips order), straight from the mapping.
    fn fill(p: *ExpertPool, pack: *const real.Pack, layer: usize, stem: ?[]const u8, id: usize, dst: []u8) !void {
        const sp = p.pool.spec;
        const mats = [3]cuda.exl3_offload.Mat{ sp.gate, sp.up, sp.down };
        for (projs, mats, p.pool.layout.regions) |pn, m, r| {
            const tr = try expertTensor(pack, layer, stem, id, pn, "trellis");
            const su = try expertTensor(pack, layer, stem, id, pn, "suh");
            const sv = try expertTensor(pack, layer, stem, id, pn, "svh");
            const words = @as(usize, m.halves) * 8;
            if (tr.rank != 3 or tr.dim(0) * 16 != m.k or tr.dim(1) * 16 != m.n or tr.dim(2) != words) return error.NonUniformExperts;
            if (su.bytes.len != r.suh_bytes or sv.bytes.len != r.svh_bytes) return error.BadShape;
            const n = tr.bytes.len / 2;
            @memcpy(std.mem.sliceAsBytes(p.aligned[0..n]), tr.bytes);
            const out: []i16 = @as([*]i16, @ptrCast(@alignCast(dst[r.trellis_off..].ptr)))[0..n];
            try strips.reorder(p.aligned[0..n], tr.dim(0), tr.dim(1), words, out);
            @memcpy(dst[r.suh_off..][0..r.suh_bytes], su.bytes);
            @memcpy(dst[r.svh_off..][0..r.svh_bytes], sv.bytes);
        }
    }
};

// -- engine --------------------------------------------------------------------------------------------------

pub const PleImpl = enum { f64_reference, rows };

pub const Options = struct {
    first_layer: usize = 0,
    layers: usize = layers_total,
    max_rows: usize = 16,
    cap: usize = 128,
    head: bool = true,
    /// Path of ngram_embedding.safetensors; null leaves the PLE block out (reported by the caller).
    ngram_path: ?[]const u8 = null,
    verbose: bool = false,
    /// 0: on-demand expert lines. N >= 10: a per-layer device pool of N lines (see the file header).
    pool_lines: usize = 0,
    /// Hyper-connection, expand/collapse and the router dots on the GPU.
    device_hc: bool = false,
    /// `f64_reference` is S5's `exl3_fwd_ple.Ple` (f64, the cross-check); `rows` is the S6b row path (default).
    ple_impl: PleImpl = .rows,
    /// Rows come from the mapping (`page_cache`) or from exact preads behind the 4096-row hot cache (`ssd`).
    ple_mode: plerows.Mode = .page_cache,
    /// Device memory kept free after the pools are sized (CUDA context growth, graph memory, other users' spikes).
    vram_reserve: usize = 512 << 20,
    /// Sync after each phase of a device-HC layer and add its wall time to `Engine.prof` (a diagnostic; slows the step).
    profile: bool = false,
    /// Extra attention KV / indexer / pooled-block slots after the 12 (slot 12 is the MTP head's); 0 for the plain model.
    extra_kv: usize = 0,
    /// Decode-step CUDA graphs (exl3_fwd_graph.zig): per-layer segments captured once and replayed. Needs device_hc and a pool.
    graph: gr.Mode = .off,
};

pub const Trace = fwd.Trace;

pub const Phase = enum { hc_mixer, post_attn, hc_gate, route_wait, moe, post_moe };

pub const Stats = struct {
    hits: u64 = 0,
    misses: u64 = 0,
    rows: u64 = 0,
    fill_ns: u64 = 0,
};

/// All fixed-size device scratch. Field names are the buffers' roles; `bytes` sizes them.
const Bufs = struct {
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
    ids: cuda.DeviceBuffer, // int32 0..9 (on-demand lines)
    ids_all: cuda.DeviceBuffer, // int32 [R][10] pool line indices, one slot a row
    y_gate: cuda.DeviceBuffer,
    y_up: cuda.DeviceBuffer,
    y_glu: cuda.DeviceBuffer,
    y_down: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer, // f32 [R][11]
    s_gate: cuda.DeviceBuffer,
    s_up: cuda.DeviceBuffer,
    s_glu: cuda.DeviceBuffer,
    logits: cuda.DeviceBuffer,
    h: cuda.DeviceBuffer, // f32 [R][10240] streams (device_hc)
    inj: cuda.DeviceBuffer, // f32 [R][4] (device_hc)
    gate_logits: cuda.DeviceBuffer, // f32 [R][513] (device_hc)
    emb: cuda.DeviceBuffer, // bf16 [R][2560] embedding rows (device_hc)

    fn bytes(comptime name: []const u8, opt: Options) usize {
        const R = opt.max_rows;
        const nb = opt.cap / 4;
        const slot = cuda.exl3_attn.slot_bytes;
        const dev = opt.device_hc;
        if (comptime (std.mem.eql(u8, name, "kv_all"))) return (12 + opt.extra_kv) * opt.cap * 2 * slot;
        if (comptime (std.mem.eql(u8, name, "ik_all"))) return (12 + opt.extra_kv) * opt.cap * 128 * 2;
        if (comptime (std.mem.eql(u8, name, "pooled_all"))) return (12 + opt.extra_kv) * (nb + 2) * 128 * 2;
        if (comptime (std.mem.eql(u8, name, "pos0"))) return 4;
        if (comptime (std.mem.eql(u8, name, "x") or std.mem.eql(u8, name, "br"))) return R * hidden * 2;
        if (comptime (std.mem.eql(u8, name, "qp"))) return R * 12288 * 2;
        if (comptime (std.mem.eql(u8, name, "kp") or std.mem.eql(u8, name, "vp"))) return R * 512 * 2;
        if (comptime (std.mem.eql(u8, name, "ip"))) return R * 640 * 2;
        if (comptime (std.mem.eql(u8, name, "q_out") or std.mem.eql(u8, name, "attn_out"))) return R * 6144 * 2;
        if (comptime (std.mem.eql(u8, name, "iq_out"))) return R * 512 * 2;
        if (comptime (std.mem.eql(u8, name, "scores"))) return R * (nb + 1) * 4;
        if (comptime (std.mem.eql(u8, name, "ids_sel"))) return R * cuda.exl3_attn.ids_stride * 4;
        if (comptime (std.mem.eql(u8, name, "nk"))) return R * 4;
        if (comptime (std.mem.eql(u8, name, "ids"))) return topk * 4;
        if (comptime (std.mem.eql(u8, name, "ids_all"))) return R * topk * 4;
        if (comptime (std.mem.eql(u8, name, "y_gate") or std.mem.eql(u8, name, "y_up") or std.mem.eql(u8, name, "y_glu"))) return topk * inter * 2;
        if (comptime (std.mem.eql(u8, name, "y_down"))) return slots * hidden * 2;
        if (comptime (std.mem.eql(u8, name, "wts"))) return R * slots * 4;
        if (comptime (std.mem.eql(u8, name, "s_gate") or std.mem.eql(u8, name, "s_up") or std.mem.eql(u8, name, "s_glu"))) return inter * 2;
        if (comptime (std.mem.eql(u8, name, "logits"))) return if (opt.head) R * vocab * 2 else 16;
        if (comptime (std.mem.eql(u8, name, "h"))) return if (dev) R * wide * 4 else 16;
        if (comptime (std.mem.eql(u8, name, "inj"))) return if (dev) R * 4 * 4 else 16;
        if (comptime (std.mem.eql(u8, name, "gate_logits"))) return if (dev) R * cuda.exl3_hc.gate_rows * 4 else 16;
        if (comptime (std.mem.eql(u8, name, "emb"))) return if (dev) R * hidden * 2 else 16;
        @compileError("unsized buffer " ++ name);
    }

    fn alloc(gpu: Gpu, opt: Options) !Bufs {
        var b: Bufs = undefined;
        var made: usize = 0;
        const names = @typeInfo(Bufs).@"struct".field_names;
        errdefer inline for (names, 0..) |name, i| {
            if (i < made) @field(b, name).free();
        };
        inline for (names) |name| {
            @field(b, name) = try dalloc(gpu, bytes(name, opt));
            made += 1;
        }
        return b;
    }

    fn free(b: *Bufs) void {
        inline for (@typeInfo(Bufs).@"struct".field_names) |name| @field(b, name).free();
    }
};

pub const PleWeights = struct {
    key: []f32,
    value: []f32,
    norms: []f32, // key | query | conv, each [wide], already 1 + stored
    conv: []f32,

    pub fn free(w: PleWeights, gpa: std.mem.Allocator) void {
        gpa.free(w.key);
        gpa.free(w.value);
        gpa.free(w.norms);
        gpa.free(w.conv);
    }

    pub fn weights(w: PleWeights) @import("exl3_ple_cpu.zig").Weights {
        return .{ .key = w.key, .value = w.value, .norm_key = w.norms[0..wide], .norm_query = w.norms[wide..][0..wide], .norm_conv = w.norms[2 * wide ..], .conv = w.conv };
    }
};

pub const Engine = struct {
    gpu: Gpu,
    pack: *const real.Pack,
    opt: Options,
    ex: cuda.exl3.Exl3,
    gdn: cuda.exl3_gdn.Gdn,
    at: cuda.exl3_attn.Attn,
    hcg: cuda.exl3_hc.Hc,
    stream: cuda.Stream,
    done: cuda.Event,
    have_done: bool = false,
    /// The one storage decision for every CenteredRMSNorm (1: gamma - 1), from the attention hc_norm means.
    offset: f32,
    layers: []Layer,
    attn_host_scales: [][4][]f32,
    mixer: HcW,
    mixer_dev: ?HcDev,
    embed_table: []const u8,
    head: ?Lin,
    ple: ?fple.Ple,
    ple_rows: ?*plerows.Rows,
    ple_w: ?PleWeights,
    ngram_file: ?st.File,
    pos: usize = 0,
    trace: ?Trace = null,
    stats: Stats = .{},

    // device state and scratch
    table: cuda.exl3_gdn.Table,
    gdn_scratch: cuda.exl3_gdn.Scratch,
    hc_scratch: cuda.exl3_hc.Scratch,
    inv_dev: cuda.DeviceBuffer,
    b: Bufs,
    lines: ExpertLines,
    /// lifetime expert use per (layer, expert), the pool policy's heat
    counts: *[layers_total][experts]u32,
    now: u32 = 0,
    pool_bytes: usize = 0,
    prof: [6]u64 = @splat(0),
    prof_t: i96 = 0,
    /// When set, every routed row appends its 10 picks (sorted) here, in call / layer / row order; `pick_n` counts rows.
    pick_buf: ?[]u16 = null,
    pick_n: usize = 0,
    /// Router logit gap between the 10th pick and the best unpicked expert, per routed row (parallel to `pick_buf`).
    gap_buf: ?[]f32 = null,
    last_gap: f32 = 0,
    graph: ?gr.State = null,

    fn initCore(gpu: Gpu, pack: *const real.Pack, opt: Options) !Engine {
        const gpa = gpu.gpa;
        if (opt.pool_lines != 0 and opt.pool_lines < topk) return error.PoolTooSmall;
        var ex = try cuda.exl3.Exl3.load(gpu.d);
        errdefer ex.unload();
        var gdn = try cuda.exl3_gdn.Gdn.load(gpu.d);
        errdefer gdn.unload();
        var at = try cuda.exl3_attn.Attn.load(gpu.d);
        errdefer at.unload();
        var hcg = try cuda.exl3_hc.Hc.load(gpu.d);
        errdefer hcg.unload();
        var stream = try cuda.Stream.init(gpu.d, true);
        errdefer stream.deinit();
        var done = try cuda.Event.init(gpu.d, false);
        errdefer done.deinit();

        // The storage convention of every CenteredRMSNorm, decided once as the reference loader does (model.py
        // `norms_stored_around_one`): the median-based locator over the 48 attention hc_norm means.
        var means: [layers_total]f32 = undefined;
        for (&means, 0..) |*m, l| {
            const w = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.attn_hyper_connection.hc_norm.weight", .{l}));
            defer gpa.free(w);
            var s: f64 = 0;
            for (w) |v| s += v;
            m.* = @floatCast(s / @as(f64, @floatFromInt(w.len)));
        }
        const offset = try scan.centredOffset(&means);
        try checkIndexerConvention(gpa, pack, opt.first_layer, opt.layers, offset);

        const n = opt.layers;
        const layers = try gpa.alloc(Layer, n);
        errdefer gpa.free(layers);
        const host_scales = try gpa.alloc([4][]f32, n);
        errdefer gpa.free(host_scales);
        for (host_scales) |*s| s.* = .{ &.{}, &.{}, &.{}, &.{} };
        var built: usize = 0;
        errdefer for (layers[0..built], host_scales[0..built]) |*l, s| {
            l.deinit(gpa);
            for (s) |a| gpa.free(a);
        };
        for (0..n) |i| {
            layers[i] = try Layer.load(gpu, pack, opt.first_layer + i, offset, opt.device_hc, &host_scales[i]);
            built += 1;
            if (opt.verbose) std.debug.print("  loaded layer {d}\n", .{opt.first_layer + i});
        }
        var mixer: HcW = HcW.empty;
        var mixer_dev: ?HcDev = null;
        if (opt.device_hc) {
            mixer_dev = try HcDev.load(gpu, pack, "model.language_model.hyper_connection_mixer", .{}, false);
        } else {
            mixer = try HcW.load(gpa, pack, "model.language_model.hyper_connection_mixer", .{}, false);
        }
        errdefer if (mixer_dev) |*m| m.free() else mixer.free(gpa);
        const embed_table = (pack.find("model.language_model.embed_tokens.weight") orelse return error.MissingTensor).bytes;
        var head: ?Lin = null;
        errdefer if (head) |*h| h.free();
        if (opt.head) head = try Lin.load(gpu, pack, "lm_head");

        var table = try cuda.exl3_gdn.Table.alloc(gpu.d, 36, 1);
        errdefer table.free();
        var gdn_scratch = try cuda.exl3_gdn.Scratch.alloc(gpu.d, opt.max_rows);
        errdefer gdn_scratch.free();
        var hc_scratch = try cuda.exl3_hc.Scratch.alloc(gpu.d, opt.max_rows);
        errdefer hc_scratch.free();
        const inv = attn_cpu.invFreq();
        var inv_dev = try dupload(gpu, std.mem.sliceAsBytes(&inv));
        errdefer inv_dev.free();
        var b = try Bufs.alloc(gpu, opt);
        errdefer b.free();
        var ids_host: [topk]i32 = undefined;
        for (&ids_host, 0..) |*d, i| d.* = @intCast(i);
        try b.ids.upload(0, std.mem.sliceAsBytes(&ids_host));
        var lines = try ExpertLines.init(gpu);
        errdefer lines.deinit(gpa);
        const counts = try gpa.create([layers_total][experts]u32);
        errdefer gpa.destroy(counts);
        counts.* = std.mem.zeroes([layers_total][experts]u32);

        const e: Engine = .{
            .gpu = gpu,
            .pack = pack,
            .opt = opt,
            .ex = ex,
            .gdn = gdn,
            .at = at,
            .hcg = hcg,
            .stream = stream,
            .done = done,
            .offset = offset,
            .layers = layers,
            .attn_host_scales = host_scales,
            .mixer = mixer,
            .mixer_dev = mixer_dev,
            .embed_table = embed_table,
            .head = head,
            .ple = null,
            .ple_rows = null,
            .ple_w = null,
            .ngram_file = null,
            .table = table,
            .gdn_scratch = gdn_scratch,
            .hc_scratch = hc_scratch,
            .inv_dev = inv_dev,
            .b = b,
            .lines = lines,
            .counts = counts,
        };
        return e;
    }

    pub fn init(gpu: Gpu, pack: *const real.Pack, opt: Options) !Engine {
        var e = try initCore(gpu, pack, opt);
        errdefer e.deinit();
        const gpa = gpu.gpa;
        if (opt.pool_lines != 0) try e.allocPools();
        if (opt.graph != .off) {
            if (!opt.device_hc or opt.pool_lines == 0) return error.GraphNeedsPool;
            e.graph = gr.State.init(opt.graph);
        }
        if (opt.ngram_path) |p| {
            e.ngram_file = try st.File.open(gpa, gpu.io, p);
            switch (opt.ple_impl) {
                .f64_reference => e.ple = try loadPleOffset(gpu, pack, &e.ngram_file.?, e.offset),
                .rows => {
                    const w = try loadPleWeights(gpa, pack, e.offset);
                    e.ple_w = w;
                    e.ple_rows = try plerows.Rows.init(gpa, &e.ngram_file.?, opt.ple_mode, w.weights());
                },
            }
        }
        return e;
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.gpu.gpa;
        e.stream.synchronize() catch {};
        if (e.graph) |*g| g.deinit(gpa);
        for (e.layers) |*l| l.deinit(gpa);
        gpa.free(e.layers);
        for (e.attn_host_scales) |s| for (s) |a| gpa.free(a);
        gpa.free(e.attn_host_scales);
        if (e.mixer_dev) |*m| m.free() else e.mixer.free(gpa);
        if (e.head) |*h| h.free();
        if (e.ple) |*p| p.deinit();
        if (e.ple_rows) |r| r.deinit();
        if (e.ple_w) |w| w.free(gpa);
        if (e.ngram_file) |*f| f.close(e.gpu.io);
        e.table.free();
        e.gdn_scratch.free();
        e.hc_scratch.free();
        e.inv_dev.free();
        e.b.free();
        e.lines.deinit(gpa);
        gpa.destroy(e.counts);
        e.done.deinit();
        e.stream.deinit();
        e.at.unload();
        e.gdn.unload();
        e.hcg.unload();
        e.ex.unload();
    }

    /// Sizes and allocates one pool per loaded layer, after everything else is resident, so the free memory is the real budget.
    /// Budget (printed): device = lines x line_bytes x layers (about 1.9 MB a line at 3.05 bpw), pinned host staging the same.
    fn allocPools(e: *Engine) !void {
        const gpa = e.gpu.gpa;
        const lines = e.opt.pool_lines;
        var want: usize = 0;
        for (e.layers) |*l| {
            const sp = try ExpertPool.shapesOf(e.pack, l.index, null);
            want += lines * (try cuda.exl3_offload.Layout.init(sp.spec)).line_bytes;
        }
        const mem = try e.gpu.ctx.memInfo();
        std.debug.print("  pool budget: {d} lines x {d} layers = {d:.0} MiB device (+ same pinned host); device free {d:.0} MiB of {d:.0} MiB, reserve {d:.0} MiB\n", .{
            lines,                                                  e.layers.len,                                     @as(f64, @floatFromInt(want)) / 1048576.0,
            @as(f64, @floatFromInt(mem.free)) / 1048576.0, @as(f64, @floatFromInt(mem.total)) / 1048576.0, @as(f64, @floatFromInt(e.opt.vram_reserve)) / 1048576.0,
        });
        if (want + e.opt.vram_reserve > mem.free) return error.NotEnoughVram;
        var made: usize = 0;
        errdefer for (e.layers[0..made]) |*l| {
            l.ep.?.deinit(gpa);
            l.ep = null;
        };
        for (e.layers) |*l| {
            l.ep = try ExpertPool.init(e.gpu, e.pack, l.index, null, lines);
            made += 1;
            e.pool_bytes += l.ep.?.pool.deviceBytes();
        }
    }

    /// Gives a layer outside the 48 (the MTP head's, `lay.stem` set) its own expert pool of `pool_lines` lines and heat table.
    /// No-op when the engine runs on-demand lines (`pool_lines == 0`).
    pub fn allocExtraPool(e: *Engine, lay: *Layer) !void {
        if (e.opt.pool_lines == 0) return;
        const gpa = e.gpu.gpa;
        const heat = try gpa.create([layers_total][experts]u32);
        errdefer gpa.destroy(heat);
        heat.* = std.mem.zeroes([layers_total][experts]u32);
        const sp = try ExpertPool.shapesOf(e.pack, lay.index, lay.stem);
        const want = e.opt.pool_lines * (try cuda.exl3_offload.Layout.init(sp.spec)).line_bytes;
        const mem = try e.gpu.ctx.memInfo();
        if (want + e.opt.vram_reserve > mem.free) return error.NotEnoughVram;
        lay.ep = try ExpertPool.init(e.gpu, e.pack, lay.index, lay.stem, e.opt.pool_lines);
        lay.own_heat = heat;
        e.pool_bytes += lay.ep.?.pool.deviceBytes();
    }

    /// New sequence: zero the GDN state, the position and the PLE state. KV needs no clearing (positions are written before
    /// they are read). The expert pools and the heat counts are not per sequence and stay.
    pub fn reset(e: *Engine) !void {
        try e.table.reset(e.stream, 0);
        try e.stream.synchronize();
        e.pos = 0;
        if (e.ple) |*p| p.reset();
        if (e.ple_rows) |r| r.reset();
    }

    /// Per-sequence PLE state hooks (the `rows` implementation): copy the 2-token id history and the 9-row conv tail out and back.
    pub fn pleState(e: *Engine) !plerows.State {
        if (e.ple_rows == null) return error.NoPleRows;
        return plerows.State.init(e.gpu.gpa);
    }

    pub fn pleSnapshot(e: *Engine, dst: *plerows.State) !void {
        (e.ple_rows orelse return error.NoPleRows).snapshot(dst);
    }

    pub fn pleRestore(e: *Engine, src: plerows.State) !void {
        (e.ple_rows orelse return error.NoPleRows).restore(src);
    }

    pub fn hasPle(e: *const Engine) bool {
        return e.ple != null or e.ple_rows != null;
    }

    // -- one call --------------------------------------------------------------------------------------

    /// Streams after the last loaded layer for `tokens` (rows consecutive from position `e.pos`), into `h` [rows][10240] f32.
    pub fn forward(e: *Engine, tokens: []const u32, h: []f32) !void {
        const rows = tokens.len;
        if (rows == 0 or rows > e.opt.max_rows or e.pos + rows > e.opt.cap or h.len != rows * wide) return error.Shape;
        const p0: i32 = @intCast(e.pos);
        try e.b.pos0.upload(0, std.mem.asBytes(&p0));
        if (e.opt.device_hc) try e.forwardDev(tokens, h) else try e.forwardHost(tokens, h);
        e.pos += rows;
    }

    fn forwardHost(e: *Engine, tokens: []const u32, h: []f32) !void {
        const rows = tokens.len;
        const gpa = e.gpu.gpa;
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
            if (e.hasPle() and l == ple_layer) try e.applyPle(tokens, h);

            // attention / GDN block
            for (0..rows) |t| {
                try hc.pre(hc.real, lay.attn_hc.weights(), e.offset, eps, h[t * wide ..][0..wide], mixed, inject[t * 4 .. t * 4 + 4], scratch);
                for (x16[t * hidden ..][0..hidden], mixed) |*d, s| d.* = @floatCast(s);
            }
            try e.b.x.upload(0, std.mem.sliceAsBytes(x16));
            try e.mixerRun(lay, rows);
            try e.stream.synchronize();
            try e.b.br.download(0, std.mem.sliceAsBytes(br16));
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
            try e.b.x.upload(0, std.mem.sliceAsBytes(x16));
            for (0..rows) |t| {
                var x32: [hidden]f32 = undefined;
                for (&x32, x16[t * hidden ..][0..hidden]) |*d, s| d.* = @floatCast(s);
                var logits: [experts + 1]f32 = undefined;
                var picks: [slots]u16 = undefined;
                var weights: [slots]f32 = undefined;
                var rscratch: [experts]f32 = undefined;
                try router.launchDtype(.f16, 0, std.mem.sliceAsBytes(lay.gate16), &x32, std.mem.sliceAsBytes(lay.sgate16), &x32, topk, &logits, &picks, &weights, &rscratch);
                e.setGap(&logits, &picks);
                try e.moeCompute(lay, &picks, &weights, t);
                if (e.trace) |tr| @memcpy(tr.picks[t * topk ..][0..topk], picks[0..topk]);
            }
            try e.stream.synchronize();
            try e.b.br.download(0, std.mem.sliceAsBytes(br16));
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
    }

    /// The same layers with the hyper-connection streams, the block inputs and the router dots on the device. Host crossings per
    /// layer: the 513 router logits per row down (one stream sync), 40 bytes of ids and 44 of weights per row up, missed expert
    /// lines up. Layer `ple_layer` also bounces the streams (the PLE block is a host path).
    fn forwardDev(e: *Engine, tokens: []const u32, h: []f32) !void {
        const rows = tokens.len;
        const gpa = e.gpu.gpa;
        const s = e.stream;
        const emb16 = try gpa.alloc(u16, rows * hidden);
        defer gpa.free(emb16);
        const glog = try gpa.alloc(f32, rows * cuda.exl3_hc.gate_rows);
        defer gpa.free(glog);
        const tr_br = try gpa.alloc(f16, rows * hidden);
        defer gpa.free(tr_br);

        for (tokens, 0..) |tk, t| {
            const row = try embed.gather(e.embed_table, tk);
            for (0..hidden) |i| emb16[t * hidden + i] = std.mem.readInt(u16, row[i * 2 ..][0..2], .little);
        }
        try e.b.emb.upload(0, std.mem.sliceAsBytes(emb16));
        try e.hcg.expand(s, e.b.emb.ptr, e.b.h.ptr, rows);

        for (e.layers) |*lay| {
            const l = lay.index;
            const hw = lay.hc_dev.?;
            if (e.hasPle() and l == ple_layer) {
                try s.synchronize();
                try e.b.h.download(0, std.mem.sliceAsBytes(h[0 .. rows * wide]));
                try e.applyPle(tokens, h);
                try e.b.h.upload(0, std.mem.sliceAsBytes(h[0 .. rows * wide]));
            }

            if (gr.applies(e, lay, rows)) {
                try gr.layerStep(e, lay);
                continue;
            }

            // attention / GDN block
            if (e.opt.profile) {
                try s.synchronize();
                e.prof_t = check.now(e.gpu.io);
            }
            try e.hcg.pre(s, hw[0].weights(), e.offset, eps, e.b.h.ptr, e.hc_scratch, e.b.x.ptr, e.b.inj.ptr, rows);
            try e.mixerRun(lay, rows);
            try e.mark(.hc_mixer);
            if (e.trace) |tr| {
                try s.synchronize();
                try e.b.br.download(0, std.mem.sliceAsBytes(tr_br));
                for (tr.attn_branch[0 .. rows * hidden], tr_br) |*d, v| d.* = @floatCast(v);
            }
            try e.hcg.post(s, e.b.h.ptr, e.b.br.ptr, e.b.inj.ptr, rows);
            try e.mark(.post_attn);
            if (e.trace) |tr| {
                try s.synchronize();
                try e.b.h.download(0, std.mem.sliceAsBytes(h[0 .. rows * wide]));
                for (tr.h_mid[0 .. rows * wide], h[0 .. rows * wide]) |*d, v| d.* = v;
            }

            // MoE block: block input, router dots, then the host picks the 10 experts a row
            try e.hcg.pre(s, hw[1].weights(), e.offset, eps, e.b.h.ptr, e.hc_scratch, e.b.x.ptr, e.b.inj.ptr, rows);
            try e.hcg.gate(s, e.b.x.ptr, lay.gate_dev.?.ptr, lay.sgate_dev.?.ptr, e.b.gate_logits.ptr, rows);
            try e.mark(.hc_gate);
            try s.synchronize();
            try e.b.gate_logits.download(0, std.mem.sliceAsBytes(glog));
            try e.mark(.route_wait);
            for (0..rows) |t| {
                var picks: [slots]u16 = undefined;
                var weights: [slots]f32 = undefined;
                var rscratch: [experts]f32 = undefined;
                try route.select(glog[t * cuda.exl3_hc.gate_rows ..][0..cuda.exl3_hc.gate_rows], topk, &picks, &weights, &rscratch);
                e.setGap(glog[t * cuda.exl3_hc.gate_rows ..][0..cuda.exl3_hc.gate_rows], &picks);
                try e.moeCompute(lay, &picks, &weights, t);
                if (e.trace) |tr| @memcpy(tr.picks[t * topk ..][0..topk], picks[0..topk]);
            }
            try e.mark(.moe);
            if (e.trace) |tr| {
                try s.synchronize();
                try e.b.br.download(0, std.mem.sliceAsBytes(tr_br));
                for (tr.mlp_branch[0 .. rows * hidden], tr_br) |*d, v| d.* = @floatCast(v);
            }
            try e.hcg.post(s, e.b.h.ptr, e.b.br.ptr, e.b.inj.ptr, rows);
            try e.mark(.post_moe);
            if (e.opt.verbose) std.debug.print("  layer {d} done\n", .{l});
        }
        try s.synchronize();
        try e.b.h.download(0, std.mem.sliceAsBytes(h[0 .. rows * wide]));
    }

    pub fn setGap(e: *Engine, logits: []const f32, picks: []const u16) void {
        var lo: f32 = std.math.inf(f32);
        var pm: [experts]bool = @splat(false);
        for (picks[0..topk]) |p| {
            pm[p] = true;
            lo = @min(lo, logits[p]);
        }
        var best: f32 = -std.math.inf(f32);
        for (logits[0..experts], 0..) |v, i| if (!pm[i]) {
            best = @max(best, v);
        };
        e.last_gap = lo - best;
    }

    fn mark(e: *Engine, ph: Phase) !void {
        if (!e.opt.profile) return;
        try e.stream.synchronize();
        const t = check.now(e.gpu.io);
        e.prof[@intFromEnum(ph)] += @intCast(t - e.prof_t);
        e.prof_t = t;
    }

    /// The block's mixer (GDN or attention) for `rows` rows of `e.b.x` into `e.b.br`, queued on the stream.
    pub fn mixerRun(e: *Engine, lay: *Layer, rows: usize) !void {
        if (lay.gdn) |*g| {
            const idx = Layer.gdnIndex(lay.index);
            try e.gdn.forward(&e.ex, e.stream, g.layer(), try e.table.convPtr(0, idx), try e.table.recPtr(0, idx), e.b.x.ptr, rows, e.gdn_scratch, e.b.br.ptr);
        } else {
            try e.attnRun(&lay.attn.?, Layer.kvIndex(lay.index), rows);
        }
    }

    fn applyPle(e: *Engine, tokens: []const u32, h: []f32) !void {
        const gpa = e.gpu.gpa;
        const rows = tokens.len;
        if (e.ple_rows) |r| {
            const out = try gpa.alloc(f32, rows * wide);
            defer gpa.free(out);
            try r.rows(tokens, h[0 .. rows * wide], out);
            for (h[0 .. rows * wide], out) |*d, s| d.* += s;
            return;
        }
        const h64 = try gpa.alloc(f64, rows * wide);
        defer gpa.free(h64);
        const out = try gpa.alloc(f64, rows * wide);
        defer gpa.free(out);
        for (h64, h[0 .. rows * wide]) |*d, s| d.* = s;
        try e.ple.?.rows(tokens, h64, out);
        for (h[0 .. rows * wide], out) |*d, s| d.* += @floatCast(s);
    }

    /// Queues the full-attention layer for `rows` rows of `e.b.x` (position `pos0` already on the device).
    pub fn attnRun(e: *Engine, a: *const AttnDev, kv_index: usize, rows: usize) !void {
        const cap = e.opt.cap;
        const nb = cap / 4;
        const dims: cuda.exl3_attn.Dims = .{ .tokens = cap };
        const s = e.stream;
        const slot = cuda.exl3_attn.slot_bytes;
        const bb = &e.b;
        const kv_ptr = bb.kv_all.ptr + kv_index * cap * 2 * slot;
        const ik_ptr = bb.ik_all.ptr + kv_index * cap * 128 * 2;
        const pooled_ptr = bb.pooled_all.ptr + kv_index * (nb + 2) * 128 * 2;
        try a.q.run(&e.ex, s, bb.x.ptr, bb.qp.ptr, rows);
        try a.k.run(&e.ex, s, bb.x.ptr, bb.kp.ptr, rows);
        try a.v.run(&e.ex, s, bb.x.ptr, bb.vp.ptr, rows);
        try a.idx.run(&e.ex, s, bb.x.ptr, bb.ip.ptr, rows);
        try e.at.prep(s, dims, rows, .{
            .qp = bb.qp.ptr,
            .kp = bb.kp.ptr,
            .vp = bb.vp.ptr,
            .ip = bb.ip.ptr,
            .q_scale = a.q_scale.ptr,
            .k_scale = a.k_scale.ptr,
            .i_scale = a.iq_scale.ptr,
            .inv_freq = e.inv_dev.ptr,
            .pos0 = bb.pos0.ptr,
            .q_out = bb.q_out.ptr,
            .kv = kv_ptr,
            .iq_out = bb.iq_out.ptr,
            .ik = ik_ptr,
        });
        try e.at.pool(s, dims, rows, ik_ptr, pooled_ptr, a.ik_scale.ptr, e.inv_dev.ptr, bb.pos0.ptr);
        try e.at.scores(s, rows, bb.iq_out.ptr, pooled_ptr, bb.scores.ptr, bb.pos0.ptr, nb, nb);
        try e.at.select(s, rows, bb.scores.ptr, bb.ids_sel.ptr, bb.nk.ptr, bb.pos0.ptr, nb);
        try e.at.decode(s, dims, rows, bb.q_out.ptr, bb.qp.ptr, kv_ptr, bb.ids_sel.ptr, bb.attn_out.ptr, bb.pos0.ptr);
        try a.o.run(&e.ex, s, bb.attn_out.ptr, bb.br.ptr, rows);
    }

    /// Pool path: make the 10 picks resident in the layer's pool and write their line indices to `ids_out`.
    /// Hits are touched; every miss is a staged H2D into the line the policy evicts (copy stream, after the compute stream's last
    /// use of that line), reaped at once, and the compute stream waits on the copy event before it reads the line. A miss never
    /// takes the CPU path: `cpu` decisions cannot occur with at least 10 lines (every miss finds a victim outside this row's picks).
    pub fn stageRow(e: *Engine, lay: *Layer, ep: *ExpertPool, picks: []const u16, ids_out: *[topk]i32) !void {
        const heat = lay.own_heat orelse e.counts;
        const l: u8 = if (lay.own_heat != null) 0 else @intCast(lay.index);
        try policy.observe(&heat[l], picks);
        var dec: [topk]policy.Decision = undefined;
        var scr: [topk]u16 = undefined;
        const lb: u64 = ep.pool.layout.line_bytes;
        const pol: policy.Policy = .{ .slots = @intCast(ep.pool.lines), .expert_bytes = lb, .copy_bytes = lb * topk };
        e.now += 1;
        try policy.scheduleUsedAt(&ep.vram, l, picks, heat, pol, e.now, &dec, &scr, policy.allOwned());
        var cps: [topk]policy.Copy = undefined;
        const n = try policy.stagedCopies(ep.vram, l, picks, &dec, &cps);
        const t0 = check.now(e.gpu.io);
        for (cps[0..n]) |c| {
            try ep.pool.wait(c.line);
            try ep.fill(e.pack, lay.index, lay.stem, c.id, ep.pool.stagingLine(c.line));
            try ep.pool.enqueue(c.line, if (e.have_done) e.done else null);
            try policy.reapAt(&ep.vram, l, c.id, e.now);
        }
        e.stats.fill_ns += @intCast(check.now(e.gpu.io) - t0);
        e.stats.misses += n;
        e.stats.hits += picks.len - n;
        e.stats.rows += 1;
        for (picks, ids_out) |id, *o| {
            const line = policy.lineOf(ep.vram, l, id) orelse return error.PoolTooSmall;
            if (ep.vram.line[line] != .resident) return error.PoolTooSmall;
            try ep.pool.gate(e.stream, line);
            o.* = @intCast(line);
        }
    }

    fn recordPicks(e: *Engine, picks: *const [slots]u16) void {
        if (e.pick_buf) |pb| {
            var srt: [topk]u16 = picks[0..topk].*;
            std.mem.sort(u16, &srt, {}, std.sort.asc(u16));
            if ((e.pick_n + 1) * topk <= pb.len) @memcpy(pb[e.pick_n * topk ..][0..topk], &srt);
            if (e.gap_buf) |gb| if (e.pick_n < gb.len) {
                gb[e.pick_n] = e.last_gap;
            };
            e.pick_n += 1;
        }
    }

    /// The host half of a one-row pool MoE (`moeCompute` without the launches): picks recorded, weights uploaded, missing lines
    /// staged, line indices written to row 0 of `ids_all`. The graph segment B reads those fixed buffers.
    pub fn moeHost(e: *Engine, lay: *Layer, picks: *const [slots]u16, weights: *const [slots]f32) !void {
        e.recordPicks(picks);
        try e.b.wts.upload(0, std.mem.sliceAsBytes(weights));
        var line_ids: [topk]i32 = undefined;
        try e.stageRow(lay, &lay.ep.?, picks[0..topk], &line_ids);
        try e.b.ids_all.upload(0, std.mem.sliceAsBytes(&line_ids));
    }

    /// One row of the MoE block given its routing: the 10 routed experts (pool lines or on-demand upload), the shared expert
    /// and the weighted combine, into row `t` of `e.b.br`. Reads row `t` of `e.b.x`. On-demand rows sync (their lines are
    /// reused by the next row); pool rows only record the compute event the next eviction waits on.
    pub fn moeCompute(e: *Engine, lay: *Layer, picks: *const [slots]u16, weights: *const [slots]f32, t: usize) !void {
        const s = e.stream;
        const bb = &e.b;
        const x_ptr = bb.x.ptr + t * hidden * 2;
        const out_ptr = bb.br.ptr + t * hidden * 2;
        e.recordPicks(picks);
        try bb.wts.upload(t * slots * 4, std.mem.sliceAsBytes(weights));
        var ids_ptr = bb.ids.ptr;
        var n_lines: usize = topk;
        var pk: [3]cuda.exl3.Packed = undefined;
        if (lay.ep) |*ep| {
            var line_ids: [topk]i32 = undefined;
            try e.stageRow(lay, ep, picks[0..topk], &line_ids);
            try bb.ids_all.upload(t * topk * 4, std.mem.sliceAsBytes(&line_ids));
            ids_ptr = bb.ids_all.ptr + t * topk * 4;
            n_lines = ep.pool.lines;
            for (&pk, 0..) |*p, i| p.* = try ep.pool.packed_(i);
        } else {
            e.lines.halves = .{ 0, 0, 0 };
            for (picks[0..topk], 0..) |p, line| try e.lines.putStem(e.pack, lay.index, lay.stem, p, line);
            for (&pk, 0..) |*p, i| p.* = try e.lines.packed_(i);
        }
        try e.ex.hit(s, x_ptr, ids_ptr, pk[0], pk[1], pk[2], bb.y_gate.ptr, bb.y_up.ptr, bb.y_glu.ptr, bb.y_down.ptr, 1, n_lines, topk);
        const sg = &lay.shared[0];
        const su = &lay.shared[1];
        const sd = &lay.shared[2];
        try sg.run(&e.ex, s, x_ptr, bb.s_gate.ptr, 1);
        try su.run(&e.ex, s, x_ptr, bb.s_up.ptr, 1);
        try e.ex.swiglu(s, bb.s_gate.ptr, bb.s_up.ptr, bb.s_glu.ptr, 1, inter);
        try sd.run(&e.ex, s, bb.s_glu.ptr, bb.y_down.ptr + topk * hidden * 2, 1);
        try e.ex.combine(s, bb.y_down.ptr, bb.wts.ptr + t * slots * 4, out_ptr, 1, hidden, slots);
        if (lay.ep == null) {
            try s.synchronize(); // the next row overwrites the expert lines
        } else {
            try e.done.record(s);
            e.have_done = true;
        }
    }

    /// Final collapse (hyper_connection_mixer, no inject) and the lm_head for `rows` stream rows. Returns f32 logits [rows][vocab].
    /// With `device_hc` the streams `h` are uploaded and collapsed on the device.
    pub fn logitsOf(e: *Engine, h: []const f32, rows: usize, out: []f32) !void {
        const gpa = e.gpu.gpa;
        const head = e.head orelse return error.NoHead;
        if (h.len != rows * wide or out.len != rows * vocab or rows > e.opt.max_rows) return error.Shape;
        if (e.opt.device_hc) {
            try e.b.h.upload(0, std.mem.sliceAsBytes(h));
            try e.hcg.pre(e.stream, e.mixer_dev.?.weights(), e.offset, eps, e.b.h.ptr, e.hc_scratch, e.b.x.ptr, null, rows);
        } else {
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
            try e.b.x.upload(0, std.mem.sliceAsBytes(x16));
        }
        try head.run(&e.ex, e.stream, e.b.x.ptr, e.b.logits.ptr, rows);
        try e.stream.synchronize();
        const l16 = try gpa.alloc(f16, rows * vocab);
        defer gpa.free(l16);
        try e.b.logits.download(0, std.mem.sliceAsBytes(l16));
        for (out, l16) |*d, s| d.* = @floatCast(s);
    }
};

pub fn dalloc(gpu: Gpu, bytes: usize) !cuda.DeviceBuffer {
    var b = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    errdefer b.free();
    try b.fill8(0, null);
    // fill8 and pageable copies are not ordered against the engine's non-blocking streams (they read all-zero rows intermittently under load)
    try gpu.ctx.synchronize();
    return b;
}

/// `DeviceBuffer.fromHost` followed by a context sync, for the same ordering reason as `dalloc`.
pub fn dupload(gpu: Gpu, bytes: []const u8) !cuda.DeviceBuffer {
    const b = try cuda.DeviceBuffer.fromHost(gpu.d, bytes);
    try gpu.ctx.synchronize();
    return b;
}

// -- PLE loaders ---------------------------------------------------------------------------------------------

/// Dense PLE weights as f32 for the `rows` implementation. Norm scales are `offset + stored` (CenteredRMSNorm).
pub fn loadPleWeights(gpa: std.mem.Allocator, pack: *const real.Pack, offset: f32) !PleWeights {
    const l = ple_layer;
    const key = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.key_proj.weight", .{l}));
    errdefer gpa.free(key);
    const value = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.value_proj.weight", .{l}));
    errdefer gpa.free(value);
    const conv = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.conv1d.weight", .{l}));
    errdefer gpa.free(conv);
    const norms = try gpa.alloc(f32, 3 * wide);
    errdefer gpa.free(norms);
    inline for (.{ "norm_key", "norm_query", "norm_conv" }, 0..) |nm, i| {
        const w = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple." ++ nm ++ ".weight", .{l}));
        defer gpa.free(w);
        if (w.len != wide) return error.BadShape;
        for (norms[i * wide ..][0..wide], w) |*d, s| d.* = s + offset;
    }
    if (key.len != wide * hidden or value.len != hidden * hidden or conv.len != wide * fple.kernel) return error.BadShape;
    return .{ .key = key, .value = value, .norms = norms, .conv = conv };
}

pub fn loadPle(gpu: Gpu, pack: *const real.Pack, ng: *st.File) !fple.Ple {
    return loadPleOffset(gpu, pack, ng, 1.0);
}

pub fn loadPleOffset(gpu: Gpu, pack: *const real.Pack, ng: *st.File, offset: f32) !fple.Ple {
    const gpa = gpu.gpa;
    var p: fple.Ple = undefined;
    p.gpa = gpa;
    const l = ple_layer;
    p.key_proj = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.key_proj.weight", .{l}));
    errdefer gpa.free(p.key_proj);
    p.value_proj = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.value_proj.weight", .{l}));
    errdefer gpa.free(p.value_proj);
    p.scale_key = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.norm_key.weight", .{l}));
    errdefer gpa.free(p.scale_key);
    p.scale_query = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.norm_query.weight", .{l}));
    errdefer gpa.free(p.scale_query);
    p.scale_conv = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.norm_conv.weight", .{l}));
    errdefer gpa.free(p.scale_conv);
    for ([_][]f32{ p.scale_key, p.scale_query, p.scale_conv }) |s| for (s) |*v| {
        v.* += offset;
    };
    p.conv_w = try f32Of(gpa, try tensor(pack, "model.language_model.layers.{d}.ple.conv1d.weight", .{l}));
    errdefer gpa.free(p.conv_w);
    if (p.key_proj.len != wide * hidden or p.value_proj.len != hidden * hidden or p.conv_w.len != wide * fple.kernel) return error.BadShape;

    const base = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.";
    const bias = ng.get(base ++ "head_bias") orelse return error.MissingTensor;
    p.head_bias = try f16Of(gpa, bias);
    errdefer gpa.free(p.head_bias);
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
