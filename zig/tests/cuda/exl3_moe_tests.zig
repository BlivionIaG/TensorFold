//! One real Flash Next MoE block on the device, with the mlp hyper-connection around it, against f64 CPU oracles.
//! Block: host router (F16 gates) -> device Exl3.hit over the 512 packed experts of the layer (10 picks) ->
//! device shared expert (linear, swiglu, linear) written into an 11th slot -> device combine (fp32 weights).
//! Hyper-connection: host f32 (`exl3_hc`), the block output bounces D2H and the mixed hidden H2D, as the
//! `exl3_hc` design says. The oracles are `exl3_moe` (Zig decoder, f64) and `exl3_hc.preF64`/`postF64`.
//! Usage: tf-cuda-test exl3-moe <pack dir> [layers, comma separated, default 0,17]

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const st = @import("safetensors.zig");
const real = @import("exl3_real_tests.zig");
const strips = @import("exl3_strips.zig");
const router = @import("exl3_router.zig");
const norm = @import("exl3_norm.zig");
const scan = @import("exl3_scan.zig");
const hc = @import("exl3_hc.zig");
const moe = @import("exl3_moe.zig");

const Gpu = check.Gpu;
const hidden = 2560;
const inter = 640;
const experts = 512;
const topk = 10;
const slots = topk + 1;
const eps: f32 = 1e-6;
const layers_total = 48;

fn f16Copy(gpa: std.mem.Allocator, t: st.Tensor) ![]f16 {
    if (t.dtype != .f16) return error.WrongDtype;
    const out = try gpa.alloc(f16, t.bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

fn f32Of(gpa: std.mem.Allocator, t: st.Tensor) ![]f32 {
    const h = try f16Copy(gpa, t);
    defer gpa.free(h);
    const out = try gpa.alloc(f32, h.len);
    for (out, h) |*d, s| d.* = @floatCast(s);
    return out;
}

fn tensor(pack: *const real.Pack, comptime fmt: []const u8, args: anytype) !st.Tensor {
    var buf: [200]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, args);
    return pack.find(name) orelse {
        std.debug.print("missing tensor {s}\n", .{name});
        return error.MissingTensor;
    };
}

fn freeMat(gpa: std.mem.Allocator, m: real.Mat) void {
    gpa.free(m.trellis);
    gpa.free(m.suh);
    gpa.free(m.svh);
}

/// All 512 experts of one projection, strips-reordered and packed into one line-indexed device matrix.
const Stack = struct {
    trellis: cuda.DeviceBuffer,
    suh: cuda.DeviceBuffer,
    svh: cuda.DeviceBuffer,
    k: usize,
    n: usize,
    halves: u8,

    fn free(s: *Stack) void {
        s.trellis.free();
        s.suh.free();
        s.svh.free();
    }

    fn packed_(s: Stack) !cuda.exl3.Packed {
        return .{
            .trellis = s.trellis.ptr,
            .suh = s.suh.ptr,
            .svh = s.svh.ptr,
            .trellis_stride = try cuda.exl3.tileSpan(s.k, s.n, s.halves),
            .suh_stride = s.k,
            .svh_stride = s.n,
            .halves = s.halves,
            .codebook = 2,
        };
    }
};

fn loadStack(gpu: Gpu, pack: *const real.Pack, layer: usize, proj: []const u8) !Stack {
    const gpa = gpu.gpa;
    const first = try tensor(pack, "model.language_model.layers.{d}.mlp.experts.0.{s}.trellis", .{ layer, proj });
    if (first.rank != 3) return error.BadShape;
    const tiles_k = first.dim(0);
    const tiles_n = first.dim(1);
    const words = first.dim(2);
    const k = tiles_k * 16;
    const n = tiles_n * 16;
    const span = tiles_k * tiles_n * words;
    const tr_stage = try gpa.alloc(i16, experts * span);
    defer gpa.free(tr_stage);
    const su_stage = try gpa.alloc(f16, experts * k);
    defer gpa.free(su_stage);
    const sv_stage = try gpa.alloc(f16, experts * n);
    defer gpa.free(sv_stage);
    const aligned = try gpa.alloc(i16, span);
    defer gpa.free(aligned);
    for (0..experts) |e| {
        const tr = try tensor(pack, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.trellis", .{ layer, e, proj });
        if (tr.rank != 3 or tr.dim(0) != tiles_k or tr.dim(1) != tiles_n or tr.dim(2) != words) {
            std.debug.print("expert {d} {s}: shape differs from expert 0 (halves not uniform)\n", .{ e, proj });
            return error.NonUniformExperts;
        }
        @memcpy(std.mem.sliceAsBytes(aligned), tr.bytes);
        try strips.reorder(aligned, tiles_k, tiles_n, words, tr_stage[e * span ..][0..span]);
        const su = try tensor(pack, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.suh", .{ layer, e, proj });
        const sv = try tensor(pack, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.svh", .{ layer, e, proj });
        @memcpy(std.mem.sliceAsBytes(su_stage[e * k ..][0..k]), su.bytes);
        @memcpy(std.mem.sliceAsBytes(sv_stage[e * n ..][0..n]), sv.bytes);
    }
    var t = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(tr_stage));
    errdefer t.free();
    var su = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(su_stage));
    errdefer su.free();
    const sv = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(sv_stage));
    return .{ .trellis = t, .suh = su, .svh = sv, .k = k, .n = n, .halves = @intCast(words / 8) };
}

const OracleExpert = struct { id: usize, gate: moe.Dense, up: moe.Dense, down: moe.Dense };

fn denseOf(gpa: std.mem.Allocator, m: real.Mat) !moe.Dense {
    return moe.Dense.init(gpa, m.halves, m.trellis, m.tiles_k, m.tiles_n, m.suh, m.svh);
}

const Hc = struct {
    norm: []f32,
    down: []f32,
    up: []f32,
    inject: ?[]f32,

    fn load(gpa: std.mem.Allocator, pack: *const real.Pack, comptime base: []const u8, args: anytype, with_inject: bool) !Hc {
        const n = try f32Of(gpa, try tensor(pack, base ++ ".hc_norm.weight", args));
        const d = try f32Of(gpa, try tensor(pack, base ++ ".input_mix_weight_down.weight", args));
        const u = try f32Of(gpa, try tensor(pack, base ++ ".input_mix_weight_up.weight", args));
        const j: ?[]f32 = if (with_inject) try f32Of(gpa, try tensor(pack, base ++ ".block_inject_weight.weight", args)) else null;
        return .{ .norm = n, .down = d, .up = u, .inject = j };
    }

    fn weights(h: Hc) hc.Weights {
        return .{ .norm = h.norm, .down = h.down, .up = h.up, .inject = h.inject };
    }

    fn free(h: Hc, gpa: std.mem.Allocator) void {
        gpa.free(h.norm);
        gpa.free(h.down);
        gpa.free(h.up);
        if (h.inject) |j| gpa.free(j);
    }
};

/// Deterministic closed form (numpy reproduces it): stream-major index i in [0, 10240).
fn streamValue(layer: usize, i: usize) f16 {
    const x: f64 = @floatFromInt(i);
    const s: f64 = @floatFromInt(i / hidden);
    const l: f64 = @floatFromInt(layer);
    const v = (@sin(0.013 * x * (1.0 + 0.1 * s)) * 2.5 + @cos(0.71 * x + l)) * (1.0 + 0.5 * @sin(0.0031 * x));
    return @floatCast(v);
}

fn branchValue(d: usize) f32 {
    const x: f64 = @floatFromInt(d);
    return @floatCast(0.75 * @cos(0.05 * x) + 0.1);
}

const Err = struct { max_abs: f64, peak: f64, rms_rel: f64 };

fn errOf(got: anytype, want: []const f64) Err {
    var max: f64 = 0;
    var peak: f64 = 0;
    var se: f64 = 0;
    var sw: f64 = 0;
    for (0..want.len) |i| {
        const g: f64 = @floatCast(got[i]);
        max = @max(max, @abs(g - want[i]));
        peak = @max(peak, @abs(want[i]));
        se += (g - want[i]) * (g - want[i]);
        sw += want[i] * want[i];
    }
    return .{ .max_abs = max, .peak = peak, .rms_rel = if (sw > 0) @sqrt(se / sw) else 0 };
}

const Ctx = struct {
    gpu: Gpu,
    ex: cuda.exl3.Exl3,
    stream: cuda.Stream,
    layer: usize,
    gate16: []f16,
    sgate16: []f16,
    gate_s: Stack,
    up_s: Stack,
    down_s: Stack,
    sh_gate: real.Mat,
    sh_up: real.Mat,
    sh_down: real.Mat,
    d_sh_gate: real.Dev,
    d_sh_up: real.Dev,
    d_sh_down: real.Dev,
    dense_sh: [3]moe.Dense,
    cache: std.ArrayList(OracleExpert),
    pack: *const real.Pack,
    // device scratch
    x: cuda.DeviceBuffer,
    ids: cuda.DeviceBuffer,
    y_gate: cuda.DeviceBuffer,
    y_up: cuda.DeviceBuffer,
    y_glu: cuda.DeviceBuffer,
    y_down: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer,
    s_gate: cuda.DeviceBuffer,
    s_up: cuda.DeviceBuffer,
    s_glu: cuda.DeviceBuffer,

    fn oracleExpert(c: *Ctx, id: usize) !*OracleExpert {
        for (c.cache.items) |*e| if (e.id == id) return e;
        const gpa = c.gpu.gpa;
        var projs: [3]moe.Dense = undefined;
        const names = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
        for (names, 0..) |p, i| {
            var nb: [160]u8 = undefined;
            const m = try real.mat(c.pack, gpa, try std.fmt.bufPrint(&nb, "model.language_model.layers.{d}.mlp.experts.{d}.{s}", .{ c.layer, id, p }));
            defer freeMat(gpa, m);
            projs[i] = try denseOf(gpa, m);
        }
        try c.cache.append(gpa, .{ .id = id, .gate = projs[0], .up = projs[1], .down = projs[2] });
        return &c.cache.items[c.cache.items.len - 1];
    }
};

const GpuOut = struct {
    out: [hidden]f16,
    slot: [slots * hidden]f16,
    picks: [slots]u16,
    weights: [slots]f32,
    logits: [experts + 1]f32,
};

fn gpuBlock(c: *Ctx, x16: []const f16) !GpuOut {
    const gpu = c.gpu;
    var r: GpuOut = undefined;
    var x32: [hidden]f32 = undefined;
    for (&x32, x16) |*d, s| d.* = @floatCast(s);
    var scratch: [experts]f32 = undefined;
    try router.launchDtype(.f16, 0, std.mem.sliceAsBytes(c.gate16), &x32, std.mem.sliceAsBytes(c.sgate16), &x32, topk, &r.logits, &r.picks, &r.weights, &scratch);

    var ids: [topk]i32 = undefined;
    for (&ids, r.picks[0..topk]) |*d, s| d.* = s;
    try c.x.upload(0, std.mem.sliceAsBytes(x16));
    try c.ids.upload(0, std.mem.sliceAsBytes(&ids));
    try c.wts.upload(0, std.mem.sliceAsBytes(&r.weights));

    const gp = try c.gate_s.packed_();
    const up = try c.up_s.packed_();
    const dp = try c.down_s.packed_();
    try c.ex.hit(c.stream, c.x.ptr, c.ids.ptr, gp, up, dp, c.y_gate.ptr, c.y_up.ptr, c.y_glu.ptr, c.y_down.ptr, 1, experts, topk);

    // shared expert into slot 10 of y_down
    const sh = c.sh_gate;
    try c.ex.linear(c.stream, c.x.ptr, c.d_sh_gate.trellis.ptr, c.d_sh_gate.suh.ptr, c.d_sh_gate.svh.ptr, c.s_gate.ptr, 1, sh.tiles_k * 16, sh.tiles_n * 16, sh.halves, 2);
    try c.ex.linear(c.stream, c.x.ptr, c.d_sh_up.trellis.ptr, c.d_sh_up.suh.ptr, c.d_sh_up.svh.ptr, c.s_up.ptr, 1, c.sh_up.tiles_k * 16, c.sh_up.tiles_n * 16, c.sh_up.halves, 2);
    try c.ex.swiglu(c.stream, c.s_gate.ptr, c.s_up.ptr, c.s_glu.ptr, 1, inter);
    const d = c.sh_down;
    try c.ex.linear(c.stream, c.s_glu.ptr, c.d_sh_down.trellis.ptr, c.d_sh_down.suh.ptr, c.d_sh_down.svh.ptr, c.y_down.ptr + topk * hidden * 2, 1, d.tiles_k * 16, d.tiles_n * 16, d.halves, 2);

    try c.ex.combine(c.stream, c.y_down.ptr, c.wts.ptr, c.out.ptr, 1, hidden, slots);
    try c.stream.synchronize();

    const ob = try check.download(gpu, c.out);
    defer gpu.gpa.free(ob);
    @memcpy(std.mem.sliceAsBytes(&r.out), ob);
    const yb = try check.download(gpu, c.y_down);
    defer gpu.gpa.free(yb);
    @memcpy(std.mem.sliceAsBytes(&r.slot), yb[0 .. slots * hidden * 2]);
    return r;
}

const OracleOut = struct {
    out: [hidden]f64,
    slot: [slots * hidden]f64,
};

/// Block oracle for the given picks and weights (f64). Slot 10 is the shared expert, scaled in `out` by `shared_w`.
fn oracleBlock(c: *Ctx, x16: []const f16, picks: []const u16, w: []const f64, shared_w: f64) !OracleOut {
    var x64: [hidden]f64 = undefined;
    for (&x64, x16) |*d, s| d.* = @floatCast(s);
    var r: OracleOut = undefined;
    @memset(&r.out, 0);
    var gb: [inter]f64 = undefined;
    var ub: [inter]f64 = undefined;
    var xb: [inter]f64 = undefined;
    for (picks, w, 0..) |p, wk, k| {
        const e = try c.oracleExpert(p);
        try moe.expert(e.gate, e.up, e.down, &x64, &gb, &ub, &xb, r.slot[k * hidden ..][0..hidden]);
        for (&r.out, r.slot[k * hidden ..][0..hidden]) |*o, v| o.* += wk * v;
    }
    try moe.expert(c.dense_sh[0], c.dense_sh[1], c.dense_sh[2], &x64, &gb, &ub, &xb, r.slot[topk * hidden ..][0..hidden]);
    for (&r.out, r.slot[topk * hidden ..][0..hidden]) |*o, v| o.* += shared_w * v;
    return r;
}

// Observed: 0.0003-0.0007 of peak (fp16 stage rounding, ~5e-4 relative); gates leave about 7x headroom.
const Tol = struct { primary: f64 = 0.005, independent: f64 = 0.007 };

fn blockCase(c: *Ctx, name: []const u8, x16: []const f16, branch64: ?*[hidden]f64, branch16: ?*[hidden]f16) !void {
    const tol: Tol = .{};
    const g = try gpuBlock(c, x16);
    // independent routing from f64 logits of the same fp16 input
    var x64: [hidden]f64 = undefined;
    for (&x64, x16) |*d, s| d.* = @floatCast(s);
    var logits64: [experts + 1]f64 = undefined;
    try moe.logitsF16(c.gate16, experts, c.sgate16, &x64, &logits64);
    var picks_o: [topk]u16 = undefined;
    var w_o: [topk]f64 = undefined;
    var sw_o: f64 = 0;
    try moe.routeFromLogits(&logits64, topk, &picks_o, &w_o, &sw_o);

    const same_picks = std.mem.eql(u16, &picks_o, g.picks[0..topk]);
    var wrel: f64 = 0;
    for (w_o, g.weights[0..topk]) |a, b| wrel = @max(wrel, @abs(a - @as(f64, b)) / a);
    const swrel = @abs(sw_o - @as(f64, g.weights[topk])) / sw_o;
    var lg_err: f64 = 0;
    for (g.logits, logits64) |a, b| lg_err = @max(lg_err, @abs(@as(f64, a) - b));
    std.debug.print("  [{s}] route: picks {s}, routed-weight max rel {e:.2} (bf16 step 3.9e-3), shared-weight rel {e:.2}, logit max_abs {e:.2}\n", .{ name, if (same_picks) "identical to the f64 oracle" else "DIFFER from the f64 oracle", wrel, swrel, lg_err });
    try check.expect(same_picks, "{s}: host router picks differ from the f64 oracle", .{name});
    try check.expect(wrel <= 1.0 / 128.0 and swrel <= 1.0 / 128.0, "{s}: router weights beyond bf16 rounding", .{name});

    // primary: oracle on the GPU's own picks and (bf16-rounded) weights, isolates the device kernels
    var wg: [topk]f64 = undefined;
    for (&wg, g.weights[0..topk]) |*d, s| d.* = s;
    const op = try oracleBlock(c, x16, g.picks[0..topk], &wg, g.weights[topk]);
    const e_out = errOf(&g.out, &op.out);
    var worst_slot: f64 = 0;
    var worst_k: usize = 0;
    var slot_peak: f64 = 0;
    for (0..slots) |k| {
        const e = errOf(g.slot[k * hidden ..][0..hidden], op.slot[k * hidden ..][0..hidden]);
        const rel = e.max_abs / e.peak;
        if (rel > worst_slot) {
            worst_slot = rel;
            worst_k = k;
        }
        slot_peak = @max(slot_peak, e.peak);
    }
    std.debug.print("  [{s}] block vs oracle(same picks+weights): max_abs {e:.3} peak {e:.3} ({d:.4} of peak) rms_rel {e:.2}; worst expert slot {d} {d:.4} of its peak\n", .{ name, e_out.max_abs, e_out.peak, e_out.max_abs / e_out.peak, e_out.rms_rel, worst_k, worst_slot });
    try check.expect(e_out.peak > 0 and e_out.max_abs <= tol.primary * e_out.peak, "{s}: block differs from the Zig-decoder oracle", .{name});
    try check.expect(worst_slot <= 0.01, "{s}: an expert slot differs from the oracle", .{name});

    // independent: f64 router weights (not bf16 rounded)
    const oi = try oracleBlock(c, x16, &picks_o, &w_o, sw_o);
    const e_ind = errOf(&g.out, &oi.out);
    std.debug.print("  [{s}] block vs fully independent oracle (f64 routing): max_abs {e:.3} ({d:.4} of peak) rms_rel {e:.2}\n", .{ name, e_ind.max_abs, e_ind.max_abs / e_ind.peak, e_ind.rms_rel });
    try check.expect(e_ind.max_abs <= tol.independent * e_ind.peak, "{s}: block differs from the independent oracle", .{name});

    if (branch64) |b| b.* = oi.out;
    if (branch16) |b| b.* = g.out;
}

fn runLayer(gpu: Gpu, pack: *const real.Pack, ex: cuda.exl3.Exl3, stream: cuda.Stream, layer: usize, offset: f32, means: []const f32) !void {
    const gpa = gpu.gpa;
    std.debug.print("== layer {d} (centred offset {d})\n", .{ layer, offset });
    const gate16 = try f16Copy(gpa, try tensor(pack, "model.language_model.layers.{d}.mlp.gate.weight", .{layer}));
    defer gpa.free(gate16);
    const sgate16 = try f16Copy(gpa, try tensor(pack, "model.language_model.layers.{d}.mlp.shared_expert_gate.weight", .{layer}));
    defer gpa.free(sgate16);
    if (gate16.len != experts * hidden or sgate16.len != hidden) return error.BadShape;

    var gs = try loadStack(gpu, pack, layer, "gate_proj");
    defer gs.free();
    var us = try loadStack(gpu, pack, layer, "up_proj");
    defer us.free();
    var ds = try loadStack(gpu, pack, layer, "down_proj");
    defer ds.free();
    try check.expect(gs.k == hidden and gs.n == inter and ds.k == inter and ds.n == hidden, "expert shapes {d}x{d}, {d}x{d}", .{ gs.k, gs.n, ds.k, ds.n });
    std.debug.print("  experts packed on the device: halves gate={d} up={d} down={d}, 512 lines each\n", .{ gs.halves, us.halves, ds.halves });

    var sb: [160]u8 = undefined;
    const stem = try std.fmt.bufPrint(&sb, "model.language_model.layers.{d}.mlp.shared_expert", .{layer});
    var nb: [3][200]u8 = undefined;
    const sh_gate = try real.mat(pack, gpa, try std.fmt.bufPrint(&nb[0], "{s}.gate_proj", .{stem}));
    const sh_up = try real.mat(pack, gpa, try std.fmt.bufPrint(&nb[1], "{s}.up_proj", .{stem}));
    const sh_down = try real.mat(pack, gpa, try std.fmt.bufPrint(&nb[2], "{s}.down_proj", .{stem}));
    defer freeMat(gpa, sh_gate);
    defer freeMat(gpa, sh_up);
    defer freeMat(gpa, sh_down);
    std.debug.print("  shared expert: halves gate={d} up={d} down={d}\n", .{ sh_gate.halves, sh_up.halves, sh_down.halves });
    var d_g = try real.upload(gpu, sh_gate);
    defer d_g.free();
    var d_u = try real.upload(gpu, sh_up);
    defer d_u.free();
    var d_d = try real.upload(gpu, sh_down);
    defer d_d.free();
    var dense_sh = [3]moe.Dense{ try denseOf(gpa, sh_gate), try denseOf(gpa, sh_up), try denseOf(gpa, sh_down) };
    defer for (&dense_sh) |*m| m.deinit(gpa);

    var x = try cuda.DeviceBuffer.alloc(gpu.d, hidden * 2);
    defer x.free();
    var ids = try cuda.DeviceBuffer.alloc(gpu.d, topk * 4);
    defer ids.free();
    var y_gate = try cuda.DeviceBuffer.alloc(gpu.d, topk * inter * 2);
    defer y_gate.free();
    var y_up = try cuda.DeviceBuffer.alloc(gpu.d, topk * inter * 2);
    defer y_up.free();
    var y_glu = try cuda.DeviceBuffer.alloc(gpu.d, topk * inter * 2);
    defer y_glu.free();
    var y_down = try cuda.DeviceBuffer.alloc(gpu.d, slots * hidden * 2);
    defer y_down.free();
    var wts = try cuda.DeviceBuffer.alloc(gpu.d, slots * 4);
    defer wts.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, hidden * 2);
    defer out.free();
    var s_gate = try cuda.DeviceBuffer.alloc(gpu.d, inter * 2);
    defer s_gate.free();
    var s_up = try cuda.DeviceBuffer.alloc(gpu.d, inter * 2);
    defer s_up.free();
    var s_glu = try cuda.DeviceBuffer.alloc(gpu.d, inter * 2);
    defer s_glu.free();

    var c: Ctx = .{
        .gpu = gpu, .ex = ex, .stream = stream, .layer = layer, .gate16 = gate16, .sgate16 = sgate16,
        .gate_s = gs, .up_s = us, .down_s = ds,
        .sh_gate = sh_gate, .sh_up = sh_up, .sh_down = sh_down, .d_sh_gate = d_g, .d_sh_up = d_u, .d_sh_down = d_d,
        .dense_sh = dense_sh, .cache = .empty, .pack = pack,
        .x = x, .ids = ids, .y_gate = y_gate, .y_up = y_up, .y_glu = y_glu, .y_down = y_down, .wts = wts, .out = out,
        .s_gate = s_gate, .s_up = s_up, .s_glu = s_glu,
    };
    defer {
        for (c.cache.items) |*e| {
            e.gate.deinit(gpa);
            e.up.deinit(gpa);
            e.down.deinit(gpa);
        }
        c.cache.deinit(gpa);
    }

    // S2 weights
    const mlp_hc = try Hc.load(gpa, pack, "model.language_model.layers.{d}.mlp_hyper_connection", .{layer}, true);
    defer mlp_hc.free(gpa);

    var rng = std.Random.DefaultPrng.init(0x5eed + layer);
    const r = rng.random();

    // (a) uniform deterministic hidden
    var xa: [hidden]f16 = undefined;
    for (&xa) |*v| v.* = @floatCast((r.float(f32) - 0.5) * 2.0);
    try blockCase(&c, "uniform", &xa, null, null);

    // (b) a hidden produced by the centred norm: heavy-tailed raw activations through hc_norm's first stream
    var raw: [hidden]f32 = undefined;
    for (&raw, 0..) |*v, i| v.* = r.floatNorm(f32) * (1.0 + 4.0 * @as(f32, @floatFromInt(@intFromBool(i % 17 == 0))));
    var normed32: [hidden]f32 = undefined;
    try norm.apply(&raw, mlp_hc.norm[0..hidden], means, eps, &normed32);
    var ms: f64 = 0;
    for (raw) |v| ms += @as(f64, v) * v;
    ms /= hidden;
    var nerr: f64 = 0;
    for (raw, mlp_hc.norm[0..hidden], normed32) |v, wv, got| {
        const want = @as(f64, v) / @sqrt(ms + 1e-6) * (@as(f64, wv) + offset);
        nerr = @max(nerr, @abs(want - got));
    }
    std.debug.print("  norm.apply (offset {d}) vs f64 formula: max_abs {e:.2}\n", .{ offset, nerr });
    try check.expect(nerr < 1e-4, "centred norm differs from the f64 formula", .{});
    var xb: [hidden]f16 = undefined;
    for (&xb, normed32) |*d, s| d.* = @floatCast(s);
    try blockCase(&c, "norm-produced", &xb, null, null);

    // (c) HC pre output of 4 residual streams, then post with the block output
    var h16: [hc.real.streams * hidden]f16 = undefined;
    var h32: [hc.real.streams * hidden]f32 = undefined;
    var h64: [hc.real.streams * hidden]f64 = undefined;
    for (0..h16.len) |i| {
        h16[i] = streamValue(layer, i);
        h32[i] = @floatCast(h16[i]);
        h64[i] = h32[i];
    }
    var mixed32: [hidden]f32 = undefined;
    var inj32: [hc.real.streams]f32 = undefined;
    const scratch = try gpa.alloc(f32, hc.scratchLen(hc.real));
    defer gpa.free(scratch);
    try hc.pre(hc.real, mlp_hc.weights(), offset, eps, &h32, &mixed32, &inj32, scratch);
    var mixed64: [hidden]f64 = undefined;
    var inj64: [hc.real.streams]f64 = undefined;
    const lowb = try gpa.alloc(f64, hc.real.lowrank);
    defer gpa.free(lowb);
    const mixb = try gpa.alloc(f64, hc.real.streams * hidden);
    defer gpa.free(mixb);
    const normb = try gpa.alloc(f64, hc.real.streams * hidden);
    defer gpa.free(normb);
    try hc.preF64(hc.real, mlp_hc.weights(), offset, eps, &h64, &mixed64, &inj64, lowb, mixb, normb);
    const e_mixed = errOf(&mixed32, &mixed64);
    const e_inj = errOf(&inj32, &inj64);
    std.debug.print("  HC pre (host f32) vs f64 oracle: mixed max_abs {e:.2} peak {e:.3}; inject max_abs {e:.2} (values {d:.4} {d:.4} {d:.4} {d:.4})\n", .{ e_mixed.max_abs, e_mixed.peak, e_inj.max_abs, inj32[0], inj32[1], inj32[2], inj32[3] });
    try check.expect(e_mixed.max_abs <= 1e-4 * e_mixed.peak and e_inj.max_abs <= 1e-5, "HC pre host f32 differs from the f64 oracle", .{});

    // the cross-check vector for the independent numpy script
    var branch_f: [hidden]f32 = undefined;
    var branch_f64: [hidden]f64 = undefined;
    for (&branch_f, &branch_f64, 0..) |*a, *b, d| {
        a.* = branchValue(d);
        b.* = a.*;
    }
    var h1x: [hc.real.streams * hidden]f32 = undefined;
    try hc.post(hc.real, &h32, &branch_f, &inj32, &h1x);
    var sum_mixed: f64 = 0;
    for (mixed64) |v| sum_mixed += v;
    var sum_h1: f64 = 0;
    for (h1x) |v| sum_h1 += v;
    std.debug.print("  XCHECK layer {d} mixed[0..3] {d:.6} {d:.6} {d:.6} sum {d:.5} inject {d:.6} {d:.6} {d:.6} {d:.6} h1[0] {d:.6} h1[7000] {d:.6} sum {d:.4}\n", .{ layer, mixed64[0], mixed64[1], mixed64[2], sum_mixed, inj64[0], inj64[1], inj64[2], inj64[3], h1x[0], h1x[7000], sum_h1 });

    var xc: [hidden]f16 = undefined;
    for (&xc, mixed32) |*d, s| d.* = @floatCast(s);
    var branch_o: [hidden]f64 = undefined;
    var branch_g: [hidden]f16 = undefined;
    try blockCase(&c, "hc-mixed", &xc, &branch_o, &branch_g);

    // post: device branch (D2H bounce) into the f32 streams, versus the oracle chain end to end in f64
    var branch_g32: [hidden]f32 = undefined;
    for (&branch_g32, branch_g) |*d, s| d.* = @floatCast(s);
    var h1: [hc.real.streams * hidden]f32 = undefined;
    try hc.post(hc.real, &h32, &branch_g32, &inj32, &h1);
    var h1_64: [hc.real.streams * hidden]f64 = undefined;
    try hc.postF64(hc.real, &h64, &branch_o, &inj64, &h1_64);
    const e_h1 = errOf(&h1, &h1_64);
    var delta_peak: f64 = 0;
    for (h1_64, h64) |a, b| delta_peak = @max(delta_peak, @abs(a - b));
    std.debug.print("  HC pre + block + post vs f64 oracle chain: streams max_abs {e:.3} peak {e:.3}; the block's own contribution to the streams peaks at {e:.3}\n", .{ e_h1.max_abs, e_h1.peak, delta_peak });
    try check.expect(e_h1.max_abs <= 0.01 * e_h1.peak, "HC chain differs from the oracle", .{});
    check.pass("layer {d}: MoE block (host router, 10 routed experts, shared expert, combine) and mlp hyper-connection pre/post match the f64 oracles", .{layer});
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    var layer_list: [8]usize = undefined;
    var n_layers: usize = 0;
    const spec: []const u8 = if (rest.len > 1) rest[1] else "0,17";
    var it = std.mem.splitScalar(u8, spec, ',');
    while (it.next()) |tok| {
        if (n_layers == layer_list.len) return error.TooManyLayers;
        layer_list[n_layers] = try std.fmt.parseInt(usize, tok, 10);
        n_layers += 1;
    }

    var pack: real.Pack = .{ .files = undefined };
    defer pack.close(gpu.io);
    try pack.openAll(gpu, dir);

    var ex = try cuda.exl3.Exl3.load(gpu.d);
    defer ex.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();

    // centred-norm offset: the same locator the loader uses, over the 48 attention hc_norm means
    var means: [layers_total]f32 = undefined;
    for (&means, 0..) |*m, l| {
        const w = try f32Of(gpu.gpa, try tensor(&pack, "model.language_model.layers.{d}.attn_hyper_connection.hc_norm.weight", .{l}));
        defer gpu.gpa.free(w);
        var s: f64 = 0;
        for (w) |v| s += v;
        m.* = @floatCast(s / @as(f64, @floatFromInt(w.len)));
    }
    const offset = try scan.centredOffset(&means);
    var sorted = means;
    std.mem.sort(f32, &sorted, {}, std.sort.asc(f32));
    std.debug.print("attn hc_norm means over 48 layers: min {d:.4} median {d:.4} max {d:.4} -> centred offset {d}\n", .{ sorted[0], sorted[24], sorted[47], offset });

    for (layer_list[0..n_layers]) |l| try runLayer(gpu, &pack, ex, stream, l, offset, &means);

    // collapse at the model end: hyper_connection_mixer (no inject), host f32 vs f64
    const mixer = try Hc.load(gpu.gpa, &pack, "model.language_model.hyper_connection_mixer", .{}, false);
    defer mixer.free(gpu.gpa);
    var h32: [hc.real.streams * hidden]f32 = undefined;
    var h64: [hc.real.streams * hidden]f64 = undefined;
    for (0..h32.len) |i| {
        h32[i] = @floatCast(streamValue(47, i));
        h64[i] = h32[i];
    }
    var out32: [hidden]f32 = undefined;
    var out64: [hidden]f64 = undefined;
    const scratch = try gpu.gpa.alloc(f32, hc.scratchLen(hc.real));
    defer gpu.gpa.free(scratch);
    try hc.pre(hc.real, mixer.weights(), offset, eps, &h32, &out32, null, scratch);
    const lowb = try gpu.gpa.alloc(f64, hc.real.lowrank);
    defer gpu.gpa.free(lowb);
    const mixb = try gpu.gpa.alloc(f64, hc.real.streams * hidden);
    defer gpu.gpa.free(mixb);
    const normb = try gpu.gpa.alloc(f64, hc.real.streams * hidden);
    defer gpu.gpa.free(normb);
    try hc.preF64(hc.real, mixer.weights(), offset, eps, &h64, &out64, null, lowb, mixb, normb);
    const e = errOf(&out32, &out64);
    std.debug.print("final collapse (hyper_connection_mixer, no inject) host f32 vs f64: max_abs {e:.2} peak {e:.3}\n", .{ e.max_abs, e.peak });
    try check.expect(e.max_abs <= 1e-4 * e.peak, "final collapse differs", .{});
    check.pass("final stream collapse matches the f64 oracle", .{});
}
