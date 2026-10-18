//! A real Qwen3.8-Flash-Next full-attention layer on the device: q/k/v/index_qk (EXL3 linears), norm + RoPE + KV write, the indexer,
//! the sparse decode, the output gate and o_proj (EXL3 linear), against `exl3_attn_cpu.zig`.
//! Usage: tf-cuda-test exl3-attn <pack dir> [layer a] [layer b]      (defaults 3 and 7)
//!
//! Part A (per layer): prefill 32 tokens, then 8 decode steps replayed through ONE captured CUDA graph (only the int32 position
//!   pointer and the input row change). The oracle consumes the device's own fp16 projections row by row, so what is compared is the
//!   code under test: norm+RoPE+KV write, pooling, dense/sparse key choice, attention, gate. The five linears are separately
//!   checked against the Zig f64 decoder (`dequantize`) on the first and last prefill row.
//! Part B (per layer): synthetic context of 2600 tokens (random q/k/v projections, the real index_qk_proj on random hidden rows,
//!   real norms): sparse rows 2592..2599 and the dense-to-sparse boundary rows 2044..2055. Checks the selected block lists against
//!   the oracle (a block that differs must sit within 1e-4 of the 512th score), the attention against the oracle on the device's own
//!   key list, and graph replay of scores+select+decode against the eager launch byte for byte.
//! Tolerances: bf16 has 8 bits of precision, so one round-to-nearest flip is at most 2^-7 = 0.78% of the stage's peak; tables and
//! normed outputs (all bf16-rounded) allow 1%. Attention outputs (softmax over 256-dim dots of flipped inputs) 2%, o_proj
//! (the existing exl3-real bound for the down projection, fp16 output) 3%, the linears 2%.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const st = @import("safetensors.zig");
const real = @import("exl3_real_tests.zig");
const format = @import("exl3_format.zig");
const scan = @import("exl3_scan.zig");
const oracle = @import("exl3_attn_cpu.zig");
const cachemod = @import("exl3_cache.zig");

const Gpu = check.Gpu;
const hidden: usize = 2560;
const q_width: usize = oracle.q_heads * 2 * oracle.head_dim; // 12288
const kv_width: usize = oracle.kv_heads * oracle.head_dim; // 512
const attn_width: usize = oracle.q_heads * oracle.head_dim; // 6144
const idx_width: usize = oracle.index_width; // 640
const slot_bytes: usize = 2 * oracle.head_dim * 2;

// -- host/device helpers ------------------------------------------------------

fn dalloc(gpu: Gpu, bytes: usize) !cuda.DeviceBuffer {
    var b = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    errdefer b.free();
    try b.fill8(0, null);
    return b;
}

fn dhost(gpu: Gpu, comptime T: type, v: []const T) !cuda.DeviceBuffer {
    return cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(v));
}

fn fetch(gpu: Gpu, comptime T: type, buf: cuda.DeviceBuffer, off_bytes: usize, n: usize) ![]T {
    const out = try gpu.gpa.alloc(T, n);
    errdefer gpu.gpa.free(out);
    try buf.download(off_bytes, std.mem.sliceAsBytes(out));
    return out;
}

fn bf16f(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

fn fetchBf16(gpu: Gpu, buf: cuda.DeviceBuffer, off_bytes: usize, n: usize) ![]f64 {
    const raw = try fetch(gpu, u16, buf, off_bytes, n);
    defer gpu.gpa.free(raw);
    const out = try gpu.gpa.alloc(f64, n);
    for (out, raw) |*o, b| o.* = bf16f(b);
    return out;
}

/// Running max error and peak over many rows of one stage.
const Acc = struct {
    name: []const u8,
    max_abs: f64 = 0,
    peak: f64 = 0,
    n: usize = 0,
    exact: usize = 0,

    fn add(a: *Acc, got: anytype, want: []const f64) void {
        for (got, want) |g, w| {
            const gv: f64 = @floatCast(g);
            a.max_abs = @max(a.max_abs, @abs(gv - w));
            a.peak = @max(a.peak, @abs(w));
            a.exact += @intFromBool(gv == w);
            a.n += 1;
        }
    }

    fn finish(a: Acc, tol: f64) !void {
        std.debug.print("  {s}: max_abs {e:.3} peak {e:.3} rel {d:.5} (tol {d:.3}), bit-equal {d}/{d}\n", .{ a.name, a.max_abs, a.peak, a.max_abs / @max(a.peak, 1e-30), tol, a.exact, a.n });
        try check.expect(a.peak > 0 and a.max_abs <= tol * a.peak, "{s} differs from the oracle", .{a.name});
    }
};

// -- weights ------------------------------------------------------------------

const Lin = struct {
    m: real.Mat,
    d: real.Dev,
    w: []f64,
    k: usize,
    n: usize,

    fn load(gpu: Gpu, pack: *const real.Pack, base: []const u8) !Lin {
        const m = try real.mat(pack, gpu.gpa, base);
        const k = m.tiles_k * 16;
        const n = m.tiles_n * 16;
        var d = try real.upload(gpu, m);
        errdefer d.free();
        const w = try gpu.gpa.alloc(f64, k * n);
        errdefer gpu.gpa.free(w);
        const scratch = try gpu.gpa.alloc(f64, 2 * k * n);
        defer gpu.gpa.free(scratch);
        format.dequantize(.{ .halves = m.halves }, .mul1, m.trellis, m.tiles_k, m.tiles_n, m.suh, m.svh, w, scratch);
        return .{ .m = m, .d = d, .w = w, .k = k, .n = n };
    }

    fn free(l: *Lin, gpa: std.mem.Allocator) void {
        l.d.free();
        gpa.free(l.w);
        gpa.free(l.m.trellis);
        gpa.free(l.m.suh);
        gpa.free(l.m.svh);
    }

    /// f64 y = x W on the Zig decoder's dequantized matrix.
    fn cpu(l: *const Lin, x: []const f16, out: []f64) void {
        @memset(out, 0);
        for (0..l.k) |i| {
            const xi: f64 = @floatCast(x[i]);
            const row = l.w[i * l.n ..][0..l.n];
            for (out, row) |*o, wv| o.* += xi * wv;
        }
    }

    fn run(l: *const Lin, ex: *const cuda.exl3.Exl3, s: cuda.Stream, x: u64, y: u64, rows: usize) !void {
        try ex.linear(s, x, l.d.trellis.ptr, l.d.suh.ptr, l.d.svh.ptr, y, rows, l.k, l.n, l.m.halves, 2);
    }
};

const Layer = struct {
    index: usize, // model layer
    kv_layer: usize, // full-attention table index
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    idx: Lin,
    q_scale: []f32,
    k_scale: []f32,
    iq_scale: []f32,
    ik_scale: []f32,
    inv: [oracle.half]f32,
    d_q_scale: cuda.DeviceBuffer,
    d_k_scale: cuda.DeviceBuffer,
    d_iq_scale: cuda.DeviceBuffer,
    d_ik_scale: cuda.DeviceBuffer,
    d_inv: cuda.DeviceBuffer,

    fn norms(l: *const Layer) oracle.Norms {
        return .{ .q = l.q_scale, .k = l.k_scale, .iq = l.iq_scale, .ik = l.ik_scale, .inv = l.inv };
    }

    fn free(l: *Layer, gpa: std.mem.Allocator) void {
        l.q.free(gpa);
        l.k.free(gpa);
        l.v.free(gpa);
        l.o.free(gpa);
        l.idx.free(gpa);
        gpa.free(l.q_scale);
        gpa.free(l.k_scale);
        gpa.free(l.iq_scale);
        gpa.free(l.ik_scale);
        l.d_q_scale.free();
        l.d_k_scale.free();
        l.d_iq_scale.free();
        l.d_ik_scale.free();
        l.d_inv.free();
    }
};

fn normWeights(gpu: Gpu, pack: *const real.Pack, layer: usize, name: []const u8, width: usize) ![]f32 {
    var buf: [160]u8 = undefined;
    const t = pack.find(try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.self_attn.{s}", .{ layer, name })) orelse return error.MissingTensor;
    try check.expect(t.numel() == width and t.bytes.len == width * 2, "{s} shape {d}", .{ name, t.numel() });
    const out = try gpu.gpa.alloc(f32, width);
    for (out, 0..) |*o, i| o.* = @floatCast(bf16f(std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little)));
    return out;
}

fn loadLayer(gpu: Gpu, pack: *const real.Pack, layer: usize) !Layer {
    var base: [96]u8 = undefined;
    const stem = try std.fmt.bufPrint(&base, "model.language_model.layers.{d}.self_attn", .{layer});
    var names: [5][128]u8 = undefined;
    var q = try Lin.load(gpu, pack, try std.fmt.bufPrint(&names[0], "{s}.q_proj", .{stem}));
    errdefer q.free(gpu.gpa);
    var k = try Lin.load(gpu, pack, try std.fmt.bufPrint(&names[1], "{s}.k_proj", .{stem}));
    errdefer k.free(gpu.gpa);
    var v = try Lin.load(gpu, pack, try std.fmt.bufPrint(&names[2], "{s}.v_proj", .{stem}));
    errdefer v.free(gpu.gpa);
    var o = try Lin.load(gpu, pack, try std.fmt.bufPrint(&names[3], "{s}.o_proj", .{stem}));
    errdefer o.free(gpu.gpa);
    var idx = try Lin.load(gpu, pack, try std.fmt.bufPrint(&names[4], "{s}.indexer.index_qk_proj", .{stem}));
    errdefer idx.free(gpu.gpa);
    std.debug.print("layer {d}: q_proj {d}x{d} (halves {d}), k/v {d}x{d}, o_proj {d}x{d}, index_qk {d}x{d}\n", .{ layer, q.k, q.n, q.m.halves, k.k, k.n, o.k, o.n, idx.k, idx.n });
    try check.expect(q.k == hidden and q.n == q_width and k.n == kv_width and v.n == kv_width and o.k == attn_width and o.n == hidden and idx.k == hidden and idx.n == idx_width, "projection shapes", .{});

    const qn = try normWeights(gpu, pack, layer, "q_norm.weight", oracle.head_dim);
    const kn = try normWeights(gpu, pack, layer, "k_norm.weight", oracle.head_dim);
    const iqn = try normWeights(gpu, pack, layer, "indexer.q_layernorm.weight", oracle.index_dim);
    const ikn = try normWeights(gpu, pack, layer, "indexer.k_layernorm.weight", oracle.index_dim);
    var means: [4]f32 = undefined;
    for ([_][]f32{ qn, kn, iqn, ikn }, 0..) |w, i| {
        var s: f64 = 0;
        for (w) |x| s += x;
        means[i] = @floatCast(s / @as(f64, @floatFromInt(w.len)));
    }
    std.debug.print("  norm weight means q {d:.4} k {d:.4} iq {d:.4} ik {d:.4}\n", .{ means[0], means[1], means[2], means[3] });
    // `scan.centredOffset` over all four means is NormAmbiguous on this pack: q_norm/k_norm means sit at 0.27-0.34, past its 0.25
    // median bound for gamma-1 storage (and nowhere near 1). The model defines all four as CenteredRMSNorm (scale 1 + w), so: the
    // indexer norms (means about -0.02..-0.04) classify as gamma-1 by the scan, and q/k are gamma-1 too because no mean is near 1.
    const ambiguous = std.meta.isError(scan.centredOffset(&means));
    const offset = try scan.centredOffset(means[2..4]);
    try check.expect(offset == 1 and means[0] < 0.5 and means[1] < 0.5, "norm storage is not gamma-1", .{});
    std.debug.print("  centredOffset over all four means: {s}\n", .{if (ambiguous) "NormAmbiguous (shim: indexer norms decide, q/k below 0.5)" else "ok"});
    std.debug.print("  centred offset {d}\n", .{offset});
    for ([_][]f32{ qn, kn, iqn, ikn }) |w| for (w) |*x| {
        x.* += offset;
    };
    const inv = oracle.invFreq();
    return .{
        .index = layer,
        .kv_layer = (layer + 1) / 4 - 1,
        .q = q,
        .k = k,
        .v = v,
        .o = o,
        .idx = idx,
        .q_scale = qn,
        .k_scale = kn,
        .iq_scale = iqn,
        .ik_scale = ikn,
        .inv = inv,
        .d_q_scale = try dhost(gpu, f32, qn),
        .d_k_scale = try dhost(gpu, f32, kn),
        .d_iq_scale = try dhost(gpu, f32, iqn),
        .d_ik_scale = try dhost(gpu, f32, ikn),
        .d_inv = try dhost(gpu, f32, &inv),
    };
}

// -- one pipeline over `rows` tokens --------------------------------------------

const Bufs = struct {
    x: u64,
    qp: u64,
    kp: u64,
    vp: u64,
    ip: u64,
    q_out: u64,
    iq_out: u64,
    scores: u64,
    ids: u64,
    nk: u64,
    attn: u64,
    y: u64,
    kv: u64, // this layer's table
    ik: u64,
    pooled: u64,
    pos0: u64,
};

fn pipeline(ex: *const cuda.exl3.Exl3, at: *const cuda.exl3_attn.Attn, s: cuda.Stream, l: *const Layer, b: Bufs, dims: cuda.exl3_attn.Dims, rows: usize) !void {
    try l.q.run(ex, s, b.x, b.qp, rows);
    try l.k.run(ex, s, b.x, b.kp, rows);
    try l.v.run(ex, s, b.x, b.vp, rows);
    try l.idx.run(ex, s, b.x, b.ip, rows);
    try at.prep(s, dims, rows, .{
        .qp = b.qp,
        .kp = b.kp,
        .vp = b.vp,
        .ip = b.ip,
        .q_scale = l.d_q_scale.ptr,
        .k_scale = l.d_k_scale.ptr,
        .i_scale = l.d_iq_scale.ptr,
        .inv_freq = l.d_inv.ptr,
        .pos0 = b.pos0,
        .q_out = b.q_out,
        .kv = b.kv,
        .iq_out = b.iq_out,
        .ik = b.ik,
    });
    try at.pool(s, dims, rows, b.ik, b.pooled, l.d_ik_scale.ptr, l.d_inv.ptr, b.pos0);
    const nb = dims.tokens / cuda.exl3_attn.ratio;
    try at.scores(s, rows, b.iq_out, b.pooled, b.scores, b.pos0, nb, nb);
    try at.select(s, rows, b.scores, b.ids, b.nk, b.pos0, nb);
    try at.decode(s, dims, rows, b.q_out, b.qp, b.kv, b.ids, b.attn, b.pos0);
    try l.o.run(ex, s, b.attn, b.y, rows);
}

fn setPos(pos0: cuda.DeviceBuffer, v: usize) !void {
    const i: i32 = @intCast(v);
    try pos0.upload(0, std.mem.asBytes(&i));
}

fn randomHalf(rng: *std.Random.DefaultPrng, out: []f16, scale: f32) void {
    for (out) |*v| v.* = @floatCast(rng.random().floatNorm(f32) * scale);
}

// -- Part A -------------------------------------------------------------------

const prefill_rows: usize = 32;
const decode_steps: usize = 8;
const cap_a: usize = 64;

fn partA(gpu: Gpu, ex: *const cuda.exl3.Exl3, at: *const cuda.exl3_attn.Attn, s: cuda.Stream, l: *const Layer, kv_all: cuda.DeviceBuffer) !void {
    const gpa = gpu.gpa;
    const total = prefill_rows + decode_steps;
    const dims: cuda.exl3_attn.Dims = .{ .tokens = cap_a };
    const nb = cap_a / 4;

    var rng = std.Random.DefaultPrng.init(0xa77e + l.index);
    const xh = try gpa.alloc(f16, total * hidden);
    defer gpa.free(xh);
    randomHalf(&rng, xh, 1.0);

    var x_dev = try dhost(gpu, f16, xh);
    defer x_dev.free();
    var qp = try dalloc(gpu, prefill_rows * q_width * 2);
    defer qp.free();
    var kp = try dalloc(gpu, prefill_rows * kv_width * 2);
    defer kp.free();
    var vp = try dalloc(gpu, prefill_rows * kv_width * 2);
    defer vp.free();
    var ip = try dalloc(gpu, prefill_rows * idx_width * 2);
    defer ip.free();
    var q_out = try dalloc(gpu, prefill_rows * attn_width * 2);
    defer q_out.free();
    var iq_out = try dalloc(gpu, prefill_rows * 4 * 128 * 2);
    defer iq_out.free();
    var scores = try dalloc(gpu, prefill_rows * nb * 4);
    defer scores.free();
    var ids = try dalloc(gpu, prefill_rows * cuda.exl3_attn.ids_stride * 4);
    defer ids.free();
    var nk = try dalloc(gpu, prefill_rows * 4);
    defer nk.free();
    var attn_out = try dalloc(gpu, prefill_rows * attn_width * 2);
    defer attn_out.free();
    var y = try dalloc(gpu, prefill_rows * hidden * 2);
    defer y.free();
    var ik = try dalloc(gpu, cap_a * 128 * 2);
    defer ik.free();
    var pooled = try dalloc(gpu, (nb + 2) * 128 * 2);
    defer pooled.free();
    var pos0 = try dhost(gpu, i32, &[_]i32{0});
    defer pos0.free();
    const kv_ptr = kv_all.ptr + l.kv_layer * cap_a * oracle.kv_heads * slot_bytes;

    // ---- prefill: 32 tokens in one pipeline ----
    const pre = Bufs{ .x = x_dev.ptr, .qp = qp.ptr, .kp = kp.ptr, .vp = vp.ptr, .ip = ip.ptr, .q_out = q_out.ptr, .iq_out = iq_out.ptr, .scores = scores.ptr, .ids = ids.ptr, .nk = nk.ptr, .attn = attn_out.ptr, .y = y.ptr, .kv = kv_ptr, .ik = ik.ptr, .pooled = pooled.ptr, .pos0 = pos0.ptr };
    try pipeline(ex, at, s, l, pre, dims, prefill_rows);
    try s.synchronize();

    // linears against the Zig f64 decoder, first and last row
    var accs = [_]Acc{ .{ .name = "q_proj" }, .{ .name = "k_proj" }, .{ .name = "v_proj" }, .{ .name = "index_qk_proj" } };
    const gq = try fetch(gpu, f16, qp, 0, prefill_rows * q_width);
    defer gpa.free(gq);
    const gk = try fetch(gpu, f16, kp, 0, prefill_rows * kv_width);
    defer gpa.free(gk);
    const gv = try fetch(gpu, f16, vp, 0, prefill_rows * kv_width);
    defer gpa.free(gv);
    const gi = try fetch(gpu, f16, ip, 0, prefill_rows * idx_width);
    defer gpa.free(gi);
    const want = try gpa.alloc(f64, q_width);
    defer gpa.free(want);
    for ([_]usize{ 0, prefill_rows - 1 }) |r| {
        const xr = xh[r * hidden ..][0..hidden];
        l.q.cpu(xr, want[0..q_width]);
        accs[0].add(gq[r * q_width ..][0..q_width], want[0..q_width]);
        l.k.cpu(xr, want[0..kv_width]);
        accs[1].add(gk[r * kv_width ..][0..kv_width], want[0..kv_width]);
        l.v.cpu(xr, want[0..kv_width]);
        accs[2].add(gv[r * kv_width ..][0..kv_width], want[0..kv_width]);
        l.idx.cpu(xr, want[0..idx_width]);
        accs[3].add(gi[r * idx_width ..][0..idx_width], want[0..idx_width]);
    }
    for (accs) |a| try a.finish(0.02);

    // oracle over the device's own projections
    var cache_o = try oracle.Cache.init(gpa, cap_a);
    defer cache_o.deinit();
    const norms = l.norms();
    var q_acc = Acc{ .name = "q (norm+rope, bf16)" };
    var iq_acc = Acc{ .name = "index q (norm+rope, bf16)" };
    var out_acc = Acc{ .name = "attention out (gated, fp16)" };
    var y_acc = Acc{ .name = "o_proj out (fp16)" };
    const gout = try fetch(gpu, f16, attn_out, 0, prefill_rows * attn_width);
    defer gpa.free(gout);
    const gy = try fetch(gpu, f16, y, 0, prefill_rows * hidden);
    defer gpa.free(gy);
    const gq_out = try fetchBf16(gpu, q_out, 0, prefill_rows * attn_width);
    defer gpa.free(gq_out);
    const giq = try fetchBf16(gpu, iq_out, 0, prefill_rows * 512);
    defer gpa.free(giq);
    var oq: [attn_width]f64 = undefined;
    var oiq: [512]f64 = undefined;
    var oout: [attn_width]f64 = undefined;
    var oscores: [cap_a / 4 + 1]f64 = undefined;
    var info = oracle.RowInfo{};
    var oy: [hidden]f64 = undefined;
    for (0..prefill_rows) |r| {
        try oracle.row(&cache_o, r, .{ .qp = gq[r * q_width ..][0..q_width], .kp = gk[r * kv_width ..][0..kv_width], .vp = gv[r * kv_width ..][0..kv_width], .ip = gi[r * idx_width ..][0..idx_width] }, &norms, &oscores, &oq, &oiq, &oout, &info);
        q_acc.add(gq_out[r * attn_width ..][0..attn_width], &oq);
        iq_acc.add(giq[r * 512 ..][0..512], &oiq);
        out_acc.add(gout[r * attn_width ..][0..attn_width], &oout);
        if (r == 0 or r == prefill_rows - 1) {
            const oh: [attn_width]f16 = blk: {
                var t: [attn_width]f16 = undefined;
                for (&t, oout) |*d, v| d.* = @floatCast(v);
                break :blk t;
            };
            l.o.cpu(&oh, &oy);
            y_acc.add(gy[r * hidden ..][0..hidden], &oy);
        }
    }
    try q_acc.finish(0.01);
    try iq_acc.finish(0.01);
    try out_acc.finish(0.02);
    try y_acc.finish(0.03);
    try tableCheck(gpu, kv_ptr, kv_all, &cache_o, prefill_rows, ik, pooled);
    check.pass("layer {d}: prefill of {d} tokens (real weights) matches the oracle", .{ l.index, prefill_rows });

    // ---- 8 decode steps through one captured graph; pos0 and the input row are the only changes ----
    var xs = try dalloc(gpu, hidden * 2);
    defer xs.free();
    var s_qp = try dalloc(gpu, q_width * 2);
    defer s_qp.free();
    var s_kp = try dalloc(gpu, kv_width * 2);
    defer s_kp.free();
    var s_vp = try dalloc(gpu, kv_width * 2);
    defer s_vp.free();
    var s_ip = try dalloc(gpu, idx_width * 2);
    defer s_ip.free();
    var s_q = try dalloc(gpu, attn_width * 2);
    defer s_q.free();
    var s_iq = try dalloc(gpu, 512 * 2);
    defer s_iq.free();
    var s_scores = try dalloc(gpu, nb * 4);
    defer s_scores.free();
    var s_ids = try dalloc(gpu, cuda.exl3_attn.ids_stride * 4);
    defer s_ids.free();
    var s_nk = try dalloc(gpu, 4);
    defer s_nk.free();
    var s_attn = try dalloc(gpu, attn_width * 2);
    defer s_attn.free();
    var s_y = try dalloc(gpu, hidden * 2);
    defer s_y.free();
    const stepb = Bufs{ .x = xs.ptr, .qp = s_qp.ptr, .kp = s_kp.ptr, .vp = s_vp.ptr, .ip = s_ip.ptr, .q_out = s_q.ptr, .iq_out = s_iq.ptr, .scores = s_scores.ptr, .ids = s_ids.ptr, .nk = s_nk.ptr, .attn = s_attn.ptr, .y = s_y.ptr, .kv = kv_ptr, .ik = ik.ptr, .pooled = pooled.ptr, .pos0 = pos0.ptr };
    try cuda.graph.beginCapture(s, .thread_local);
    try pipeline(ex, at, s, l, stepb, dims, 1);
    var captured = try cuda.graph.endCapture(s);
    defer captured.deinit();
    var exec = try captured.instantiate();
    defer exec.deinit();
    try exec.upload(s);

    var q2 = Acc{ .name = "decode q" };
    var out2 = Acc{ .name = "decode attention out" };
    var y2 = Acc{ .name = "decode o_proj out" };
    for (0..decode_steps) |i| {
        const pos = prefill_rows + i;
        try setPos(pos0, pos);
        try xs.upload(0, std.mem.sliceAsBytes(xh[pos * hidden ..][0..hidden]));
        try exec.launchOn(s);
        try s.synchronize();
        const sq = try fetch(gpu, f16, s_qp, 0, q_width);
        defer gpa.free(sq);
        const sk = try fetch(gpu, f16, s_kp, 0, kv_width);
        defer gpa.free(sk);
        const sv = try fetch(gpu, f16, s_vp, 0, kv_width);
        defer gpa.free(sv);
        const si = try fetch(gpu, f16, s_ip, 0, idx_width);
        defer gpa.free(si);
        const sout = try fetch(gpu, f16, s_attn, 0, attn_width);
        defer gpa.free(sout);
        const sy = try fetch(gpu, f16, s_y, 0, hidden);
        defer gpa.free(sy);
        const sqo = try fetchBf16(gpu, s_q, 0, attn_width);
        defer gpa.free(sqo);
        try oracle.row(&cache_o, pos, .{ .qp = sq, .kp = sk, .vp = sv, .ip = si }, &norms, &oscores, &oq, &oiq, &oout, &info);
        q2.add(sqo, &oq);
        out2.add(sout, &oout);
        var oh: [attn_width]f16 = undefined;
        for (&oh, oout) |*d, v| d.* = @floatCast(v);
        l.o.cpu(&oh, &oy);
        y2.add(sy, &oy);
        // the projections of step i also match the f64 decoder
        l.q.cpu(xh[pos * hidden ..][0..hidden], want[0..q_width]);
        var lin = Acc{ .name = "decode q_proj" };
        lin.add(sq, want[0..q_width]);
        try lin.finish(0.02);
    }
    try q2.finish(0.01);
    try out2.finish(0.02);
    try y2.finish(0.03);
    try tableCheck(gpu, kv_ptr, kv_all, &cache_o, total, ik, pooled);
    check.pass("layer {d}: {d} decode steps replayed through one CUDA graph match the oracle (KV table, raw index keys, pooled blocks too)", .{ l.index, decode_steps });
}

/// The device KV table (this layer), raw index keys and pooled blocks against the oracle's cache for `tokens` positions.
fn tableCheck(gpu: Gpu, kv_ptr: u64, kv_all: cuda.DeviceBuffer, c: *const oracle.Cache, tokens: usize, ik: cuda.DeviceBuffer, pooled: cuda.DeviceBuffer) !void {
    const gpa = gpu.gpa;
    const off = kv_ptr - kv_all.ptr;
    const raw = try fetch(gpu, u16, kv_all, off, tokens * oracle.kv_heads * 2 * oracle.head_dim);
    defer gpa.free(raw);
    var ka = Acc{ .name = "K table (bf16)" };
    var va = Acc{ .name = "V table (bf16)" };
    const kg = try gpa.alloc(f64, oracle.head_dim);
    defer gpa.free(kg);
    for (0..tokens) |t| for (0..oracle.kv_heads) |j| {
        const slot = raw[(t * oracle.kv_heads + j) * 2 * oracle.head_dim ..][0 .. 2 * oracle.head_dim];
        for (0..oracle.head_dim) |d| kg[d] = bf16f(slot[d]);
        ka.add(kg, c.keys[(t * oracle.kv_heads + j) * oracle.head_dim ..][0..oracle.head_dim]);
        for (0..oracle.head_dim) |d| kg[d] = bf16f(slot[oracle.head_dim + d]);
        va.add(kg, c.values[(t * oracle.kv_heads + j) * oracle.head_dim ..][0..oracle.head_dim]);
    };
    try ka.finish(0.01);
    try va.finish(0.0);
    const gik = try fetchBf16(gpu, ik, 0, tokens * 128);
    defer gpa.free(gik);
    var ia = Acc{ .name = "raw index keys (bf16)" };
    ia.add(gik, c.raw[0 .. tokens * 128]);
    try ia.finish(0.0);
    const blocks = tokens / oracle.ratio;
    if (blocks > 0) {
        const gp = try fetchBf16(gpu, pooled, 0, blocks * 128);
        defer gpa.free(gp);
        var pa = Acc{ .name = "pooled blocks (bf16)" };
        pa.add(gp, c.pooled[0 .. blocks * 128]);
        try pa.finish(0.01);
    }
}

// -- Part B -------------------------------------------------------------------

const syn_tokens: usize = 2600;
const syn_cap: usize = 2688;

const Batch = struct { start: usize, rows: usize };

fn partB(gpu: Gpu, ex: *const cuda.exl3.Exl3, at: *const cuda.exl3_attn.Attn, s: cuda.Stream, l: *const Layer) !void {
    const gpa = gpu.gpa;
    const dims: cuda.exl3_attn.Dims = .{ .tokens = syn_cap };
    const nb = syn_cap / 4;
    var rng = std.Random.DefaultPrng.init(0x5b5b + l.index);

    const xh = try gpa.alloc(f16, syn_tokens * hidden);
    defer gpa.free(xh);
    randomHalf(&rng, xh, 1.0);
    const qph = try gpa.alloc(f16, syn_tokens * q_width);
    defer gpa.free(qph);
    randomHalf(&rng, qph, 1.0);
    const kph = try gpa.alloc(f16, syn_tokens * kv_width);
    defer gpa.free(kph);
    randomHalf(&rng, kph, 1.0);
    const vph = try gpa.alloc(f16, syn_tokens * kv_width);
    defer gpa.free(vph);
    randomHalf(&rng, vph, 1.0);

    var x_dev = try dhost(gpu, f16, xh);
    defer x_dev.free();
    var qp = try dhost(gpu, f16, qph);
    defer qp.free();
    var kp = try dhost(gpu, f16, kph);
    defer kp.free();
    var vp = try dhost(gpu, f16, vph);
    defer vp.free();
    var ip = try dalloc(gpu, syn_tokens * idx_width * 2);
    defer ip.free();
    var q_out = try dalloc(gpu, syn_tokens * attn_width * 2);
    defer q_out.free();
    var iq_out = try dalloc(gpu, syn_tokens * 512 * 2);
    defer iq_out.free();
    var kv = try dalloc(gpu, try tableBytes());
    defer kv.free();
    var ik = try dalloc(gpu, syn_cap * 128 * 2);
    defer ik.free();
    var pooled = try dalloc(gpu, (nb + 2) * 128 * 2);
    defer pooled.free();
    var pos0 = try dhost(gpu, i32, &[_]i32{0});
    defer pos0.free();
    const kv_ptr = kv.ptr + l.kv_layer * syn_cap * oracle.kv_heads * slot_bytes;

    // the real index_qk_proj on every random hidden row, then the real prep and pool over the whole context
    try l.idx.run(ex, s, x_dev.ptr, ip.ptr, syn_tokens);
    try at.prep(s, dims, syn_tokens, .{ .qp = qp.ptr, .kp = kp.ptr, .vp = vp.ptr, .ip = ip.ptr, .q_scale = l.d_q_scale.ptr, .k_scale = l.d_k_scale.ptr, .i_scale = l.d_iq_scale.ptr, .inv_freq = l.d_inv.ptr, .pos0 = pos0.ptr, .q_out = q_out.ptr, .kv = kv_ptr, .iq_out = iq_out.ptr, .ik = ik.ptr });
    try at.pool(s, dims, syn_tokens, ik.ptr, pooled.ptr, l.d_ik_scale.ptr, l.d_inv.ptr, pos0.ptr);
    try s.synchronize();

    // oracle context from the device's index projections
    const gi = try fetch(gpu, f16, ip, 0, syn_tokens * idx_width);
    defer gpa.free(gi);
    var oc = try oracle.Cache.init(gpa, syn_cap);
    defer oc.deinit();
    const norms = l.norms();
    var oq: [attn_width]f64 = undefined;
    var oiq: [512]f64 = undefined;
    for (0..syn_tokens) |pos| {
        oracle.prepare(&oc, pos, .{ .qp = qph[pos * q_width ..][0..q_width], .kp = kph[pos * kv_width ..][0..kv_width], .vp = vph[pos * kv_width ..][0..kv_width], .ip = gi[pos * idx_width ..][0..idx_width] }, &norms, &oq, &oiq);
        if ((pos + 1) % oracle.ratio == 0) oracle.poolBlock(&oc, (pos + 1) / oracle.ratio - 1, &norms);
    }
    try tableCheck(gpu, kv_ptr, kv, &oc, syn_tokens, ik, pooled);

    const batches = [_]Batch{ .{ .start = 2592, .rows = 8 }, .{ .start = 2044, .rows = 12 } };
    var scores = try dalloc(gpu, 12 * nb * 4);
    defer scores.free();
    var ids = try dalloc(gpu, 12 * cuda.exl3_attn.ids_stride * 4);
    defer ids.free();
    var nk = try dalloc(gpu, 12 * 4);
    defer nk.free();
    var attn_out = try dalloc(gpu, 12 * attn_width * 2);
    defer attn_out.free();

    // single-row graph of scores+select+decode, fixed buffers
    var s_q = try dalloc(gpu, attn_width * 2);
    defer s_q.free();
    var s_qp = try dalloc(gpu, q_width * 2);
    defer s_qp.free();
    var s_iq = try dalloc(gpu, 512 * 2);
    defer s_iq.free();
    var s_scores = try dalloc(gpu, nb * 4);
    defer s_scores.free();
    var s_ids = try dalloc(gpu, cuda.exl3_attn.ids_stride * 4);
    defer s_ids.free();
    var s_nk = try dalloc(gpu, 4);
    defer s_nk.free();
    var s_out = try dalloc(gpu, attn_width * 2);
    defer s_out.free();
    try cuda.graph.beginCapture(s, .thread_local);
    try at.scores(s, 1, s_iq.ptr, pooled.ptr, s_scores.ptr, pos0.ptr, nb, nb);
    try at.select(s, 1, s_scores.ptr, s_ids.ptr, s_nk.ptr, pos0.ptr, nb);
    try at.decode(s, dims, 1, s_q.ptr, s_qp.ptr, kv_ptr, s_ids.ptr, s_out.ptr, pos0.ptr);
    var captured = try cuda.graph.endCapture(s);
    defer captured.deinit();
    var exec = try captured.instantiate();
    defer exec.deinit();
    try exec.upload(s);

    var sel_diff_total: usize = 0;
    var sel_rows: usize = 0;
    var own = Acc{ .name = "attention out vs oracle on the device's key list (sparse and dense rows)" };
    var full = Acc{ .name = "sparse attention out vs oracle with its own selection" };
    var graph_equal: usize = 0;
    var graph_rows: usize = 0;
    var dense_rows: usize = 0;
    for (batches) |bt| {
        try setPos(pos0, bt.start);
        try at.scores(s, bt.rows, iq_out.ptr + bt.start * 512 * 2, pooled.ptr, scores.ptr, pos0.ptr, nb, nb);
        try at.select(s, bt.rows, scores.ptr, ids.ptr, nk.ptr, pos0.ptr, nb);
        try at.decode(s, dims, bt.rows, q_out.ptr + bt.start * attn_width * 2, qp.ptr + bt.start * q_width * 2, kv_ptr, ids.ptr, attn_out.ptr, pos0.ptr);
        try s.synchronize();
        const gids = try fetch(gpu, i32, ids, 0, bt.rows * cuda.exl3_attn.ids_stride);
        defer gpa.free(gids);
        const gnk = try fetch(gpu, i32, nk, 0, bt.rows);
        defer gpa.free(gnk);
        const gout = try fetch(gpu, f16, attn_out, 0, bt.rows * attn_width);
        defer gpa.free(gout);

        for (0..bt.rows) |r| {
            const pos = bt.start + r;
            const end = pos + 1;
            const complete = end / oracle.ratio;
            const sparse = complete > oracle.top_blocks;
            // oracle for this row
            oracle.prepare(&oc, pos, .{ .qp = qph[pos * q_width ..][0..q_width], .kp = kph[pos * kv_width ..][0..kv_width], .vp = vph[pos * kv_width ..][0..kv_width], .ip = gi[pos * idx_width ..][0..idx_width] }, &norms, &oq, &oiq);
            var okeys: [oracle.max_keys]usize = undefined;
            var gkeys: [oracle.max_keys]usize = undefined;
            var onk: usize = undefined;
            const gn: usize = @intCast(gnk[r]);
            if (sparse) {
                sel_rows += 1;
                const osc = try gpa.alloc(f64, complete);
                defer gpa.free(osc);
                oracle.blockScores(&oc, &oiq, complete, osc);
                var blocks: [oracle.top_blocks]usize = undefined;
                try oracle.selectBlocks(gpa, osc, &blocks);
                onk = oracle.keyList(end, &blocks, &okeys);
                try check.expect(gn == onk, "row {d}: device key count {d}, oracle {d}", .{ pos, gn, onk });
                // device block list: ids[j * 4] / 4, in block order
                var diff: usize = 0;
                const sorted = try gpa.dupe(f64, osc);
                defer gpa.free(sorted);
                std.mem.sort(f64, sorted, {}, std.sort.desc(f64));
                const cut = sorted[oracle.top_blocks - 1];
                const in_o = try gpa.alloc(bool, complete);
                defer gpa.free(in_o);
                @memset(in_o, false);
                for (blocks) |b| in_o[b] = true;
                const in_g = try gpa.alloc(bool, complete);
                defer gpa.free(in_g);
                @memset(in_g, false);
                for (0..oracle.top_blocks) |j| {
                    const first: usize = @intCast(gids[r * cuda.exl3_attn.ids_stride + j * 4]);
                    try check.expect(first % 4 == 0 and first / 4 < complete, "row {d}: bad device id {d}", .{ pos, first });
                    for (0..4) |k| try check.expect(gids[r * cuda.exl3_attn.ids_stride + j * 4 + k] == first + k, "row {d}: ids not four consecutive keys", .{pos});
                    if (j > 0) try check.expect(first > @as(usize, @intCast(gids[r * cuda.exl3_attn.ids_stride + (j - 1) * 4])), "row {d}: ids not in block order", .{pos});
                    in_g[first / 4] = true;
                }
                for (0..complete) |b| if (in_o[b] != in_g[b]) {
                    diff += 1;
                    const near = @abs(osc[b] - cut) <= 1e-4 * @abs(cut);
                    try check.expect(near, "row {d}: block {d} differs and its score {d} is not within 1e-4 of the cut {d}", .{ pos, b, osc[b], cut });
                };
                sel_diff_total += diff;
                const tail_n = end - oracle.ratio * complete;
                for (0..tail_n) |t| try check.expect(gids[r * cuda.exl3_attn.ids_stride + oracle.top_blocks * 4 + t] == @as(i32, @intCast(oracle.ratio * complete + t)), "row {d}: tail", .{pos});
                // negative control: the same row attending to the 512 WORST blocks must be far outside the tolerance, so the
                // agreement above is not vacuous (the selection really changes the output)
                if (r == bt.rows - 1) {
                    const idx = try gpa.alloc(usize, complete);
                    defer gpa.free(idx);
                    for (idx, 0..) |*v, i| v.* = i;
                    const Ctx = struct {
                        s: []const f64,
                        fn less(c: @This(), a: usize, b: usize) bool {
                            return if (c.s[a] != c.s[b]) c.s[a] < c.s[b] else a < b;
                        }
                    };
                    std.mem.sort(usize, idx, Ctx{ .s = osc }, Ctx.less);
                    var worst: [oracle.top_blocks]usize = undefined;
                    @memcpy(&worst, idx[0..oracle.top_blocks]);
                    std.mem.sort(usize, &worst, {}, std.sort.asc(usize));
                    var wkeys: [oracle.max_keys]usize = undefined;
                    const wn = oracle.keyList(end, &worst, &wkeys);
                    var wout: [attn_width]f64 = undefined;
                    oracle.attend(&oc, &oq, qph[pos * q_width ..][0..q_width], wkeys[0..wn], &wout);
                    var ctl = Acc{ .name = "control (worst 512 blocks)" };
                    ctl.add(gout[r * attn_width ..][0..attn_width], &wout);
                    std.debug.print("  negative control, worst-512 selection: max_abs {e:.3} peak {e:.3} rel {d:.3}\n", .{ ctl.max_abs, ctl.peak, ctl.max_abs / ctl.peak });
                    // above the 2% attention tolerance with margin (layer 7's last row only moves 4.7%: its output depends weakly on which blocks are read)
                    try check.expect(ctl.max_abs > 0.03 * ctl.peak, "the worst-block control is not distinguishable", .{});
                }
                // attention on the oracle's own selection
                var oout: [attn_width]f64 = undefined;
                oracle.attend(&oc, &oq, qph[pos * q_width ..][0..q_width], okeys[0..onk], &oout);
                full.add(gout[r * attn_width ..][0..attn_width], &oout);
            } else {
                dense_rows += 1;
                onk = end;
                try check.expect(gn == end, "row {d}: dense key count {d}", .{ pos, gn });
                for (0..end) |i| okeys[i] = i;
                var oout: [attn_width]f64 = undefined;
                oracle.attend(&oc, &oq, qph[pos * q_width ..][0..q_width], okeys[0..onk], &oout);
                own.add(gout[r * attn_width ..][0..attn_width], &oout);
                continue;
            }
            // attention on the device's key list (isolates the decode kernel from near-tie selection flips)
            for (0..gn) |j| gkeys[j] = @intCast(gids[r * cuda.exl3_attn.ids_stride + j]);
            var gout_o: [attn_width]f64 = undefined;
            oracle.attend(&oc, &oq, qph[pos * q_width ..][0..q_width], gkeys[0..gn], &gout_o);
            own.add(gout[r * attn_width ..][0..attn_width], &gout_o);

            // graph replay of this row against the eager launch, byte for byte
            try s_q.copyFrom(0, q_out.ptr + pos * attn_width * 2, attn_width * 2, s.handle);
            try s_qp.copyFrom(0, qp.ptr + pos * q_width * 2, q_width * 2, s.handle);
            try s_iq.copyFrom(0, iq_out.ptr + pos * 512 * 2, 512 * 2, s.handle);
            try s.synchronize();
            try setPos(pos0, pos);
            try exec.launchOn(s);
            try s.synchronize();
            const row_bytes = try fetch(gpu, u16, s_out, 0, attn_width);
            defer gpa.free(row_bytes);
            const eager = std.mem.sliceAsBytes(gout[r * attn_width ..][0..attn_width]);
            graph_equal += @intFromBool(std.mem.eql(u8, eager, std.mem.sliceAsBytes(row_bytes)));
            graph_rows += 1;
        }
        try setPos(pos0, 0);
    }
    std.debug.print("  selection: {d} sparse rows, {d} dense rows, {d} blocks differing from the oracle (each within 1e-4 of the cut)\n", .{ sel_rows, dense_rows, sel_diff_total });
    try own.finish(0.02);
    try full.finish(0.02);
    std.debug.print("  graph replay of scores+select+decode equals the eager launch byte for byte on {d}/{d} sparse rows\n", .{ graph_equal, graph_rows });
    try check.expect(graph_equal == graph_rows, "a replayed row differs from the eager launch", .{});
    try check.expect(sel_rows == 8 + 5 and dense_rows == 7, "rows {d} sparse {d} dense", .{ sel_rows, dense_rows });
    check.pass("layer {d}: {d}-token synthetic context: top-k selection (512 of up to 650 blocks + tail), dense/sparse boundary at 2051/2052 keys, and sparse decode match the oracle", .{ l.index, syn_tokens });
}

fn tableBytes() !usize {
    return @intCast(try cachemod.contextKvRank(syn_cap, .bf16, 1));
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    const layer_a: usize = if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 3;
    const layer_b: usize = if (rest.len > 2) try std.fmt.parseInt(usize, rest[2], 10) else 7;
    for ([_]usize{ layer_a, layer_b }) |l| try check.expect(l % 4 == 3 and l < 48, "layer {d} is not a full-attention layer", .{l});

    var pack: real.Pack = .{ .files = undefined };
    defer for (pack.files[0..pack.open]) |*f| f.close(gpu.io);
    for (0..real.shards) |i| {
        var path: [512]u8 = undefined;
        const p = try std.fmt.bufPrint(&path, "{s}/model-{d:0>5}-of-{d:0>5}.safetensors", .{ dir, i + 1, real.shards });
        pack.files[i] = try st.File.open(gpu.gpa, gpu.io, p);
        pack.open += 1;
    }

    var ex = try cuda.exl3.Exl3.load(gpu.d);
    defer ex.unload();
    var at = try cuda.exl3_attn.Attn.load(gpu.d);
    defer at.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();

    // Part A's table holds every full-attention layer (as the engine's does); the second layer must not touch the first's slice.
    var kv_all = try dalloc(gpu, @intCast(try cachemod.contextKvRank(cap_a, .bf16, 1)));
    defer kv_all.free();

    var guard: ?[]u8 = null;
    defer if (guard) |g| gpu.gpa.free(g);
    for ([_]usize{ layer_a, layer_b }, 0..) |layer, n| {
        var l = try loadLayer(gpu, &pack, layer);
        defer l.free(gpu.gpa);
        try partA(gpu, &ex, &at, stream, &l, kv_all);
        if (n == 0) {
            const first = l.kv_layer * cap_a * oracle.kv_heads * slot_bytes;
            const snap = try gpu.gpa.alloc(u8, cap_a * oracle.kv_heads * slot_bytes);
            try kv_all.download(first, snap);
            guard = snap;
        }
        try partB(gpu, &ex, &at, stream, &l);
    }
    // layer a's slice of the shared table is untouched by layer b
    if (guard) |g| {
        const a_off = ((layer_a + 1) / 4 - 1) * cap_a * oracle.kv_heads * slot_bytes;
        const now = try gpu.gpa.alloc(u8, g.len);
        defer gpu.gpa.free(now);
        try kv_all.download(a_off, now);
        try check.sameBytes("layer a KV slice after layer b ran", now, g);
    }
    check.pass("exl3-attn: layers {d} and {d} (real weights) prefill, graph decode and sparse indexer match the CPU oracle", .{ layer_a, layer_b });
}
