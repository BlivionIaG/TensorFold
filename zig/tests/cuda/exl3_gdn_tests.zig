//! A real Gated DeltaNet layer of the EXL3 pack on the device: in_proj_qkv / in_proj_z / out_proj (EXL3), dense a/b, conv, delta
//! rule, gated norm, with the conv and recurrent state in the device table. A 16-token prefill then 8 decode steps, against
//! (1) the same 24 tokens decoded one at a time, byte for byte, (2) the f64 oracle fed the device's own projection outputs
//! (the conv / gate / delta / norm maths and the final state), (3) the f64 oracle end to end from the dequantized weights.
//! Usage: tf-cuda-test exl3-gdn <pack dir> [layer ...]   (default layers 0 1 2)

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const format = @import("exl3_format.zig");
const oracle = @import("exl3_gdn_cpu.zig");
const real = @import("exl3_real_tests.zig");

const Gpu = check.Gpu;
const dims = oracle.qwen;
const hidden = 2560;
const prefill = 16;
const decode = 8;
const total = prefill + decode;
const c = dims.channels();
const zw = dims.zWidth();

const Diff = struct { max_abs: f64, peak: f64 };

fn diff(got: anytype, want: []const f64) Diff {
    var d: Diff = .{ .max_abs = 0, .peak = 0 };
    for (got, want) |g, w| {
        d.max_abs = @max(d.max_abs, @abs(@as(f64, @floatCast(g)) - w));
        d.peak = @max(d.peak, @abs(w));
    }
    return d;
}

fn bytesOf(gpu: Gpu, pack: *const real.Pack, name: []const u8, dtype: @import("safetensors.zig").DType, count: usize) ![]align(8) u8 {
    const t = pack.find(name) orelse return error.MissingTensor;
    try check.expect(t.dtype == dtype and t.numel() == count, "{s}: dtype/size differ from the layer shape", .{name});
    const out = try gpu.gpa.alignedAlloc(u8, .@"8", t.bytes.len);
    @memcpy(out, t.bytes);
    return out;
}

fn halfs(gpa: std.mem.Allocator, bytes: []align(8) const u8, bf16: bool) ![]f64 {
    const n = bytes.len / 2;
    const out = try gpa.alloc(f64, n);
    const w: [*]const u16 = @ptrCast(bytes.ptr);
    for (out, 0..) |*o, i| {
        o.* = if (bf16) @as(f64, @floatCast(@as(f32, @bitCast(@as(u32, w[i]) << 16)))) else @as(f64, @floatCast(@as(f16, @bitCast(w[i]))));
    }
    return out;
}

/// Dequantized W[k][n] in f64, y = x W.
fn dequant(gpa: std.mem.Allocator, m: real.Mat) ![]f64 {
    const k = m.tiles_k * 16;
    const n = m.tiles_n * 16;
    const w = try gpa.alloc(f64, k * n);
    errdefer gpa.free(w);
    const scratch = try gpa.alloc(f64, 2 * k * n);
    defer gpa.free(scratch);
    format.dequantize(.{ .halves = m.halves }, .mul1, m.trellis, m.tiles_k, m.tiles_n, m.suh, m.svh, w, scratch);
    return w;
}

fn matvec(w: []const f64, k: usize, n: usize, x: []const f64, y: []f64) void {
    @memset(y, 0);
    for (0..k) |i| {
        const xi = x[i];
        const row = w[i * n ..][0..n];
        for (y, row) |*o, wi| o.* += xi * wi;
    }
}

const Run = struct {
    out: []f16, // [total][2560]
    qkv: []f16, // [total][c]  device projection outputs
    z: []f16, // [total][zw]
    ab: []f32, // [total][96]
    gated: []f16, // [total][zw]
    conv: []f16, // final conv state [3][c]
    rec: []f32, // final recurrent state [48][128][128]

    fn deinit(r: Run, gpa: std.mem.Allocator) void {
        gpa.free(r.out);
        gpa.free(r.qkv);
        gpa.free(r.z);
        gpa.free(r.ab);
        gpa.free(r.gated);
        gpa.free(r.conv);
        gpa.free(r.rec);
    }
};

const Ctx = struct {
    gpu: Gpu,
    gdn: *const cuda.exl3_gdn.Gdn,
    ex: *const cuda.exl3.Exl3,
    stream: cuda.Stream,
    layer: cuda.exl3_gdn.Layer,
    table: cuda.exl3_gdn.Table,
    scratch: cuda.exl3_gdn.Scratch,
    x: u64,
    out: u64,
};

fn download(dev: cuda.DeviceBuffer, offset: usize, dst: anytype) !void {
    try dev.download(offset, std.mem.sliceAsBytes(dst));
}

/// Runs `segs` (token start, count) in order against `slot` from a zeroed state, collecting each call's scratch.
fn runSegments(cx: Ctx, slot: usize, segs: []const [2]usize) !Run {
    const gpa = cx.gpu.gpa;
    const r: Run = .{
        .out = try gpa.alloc(f16, total * hidden),
        .qkv = try gpa.alloc(f16, total * c),
        .z = try gpa.alloc(f16, total * zw),
        .ab = try gpa.alloc(f32, total * 2 * dims.nv),
        .gated = try gpa.alloc(f16, total * zw),
        .conv = try gpa.alloc(f16, dims.convElems()),
        .rec = try gpa.alloc(f32, dims.recElems()),
    };
    try cx.table.reset(cx.stream, slot);
    try cx.stream.synchronize();
    const conv_ptr = try cx.table.convPtr(slot, 0);
    const rec_ptr = try cx.table.recPtr(slot, 0);
    for (segs) |seg| {
        const t0 = seg[0];
        const n = seg[1];
        try cx.gdn.forward(cx.ex, cx.stream, cx.layer, conv_ptr, rec_ptr, cx.x + t0 * hidden * 2, n, cx.scratch, cx.out + t0 * hidden * 2);
        try cx.stream.synchronize();
        try download(cx.scratch.qkv, 0, r.qkv[t0 * c ..][0 .. n * c]);
        try download(cx.scratch.z, 0, r.z[t0 * zw ..][0 .. n * zw]);
        try download(cx.scratch.ab, 0, r.ab[t0 * 96 ..][0 .. n * 96]);
        try download(cx.scratch.gated, 0, r.gated[t0 * zw ..][0 .. n * zw]);
        // out lives in a caller buffer of total rows
        const tmp = cuda.DeviceBuffer{ .d = cx.gpu.d, .ptr = cx.out, .len = total * hidden * 2, .borrowed = true };
        try download(tmp, t0 * hidden * 2, r.out[t0 * hidden ..][0 .. n * hidden]);
    }
    const cb = cuda.DeviceBuffer{ .d = cx.gpu.d, .ptr = conv_ptr, .len = cuda.exl3_gdn.conv_bytes, .borrowed = true };
    try download(cb, 0, r.conv);
    const rb = cuda.DeviceBuffer{ .d = cx.gpu.d, .ptr = rec_ptr, .len = cuda.exl3_gdn.rec_bytes, .borrowed = true };
    try download(rb, 0, r.rec);
    return r;
}

fn toF64(gpa: std.mem.Allocator, xs: anytype) ![]f64 {
    const out = try gpa.alloc(f64, xs.len);
    for (out, xs) |*o, v| o.* = @floatCast(v);
    return out;
}

fn layerCheck(gpu: Gpu, gdn: *const cuda.exl3_gdn.Gdn, ex: *const cuda.exl3.Exl3, stream: cuda.Stream, pack: *const real.Pack, layer: usize) !void {
    const gpa = gpu.gpa;
    var stem_buf: [96]u8 = undefined;
    const stem = try std.fmt.bufPrint(&stem_buf, "model.language_model.layers.{d}.linear_attn", .{layer});
    var nb: [128]u8 = undefined;
    const qkv_m = try real.mat(pack, gpa, try std.fmt.bufPrint(&nb, "{s}.in_proj_qkv", .{stem}));
    const z_m = try real.mat(pack, gpa, try std.fmt.bufPrint(&nb, "{s}.in_proj_z", .{stem}));
    const out_m = try real.mat(pack, gpa, try std.fmt.bufPrint(&nb, "{s}.out_proj", .{stem}));
    try check.expect(qkv_m.tiles_k * 16 == hidden and qkv_m.tiles_n * 16 == c, "in_proj_qkv is {d}x{d}", .{ qkv_m.tiles_k * 16, qkv_m.tiles_n * 16 });
    try check.expect(z_m.tiles_k * 16 == hidden and z_m.tiles_n * 16 == zw, "in_proj_z is {d}x{d}", .{ z_m.tiles_k * 16, z_m.tiles_n * 16 });
    try check.expect(out_m.tiles_k * 16 == zw and out_m.tiles_n * 16 == hidden, "out_proj is {d}x{d}", .{ out_m.tiles_k * 16, out_m.tiles_n * 16 });
    std.debug.print("layer {d}: halves qkv={d} z={d} out={d}\n", .{ layer, qkv_m.halves, z_m.halves, out_m.halves });

    const f16t = @import("safetensors.zig").DType.f16;
    const bf16t = @import("safetensors.zig").DType.bf16;
    const a_b = try bytesOf(gpu, pack, try std.fmt.bufPrint(&nb, "{s}.in_proj_a.weight", .{stem}), f16t, dims.nv * hidden);
    defer gpa.free(a_b);
    const b_b = try bytesOf(gpu, pack, try std.fmt.bufPrint(&nb, "{s}.in_proj_b.weight", .{stem}), f16t, dims.nv * hidden);
    defer gpa.free(b_b);
    const conv_b = try bytesOf(gpu, pack, try std.fmt.bufPrint(&nb, "{s}.conv1d.weight", .{stem}), bf16t, c * dims.taps);
    defer gpa.free(conv_b);
    const alog_b = try bytesOf(gpu, pack, try std.fmt.bufPrint(&nb, "{s}.A_log", .{stem}), bf16t, dims.nv);
    defer gpa.free(alog_b);
    const dt_b = try bytesOf(gpu, pack, try std.fmt.bufPrint(&nb, "{s}.dt_bias", .{stem}), bf16t, dims.nv);
    defer gpa.free(dt_b);
    const norm_b = try bytesOf(gpu, pack, try std.fmt.bufPrint(&nb, "{s}.norm.weight", .{stem}), bf16t, dims.dv);
    defer gpa.free(norm_b);

    const a_w = try halfs(gpa, a_b, false);
    defer gpa.free(a_w);
    const b_w = try halfs(gpa, b_b, false);
    defer gpa.free(b_w);
    const conv_w = try halfs(gpa, conv_b, true);
    defer gpa.free(conv_w);
    const a_log = try halfs(gpa, alog_b, true);
    defer gpa.free(a_log);
    const dt_bias = try halfs(gpa, dt_b, true);
    defer gpa.free(dt_bias);
    const norm_w = try halfs(gpa, norm_b, true);
    defer gpa.free(norm_w);
    var nmean: f64 = 0;
    for (norm_w) |w| nmean += w / @as(f64, @floatFromInt(norm_w.len));
    std.debug.print("  linear_attn.norm.weight mean {d:.4} (the model applies it as stored, not 1 + w); A_log [{d:.3}, {d:.3}]; dt_bias [{d:.3}, {d:.3}]\n", .{
        nmean,
        std.mem.min(f64, a_log),
        std.mem.max(f64, a_log),
        std.mem.min(f64, dt_bias),
        std.mem.max(f64, dt_bias),
    });
    const params: oracle.Params = .{ .conv_w = conv_w, .a_log = a_log, .dt_bias = dt_bias, .norm_w = norm_w };

    // device weights
    var dq = try real.upload(gpu, qkv_m);
    defer dq.free();
    var dz = try real.upload(gpu, z_m);
    defer dz.free();
    var do_ = try real.upload(gpu, out_m);
    defer do_.free();
    var da = try cuda.DeviceBuffer.fromHost(gpu.d, a_b);
    defer da.free();
    var db = try cuda.DeviceBuffer.fromHost(gpu.d, b_b);
    defer db.free();
    var dc = try cuda.DeviceBuffer.fromHost(gpu.d, conv_b);
    defer dc.free();
    var dal = try cuda.DeviceBuffer.fromHost(gpu.d, alog_b);
    defer dal.free();
    var ddt = try cuda.DeviceBuffer.fromHost(gpu.d, dt_b);
    defer ddt.free();
    var dn = try cuda.DeviceBuffer.fromHost(gpu.d, norm_b);
    defer dn.free();
    const lw: cuda.exl3_gdn.Layer = .{
        .qkv = .{ .trellis = dq.trellis.ptr, .suh = dq.suh.ptr, .svh = dq.svh.ptr, .k = hidden, .n = c, .halves = qkv_m.halves },
        .z = .{ .trellis = dz.trellis.ptr, .suh = dz.suh.ptr, .svh = dz.svh.ptr, .k = hidden, .n = zw, .halves = z_m.halves },
        .out = .{ .trellis = do_.trellis.ptr, .suh = do_.suh.ptr, .svh = do_.svh.ptr, .k = zw, .n = hidden, .halves = out_m.halves },
        .a_w = da.ptr,
        .b_w = db.ptr,
        .conv_w = dc.ptr,
        .a_log = dal.ptr,
        .dt_bias = ddt.ptr,
        .norm_w = dn.ptr,
    };

    // input: unit-variance rows (a sum of four uniforms), rounded to fp16 as the EXL3 path hands them over
    const x = try gpa.alloc(f16, total * hidden);
    defer gpa.free(x);
    var rng = std.Random.DefaultPrng.init(0x6d5 + layer);
    for (x) |*v| {
        var s: f32 = 0;
        for (0..4) |_| s += rng.random().float(f32);
        v.* = @floatCast((s - 2.0) * 1.7320508);
    }
    const x64 = try toF64(gpa, x);
    defer gpa.free(x64);
    var x_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer x_dev.free();
    var out_dev = try cuda.DeviceBuffer.alloc(gpu.d, total * hidden * 2);
    defer out_dev.free();
    var scratch = try cuda.exl3_gdn.Scratch.alloc(gpu.d, prefill);
    defer scratch.free();
    var table = try cuda.exl3_gdn.Table.alloc(gpu.d, 1, 2);
    defer table.free();

    const cx: Ctx = .{ .gpu = gpu, .gdn = gdn, .ex = ex, .stream = stream, .layer = lw, .table = table, .scratch = scratch, .x = x_dev.ptr, .out = out_dev.ptr };

    // A: prefill 16 then 8 single-token decode steps (slot 0). B: all 24 one token at a time (slot 1).
    var segs_a: [1 + decode][2]usize = undefined;
    segs_a[0] = .{ 0, prefill };
    for (0..decode) |i| segs_a[1 + i] = .{ prefill + i, 1 };
    var segs_b: [total][2]usize = undefined;
    for (0..total) |i| segs_b[i] = .{ i, 1 };
    const run_a = try runSegments(cx, 0, &segs_a);
    defer run_a.deinit(gpa);
    const run_b = try runSegments(cx, 1, &segs_b);
    defer run_b.deinit(gpa);

    // 1. prefill-then-decode == one token at a time: same kernels per token, so the bytes must be equal
    try check.sameBytes("qkv rows", std.mem.sliceAsBytes(run_a.qkv), std.mem.sliceAsBytes(run_b.qkv));
    try check.sameBytes("a/b rows", std.mem.sliceAsBytes(run_a.ab), std.mem.sliceAsBytes(run_b.ab));
    try check.sameBytes("gated rows", std.mem.sliceAsBytes(run_a.gated), std.mem.sliceAsBytes(run_b.gated));
    try check.sameBytes("layer outputs", std.mem.sliceAsBytes(run_a.out), std.mem.sliceAsBytes(run_b.out));
    try check.sameBytes("conv state", std.mem.sliceAsBytes(run_a.conv), std.mem.sliceAsBytes(run_b.conv));
    try check.sameBytes("recurrent state", std.mem.sliceAsBytes(run_a.rec), std.mem.sliceAsBytes(run_b.rec));
    check.pass("layer {d}: prefill(16)+decode(8) equals 24 single-token steps byte for byte (outputs and both states)", .{layer});

    // 2. oracle fed the device's projection outputs
    const qkv_g = try toF64(gpa, run_a.qkv);
    defer gpa.free(qkv_g);
    const z_g = try toF64(gpa, run_a.z);
    defer gpa.free(z_g);
    const a_g = try gpa.alloc(f64, total * dims.nv);
    defer gpa.free(a_g);
    const b_g = try gpa.alloc(f64, total * dims.nv);
    defer gpa.free(b_g);
    for (0..total) |t| for (0..dims.nv) |h| {
        a_g[t * dims.nv + h] = run_a.ab[t * 96 + h];
        b_g[t * dims.nv + h] = run_a.ab[t * 96 + dims.nv + h];
    };
    // a and b against f64 dense dot products of the fp16 input
    {
        const want = try gpa.alloc(f64, 2 * dims.nv);
        defer gpa.free(want);
        var worst: Diff = .{ .max_abs = 0, .peak = 0 };
        for (0..total) |t| {
            oracle.rowsDot(a_w, x64[t * hidden ..][0..hidden], want[0..dims.nv]);
            oracle.rowsDot(b_w, x64[t * hidden ..][0..hidden], want[dims.nv..]);
            const dd = diff(run_a.ab[t * 96 ..][0..96], want);
            worst.max_abs = @max(worst.max_abs, dd.max_abs);
            worst.peak = @max(worst.peak, dd.peak);
        }
        std.debug.print("  in_proj_a/b: max_abs {e} peak {e}\n", .{ worst.max_abs, worst.peak });
        try check.expect(worst.max_abs <= 1e-4 * worst.peak and worst.peak > 0, "in_proj_a/b differ from the f64 dot", .{});
    }
    var st = try oracle.State.init(gpa, dims);
    defer st.deinit(gpa);
    const gated_c = try gpa.alloc(f64, total * zw);
    defer gpa.free(gated_c);
    try oracle.tokens(gpa, dims, params, st, total, qkv_g, z_g, a_g, b_g, gated_c);
    const dg = diff(run_a.gated, gated_c);
    const rec_want = st.rec;
    const dr = diff(run_a.rec, rec_want);
    std.debug.print("  conv+gates+delta+norm vs oracle (device projections): gated max_abs {e} peak {e} (rel {e}); recurrent state max_abs {e} peak {e} (rel {e})\n", .{ dg.max_abs, dg.peak, dg.max_abs / dg.peak, dr.max_abs, dr.peak, dr.max_abs / dr.peak });
    // the gated rows are fp16: half an ulp is 2^-11 of the element, so 1e-3 of the peak bounds rounding with fp32 sums
    try check.expect(dg.max_abs <= 1e-3 * dg.peak and dg.peak > 0, "gated rows differ from the oracle", .{});
    // the state is fp32 end to end: 24 tokens of fp32 sums against f64 (observed 2-3e-7)
    try check.expect(dr.max_abs <= 1e-5 * dr.peak and dr.peak > 0, "recurrent state differs from the oracle", .{});
    // the conv state holds the last three raw rows, exactly
    for (0..dims.keep()) |r| {
        const row = total - dims.keep() + r;
        try check.sameBytes("conv state rows", std.mem.sliceAsBytes(run_a.conv[r * c ..][0..c]), std.mem.sliceAsBytes(run_a.qkv[row * c ..][0..c]));
    }
    check.pass("layer {d}: conv/gates/delta/norm and both final states match the f64 oracle (gated rel {e}, state rel {e})", .{ layer, dg.max_abs / dg.peak, dr.max_abs / dr.peak });

    // 3. out_proj on the device's own gated rows against the dequantized matrix; and the whole layer end to end
    const w_out = try dequant(gpa, out_m);
    defer gpa.free(w_out);
    const gated_g = try toF64(gpa, run_a.gated);
    defer gpa.free(gated_g);
    const out_c = try gpa.alloc(f64, total * hidden);
    defer gpa.free(out_c);
    for (0..total) |t| matvec(w_out, zw, hidden, gated_g[t * zw ..][0..zw], out_c[t * hidden ..][0..hidden]);
    const do1 = diff(run_a.out, out_c);
    std.debug.print("  out_proj vs Zig decoder: max_abs {e} peak {e} (rel {e})\n", .{ do1.max_abs, do1.peak, do1.max_abs / do1.peak });
    try check.expect(do1.max_abs <= 2e-3 * do1.peak and do1.peak > 0, "out_proj differs from the Zig decoder", .{});

    const w_qkv = try dequant(gpa, qkv_m);
    defer gpa.free(w_qkv);
    const w_z = try dequant(gpa, z_m);
    defer gpa.free(w_z);
    const qkv_f = try gpa.alloc(f64, total * c);
    defer gpa.free(qkv_f);
    const z_f = try gpa.alloc(f64, total * zw);
    defer gpa.free(z_f);
    const a_f = try gpa.alloc(f64, total * dims.nv);
    defer gpa.free(a_f);
    const b_f = try gpa.alloc(f64, total * dims.nv);
    defer gpa.free(b_f);
    for (0..total) |t| {
        const xr = x64[t * hidden ..][0..hidden];
        matvec(w_qkv, hidden, c, xr, qkv_f[t * c ..][0..c]);
        matvec(w_z, hidden, zw, xr, z_f[t * zw ..][0..zw]);
        oracle.rowsDot(a_w, xr, a_f[t * dims.nv ..][0..dims.nv]);
        oracle.rowsDot(b_w, xr, b_f[t * dims.nv ..][0..dims.nv]);
    }
    const dqk = diff(run_a.qkv, qkv_f);
    const dzz = diff(run_a.z, z_f);
    std.debug.print("  in_proj_qkv vs Zig decoder: max_abs {e} peak {e} (rel {e}); in_proj_z: max_abs {e} peak {e} (rel {e})\n", .{ dqk.max_abs, dqk.peak, dqk.max_abs / dqk.peak, dzz.max_abs, dzz.peak, dzz.max_abs / dzz.peak });
    var st2 = try oracle.State.init(gpa, dims);
    defer st2.deinit(gpa);
    const gated_f = try gpa.alloc(f64, total * zw);
    defer gpa.free(gated_f);
    try oracle.tokens(gpa, dims, params, st2, total, qkv_f, z_f, a_f, b_f, gated_f);
    const full_c = try gpa.alloc(f64, total * hidden);
    defer gpa.free(full_c);
    for (0..total) |t| matvec(w_out, zw, hidden, gated_f[t * zw ..][0..zw], full_c[t * hidden ..][0..hidden]);
    const dfull = diff(run_a.out, full_c);
    const dgf = diff(run_a.gated, gated_f);
    const drf = diff(run_a.rec, st2.rec);
    std.debug.print("  end to end vs f64 oracle on the dequantized weights: layer out max_abs {e} peak {e} (rel {e}); gated rel {e}; state rel {e}\n", .{ dfull.max_abs, dfull.peak, dfull.max_abs / dfull.peak, dgf.max_abs / dgf.peak, drf.max_abs / drf.peak });
    // fp16 rounds the qkv/z rows, the gated rows and the output (half an ulp is 2^-11 = 4.9e-4 of an element), the recurrence
    // carries the first two forward: 4e-3 of the peak is about eight half-ulps, observed 3-5e-4
    try check.expect(dfull.max_abs <= 4e-3 * dfull.peak and dfull.peak > 0, "layer output differs from the f64 oracle", .{});
    check.pass("layer {d}: 16-token prefill + 8 decode steps match the f64 oracle end to end (rel {e})", .{ layer, dfull.max_abs / dfull.peak });
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    var pack: real.Pack = .{ .files = undefined };
    defer pack.close(gpu.io);
    try pack.openAll(gpu, rest[0]);

    var ex = try cuda.exl3.Exl3.load(gpu.d);
    defer ex.unload();
    var gdn = try cuda.exl3_gdn.Gdn.load(gpu.d);
    defer gdn.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();

    // the device sizes are the host table's
    const tbl: oracle.Table = .{ .dims = dims, .layers = 1, .slots = 1 };
    try check.expect(tbl.convBytesPer() == cuda.exl3_gdn.conv_bytes and tbl.recBytesPer() == cuda.exl3_gdn.rec_bytes, "device state sizes differ from the host table", .{});

    if (rest.len > 1) {
        for (rest[1..]) |a| try layerCheck(gpu, &gdn, &ex, stream, &pack, try std.fmt.parseInt(usize, a, 10));
    } else {
        for ([_]usize{ 0, 1, 2 }) |l| try layerCheck(gpu, &gdn, &ex, stream, &pack, l);
    }
}
