//! Tensor parallel (tp=2) layers of the EXL3 Flash Next path on the two RTX 5060 Ti (no peer access: every exchange is a host bounce,
//! rank 0 then rank 1, as exl3_tp.zig / exl3_lane.zig state). Each rank has its own CUDA context, kernel modules, stream, weights
//! and state; the ranks run one after the other on the host thread.
//!   exl3-gdn-tp  <pack> [layer ...]        GDN layers (default 0 1 2): tp=2 against tp=1 on rank 0, 16-token prefill + 8 decode steps
//!   exl3-attn-tp <pack> [layer a] [layer b] full attention (default 3 7): prefill + graph decode + the synthetic sparse-indexer context
//! What must be byte-identical (an N-split column block, a conv channel, a delta-rule row and a head norm do not depend on the cut):
//! the projections, gated rows, attention outputs and states of each rank against the matching slice of tp=1. What may differ: the
//! K-split o_proj / out_proj, whose two fp16 partials are added in fp32 on the host (rank 0 + rank 1) instead of one fp32 K sum
//! rounded once; that is checked against tp=1 within the stated fp16 bound.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const st = @import("safetensors.zig");
const real = @import("exl3_real_tests.zig");
const split = @import("exl3_tp_split.zig");
const gdn_host = @import("exl3_gdn.zig");

const Gpu = check.Gpu;
const hidden: usize = 2560;
const tp = cuda.exl3_tp;

// -- two ranks, one process ------------------------------------------------------------

pub const Rank = struct {
    index: usize,
    ctx: cuda.Context,
    ex: cuda.exl3.Exl3,
    stream: cuda.Stream,
    /// bytes this test holds in DeviceBuffers on this rank (weights, state, scratch), by hand since the driver counter is process wide
    held: usize = 0,
    free_before: usize = 0,

    pub fn enter(r: *const Rank) !void {
        try r.ctx.makeCurrent();
    }

    pub fn freeNow(r: *const Rank) !usize {
        try r.enter();
        return (try r.ctx.memInfo()).free;
    }

    pub fn alloc(r: *Rank, bytes: usize) !cuda.DeviceBuffer {
        try r.enter();
        const b = try cuda.DeviceBuffer.alloc(r.ctx.d, bytes);
        r.held += bytes;
        return b;
    }

    pub fn fromHost(r: *Rank, bytes: []const u8) !cuda.DeviceBuffer {
        try r.enter();
        const b = try cuda.DeviceBuffer.fromHost(r.ctx.d, bytes);
        // A pageable-memory cuMemcpyHtoD can return before its DMA has landed, and it is ordered only against the legacy default
        // stream, not against this rank's non-blocking stream: a large upload followed at once by a launch can read stale
        // memory (seen as intermittent all-zero index-key rows on the shared GPU). Wait for the copy.
        try r.ctx.synchronize();
        r.held += bytes.len;
        return b;
    }
};

pub const Ranks = struct {
    r: [2]*Rank,
    gpa: std.mem.Allocator,

    pub fn open(gpu: Gpu) !Ranks {
        const n = try gpu.d.deviceCount();
        if (n < 2) return error.NeedTwoDevices;
        var out: Ranks = .{ .r = undefined, .gpa = gpu.gpa };
        var opened: usize = 0;
        errdefer for (out.r[0..opened]) |p| destroy(gpu.gpa, p);
        for (0..2) |i| {
            const p = try gpu.gpa.create(Rank);
            errdefer gpu.gpa.destroy(p);
            p.* = .{ .index = i, .ctx = try cuda.Context.init(gpu.d, @intCast(i)), .ex = undefined, .stream = undefined };
            errdefer p.ctx.deinit();
            p.ex = try cuda.exl3.Exl3.load(gpu.d);
            errdefer p.ex.unload();
            p.stream = try cuda.Stream.init(gpu.d, true);
            out.r[i] = p;
            opened += 1;
        }
        return out;
    }

    fn destroy(gpa: std.mem.Allocator, p: *Rank) void {
        p.enter() catch {};
        p.stream.deinit();
        p.ex.unload();
        p.ctx.deinit();
        gpa.destroy(p);
    }

    pub fn close(rs: *Ranks, gpu: Gpu) void {
        for (rs.r) |p| destroy(rs.gpa, p);
        gpu.ctx.makeCurrent() catch {};
    }
};

// -- small helpers -------------------------------------------------------------------

fn bf16f(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

fn bytesOf(gpa: std.mem.Allocator, pack: *const real.Pack, name: []const u8, dtype: st.DType, count: usize) ![]align(8) u8 {
    const t = pack.find(name) orelse return error.MissingTensor;
    try check.expect(t.dtype == dtype and t.numel() == count, "{s}: dtype/size differ from the layer shape", .{name});
    const out = try gpa.alignedAlloc(u8, .@"8", t.bytes.len);
    @memcpy(out, t.bytes);
    return out;
}

pub fn toStored(m: real.Mat) split.Stored {
    return .{ .trellis = m.trellis, .suh = m.suh, .svh = m.svh, .tiles_k = m.tiles_k, .tiles_n = m.tiles_n, .halves = m.halves };
}

pub fn toMat(s: split.Stored) real.Mat {
    return .{ .trellis = s.trellis, .suh = s.suh, .svh = s.svh, .tiles_k = s.tiles_k, .tiles_n = s.tiles_n, .halves = s.halves };
}

/// An EXL3 matrix cut for one rank and uploaded (strips order) to that rank. Frees the host slice.
pub const RankMat = struct {
    dev: real.Dev,
    k: usize,
    n: usize,
    halves: u8,

    pub fn upload(r: *Rank, gpu: Gpu, m: real.Mat) !RankMat {
        try r.enter();
        const d = try real.upload(gpu, m);
        try r.ctx.synchronize(); // see Rank.fromHost
        r.held += d.trellis.len + d.suh.len + d.svh.len;
        return .{ .dev = d, .k = m.tiles_k * 16, .n = m.tiles_n * 16, .halves = m.halves };
    }

    /// `kr` x `nr` of `whole`, on rank `r`.
    pub fn cut(r: *Rank, gpu: Gpu, whole: real.Mat, kr: split.Range, nr: []const split.Range) !RankMat {
        const s = try split.slice(gpu.gpa, toStored(whole), kr, nr);
        defer split.free(gpu.gpa, s);
        return upload(r, gpu, toMat(s));
    }

    pub fn free(m: *RankMat) void {
        m.dev.free();
    }

    pub fn run(m: *const RankMat, r: *const Rank, x: u64, y: u64, rows: usize) !void {
        try r.ex.linear(r.stream, x, m.dev.trellis.ptr, m.dev.suh.ptr, m.dev.svh.ptr, y, rows, m.k, m.n, m.halves, 2);
    }

    pub fn proj(m: *const RankMat) cuda.exl3_gdn.Proj {
        return .{ .trellis = m.dev.trellis.ptr, .suh = m.dev.suh.ptr, .svh = m.dev.svh.ptr, .k = m.k, .n = m.n, .halves = m.halves };
    }
};

fn download(comptime T: type, gpa: std.mem.Allocator, ptr: u64, d: *const cuda.Driver, count: usize) ![]T {
    const out = try gpa.alloc(T, count);
    errdefer gpa.free(out);
    const b = cuda.DeviceBuffer{ .d = d, .ptr = ptr, .len = count * @sizeOf(T), .borrowed = true };
    try b.download(0, std.mem.sliceAsBytes(out));
    return out;
}

const Stat = struct {
    max_abs: f64 = 0,
    peak: f64 = 0,
    equal: usize = 0,
    n: usize = 0,

    fn add(s: *Stat, got: anytype, want: anytype) void {
        for (got, want) |g, w| {
            const gv: f64 = @floatCast(g);
            const wv: f64 = @floatCast(w);
            s.max_abs = @max(s.max_abs, @abs(gv - wv));
            s.peak = @max(s.peak, @abs(wv));
            s.equal += @intFromBool(gv == wv);
            s.n += 1;
        }
    }
};

fn randomFill(rng: *std.Random.DefaultPrng, out: []f16) void {
    for (out) |*v| {
        var s: f32 = 0;
        for (0..4) |_| s += rng.random().float(f32);
        v.* = @floatCast((s - 2.0) * 1.7320508);
    }
}

pub fn openPack(gpu: Gpu, dir: []const u8) !real.Pack {
    var pack: real.Pack = .{ .files = undefined };
    errdefer pack.close(gpu.io);
    try pack.openAll(gpu, dir);
    return pack;
}

// -- GDN ---------------------------------------------------------------------------------

const prefill = 16;
const decode = 8;
const total = prefill + decode;
const c_full = cuda.exl3_gdn.channels;
const zw_full = cuda.exl3_gdn.z_width;

/// Everything of one GDN layer on the host (the pack's own tensors).
pub const GdnHost = struct {
    qkv: real.Mat,
    z: real.Mat,
    out: real.Mat,
    a: []align(8) u8,
    b: []align(8) u8,
    conv: []align(8) u8,
    alog: []align(8) u8,
    dt: []align(8) u8,
    norm: []align(8) u8,

    pub fn load(gpu: Gpu, pack: *const real.Pack, layer: usize) !GdnHost {
        const gpa = gpu.gpa;
        var sb: [96]u8 = undefined;
        const stem = try std.fmt.bufPrint(&sb, "model.language_model.layers.{d}.linear_attn", .{layer});
        var nb: [128]u8 = undefined;
        const qkv = try real.mat(pack, gpa, try std.fmt.bufPrint(&nb, "{s}.in_proj_qkv", .{stem}));
        const z = try real.mat(pack, gpa, try std.fmt.bufPrint(&nb, "{s}.in_proj_z", .{stem}));
        const out = try real.mat(pack, gpa, try std.fmt.bufPrint(&nb, "{s}.out_proj", .{stem}));
        try check.expect(qkv.tiles_k * 16 == hidden and qkv.tiles_n * 16 == c_full and z.tiles_n * 16 == zw_full and out.tiles_k * 16 == zw_full and out.tiles_n * 16 == hidden, "GDN projection shapes", .{});
        const nv = cuda.exl3_gdn.nv;
        return .{
            .qkv = qkv,
            .z = z,
            .out = out,
            .a = try bytesOf(gpa, pack, try std.fmt.bufPrint(&nb, "{s}.in_proj_a.weight", .{stem}), .f16, nv * hidden),
            .b = try bytesOf(gpa, pack, try std.fmt.bufPrint(&nb, "{s}.in_proj_b.weight", .{stem}), .f16, nv * hidden),
            .conv = try bytesOf(gpa, pack, try std.fmt.bufPrint(&nb, "{s}.conv1d.weight", .{stem}), .bf16, c_full * 4),
            .alog = try bytesOf(gpa, pack, try std.fmt.bufPrint(&nb, "{s}.A_log", .{stem}), .bf16, nv),
            .dt = try bytesOf(gpa, pack, try std.fmt.bufPrint(&nb, "{s}.dt_bias", .{stem}), .bf16, nv),
            .norm = try bytesOf(gpa, pack, try std.fmt.bufPrint(&nb, "{s}.norm.weight", .{stem}), .bf16, cuda.exl3_gdn.dv),
        };
    }

    pub fn free(h: GdnHost, gpa: std.mem.Allocator) void {
        for ([_]real.Mat{ h.qkv, h.z, h.out }) |m| {
            gpa.free(m.trellis);
            gpa.free(m.suh);
            gpa.free(m.svh);
        }
        gpa.free(h.a);
        gpa.free(h.b);
        gpa.free(h.conv);
        gpa.free(h.alog);
        gpa.free(h.dt);
        gpa.free(h.norm);
    }
};

/// One rank's GDN layer weights on that rank: `ranks` = 1 is the whole layer (tp=1, exl3_gdn.zig), 2 the rank's half.
pub const GdnW = struct {
    r: *Rank,
    mats: [3]RankMat,
    bufs: [6]cuda.DeviceBuffer, // a, b, conv, alog, dt, norm
    layer: cuda.exl3_gdn.Layer,

    pub fn load(r: *Rank, gpu: Gpu, h: GdnHost, comptime ranks: usize, rank: usize) !GdnW {
        const gpa = gpu.gpa;
        const nvl = cuda.exl3_gdn.nv / ranks;
        const zl = nvl * cuda.exl3_gdn.dv;
        const all: split.Range = .{ .lo = 0, .hi = hidden };
        var qkv_ranges: [3]split.Range = undefined;
        if (ranks == 1) {
            qkv_ranges = .{ .{ .lo = 0, .hi = c_full }, undefined, undefined };
        } else {
            for (tp.qkvSlice(rank), 0..) |sl, i| qkv_ranges[i] = .{ .lo = sl.lo, .hi = sl.lo + sl.width };
        }
        const nq: usize = if (ranks == 1) 1 else 3;
        var mq = try RankMat.cut(r, gpu, h.qkv, all, qkv_ranges[0..nq]);
        errdefer mq.free();
        var mz = try RankMat.cut(r, gpu, h.z, all, &.{.{ .lo = rank * zl, .hi = (rank + 1) * zl }});
        errdefer mz.free();
        var mo = try RankMat.cut(r, gpu, h.out, .{ .lo = rank * zl, .hi = (rank + 1) * zl }, &.{all});
        errdefer mo.free();

        const hv: split.Range = .{ .lo = rank * nvl, .hi = (rank + 1) * nvl };
        // the conv channels of a rank are its q, k, v slices in that order (the same ranges as the qkv columns)
        const conv_b = try split.gatherRows(gpa, h.conv, 8, qkv_ranges[0..nq]);
        defer gpa.free(conv_b);
        var bufs: [6]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        errdefer for (bufs[0..made]) |*b| b.free();
        const parts = [6][]const u8{
            h.a[hv.lo * hidden * 2 .. hv.hi * hidden * 2],
            h.b[hv.lo * hidden * 2 .. hv.hi * hidden * 2],
            conv_b,
            h.alog[hv.lo * 2 .. hv.hi * 2],
            h.dt[hv.lo * 2 .. hv.hi * 2],
            h.norm,
        };
        for (parts, 0..) |p, i| {
            bufs[i] = try r.fromHost(p);
            made += 1;
        }
        return .{
            .r = r,
            .mats = .{ mq, mz, mo },
            .bufs = bufs,
            .layer = .{
                .qkv = mq.proj(),
                .z = mz.proj(),
                .out = mo.proj(),
                .a_w = bufs[0].ptr,
                .b_w = bufs[1].ptr,
                .conv_w = bufs[2].ptr,
                .a_log = bufs[3].ptr,
                .dt_bias = bufs[4].ptr,
                .norm_w = bufs[5].ptr,
            },
        };
    }

    pub fn free(g: *GdnW) void {
        g.r.enter() catch {};
        for (&g.mats) |*m| m.free();
        for (&g.bufs) |*b| b.free();
    }
};

/// The weights plus the test's input rows and output buffer.
const GdnRank = struct {
    w: GdnW,
    layer: cuda.exl3_gdn.Layer,
    x: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer,

    fn build(r: *Rank, gpu: Gpu, h: GdnHost, comptime ranks: usize, rank: usize, x_host: []const f16) !GdnRank {
        var w = try GdnW.load(r, gpu, h, ranks, rank);
        errdefer w.free();
        var x = try r.fromHost(std.mem.sliceAsBytes(x_host));
        errdefer x.free();
        const out = try r.alloc(total * hidden * 2);
        return .{ .w = w, .layer = w.layer, .x = x, .out = out };
    }

    fn free(g: *GdnRank) void {
        g.w.r.enter() catch {};
        g.w.free();
        g.x.free();
        g.out.free();
    }
};

/// What one rank (or tp=1) produced over the 24 tokens.
const GdnRun = struct {
    out: []f16, // [total][2560] (a partial when tp=2)
    qkv: []f16, // [total][channels]
    gated: []f16, // [total][z_width]
    conv: []f16, // [3][channels]
    rec: []f32, // [nv][128][128]
    channels: usize,
    zw: usize,

    fn deinit(g: GdnRun, gpa: std.mem.Allocator) void {
        gpa.free(g.out);
        gpa.free(g.qkv);
        gpa.free(g.gated);
        gpa.free(g.conv);
        gpa.free(g.rec);
    }
};

/// tp=1: the existing Gdn.forward on `r`. tp=2: GdnTp.forward on `r`. 16-token prefill then 8 one-token steps from zero state.
fn gdnRun(comptime ranks: usize, gpu: Gpu, r: *Rank, g: *GdnRank, gdn1: ?*const cuda.exl3_gdn.Gdn, gdn2: ?*const tp.GdnTp) !GdnRun {
    const gpa = gpu.gpa;
    try r.enter();
    const chan = if (ranks == 1) c_full else tp.channels;
    const zw = if (ranks == 1) zw_full else tp.z_width;
    const conv_bytes = if (ranks == 1) cuda.exl3_gdn.conv_bytes else tp.conv_bytes;
    const rec_bytes = if (ranks == 1) cuda.exl3_gdn.rec_bytes else tp.rec_bytes;
    const nvl = if (ranks == 1) cuda.exl3_gdn.nv else tp.nvl;
    var conv_state = try r.alloc(conv_bytes);
    defer conv_state.free();
    var rec_state = try r.alloc(rec_bytes);
    defer rec_state.free();
    try conv_state.fill8(0, null);
    try rec_state.fill8(0, null);
    try r.ctx.synchronize(); // see zeroed()
    var sc1: cuda.exl3_gdn.Scratch = undefined;
    var sc2: tp.Scratch = undefined;
    if (ranks == 1) sc1 = try cuda.exl3_gdn.Scratch.alloc(gpu.d, prefill) else sc2 = try tp.Scratch.alloc(gpu.d, prefill);
    defer if (ranks == 1) sc1.free() else sc2.free();
    r.held += prefill * (chan * 2 + zw * 2 + 2 * nvl * 4 + chan * 4 + zw * 4 + zw * 2);

    const res: GdnRun = .{
        .out = try gpa.alloc(f16, total * hidden),
        .qkv = try gpa.alloc(f16, total * chan),
        .gated = try gpa.alloc(f16, total * zw),
        .conv = try gpa.alloc(f16, 3 * chan),
        .rec = try gpa.alloc(f32, nvl * 128 * 128),
        .channels = chan,
        .zw = zw,
    };
    var segs: [1 + decode][2]usize = undefined;
    segs[0] = .{ 0, prefill };
    for (0..decode) |i| segs[1 + i] = .{ prefill + i, 1 };
    for (segs) |seg| {
        const t0 = seg[0];
        const n = seg[1];
        const xp = g.x.ptr + t0 * hidden * 2;
        const op = g.out.ptr + t0 * hidden * 2;
        if (ranks == 1) {
            try gdn1.?.forward(&r.ex, r.stream, g.layer, conv_state.ptr, rec_state.ptr, xp, n, sc1, op);
        } else {
            try gdn2.?.forward(&r.ex, r.stream, g.layer, conv_state.ptr, rec_state.ptr, xp, n, sc2, op);
        }
        try r.stream.synchronize();
        const qkv_ptr = if (ranks == 1) sc1.qkv.ptr else sc2.qkv.ptr;
        const gated_ptr = if (ranks == 1) sc1.gated.ptr else sc2.gated.ptr;
        const b1 = cuda.DeviceBuffer{ .d = gpu.d, .ptr = qkv_ptr, .len = n * chan * 2, .borrowed = true };
        try b1.download(0, std.mem.sliceAsBytes(res.qkv[t0 * chan ..][0 .. n * chan]));
        const b2 = cuda.DeviceBuffer{ .d = gpu.d, .ptr = gated_ptr, .len = n * zw * 2, .borrowed = true };
        try b2.download(0, std.mem.sliceAsBytes(res.gated[t0 * zw ..][0 .. n * zw]));
        const b3 = cuda.DeviceBuffer{ .d = gpu.d, .ptr = op, .len = n * hidden * 2, .borrowed = true };
        try b3.download(0, std.mem.sliceAsBytes(res.out[t0 * hidden ..][0 .. n * hidden]));
    }
    try conv_state.download(0, std.mem.sliceAsBytes(res.conv));
    try rec_state.download(0, std.mem.sliceAsBytes(res.rec));
    return res;
}

fn bytesEqual(what: []const u8, got: anytype, want: anytype) !void {
    try check.sameBytes(what, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want));
}

fn gdnLayer(gpu: Gpu, rs: *Ranks, gdn1: *const cuda.exl3_gdn.Gdn, gdn2: [2]*const tp.GdnTp, pack: *const real.Pack, layer: usize) !void {
    const gpa = gpu.gpa;
    const h = try GdnHost.load(gpu, pack, layer);
    defer h.free(gpa);
    std.debug.print("layer {d}: halves qkv={d} z={d} out={d}\n", .{ layer, h.qkv.halves, h.z.halves, h.out.halves });
    const x = try gpa.alloc(f16, total * hidden);
    defer gpa.free(x);
    var rng = std.Random.DefaultPrng.init(0x6d5 + layer);
    randomFill(&rng, x);

    // tp=1 on rank 0
    var t1 = try GdnRank.build(rs.r[0], gpu, h, 1, 0, x);
    const held1 = rs.r[0].held;
    const run1 = try gdnRun(1, gpu, rs.r[0], &t1, gdn1, null);
    defer run1.deinit(gpa);
    t1.free();
    rs.r[0].held = 0;

    // tp=2: rank 0 then rank 1
    var runs: [2]GdnRun = undefined;
    var mem: [2]struct { held: usize, freed: usize } = undefined;
    var built: usize = 0;
    defer for (runs[0..built]) |q| q.deinit(gpa);
    for (0..2) |i| {
        const r = rs.r[i];
        r.held = 0;
        const f0 = try r.freeNow();
        var g = try GdnRank.build(r, gpu, h, 2, i, x);
        const weights_held = r.held;
        runs[i] = try gdnRun(2, gpu, r, &g, null, gdn2[i]);
        built += 1;
        mem[i] = .{ .held = weights_held, .freed = f0 -| try r.freeNow() };
        g.free();
        r.held = 0;
    }

    // 1. byte identities against the tp=1 slices
    const nv = cuda.exl3_gdn.nv;
    for (0..2) |i| {
        const q = runs[i];
        // projections: q | k | v column slices of every row
        var off: usize = 0;
        for (tp.qkvSlice(i)) |sl| {
            for (0..total) |t| try check.sameBytes("in_proj_qkv slice", std.mem.sliceAsBytes(q.qkv[t * tp.channels + off ..][0..sl.width]), std.mem.sliceAsBytes(run1.qkv[t * c_full + sl.lo ..][0..sl.width]));
            off += sl.width;
        }
        // gated rows (after conv, delta rule, norm): the rank's value heads
        for (0..total) |t| try check.sameBytes("gated rows", std.mem.sliceAsBytes(q.gated[t * tp.z_width ..][0..tp.z_width]), std.mem.sliceAsBytes(run1.gated[t * zw_full + i * tp.z_width ..][0..tp.z_width]));
        // recurrent state: value heads 24 i ..
        try check.sameBytes("recurrent state", std.mem.sliceAsBytes(q.rec), std.mem.sliceAsBytes(run1.rec[i * tp.nvl * 128 * 128 ..][0 .. tp.nvl * 128 * 128]));
        // conv state: three rows, the rank's channels
        for (0..3) |row| {
            off = 0;
            for (tp.qkvSlice(i)) |sl| {
                try check.sameBytes("conv state", std.mem.sliceAsBytes(q.conv[row * tp.channels + off ..][0..sl.width]), std.mem.sliceAsBytes(run1.conv[row * c_full + sl.lo ..][0..sl.width]));
                off += sl.width;
            }
        }
    }
    _ = nv;
    check.pass("layer {d}: tp=2 in_proj_qkv/in_proj_z/conv/delta/gated-norm rows ({d} tokens) and both ranks' conv + recurrent states are byte-identical to the tp=1 slices", .{ layer, total });

    // 2. the K-split out_proj: host sum of the two fp16 partials, rank 0 then rank 1, against tp=1's single fp16 output
    const p0 = try gpa.alloc(f32, total * hidden);
    defer gpa.free(p0);
    const p1 = try gpa.alloc(f32, total * hidden);
    defer gpa.free(p1);
    const sum = try gpa.alloc(f32, total * hidden);
    defer gpa.free(sum);
    for (p0, runs[0].out) |*d, s| d.* = s;
    for (p1, runs[1].out) |*d, s| d.* = s;
    try gdn_host.reduceOut(p0, p1, true, sum);
    var stat: Stat = .{};
    stat.add(sum, run1.out);
    // once the sum is rounded to fp16 (what the next stage would read) how many elements equal tp=1's
    var eq16: usize = 0;
    for (sum, run1.out) |a, b| eq16 += @intFromBool(@as(f16, @floatCast(a)) == b);
    const rel = stat.max_abs / stat.peak;
    std.debug.print("  out_proj host sum vs tp=1: max_abs {e:.3} peak {e:.3} rel {e:.3} (bound 1.5e-3); fp32 sum rounded to fp16 equals tp=1 on {d}/{d} elements\n", .{ stat.max_abs, stat.peak, rel, eq16, stat.n });
    // tp=1 rounds the full K sum to fp16 once (half an ulp = 2^-11 = 4.9e-4 of an element); tp=2 rounds each K half and then adds in fp32:
    // two half-ulp roundings of partials that can exceed the sum, plus tp=1's own: 1.5e-3 of the peak is about three half-ulps (observed under one).
    try check.expect(stat.peak > 0 and rel <= 1.5e-3, "tp=2 out_proj sum differs from tp=1 beyond the fp16 bound", .{});
    check.pass("layer {d}: tp=2 layer output (prefill 16 + 8 decode steps) equals tp=1 within {e:.2} of the peak; per-rank weights {d:.1} MiB (rank 0) / {d:.1} MiB (rank 1) vs tp=1 {d:.1} MiB (this layer, incl. the 24-row x/out buffers)", .{ layer, rel, mib(mem[0].held), mib(mem[1].held), mib(held1) });
    std.debug.print("  memInfo free delta while resident (includes scratch, state and context growth): rank 0 {d:.1} MiB, rank 1 {d:.1} MiB\n", .{ mib(mem[0].freed), mib(mem[1].freed) });
}

pub fn mib(b: usize) f64 {
    return @as(f64, @floatFromInt(b)) / (1024.0 * 1024.0);
}

pub fn runGdn(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    var pack = try openPack(gpu, rest[0]);
    defer pack.close(gpu.io);
    var rs = try Ranks.open(gpu);
    defer rs.close(gpu);
    var gdn1: [1]cuda.exl3_gdn.Gdn = undefined;
    var gdn2: [2]tp.GdnTp = undefined;
    try rs.r[0].enter();
    gdn1[0] = try cuda.exl3_gdn.Gdn.load(gpu.d);
    defer {
        rs.r[0].enter() catch {};
        gdn1[0].unload();
    }
    var loaded: usize = 0;
    defer for (0..loaded) |i| {
        rs.r[i].enter() catch {};
        gdn2[i].unload();
    };
    for (0..2) |i| {
        try rs.r[i].enter();
        gdn2[i] = try tp.GdnTp.load(gpu.d);
        loaded += 1;
    }
    const ptrs = [2]*const tp.GdnTp{ &gdn2[0], &gdn2[1] };
    if (rest.len > 1) {
        for (rest[1..]) |a| try gdnLayer(gpu, &rs, &gdn1[0], ptrs, &pack, try std.fmt.parseInt(usize, a, 10));
    } else {
        for ([_]usize{ 0, 1, 2 }) |l| try gdnLayer(gpu, &rs, &gdn1[0], ptrs, &pack, l);
    }
}

// -- Attention ---------------------------------------------------------------------------

const oracle = @import("exl3_attn_cpu.zig");
const scan = @import("exl3_scan.zig");
const cachemod = @import("exl3_cache.zig");
const attn_ = cuda.exl3_attn;

const slot_bytes = attn_.slot_bytes;
const idx_width: usize = oracle.index_width; // 640

pub const AttnHost = struct {
    q: real.Mat,
    k: real.Mat,
    v: real.Mat,
    o: real.Mat,
    idx: real.Mat,
    q_scale: []f32,
    k_scale: []f32,
    iq_scale: []f32,
    ik_scale: []f32,
    inv: [oracle.half]f32,

    pub fn load(gpu: Gpu, pack: *const real.Pack, layer: usize) !AttnHost {
        const gpa = gpu.gpa;
        var base: [96]u8 = undefined;
        const stem = try std.fmt.bufPrint(&base, "model.language_model.layers.{d}.self_attn", .{layer});
        var names: [5][128]u8 = undefined;
        const q = try real.mat(pack, gpa, try std.fmt.bufPrint(&names[0], "{s}.q_proj", .{stem}));
        const k = try real.mat(pack, gpa, try std.fmt.bufPrint(&names[1], "{s}.k_proj", .{stem}));
        const v = try real.mat(pack, gpa, try std.fmt.bufPrint(&names[2], "{s}.v_proj", .{stem}));
        const o = try real.mat(pack, gpa, try std.fmt.bufPrint(&names[3], "{s}.o_proj", .{stem}));
        const idx = try real.mat(pack, gpa, try std.fmt.bufPrint(&names[4], "{s}.indexer.index_qk_proj", .{stem}));
        try check.expect(q.tiles_k * 16 == hidden and q.tiles_n * 16 == 12288 and k.tiles_n * 16 == 512 and v.tiles_n * 16 == 512 and o.tiles_k * 16 == 6144 and o.tiles_n * 16 == hidden and idx.tiles_n * 16 == idx_width, "attention projection shapes", .{});
        std.debug.print("layer {d}: halves q={d} k={d} v={d} o={d} index_qk={d}\n", .{ layer, q.halves, k.halves, v.halves, o.halves, idx.halves });
        var means: [4]f32 = undefined;
        const norms = [4][]f32{
            try normWeights(gpa, pack, layer, "q_norm.weight", oracle.head_dim),
            try normWeights(gpa, pack, layer, "k_norm.weight", oracle.head_dim),
            try normWeights(gpa, pack, layer, "indexer.q_layernorm.weight", oracle.index_dim),
            try normWeights(gpa, pack, layer, "indexer.k_layernorm.weight", oracle.index_dim),
        };
        for (norms, 0..) |w, i| {
            var s: f64 = 0;
            for (w) |x| s += x;
            means[i] = @floatCast(s / @as(f64, @floatFromInt(w.len)));
        }
        // the same shim as exl3-attn: the indexer norms decide (gamma-1 storage), q/k means sit at 0.27-0.59 and are gamma-1 too
        const offset = try scan.centredOffset(means[2..4]);
        try check.expect(offset == 1 and means[0] < 0.6 and means[1] < 0.6, "norm storage is not gamma-1", .{});
        for (norms) |w| for (w) |*x| {
            x.* += offset;
        };
        return .{ .q = q, .k = k, .v = v, .o = o, .idx = idx, .q_scale = norms[0], .k_scale = norms[1], .iq_scale = norms[2], .ik_scale = norms[3], .inv = oracle.invFreq() };
    }

    pub fn free(h: AttnHost, gpa: std.mem.Allocator) void {
        for ([_]real.Mat{ h.q, h.k, h.v, h.o, h.idx }) |m| {
            gpa.free(m.trellis);
            gpa.free(m.suh);
            gpa.free(m.svh);
        }
        gpa.free(h.q_scale);
        gpa.free(h.k_scale);
        gpa.free(h.iq_scale);
        gpa.free(h.ik_scale);
    }
};

fn normWeights(gpa: std.mem.Allocator, pack: *const real.Pack, layer: usize, name: []const u8, width: usize) ![]f32 {
    var buf: [160]u8 = undefined;
    const t = pack.find(try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.self_attn.{s}", .{ layer, name })) orelse return error.MissingTensor;
    try check.expect(t.numel() == width and t.bytes.len == width * 2, "{s} shape {d}", .{ name, t.numel() });
    const out = try gpa.alloc(f32, width);
    for (out, 0..) |*o, i| o.* = @floatCast(bf16f(std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little)));
    return out;
}

/// One side of the comparison: tp=1 (24 q heads, 2 kv heads, rank 0 holds everything) or one tp=2 rank (12 q heads, 1 kv head).
pub const AttnSide = struct {
    r: *Rank,
    at: *const attn_.Attn,
    qh: usize,
    kvh: usize,
    q: RankMat,
    k: RankMat,
    v: RankMat,
    o: RankMat,
    idx: RankMat,
    scales: [5]cuda.DeviceBuffer, // q, k, iq, ik, inv_freq

    pub fn qw(s: AttnSide) usize {
        return s.qh * 512;
    }
    pub fn kvw(s: AttnSide) usize {
        return s.kvh * 256;
    }
    pub fn aw(s: AttnSide) usize {
        return s.qh * 256;
    }

    pub fn build(r: *Rank, at: *const attn_.Attn, gpu: Gpu, h: AttnHost, rank: ?usize) !AttnSide {
        const all: split.Range = .{ .lo = 0, .hi = hidden };
        const qh: usize = if (rank == null) 24 else 12;
        const kvh: usize = if (rank == null) 2 else 1;
        const rk = rank orelse 0;
        const qr: split.Range = .{ .lo = rk * qh * 512, .hi = (rk + 1) * qh * 512 };
        if (rank == null) std.debug.assert(qr.hi == 12288);
        const kr: split.Range = .{ .lo = rk * kvh * 256, .hi = (rk + 1) * kvh * 256 };
        const orr: split.Range = .{ .lo = rk * qh * 256, .hi = (rk + 1) * qh * 256 };
        var q = try RankMat.cut(r, gpu, h.q, all, &.{qr});
        errdefer q.free();
        var k = try RankMat.cut(r, gpu, h.k, all, &.{kr});
        errdefer k.free();
        var v = try RankMat.cut(r, gpu, h.v, all, &.{kr});
        errdefer v.free();
        var o = try RankMat.cut(r, gpu, h.o, orr, &.{all});
        errdefer o.free();
        var idx = try RankMat.cut(r, gpu, h.idx, all, &.{.{ .lo = 0, .hi = idx_width }});
        errdefer idx.free();
        var sc: [5]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        errdefer for (sc[0..made]) |*b| b.free();
        const srcs = [5][]const u8{ std.mem.sliceAsBytes(h.q_scale), std.mem.sliceAsBytes(h.k_scale), std.mem.sliceAsBytes(h.iq_scale), std.mem.sliceAsBytes(h.ik_scale), std.mem.sliceAsBytes(&h.inv) };
        for (srcs, 0..) |s, i| {
            sc[i] = try r.fromHost(s);
            made += 1;
        }
        return .{ .r = r, .at = at, .qh = qh, .kvh = kvh, .q = q, .k = k, .v = v, .o = o, .idx = idx, .scales = sc };
    }

    pub fn free(s: *AttnSide) void {
        s.r.enter() catch {};
        s.q.free();
        s.k.free();
        s.v.free();
        s.o.free();
        s.idx.free();
        for (&s.scales) |*b| b.free();
    }
};

pub const Bufs = struct {
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
    kv: u64,
    ik: u64,
    pooled: u64,
    pos0: u64,
};

pub fn pipeline(s: *const AttnSide, b: Bufs, dims: attn_.Dims, rows: usize) !void {
    const r = s.r;
    const st_ = r.stream;
    try s.q.run(r, b.x, b.qp, rows);
    try s.k.run(r, b.x, b.kp, rows);
    try s.v.run(r, b.x, b.vp, rows);
    try s.idx.run(r, b.x, b.ip, rows);
    try s.at.prep(st_, dims, rows, .{ .qp = b.qp, .kp = b.kp, .vp = b.vp, .ip = b.ip, .q_scale = s.scales[0].ptr, .k_scale = s.scales[1].ptr, .i_scale = s.scales[2].ptr, .inv_freq = s.scales[4].ptr, .pos0 = b.pos0, .q_out = b.q_out, .kv = b.kv, .iq_out = b.iq_out, .ik = b.ik });
    try s.at.pool(st_, dims, rows, b.ik, b.pooled, s.scales[3].ptr, s.scales[4].ptr, b.pos0);
    const nb = dims.tokens / attn_.ratio;
    try s.at.scores(st_, rows, b.iq_out, b.pooled, b.scores, b.pos0, nb, nb);
    try s.at.select(st_, rows, b.scores, b.ids, b.nk, b.pos0, nb);
    try s.at.decode(st_, dims, rows, b.q_out, b.qp, b.kv, b.ids, b.attn, b.pos0);
    try s.o.run(r, b.attn, b.y, rows);
}

fn setPos(pos0: cuda.DeviceBuffer, v: usize) !void {
    const i: i32 = @intCast(v);
    try pos0.upload(0, std.mem.asBytes(&i));
}

const prefill_rows: usize = 32;
const decode_steps: usize = 8;
const cap_a: usize = 64;

/// Everything Part A (real projections) leaves on the host.
const ARes = struct {
    qp: []f16,
    kp: []f16,
    vp: []f16,
    q_out: []u16,
    attn: []f16,
    y: []f16,
    kv: []u8, // [cap][kvh][K|V] after the decode steps
    ik: []u16,
    pooled: []u16,
    d_attn: []f16, // [decode][attn_w]
    d_y: []f16,
    d_q: []u16,

    fn deinit(a: ARes, gpa: std.mem.Allocator) void {
        gpa.free(a.qp);
        gpa.free(a.kp);
        gpa.free(a.vp);
        gpa.free(a.q_out);
        gpa.free(a.attn);
        gpa.free(a.y);
        gpa.free(a.kv);
        gpa.free(a.ik);
        gpa.free(a.pooled);
        gpa.free(a.d_attn);
        gpa.free(a.d_y);
        gpa.free(a.d_q);
    }
};

fn fetch(comptime T: type, gpu: Gpu, buf: cuda.DeviceBuffer, off_bytes: usize, n: usize) ![]T {
    const out = try gpu.gpa.alloc(T, n);
    errdefer gpu.gpa.free(out);
    try buf.download(off_bytes, std.mem.sliceAsBytes(out));
    return out;
}

fn zeroed(r: *Rank, bytes: usize) !cuda.DeviceBuffer {
    var b = try r.alloc(bytes);
    errdefer b.free();
    try b.fill8(0, null);
    // cuMemsetD8 is asynchronous w.r.t. the host and ordered only against the legacy default stream, not this rank's non-blocking
    // stream: wait, or a late memset can zero what the first kernels already wrote (or leave pos0 garbage for them).
    try r.ctx.synchronize();
    return b;
}

fn attnPartA(gpu: Gpu, s: *const AttnSide, xh: []const f16) !ARes {
    const r = s.r;
    try r.enter();
    const total_rows = prefill_rows + decode_steps;
    const dims: attn_.Dims = .{ .q_heads = s.qh, .kv_heads = s.kvh, .tokens = cap_a };
    const nb = cap_a / 4;
    const qw = s.qw();
    const kvw = s.kvw();
    const aw = s.aw();
    var bufs: [16]cuda.DeviceBuffer = undefined;
    var n: usize = 0;
    defer for (bufs[0..n]) |*b| b.free();
    const sizes = [_]usize{
        total_rows * hidden * 2, // 0 x
        prefill_rows * qw * 2, // 1 qp
        prefill_rows * kvw * 2, // 2 kp
        prefill_rows * kvw * 2, // 3 vp
        prefill_rows * idx_width * 2, // 4 ip
        prefill_rows * aw * 2, // 5 q_out
        prefill_rows * 512 * 2, // 6 iq_out
        prefill_rows * nb * 4, // 7 scores
        prefill_rows * attn_.ids_stride * 4, // 8 ids
        prefill_rows * 4, // 9 nk
        prefill_rows * aw * 2, // 10 attn
        prefill_rows * hidden * 2, // 11 y
        cap_a * s.kvh * slot_bytes, // 12 kv
        cap_a * 128 * 2, // 13 ik
        (nb + 2) * 128 * 2, // 14 pooled
        4, // 15 pos0
    };
    for (sizes) |sz| {
        bufs[n] = try zeroed(r, sz);
        n += 1;
    }
    try bufs[0].upload(0, std.mem.sliceAsBytes(xh));
    try setPos(bufs[15], 0);
    try r.ctx.synchronize();
    const pre = Bufs{ .x = bufs[0].ptr, .qp = bufs[1].ptr, .kp = bufs[2].ptr, .vp = bufs[3].ptr, .ip = bufs[4].ptr, .q_out = bufs[5].ptr, .iq_out = bufs[6].ptr, .scores = bufs[7].ptr, .ids = bufs[8].ptr, .nk = bufs[9].ptr, .attn = bufs[10].ptr, .y = bufs[11].ptr, .kv = bufs[12].ptr, .ik = bufs[13].ptr, .pooled = bufs[14].ptr, .pos0 = bufs[15].ptr };
    try pipeline(s, pre, dims, prefill_rows);
    try r.stream.synchronize();

    // decode steps: ONE captured graph; only pos0 and the input row change
    var sx = try zeroed(r, hidden * 2);
    defer sx.free();
    var sqp = try zeroed(r, qw * 2);
    defer sqp.free();
    var skp = try zeroed(r, kvw * 2);
    defer skp.free();
    var svp = try zeroed(r, kvw * 2);
    defer svp.free();
    var sip = try zeroed(r, idx_width * 2);
    defer sip.free();
    var sq = try zeroed(r, aw * 2);
    defer sq.free();
    var siq = try zeroed(r, 512 * 2);
    defer siq.free();
    var ssc = try zeroed(r, nb * 4);
    defer ssc.free();
    var sids = try zeroed(r, attn_.ids_stride * 4);
    defer sids.free();
    var snk = try zeroed(r, 4);
    defer snk.free();
    var sat = try zeroed(r, aw * 2);
    defer sat.free();
    var sy = try zeroed(r, hidden * 2);
    defer sy.free();
    const stepb = Bufs{ .x = sx.ptr, .qp = sqp.ptr, .kp = skp.ptr, .vp = svp.ptr, .ip = sip.ptr, .q_out = sq.ptr, .iq_out = siq.ptr, .scores = ssc.ptr, .ids = sids.ptr, .nk = snk.ptr, .attn = sat.ptr, .y = sy.ptr, .kv = pre.kv, .ik = pre.ik, .pooled = pre.pooled, .pos0 = pre.pos0 };
    try cuda.graph.beginCapture(r.stream, .thread_local);
    try pipeline(s, stepb, dims, 1);
    var captured = try cuda.graph.endCapture(r.stream);
    defer captured.deinit();
    var exec = try captured.instantiate();
    defer exec.deinit();
    try exec.upload(r.stream);

    const gpa = gpu.gpa;
    const res: ARes = .{
        .qp = try fetch(f16, gpu, bufs[1], 0, prefill_rows * qw),
        .kp = try fetch(f16, gpu, bufs[2], 0, prefill_rows * kvw),
        .vp = try fetch(f16, gpu, bufs[3], 0, prefill_rows * kvw),
        .q_out = try fetch(u16, gpu, bufs[5], 0, prefill_rows * aw),
        .attn = try fetch(f16, gpu, bufs[10], 0, prefill_rows * aw),
        .y = try fetch(f16, gpu, bufs[11], 0, prefill_rows * hidden),
        .kv = undefined,
        .ik = undefined,
        .pooled = undefined,
        .d_attn = try gpa.alloc(f16, decode_steps * aw),
        .d_y = try gpa.alloc(f16, decode_steps * hidden),
        .d_q = try gpa.alloc(u16, decode_steps * aw),
    };
    for (0..decode_steps) |i| {
        const pos = prefill_rows + i;
        try setPos(bufs[15], pos);
        try sx.upload(0, std.mem.sliceAsBytes(xh[pos * hidden ..][0..hidden]));
        try r.ctx.synchronize();
        try exec.launchOn(r.stream);
        try r.stream.synchronize();
        try sat.download(0, std.mem.sliceAsBytes(res.d_attn[i * aw ..][0..aw]));
        try sy.download(0, std.mem.sliceAsBytes(res.d_y[i * hidden ..][0..hidden]));
        try sq.download(0, std.mem.sliceAsBytes(res.d_q[i * aw ..][0..aw]));
    }
    var out = res;
    out.kv = try gpa.alloc(u8, cap_a * s.kvh * slot_bytes);
    try bufs[12].download(0, out.kv);
    out.ik = try fetch(u16, gpu, bufs[13], 0, cap_a * 128);
    out.pooled = try fetch(u16, gpu, bufs[14], 0, (nb + 2) * 128);
    return out;
}

const syn_tokens: usize = 2600;
const syn_cap: usize = 2688;

const BRes = struct {
    ids: []i32, // [20][ids_stride] (batch 1 then batch 2)
    nk: []i32,
    attn: []f16, // [20][aw]
    y: []f16, // [20][hidden]
    kv: []u8,
    ik: []u16,
    pooled: []u16,

    fn deinit(b: BRes, gpa: std.mem.Allocator) void {
        gpa.free(b.ids);
        gpa.free(b.nk);
        gpa.free(b.attn);
        gpa.free(b.y);
        gpa.free(b.kv);
        gpa.free(b.ik);
        gpa.free(b.pooled);
    }
};

const Batch = struct { start: usize, rows: usize };
const batches = [_]Batch{ .{ .start = 2592, .rows = 8 }, .{ .start = 2044, .rows = 12 } };
const batch_rows = 20;

/// The 2600-token synthetic context: the host supplies this side's q/k/v projections (random) and the shared hidden rows; the real
/// index_qk_proj, prep, pool, scores, select and decode run on `s`.
fn attnPartB(gpu: Gpu, s: *const AttnSide, xh: []const f16, qph: []const f16, kph: []const f16, vph: []const f16) !BRes {
    const r = s.r;
    try r.enter();
    const gpa = gpu.gpa;
    const dims: attn_.Dims = .{ .q_heads = s.qh, .kv_heads = s.kvh, .tokens = syn_cap };
    const nb = syn_cap / 4;
    const qw = s.qw();
    const aw = s.aw();
    var x = try r.fromHost(std.mem.sliceAsBytes(xh));
    defer x.free();
    var qp = try r.fromHost(std.mem.sliceAsBytes(qph));
    defer qp.free();
    var kp = try r.fromHost(std.mem.sliceAsBytes(kph));
    defer kp.free();
    var vp = try r.fromHost(std.mem.sliceAsBytes(vph));
    defer vp.free();
    var ip = try zeroed(r, syn_tokens * idx_width * 2);
    defer ip.free();
    var q_out = try zeroed(r, syn_tokens * aw * 2);
    defer q_out.free();
    var iq_out = try zeroed(r, syn_tokens * 512 * 2);
    defer iq_out.free();
    var kv = try zeroed(r, syn_cap * s.kvh * slot_bytes);
    defer kv.free();
    var ik = try zeroed(r, syn_cap * 128 * 2);
    defer ik.free();
    var pooled = try zeroed(r, (nb + 2) * 128 * 2);
    defer pooled.free();
    var pos0 = try zeroed(r, 4);
    defer pos0.free();
    var scores = try zeroed(r, 12 * nb * 4);
    defer scores.free();
    var ids = try zeroed(r, 12 * attn_.ids_stride * 4);
    defer ids.free();
    var nk = try zeroed(r, 12 * 4);
    defer nk.free();
    var attn_out = try zeroed(r, 12 * aw * 2);
    defer attn_out.free();
    var y = try zeroed(r, 12 * hidden * 2);
    defer y.free();

    const stream = r.stream;
    try s.idx.run(r, x.ptr, ip.ptr, syn_tokens);
    try s.at.prep(stream, dims, syn_tokens, .{ .qp = qp.ptr, .kp = kp.ptr, .vp = vp.ptr, .ip = ip.ptr, .q_scale = s.scales[0].ptr, .k_scale = s.scales[1].ptr, .i_scale = s.scales[2].ptr, .inv_freq = s.scales[4].ptr, .pos0 = pos0.ptr, .q_out = q_out.ptr, .kv = kv.ptr, .iq_out = iq_out.ptr, .ik = ik.ptr });
    try s.at.pool(stream, dims, syn_tokens, ik.ptr, pooled.ptr, s.scales[3].ptr, s.scales[4].ptr, pos0.ptr);
    try stream.synchronize();

    const res: BRes = .{
        .ids = try gpa.alloc(i32, batch_rows * attn_.ids_stride),
        .nk = try gpa.alloc(i32, batch_rows),
        .attn = try gpa.alloc(f16, batch_rows * aw),
        .y = try gpa.alloc(f16, batch_rows * hidden),
        .kv = try gpa.alloc(u8, syn_tokens * s.kvh * slot_bytes),
        .ik = try fetch(u16, gpu, ik, 0, syn_tokens * 128),
        .pooled = try fetch(u16, gpu, pooled, 0, (syn_tokens / 4) * 128),
    };
    try kv.download(0, res.kv);
    var at_row: usize = 0;
    for (batches) |bt| {
        try setPos(pos0, bt.start);
        try r.ctx.synchronize();
        try s.at.scores(stream, bt.rows, iq_out.ptr + bt.start * 512 * 2, pooled.ptr, scores.ptr, pos0.ptr, nb, nb);
        try s.at.select(stream, bt.rows, scores.ptr, ids.ptr, nk.ptr, pos0.ptr, nb);
        try s.at.decode(stream, dims, bt.rows, q_out.ptr + bt.start * aw * 2, qp.ptr + bt.start * qw * 2, kv.ptr, ids.ptr, attn_out.ptr, pos0.ptr);
        try s.o.run(r, attn_out.ptr, y.ptr, bt.rows);
        try stream.synchronize();
        try ids.download(0, std.mem.sliceAsBytes(res.ids[at_row * attn_.ids_stride ..][0 .. bt.rows * attn_.ids_stride]));
        try nk.download(0, std.mem.sliceAsBytes(res.nk[at_row..][0..bt.rows]));
        try attn_out.download(0, std.mem.sliceAsBytes(res.attn[at_row * aw ..][0 .. bt.rows * aw]));
        try y.download(0, std.mem.sliceAsBytes(res.y[at_row * hidden ..][0 .. bt.rows * hidden]));
        at_row += bt.rows;
    }
    return res;
}

fn cols(comptime T: type, rows_n: usize, width: usize, lo: usize, w: usize, src: []const T, dst: []T) void {
    for (0..rows_n) |t| @memcpy(dst[t * w ..][0..w], src[t * width + lo ..][0..w]);
}

/// Rank `r`'s KV slice of a [token][2][K|V] tp=1 table: kv head r.
fn kvSlice(gpa: std.mem.Allocator, kv1: []const u8, tokens: usize, rk: usize) ![]u8 {
    const out = try gpa.alloc(u8, tokens * slot_bytes);
    for (0..tokens) |t| @memcpy(out[t * slot_bytes ..][0..slot_bytes], kv1[(t * 2 + rk) * slot_bytes ..][0..slot_bytes]);
    return out;
}

fn sumPartials(gpa: std.mem.Allocator, a: []const f16, b: []const f16) ![]f32 {
    const p0 = try gpa.alloc(f32, a.len);
    defer gpa.free(p0);
    const p1 = try gpa.alloc(f32, a.len);
    defer gpa.free(p1);
    for (p0, a) |*d, s| d.* = s;
    for (p1, b) |*d, s| d.* = s;
    const out = try gpa.alloc(f32, a.len);
    errdefer gpa.free(out);
    try gdn_host.reduceOut(p0, p1, true, out);
    return out;
}

fn yBound(what: []const u8, gpa: std.mem.Allocator, y0: []const f16, y1: []const f16, want: []const f16) !f64 {
    const sum = try sumPartials(gpa, y0, y1);
    defer gpa.free(sum);
    var stat: Stat = .{};
    stat.add(sum, want);
    var eq16: usize = 0;
    for (sum, want) |a, b| eq16 += @intFromBool(@as(f16, @floatCast(a)) == b);
    const rel = stat.max_abs / stat.peak;
    std.debug.print("  {s}: host sum of the two o_proj partials vs tp=1: max_abs {e:.3} peak {e:.3} rel {e:.3} (bound 1.5e-3); fp16 of the sum equals tp=1 on {d}/{d}\n", .{ what, stat.max_abs, stat.peak, rel, eq16, stat.n });
    // one fp16 rounding of the full K sum (tp=1) against one per K half plus the fp32 add (tp=2): 1.5e-3 of the peak is about three half-ulps (observed under one)
    try check.expect(stat.peak > 0 and rel <= 1.5e-3, "{s}: tp=2 o_proj sum differs from tp=1 beyond the fp16 bound", .{what});
    return rel;
}

fn attnLayer(gpu: Gpu, rs: *Ranks, at: [2]*const attn_.Attn, pack: *const real.Pack, layer: usize) !void {
    const gpa = gpu.gpa;
    const h = try AttnHost.load(gpu, pack, layer);
    defer h.free(gpa);
    // -- Part A inputs
    const total_rows = prefill_rows + decode_steps;
    const xa = try gpa.alloc(f16, total_rows * hidden);
    defer gpa.free(xa);
    var rng = std.Random.DefaultPrng.init(0xa77e + layer);
    for (xa) |*v| v.* = @floatCast(rng.random().floatNorm(f32));

    var s1 = try AttnSide.build(rs.r[0], at[0], gpu, h, null);
    const held1 = rs.r[0].held;
    const a1 = try attnPartA(gpu, &s1, xa);
    defer a1.deinit(gpa);
    s1.free();
    rs.r[0].held = 0;

    var a2: [2]ARes = undefined;
    var made: usize = 0;
    defer for (a2[0..made]) |q| q.deinit(gpa);
    var held2: [2]usize = undefined;
    for (0..2) |i| {
        const r = rs.r[i];
        r.held = 0;
        var s = try AttnSide.build(r, at[i], gpu, h, i);
        held2[i] = r.held;
        a2[i] = try attnPartA(gpu, &s, xa);
        made += 1;
        s.free();
        r.held = 0;
    }
    for (0..2) |i| {
        const q = a2[i];
        // projections (the N-split q_proj [query|gate] per head, k, v) and the q after norm+RoPE: byte slices of tp=1
        for (0..prefill_rows) |t| {
            try check.sameBytes("q_proj slice", std.mem.sliceAsBytes(q.qp[t * 6144 ..][0..6144]), std.mem.sliceAsBytes(a1.qp[t * 12288 + i * 6144 ..][0..6144]));
            try check.sameBytes("k_proj slice", std.mem.sliceAsBytes(q.kp[t * 256 ..][0..256]), std.mem.sliceAsBytes(a1.kp[t * 512 + i * 256 ..][0..256]));
            try check.sameBytes("v_proj slice", std.mem.sliceAsBytes(q.vp[t * 256 ..][0..256]), std.mem.sliceAsBytes(a1.vp[t * 512 + i * 256 ..][0..256]));
            try check.sameBytes("q (norm+rope)", std.mem.sliceAsBytes(q.q_out[t * 3072 ..][0..3072]), std.mem.sliceAsBytes(a1.q_out[t * 6144 + i * 3072 ..][0..3072]));
            // the attention output (after the indexer, sparse/dense choice, softmax, gate): rank i's heads
            try check.sameBytes("attention out (prefill)", std.mem.sliceAsBytes(q.attn[t * 3072 ..][0..3072]), std.mem.sliceAsBytes(a1.attn[t * 6144 + i * 3072 ..][0..3072]));
        }
        for (0..decode_steps) |t| {
            try check.sameBytes("decode q", std.mem.sliceAsBytes(q.d_q[t * 3072 ..][0..3072]), std.mem.sliceAsBytes(a1.d_q[t * 6144 + i * 3072 ..][0..3072]));
            try check.sameBytes("decode attention out (graph)", std.mem.sliceAsBytes(q.d_attn[t * 3072 ..][0..3072]), std.mem.sliceAsBytes(a1.d_attn[t * 6144 + i * 3072 ..][0..3072]));
        }
        // KV table: kv head i of tp=1; replicated indexer state
        const kv_i = try kvSlice(gpa, a1.kv, cap_a, i);
        defer gpa.free(kv_i);
        try check.sameBytes("KV table", q.kv, kv_i);
        try check.sameBytes("raw index keys", std.mem.sliceAsBytes(q.ik), std.mem.sliceAsBytes(a1.ik));
        try check.sameBytes("pooled index blocks", std.mem.sliceAsBytes(q.pooled), std.mem.sliceAsBytes(a1.pooled));
    }
    const rel_a = try yBound("prefill (32 rows)", gpa, a2[0].y, a2[1].y, a1.y);
    const rel_d = try yBound("graph decode (8 steps)", gpa, a2[0].d_y, a2[1].d_y, a1.d_y);
    check.pass("layer {d}: tp=2 attention (32-token prefill + 8 graph-replayed decode steps, real weights): projections, q, attention outputs, KV table (kv head per rank), index keys and pooled blocks byte-identical to tp=1; o_proj sum rel {e:.2} / {e:.2} of the peak", .{ layer, rel_a, rel_d });

    // -- Part B: 2600-token synthetic context (sparse indexer: 512 best of up to 650 blocks + tail, dense/sparse boundary at 2051/2052 keys)
    const xb = try gpa.alloc(f16, syn_tokens * hidden);
    defer gpa.free(xb);
    const qb = try gpa.alloc(f16, syn_tokens * 12288);
    defer gpa.free(qb);
    const kb = try gpa.alloc(f16, syn_tokens * 512);
    defer gpa.free(kb);
    const vb = try gpa.alloc(f16, syn_tokens * 512);
    defer gpa.free(vb);
    var rng_b = std.Random.DefaultPrng.init(0x5b5b + layer);
    for ([_][]f16{ xb, qb, kb, vb }) |buf| for (buf) |*v| {
        v.* = @floatCast(rng_b.random().floatNorm(f32));
    };
    var s1b = try AttnSide.build(rs.r[0], at[0], gpu, h, null);
    const b1 = try attnPartB(gpu, &s1b, xb, qb, kb, vb);
    defer b1.deinit(gpa);
    s1b.free();
    rs.r[0].held = 0;

    var b2: [2]BRes = undefined;
    var made_b: usize = 0;
    defer for (b2[0..made_b]) |q| q.deinit(gpa);
    for (0..2) |i| {
        const r = rs.r[i];
        r.held = 0;
        var s = try AttnSide.build(r, at[i], gpu, h, i);
        const qs = try gpa.alloc(f16, syn_tokens * 6144);
        defer gpa.free(qs);
        const ks = try gpa.alloc(f16, syn_tokens * 256);
        defer gpa.free(ks);
        const vs = try gpa.alloc(f16, syn_tokens * 256);
        defer gpa.free(vs);
        cols(f16, syn_tokens, 12288, i * 6144, 6144, qb, qs);
        cols(f16, syn_tokens, 512, i * 256, 256, kb, ks);
        cols(f16, syn_tokens, 512, i * 256, 256, vb, vs);
        b2[i] = try attnPartB(gpu, &s, xb, qs, ks, vs);
        made_b += 1;
        s.free();
        r.held = 0;
    }
    for (0..2) |i| {
        const q = b2[i];
        // the indexer is recomputed per rank: same raw keys, pooled blocks, selected key ids and counts as tp=1
        {
            var bad_rows: usize = 0;
            for (0..syn_tokens) |t| {
                if (!std.mem.eql(u16, q.ik[t * 128 ..][0..128], b1.ik[t * 128 ..][0..128])) {
                    bad_rows += 1;
                    if (bad_rows <= 3) std.debug.print("  ik row {d} differs on rank {d}: rank {any} vs tp=1 {any}\n", .{ t, i, q.ik[t * 128 ..][0..4], b1.ik[t * 128 ..][0..4] });
                }
            }
            if (bad_rows > 0) std.debug.print("  {d} index-key rows differ on rank {d}\n", .{ bad_rows, i });
        }
        try check.sameBytes("synthetic raw index keys", std.mem.sliceAsBytes(q.ik), std.mem.sliceAsBytes(b1.ik));
        try check.sameBytes("synthetic pooled blocks", std.mem.sliceAsBytes(q.pooled), std.mem.sliceAsBytes(b1.pooled));
        try check.sameBytes("synthetic selected key counts", std.mem.sliceAsBytes(q.nk), std.mem.sliceAsBytes(b1.nk));
        try check.sameBytes("synthetic selected key ids", std.mem.sliceAsBytes(q.ids), std.mem.sliceAsBytes(b1.ids));
        const kv_i = try kvSlice(gpa, b1.kv, syn_tokens, i);
        defer gpa.free(kv_i);
        try check.sameBytes("synthetic KV table", q.kv, kv_i);
        for (0..batch_rows) |t| try check.sameBytes("synthetic attention out", std.mem.sliceAsBytes(q.attn[t * 3072 ..][0..3072]), std.mem.sliceAsBytes(b1.attn[t * 6144 + i * 3072 ..][0..3072]));
    }
    var sparse_rows: usize = 0;
    var non_prefix: usize = 0;
    for (batches) |bt| for (0..bt.rows) |rr| {
        if ((bt.start + rr + 1) / 4 > 512) sparse_rows += 1;
    };
    // a sparse row whose 512 chosen blocks are not simply blocks 0..511 really selected something
    {
        var row_at: usize = 0;
        for (batches) |bt| {
            for (0..bt.rows) |rr| {
                if ((bt.start + rr + 1) / 4 > 512) {
                    const first_ids = b1.ids[(row_at + rr) * attn_.ids_stride ..];
                    if (first_ids[511 * 4] != 511 * 4) non_prefix += 1;
                }
            }
            row_at += bt.rows;
        }
    }
    try check.expect(sparse_rows == 13 and non_prefix * 2 >= sparse_rows, "synthetic rows: {d} sparse, {d} with a non-prefix block set (at least half expected)", .{ sparse_rows, non_prefix });
    var min_nk: i32 = std.math.maxInt(i32);
    var max_nk: i32 = 0;
    for (b1.nk) |n| {
        min_nk = @min(min_nk, n);
        max_nk = @max(max_nk, n);
    }
    const rel_b = try yBound("synthetic 20 rows", gpa, b2[0].y, b2[1].y, b1.y);
    check.pass("layer {d}: tp=2 synthetic {d}-token context (indexer recomputed per rank): pooled blocks, selected key lists ({d} rows, {d} of them sparse with a non-prefix 512-block choice, key counts {d}..{d}), KV slices and attention outputs byte-identical to tp=1 across the dense/sparse boundary; o_proj sum rel {e:.2}", .{ layer, syn_tokens, batch_rows, sparse_rows, min_nk, max_nk, rel_b });

    const kv_tp1 = try cachemod.contextKvRank(syn_cap, .bf16, 1);
    const kv_tp2 = try cachemod.contextKvRank(syn_cap, .bf16, 2);
    std.debug.print("  resident (weights only, this layer) tp=1 {d:.1} MiB; tp=2 rank 0 {d:.1} MiB, rank 1 {d:.1} MiB; all-12-layer KV at {d} tokens: tp=1 {d:.1} MiB, tp=2 {d:.1} MiB per rank (index keys + pooled blocks are replicated, ~{d:.1} MiB per layer)\n", .{ mib(held1), mib(held2[0]), mib(held2[1]), syn_cap, mib(@intCast(kv_tp1)), mib(@intCast(kv_tp2)), mib(syn_cap * 128 * 2 + (syn_cap / 4 + 2) * 128 * 2) });
}

pub fn runAttn(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const layer_a: usize = if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 3;
    const layer_b: usize = if (rest.len > 2) try std.fmt.parseInt(usize, rest[2], 10) else 7;
    for ([_]usize{ layer_a, layer_b }) |l| try check.expect(l % 4 == 3 and l < 48, "layer {d} is not a full-attention layer", .{l});
    var pack = try openPack(gpu, rest[0]);
    defer pack.close(gpu.io);
    var rs = try Ranks.open(gpu);
    defer rs.close(gpu);
    var at: [2]attn_.Attn = undefined;
    var loaded: usize = 0;
    defer for (0..loaded) |i| {
        rs.r[i].enter() catch {};
        at[i].unload();
    };
    for (0..2) |i| {
        try rs.r[i].enter();
        at[i] = try attn_.Attn.load(gpu.d);
        loaded += 1;
    }
    const ptrs = [2]*const attn_.Attn{ &at[0], &at[1] };
    for ([_]usize{ layer_a, layer_b }) |l| try attnLayer(gpu, &rs, ptrs, &pack, l);
    check.pass("exl3-attn-tp: layers {d} and {d} (real weights) tp=2 equals tp=1", .{ layer_a, layer_b });
}
