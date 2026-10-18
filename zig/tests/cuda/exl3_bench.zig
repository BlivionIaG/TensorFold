//! exl3-bench: old (`_ref`, one warp per block) versus new (decode-path) EXL3 projections on the pack's real shapes.
//! For each matrix: byte equality of the two outputs (rows 1 and 5), then cudaEvent timing over many launches
//! (min over batches of the mean launch time, because the GPUs are shared). Also an exhaustive check of the
//! fp32 mul1 codebook against the binary64 reference.
//! Usage: tf-cuda-test exl3-bench <pack dir> [launches per batch, default 20]

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const real = @import("exl3_real_tests.zig");
const strips = @import("exl3_strips.zig");

const Gpu = check.Gpu;
const lines = 10;

const Bench = struct {
    gpu: Gpu,
    ex: *const cuda.exl3.Exl3,
    stream: cuda.Stream,
    ev0: cuda.Event,
    ev1: cuda.Event,
    reps: usize,
    all_equal: bool = true,
    sum_ref: f64 = 0,
    sum_new: f64 = 0,

    fn time(b: *Bench, ctx: anytype, comptime f: fn (@TypeOf(ctx)) anyerror!void) !f64 {
        for (0..3) |_| try f(ctx);
        try b.stream.synchronize();
        var best: f64 = std.math.inf(f64);
        for (0..5) |_| {
            try b.ev0.record(b.stream);
            for (0..b.reps) |_| try f(ctx);
            try b.ev1.record(b.stream);
            try b.ev1.synchronize();
            const ms = try b.ev0.elapsedMs(b.ev1);
            best = @min(best, @as(f64, ms) / @as(f64, @floatFromInt(b.reps)));
        }
        return best;
    }

    fn report(b: *Bench, name: []const u8, k: usize, n: usize, halves: u8, rows: usize, equal: bool, ref_ms: f64, new_ms: f64) void {
        std.debug.print("{s:<44} K={d:<5} N={d:<6} halves={d:<2} rows={d:<2} equal={s} ref {d:>9.4} ms  new {d:>9.4} ms  x{d:.1}\n", .{ name, k, n, halves, rows, if (equal) "yes" else "NO", ref_ms, new_ms, ref_ms / new_ms });
        if (!equal) b.all_equal = false;
        b.sum_ref += ref_ms;
        b.sum_new += new_ms;
    }
};

fn fillX(gpa: std.mem.Allocator, n: usize, seed: u64) ![]f16 {
    const x = try gpa.alloc(f16, n);
    var rng = std.Random.DefaultPrng.init(seed);
    for (x) |*v| v.* = @floatCast((rng.random().float(f32) - 0.5) * 2.0);
    return x;
}

const Lin = struct {
    b: *Bench,
    m: real.Mat,
    d: real.Dev,
    x: u64,
    y: u64,
    rows: usize,
    ref: bool,
    fn go(l: Lin) anyerror!void {
        const k = l.m.tiles_k * 16;
        const n = l.m.tiles_n * 16;
        if (l.ref) {
            try l.b.ex.linearRef(l.b.stream, l.x, l.d.trellis.ptr, l.d.suh.ptr, l.d.svh.ptr, l.y, l.rows, k, n, l.m.halves, 2);
        } else {
            try l.b.ex.linear(l.b.stream, l.x, l.d.trellis.ptr, l.d.suh.ptr, l.d.svh.ptr, l.y, l.rows, k, n, l.m.halves, 2);
        }
    }
};

fn benchLinear(b: *Bench, pack: *const real.Pack, name: []const u8) !void {
    const gpa = b.gpu.gpa;
    const m = real.mat(pack, gpa, name) catch |e| {
        std.debug.print("{s}: skipped ({s})\n", .{ name, @errorName(e) });
        return;
    };
    defer {
        gpa.free(m.trellis);
        gpa.free(m.suh);
        gpa.free(m.svh);
    }
    var d = try real.upload(b.gpu, m);
    defer d.free();
    const k = m.tiles_k * 16;
    const n = m.tiles_n * 16;
    for ([_]usize{ 1, 5 }) |rows| {
        const x = try fillX(gpa, rows * k, 0x5eed + rows);
        defer gpa.free(x);
        var xd = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(x));
        defer xd.free();
        var y_ref = try cuda.DeviceBuffer.alloc(b.gpu.d, rows * n * 2);
        defer y_ref.free();
        var y_new = try cuda.DeviceBuffer.alloc(b.gpu.d, rows * n * 2);
        defer y_new.free();
        try y_ref.fill8(0xff, null);
        try y_new.fill8(0xff, null);
        try b.gpu.ctx.synchronize();
        const lr: Lin = .{ .b = b, .m = m, .d = d, .x = xd.ptr, .y = y_ref.ptr, .rows = rows, .ref = true };
        const ln: Lin = .{ .b = b, .m = m, .d = d, .x = xd.ptr, .y = y_new.ptr, .rows = rows, .ref = false };
        try lr.go();
        try ln.go();
        try b.stream.synchronize();
        const a = try check.download(b.gpu, y_ref);
        defer gpa.free(a);
        const c = try check.download(b.gpu, y_new);
        defer gpa.free(c);
        const equal = std.mem.eql(u8, a, c);
        const t_ref = try b.time(lr, Lin.go);
        const t_new = try b.time(ln, Lin.go);
        b.report(name[0..@min(name.len, 44)], k, n, m.halves, rows, equal, t_ref, t_new);
    }
}

const Exp = struct {
    b: *Bench,
    which: cuda.exl3.Expert,
    x: u64,
    ids: u64,
    t: u64,
    su: u64,
    sv: u64,
    y: u64,
    tokens: usize,
    topk: usize,
    k: usize,
    n: usize,
    halves: u8,
    ref: bool,
    fn go(e: Exp) anyerror!void {
        if (e.ref) {
            try e.b.ex.expertsRef(e.b.stream, e.which, e.x, e.ids, e.t, e.su, e.sv, e.y, e.tokens, e.k, e.n, lines, e.topk, e.halves, 2);
        } else {
            try e.b.ex.experts(e.b.stream, e.which, e.x, e.ids, e.t, e.su, e.sv, e.y, e.tokens, e.k, e.n, lines, e.topk, e.halves, 2);
        }
    }
};

fn benchExperts(b: *Bench, pack: *const real.Pack, layer: usize, proj: []const u8, which: cuda.exl3.Expert) !void {
    const gpa = b.gpu.gpa;
    var nbuf: [160]u8 = undefined;
    var first: ?real.Mat = null;
    defer if (first) |f| {
        gpa.free(f.trellis);
        gpa.free(f.suh);
        gpa.free(f.svh);
    };
    var tr_all: std.ArrayList(i16) = .empty;
    defer tr_all.deinit(gpa);
    var su_all: std.ArrayList(f16) = .empty;
    defer su_all.deinit(gpa);
    var sv_all: std.ArrayList(f16) = .empty;
    defer sv_all.deinit(gpa);
    for (0..lines) |e| {
        const m = try real.mat(pack, gpa, try std.fmt.bufPrint(&nbuf, "model.language_model.layers.{d}.mlp.experts.{d}.{s}", .{ layer, e, proj }));
        const words = @as(usize, m.halves) * 8;
        const ordered = try gpa.alloc(i16, m.trellis.len);
        defer gpa.free(ordered);
        try strips.reorder(m.trellis, m.tiles_k, m.tiles_n, words, ordered);
        try tr_all.appendSlice(gpa, ordered);
        try su_all.appendSlice(gpa, m.suh);
        try sv_all.appendSlice(gpa, m.svh);
        if (first == null) {
            first = m;
        } else {
            gpa.free(m.trellis);
            gpa.free(m.suh);
            gpa.free(m.svh);
        }
    }
    const f = first.?;
    const k = f.tiles_k * 16;
    const n = f.tiles_n * 16;
    var t = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(tr_all.items));
    defer t.free();
    var su = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(su_all.items));
    defer su.free();
    var sv = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(sv_all.items));
    defer sv.free();
    // gate_up: 1 token, topk 10 (x is [1, K]); down: 10 tokens, topk 1 (x is [10, K]).
    const tokens: usize = if (which == .gate_up) 1 else lines;
    const topk: usize = if (which == .gate_up) lines else 1;
    const x = try fillX(gpa, tokens * k, 0xbeef);
    defer gpa.free(x);
    var xd = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(x));
    defer xd.free();
    var ids: [lines]i32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast((i * 7) % lines);
    var idd = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(&ids));
    defer idd.free();
    const slots = tokens * topk;
    var y_ref = try cuda.DeviceBuffer.alloc(b.gpu.d, slots * n * 2);
    defer y_ref.free();
    var y_new = try cuda.DeviceBuffer.alloc(b.gpu.d, slots * n * 2);
    defer y_new.free();
    try y_ref.fill8(0xff, null);
    try y_new.fill8(0xff, null);
    try b.gpu.ctx.synchronize();
    const er: Exp = .{ .b = b, .which = which, .x = xd.ptr, .ids = idd.ptr, .t = t.ptr, .su = su.ptr, .sv = sv.ptr, .y = y_ref.ptr, .tokens = tokens, .topk = topk, .k = k, .n = n, .halves = f.halves, .ref = true };
    var en = er;
    en.y = y_new.ptr;
    en.ref = false;
    try er.go();
    try en.go();
    try b.stream.synchronize();
    const a = try check.download(b.gpu, y_ref);
    defer gpa.free(a);
    const c = try check.download(b.gpu, y_new);
    defer gpa.free(c);
    const equal = std.mem.eql(u8, a, c);
    const t_ref = try b.time(er, Exp.go);
    const t_new = try b.time(en, Exp.go);
    var label: [64]u8 = undefined;
    const nm = try std.fmt.bufPrint(&label, "experts {s} (L{d}, {d} slots)", .{ proj, layer, slots });
    b.report(nm, k, n, f.halves, slots, equal, t_ref, t_new);
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    const reps: usize = if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 20;
    var pack: real.Pack = .{ .files = undefined };
    defer pack.close(gpu.io);
    try pack.openAll(gpu, dir);
    var ex = try cuda.exl3.Exl3.load(gpu.d);
    defer ex.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var b: Bench = .{ .gpu = gpu, .ex = &ex, .stream = stream, .ev0 = try cuda.Event.init(gpu.d, true), .ev1 = try cuda.Event.init(gpu.d, true), .reps = reps };
    defer b.ev0.deinit();
    defer b.ev1.deinit();

    // mul1 codebook: all 65536 states, fp32 path vs the binary64 reference
    var bad = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&@as(u32, 0)));
    defer bad.free();
    try ex.probeMul1(stream, bad.ptr);
    try stream.synchronize();
    var bad_n: u32 = 0;
    try bad.download(0, std.mem.asBytes(&bad_n));
    try check.expect(bad_n == 0, "fp32 mul1 codebook differs from the binary64 reference in {d} of 65536 states", .{bad_n});
    check.pass("fp32 mul1 codebook equals the binary64 reference for all 65536 states", .{});

    const names = [_][]const u8{
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.v_proj",
        "model.language_model.layers.3.self_attn.o_proj",
        "model.language_model.layers.3.mlp.shared_expert.gate_proj",
        "model.language_model.layers.3.mlp.shared_expert.down_proj",
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj",
        "lm_head",
    };
    for (names) |nm| try benchLinear(&b, &pack, nm);
    try benchExperts(&b, &pack, 0, "gate_proj", .gate_up);
    try benchExperts(&b, &pack, 0, "up_proj", .gate_up);
    try benchExperts(&b, &pack, 0, "down_proj", .down);
    try benchExperts(&b, &pack, 17, "gate_proj", .gate_up);
    try benchExperts(&b, &pack, 17, "down_proj", .down);
    std.debug.print("sum of per-shape ms: ref {d:.3}  new {d:.3}  x{d:.1}\n", .{ b.sum_ref, b.sum_new, b.sum_ref / b.sum_new });
    try check.expect(b.all_equal, "new kernels are not bit-identical to the reference on some shape", .{});
    check.pass("exl3-bench: new kernels are bit-identical to the reference on every shape", .{});
}
