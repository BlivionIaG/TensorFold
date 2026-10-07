//! The prefill GEMM tile against the previous one on the engine's projection shapes: random packed products, each
//! tile's error against a float64 dequant(W) . x on sampled outputs, best-of-n milliseconds and TFLOPS of each.
//! `tf-hip-test gemm [reps] [name filter]`, and `gemm sweep` for ragged shapes (errors and run-to-run bytes only);
//! `gemm tiers` and `gemm short` are gemm_short.zig's.

const std = @import("std");
const hip = @import("hip");
const check = @import("check.zig");
const ref = @import("gemm_ref.zig");
const short = @import("gemm_short.zig");
const Gpu = check.Gpu;

const Case = struct { name: []const u8, m: usize, n: usize, k: usize, bits: c_int = 4, group: c_int = 64, experts: usize = 0, x_div: c_int = 1, even: bool = false };

// Qwen3.6-35B-A3B (hidden 2048), Qwen3.5-9B (4096) and Qwen3.8-27B (5120) prefill projections, 4-bit groups of 64
// unless named; the expert rows are a prompt's pairs over 256 experts (m * 8 / 256).
const cases = [_]Case{
    .{ .name = "35b qkv", .m = 2048, .n = 8192, .k = 2048 },
    .{ .name = "35b qkv", .m = 8192, .n = 8192, .k = 2048 },
    .{ .name = "35b o", .m = 2048, .n = 2048, .k = 4096 },
    .{ .name = "35b o", .m = 8192, .n = 2048, .k = 4096 },
    .{ .name = "35b shared gate_up", .m = 8192, .n = 1024, .k = 2048 },
    .{ .name = "35b expert gate_up", .m = 64, .n = 1024, .k = 2048 },
    .{ .name = "35b expert gate_up", .m = 256, .n = 1024, .k = 2048 },
    .{ .name = "35b expert down", .m = 256, .n = 2048, .k = 512 },
    .{ .name = "35b routed gate_up", .m = 2048 * 8, .n = 1024, .k = 2048, .experts = 256, .x_div = 8 },
    .{ .name = "35b routed gate_up", .m = 8192 * 8, .n = 1024, .k = 2048, .experts = 256, .x_div = 8 },
    .{ .name = "35b routed even 32", .m = 256 * 32, .n = 1024, .k = 2048, .experts = 256, .x_div = 8, .even = true },
    .{ .name = "35b routed even 64", .m = 256 * 64, .n = 1024, .k = 2048, .experts = 256, .x_div = 8, .even = true },
    .{ .name = "35b routed down", .m = 2048 * 8, .n = 2048, .k = 512, .experts = 256 },
    .{ .name = "35b routed down", .m = 8192 * 8, .n = 2048, .k = 512, .experts = 256 },
    .{ .name = "9b qkv", .m = 2048, .n = 8192, .k = 4096 },
    .{ .name = "9b gate_up", .m = 2048, .n = 24576, .k = 4096 },
    .{ .name = "9b gate_up", .m = 8192, .n = 12288, .k = 4096 },
    .{ .name = "9b down", .m = 2048, .n = 4096, .k = 12288 },
    .{ .name = "9b down", .m = 8192, .n = 4096, .k = 12288 },
    .{ .name = "27b gate_up", .m = 2048, .n = 17408, .k = 5120 },
    .{ .name = "27b down", .m = 2048, .n = 5120, .k = 17408 },
    .{ .name = "27b q", .m = 8192, .n = 12288, .k = 5120 },
    .{ .name = "vocab", .m = 128, .n = 248320, .k = 2048 },
    .{ .name = "3-bit", .m = 2048, .n = 4096, .k = 2048, .bits = 3 },
    .{ .name = "6-bit", .m = 2048, .n = 4096, .k = 2048, .bits = 6 },
    .{ .name = "8-bit", .m = 2048, .n = 4096, .k = 2048, .bits = 8 },
    .{ .name = "group 32", .m = 2048, .n = 4096, .k = 2048, .group = 32 },
    .{ .name = "group 128", .m = 2048, .n = 4096, .k = 2048, .group = 128 },
    .{ .name = "edges", .m = 1000, .n = 1001, .k = 2112 },
};

pub const Rng = struct {
    state: u64,

    pub fn next(r: *Rng) u64 {
        r.state ^= r.state >> 12;
        r.state ^= r.state << 25;
        r.state ^= r.state >> 27;
        return r.state *% 0x2545F4914F6CDD1D;
    }

    /// A value in [-1, 1) with a few mantissa bits, so activation sums stay far from overflow.
    fn unit(r: *Rng) f32 {
        const v: i32 = @intCast(r.next() >> 40 & 0x7ff);
        return @as(f32, @floatFromInt(v - 1024)) / 1024.0;
    }
};

fn bf16Bits(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16);
}

fn f16Bits(v: f32) u16 {
    const h: f16 = @floatCast(v);
    return @bitCast(h);
}

pub fn fill(gpa: std.mem.Allocator, comptime T: type, n: usize, rng: *Rng, make: *const fn (*Rng) T) ![]T {
    const host = try gpa.alloc(T, n);
    for (host) |*v| v.* = make(rng);
    return host;
}

pub fn toDevice(gpu: Gpu, host: anytype) !hip.DeviceBuffer {
    return hip.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
}

pub fn makeWord(r: *Rng) u32 {
    return @truncate(r.next() >> 16);
}

pub fn makeScale(r: *Rng) u16 {
    return bf16Bits((r.unit() + 1.0) / 16.0);
}

pub fn makeBias(r: *Rng) u16 {
    return bf16Bits(r.unit() / 2.0);
}

/// A full-mantissa activation in [-1, 1) at one of six scales: sums of products then round, in any order.
pub fn fine(r: *Rng) f32 {
    const v = r.next();
    const m: i32 = @intCast(v >> 41 & 0x3fffff);
    return @as(f32, @floatFromInt(m - 0x200000)) / 2097152.0 * std.math.ldexp(@as(f32, 1), -@as(i32, @intCast(v % 6)));
}

pub fn makeX16(r: *Rng) u16 {
    return f16Bits(fine(r));
}

pub fn makeXB(r: *Rng) u16 {
    return bf16Bits(fine(r));
}

/// Words of two device buffers that differ, compared through 32 MiB windows.
pub fn sameOnDevice(gpu: Gpu, a: hip.DeviceBuffer, b: hip.DeviceBuffer, len: usize) !usize {
    const win = 32 << 20;
    const ha = try gpu.gpa.alloc(u8, win);
    defer gpu.gpa.free(ha);
    const hb = try gpu.gpa.alloc(u8, win);
    defer gpu.gpa.free(hb);
    var diff: usize = 0;
    var at: usize = 0;
    while (at < len) : (at += win) {
        const n = @min(win, len - at);
        try a.download(at, ha[0..n]);
        try b.download(at, hb[0..n]);
        var i: usize = 0;
        while (i + 4 <= n) : (i += 4) {
            if (!std.mem.eql(u8, ha[i..][0..4], hb[i..][0..4])) diff += 1;
        }
    }
    return diff;
}

pub const Timing = struct { best: f32, median: f32 };

pub fn time(gpu: Gpu, stream: hip.Stream, reps: usize, ctx: anytype, comptime go: fn (@TypeOf(ctx)) hip.Error!void) !Timing {
    try go(ctx);
    try stream.synchronize();
    var start = try hip.Event.init(gpu.d, true);
    defer start.deinit();
    var stop = try hip.Event.init(gpu.d, true);
    defer stop.deinit();
    const ms = try gpu.gpa.alloc(f64, reps);
    defer gpu.gpa.free(ms);
    for (ms) |*t| {
        try start.record(stream);
        try go(ctx);
        try stop.record(stream);
        try stop.synchronize();
        t.* = try hip.Event.elapsedMs(start, stop);
    }
    const best = std.mem.min(f64, ms);
    return .{ .best = @floatCast(best), .median = @floatCast(check.median(ms)) };
}

const Job = struct {
    k: *const hip.affine.Kernels,
    d: *const hip.Driver,
    arg: hip.affine.Arg,
    stream: hip.abi.Stream,
    tile: hip.affine.Tile,
    items: c_int,
    routed: bool,
};

fn launch(j: Job) hip.Error!void {
    if (j.routed) return j.k.routedWith(j.d, j.arg, j.items, j.stream, j.tile);
    return j.k.blockWith(j.d, j.arg, j.stream, 1, j.tile);
}

const Rig = struct {
    gpu: Gpu,
    kernels: *const hip.affine.Kernels,
    stream: hip.Stream,
    fp16: bool,
    rng: Rng,
};

/// Evenly spaced indices below `n` ending on the last one: the edge rows and columns are sampled too.
fn sample(i: usize, count: usize, n: usize) usize {
    return if (count <= 1 or n <= count) @min(i, n - 1) else i * (n - 1) / (count - 1);
}

/// Errors of both tiles against the float64 reference on a grid of sampled outputs.
fn errors(t: *Rig, c: Case, p: ref.Problem, expert_of: []const u32, total: usize, out: [2]hip.DeviceBuffer) ![2]ref.Stat {
    var stats: [2]ref.Stat = .{ .{}, .{} };
    const pick = 24;
    for (0..@min(pick, total)) |i| {
        const row = sample(i, pick, total);
        for (0..@min(pick, c.n)) |j| {
            const col = sample(j, pick, c.n);
            const v = ref.reference(p, if (c.experts > 0) row / @as(usize, @intCast(c.x_div)) else row, if (c.experts > 0) expert_of[row] else 0, col);
            for (out, &stats) |buf, *st| {
                var y: f32 = 0;
                try buf.download((row * c.n + col) * 4, std.mem.asBytes(&y));
                st.add(y, v);
            }
        }
    }
    _ = t;
    return stats;
}

/// One product on both tiles: speed, each tile's error against the reference, and an error when the new tile is more
/// than twice as far from the reference as the previous one or differs from run to run.
fn runCase(t: *Rig, c: Case, reps: usize, quiet: bool) !f64 {
    const gpa = t.gpu.gpa;
    const experts = @max(c.experts, 1);
    const words_row = c.k * @as(usize, @intCast(c.bits)) / 32;
    const groups = c.k / @as(usize, @intCast(c.group));
    // a routed case: about `m` pairs (x row = pair / x_div) over `experts` experts, counts spread over 1/4 .. 7/4
    // of the mean and cut into items of at most 128 rows like the engine's plan
    const mean = c.m / experts;
    var total: usize = c.m;
    var item_count: usize = 1;
    var max_rows: usize = c.m;
    var plan: ?hip.DeviceBuffer = null;
    var members: ?hip.DeviceBuffer = null;
    defer if (plan) |*p| p.free();
    defer if (members) |*p| p.free();
    var expert_of: []u32 = &.{};
    defer gpa.free(expert_of);
    if (c.experts > 0) {
        const counts = try gpa.alloc(usize, experts);
        defer gpa.free(counts);
        total = 0;
        for (counts) |*n| {
            n.* = if (c.even) mean else mean / 4 + t.rng.next() % (mean * 3 / 2 + 1);
            total += n.*;
        }
        const items = try gpa.alloc(i32, (total / 128 + 2 * experts) * 3);
        defer gpa.free(items);
        const pairs = try gpa.alloc(i32, total);
        defer gpa.free(pairs);
        expert_of = try gpa.alloc(u32, total);
        for (pairs, 0..) |*p, i| p.* = @intCast(i);
        item_count = 0;
        var first: usize = 0;
        max_rows = 1;
        for (counts, 0..) |n, e| {
            var left = n;
            while (left > 0) {
                const take = @min(left, 128);
                items[item_count * 3 ..][0..3].* = .{ @intCast(e), @intCast(first), @intCast(take) };
                @memset(expert_of[first..][0..take], @intCast(e));
                item_count += 1;
                first += take;
                left -= take;
                max_rows = @max(max_rows, take);
            }
        }
        plan = try toDevice(t.gpu, items[0 .. item_count * 3]);
        members = try toDevice(t.gpu, pairs);
    }
    const x_rows = if (c.experts > 0) total / @as(usize, @intCast(c.x_div)) + 1 else c.m;
    const hx = try fill(gpa, u16, x_rows * c.k, &t.rng, if (t.fp16) makeX16 else makeXB);
    defer gpa.free(hx);
    const hw = try fill(gpa, u32, experts * c.n * words_row, &t.rng, makeWord);
    defer gpa.free(hw);
    const hs = try fill(gpa, u16, experts * c.n * groups, &t.rng, makeScale);
    defer gpa.free(hs);
    const hb = try fill(gpa, u16, experts * c.n * groups, &t.rng, makeBias);
    defer gpa.free(hb);
    var x = try toDevice(t.gpu, hx);
    defer x.free();
    var words = try toDevice(t.gpu, hw);
    defer words.free();
    var scale = try toDevice(t.gpu, hs);
    defer scale.free();
    var bias = try toDevice(t.gpu, hb);
    defer bias.free();
    var out_old = try hip.DeviceBuffer.alloc(t.gpu.d, total * c.n * 4);
    defer out_old.free();
    var out_new = try hip.DeviceBuffer.alloc(t.gpu.d, total * c.n * 4);
    defer out_new.free();
    try out_old.fill8(0, null);
    try out_new.fill8(0, null);
    const old: hip.affine.Arg = .{
        .x = x.ptr,
        .words = words.ptr,
        .scale = .{ .p = scale.ptr, .kind = 1 },
        .bias = .{ .p = bias.ptr, .kind = 1 },
        .out = out_old.ptr,
        .m = @intCast(max_rows),
        .n = @intCast(c.n),
        .k = @intCast(c.k),
        .bits = c.bits,
        .group = c.group,
        .fp16 = @intFromBool(t.fp16),
        .route = if (c.experts > 0) .{ .items = plan.?.ptr, .members = members.?.ptr, .x_div = c.x_div } else .{},
    };
    var new = old;
    new.out = out_new.ptr;
    const job = Job{ .k = t.kernels, .d = t.gpu.d, .arg = old, .stream = t.stream.handle, .tile = .block, .items = @intCast(item_count), .routed = c.experts > 0 };
    var job_new = job;
    job_new.arg = new;
    job_new.tile = .gemm;
    // old and new in turn, twice: the card's clocks fall as it stays busy, so none of them goes last alone
    var t_old: Timing = undefined;
    var t_new: Timing = undefined;
    for (0..2) |round| {
        const o = try time(t.gpu, t.stream, reps, job, launch);
        const n = try time(t.gpu, t.stream, reps, job_new, launch);
        if (round == 0) {
            t_old = o;
            t_new = n;
        } else {
            t_old.best = @min(t_old.best, o.best);
            t_new.best = @min(t_new.best, n.best);
        }
    }
    const diff = try sameOnDevice(t.gpu, out_old, out_new, total * c.n * 4);
    // a race shows between launches: launch the new tile again from a cleared output a few times
    var again: usize = 0;
    for (0..if (quiet) 4 else 1) |_| {
        var out_next = try hip.DeviceBuffer.alloc(t.gpu.d, total * c.n * 4);
        defer out_next.free();
        try out_next.fill8(0, null);
        var next = job_new;
        next.arg.out = out_next.ptr;
        try launch(next);
        try t.stream.synchronize();
        again += try sameOnDevice(t.gpu, out_new, out_next, total * c.n * 4);
    }
    const problem: ref.Problem = .{ .fp16 = t.fp16, .n = c.n, .k = c.k, .bits = @intCast(c.bits), .group = @intCast(c.group), .x = hx, .words = hw, .scale = hs, .bias = hb };
    const err = try errors(t, c, problem, expert_of, total, .{ out_old, out_new });
    const flops = 2.0 * @as(f64, @floatFromInt(total)) * @as(f64, @floatFromInt(c.n)) * @as(f64, @floatFromInt(c.k));
    const tf_old = flops / t_old.best / 1e9;
    const tf_new = flops / t_new.best / 1e9;
    if (!quiet) std.debug.print("RESULT gemm {s} m{d} n{d} k{d} b{d} g{d}: old {d:.3} ms {d:.1} TFLOPS, new {d:.3} ms {d:.1} TFLOPS, x{d:.2}; max|y-ref|/sum|terms| old {e:.1} new {e:.1}, rms old {e:.2} new {e:.2}, {d} words differ\n", .{
        c.name,         total,        c.n,          c.k,    c.bits,          c.group,
        t_old.best,     tf_old,       t_new.best,   tf_new, tf_new / tf_old, err[0].max_rel,
        err[1].max_rel, err[0].rms(), err[1].rms(), diff,
    });
    const what = .{ c.name, total, c.n, c.k, c.bits, c.group };
    try check.expect(again == 0, "gemm {s} m{d} n{d} k{d} b{d} g{d}: the new tile's output changes from run to run ({d} words)", what ++ .{again});
    try check.expect(err[1].max_rel <= 2 * err[0].max_rel + 1e-12 and err[1].rms() <= 2 * err[0].rms() + 1e-12, "gemm {s} m{d} n{d} k{d} b{d} g{d}: the new tile is further from the float64 reference than twice the previous one", what);
    return tf_new / tf_old;
}

/// Every width and group on ragged sizes, odd stage counts and short routed items.
fn sweep(t: *Rig) !usize {
    var ran: usize = 0;
    const widths = [_]c_int{ 2, 3, 4, 5, 6, 8 };
    const groups = [_]c_int{ 32, 64, 128 };
    const rows = [_]usize{ 64, 100, 129, 300 };
    const cols = [_]usize{ 1, 33, 130, 257 };
    for (widths) |bits| for (groups) |group| for (rows, 0..) |m, i| {
        // k in groups: one, an odd stage count of 32-code stages (group 32 only) and a few
        const ks = [_]usize{ @intCast(group), @as(usize, @intCast(group)) * 3, if (group == 32) 2080 else 4 * @as(usize, @intCast(group)) };
        for (ks, 0..) |k, j| {
            const n = cols[(i + j) % cols.len];
            _ = try runCase(t, .{ .name = "sweep", .m = m, .n = n, .k = k, .bits = bits, .group = group }, 1, true);
            ran += 1;
        }
    };
    // routed: short items (1 to 40 rows a 128-row tile) over a few experts
    for (widths) |bits| {
        _ = try runCase(t, .{ .name = "sweep routed", .m = 600, .n = 200, .k = 256, .bits = bits, .experts = 40, .x_div = 2 }, 1, true);
        ran += 1;
    }
    return ran;
}

pub fn run(gpu: Gpu, args: []const [:0]const u8) !void {
    if (args.len > 0 and (std.mem.eql(u8, args[0], "tiers") or std.mem.eql(u8, args[0], "short"))) return short.run(gpu, args);
    const sweep_only = args.len > 0 and std.mem.eql(u8, args[0], "sweep");
    const reps: usize = if (args.len > 0 and !sweep_only) try std.fmt.parseInt(usize, args[0], 10) else 5;
    const filter: []const u8 = if (args.len > 1) args[1] else "";
    const family = hip.rocm.familyOf(try gpu.ctx.capability()) orelse return error.UnsupportedGpu;
    var lib = try hip.rocm.Library.open(gpu.d, family);
    defer lib.close();
    const launcher = &(lib.zig orelse return error.LibraryUnavailable);
    var t: Rig = .{
        .gpu = gpu,
        .kernels = &launcher.affine,
        .stream = try hip.Stream.init(gpu.d, true),
        .fp16 = family == .rdna2,
        .rng = .{ .state = 0x9E3779B97F4A7C15 },
    };
    defer t.stream.deinit();
    if (sweep_only) {
        const ran = try sweep(&t);
        check.pass("gemm sweep: {d} ragged products (every width and group, odd stage counts, short routed items), steady and within twice the previous tile's error of the float64 reference", .{ran});
        return;
    }
    var ran: usize = 0;
    var worst: f64 = std.math.inf(f64);
    for (cases) |c| {
        if (filter.len > 0 and std.mem.indexOf(u8, c.name, filter) == null) continue;
        worst = @min(worst, try runCase(&t, c, reps, false));
        ran += 1;
    }
    try check.expect(ran > 0, "gemm: no case matches '{s}'", .{filter});
    check.pass("gemm: {d} products, the new tile within twice the previous tile's error of the float64 reference (slowest x{d:.2})", .{ ran, worst });
}
