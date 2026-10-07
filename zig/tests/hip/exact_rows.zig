//! Row exactness of the decode stream tile: the rows of a product launched together are, byte for byte, what each row is
//! launched alone, at every row count a lane round holds, with fp32 outputs and with outputs rounded to the activation type.
//! `tf-hip-test exact`.

const std = @import("std");
const hip = @import("hip");
const check = @import("check.zig");
const bench = @import("gemm_bench.zig");
const Gpu = check.Gpu;

const Case = struct { name: []const u8, n: usize, k: usize, bits: c_int = 4, group: c_int = 64 };

const cases = [_]Case{
    .{ .name = "vocab", .n = 248320, .k = 2048 },
    .{ .name = "qkv", .n = 8192, .k = 2048 },
    .{ .name = "down", .n = 2048, .k = 512 },
    .{ .name = "9b gate_up", .n = 24576, .k = 4096 },
    .{ .name = "ragged", .n = 1001, .k = 2112 },
    .{ .name = "2-bit", .n = 4096, .k = 2048, .bits = 2 },
    .{ .name = "3-bit", .n = 4096, .k = 2048, .bits = 3 },
    .{ .name = "5-bit", .n = 4096, .k = 2048, .bits = 5 },
    .{ .name = "6-bit group 32", .n = 4096, .k = 2048, .bits = 6, .group = 32 },
    .{ .name = "8-bit group 128", .n = 4096, .k = 2048, .bits = 8, .group = 128 },
};

/// Row counts launched together: every count of a lane round up to 16, then wider rounds.
const together = [_]usize{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 32 };
const most_rows = 32;

const Rig = struct {
    gpu: Gpu,
    kernels: *const hip.affine.Kernels,
    stream: hip.Stream,
    fp16: bool,
};

/// One product at every row count: the number of rows whose bytes differ from the same row alone.
fn product(t: *Rig, rng: *bench.Rng, c: Case, half: bool) !usize {
    const gpa = t.gpu.gpa;
    const words_row = c.k * @as(usize, @intCast(c.bits)) / 32;
    const groups = c.k / @as(usize, @intCast(c.group));
    const hx = try bench.fill(gpa, u16, most_rows * c.k, rng, if (t.fp16) bench.makeX16 else bench.makeXB);
    defer gpa.free(hx);
    const hw = try bench.fill(gpa, u32, c.n * words_row, rng, bench.makeWord);
    defer gpa.free(hw);
    const hs = try bench.fill(gpa, u16, c.n * groups, rng, bench.makeScale);
    defer gpa.free(hs);
    const hb = try bench.fill(gpa, u16, c.n * groups, rng, bench.makeBias);
    defer gpa.free(hb);
    var x = try bench.toDevice(t.gpu, hx);
    defer x.free();
    var words = try bench.toDevice(t.gpu, hw);
    defer words.free();
    var scale = try bench.toDevice(t.gpu, hs);
    defer scale.free();
    var bias = try bench.toDevice(t.gpu, hb);
    defer bias.free();
    const size: usize = if (half) 2 else 4;
    var out = try hip.DeviceBuffer.alloc(t.gpu.d, most_rows * c.n * size);
    defer out.free();
    const alone = try gpa.alloc(u8, most_rows * c.n * size);
    defer gpa.free(alone);
    const got = try gpa.alloc(u8, most_rows * c.n * size);
    defer gpa.free(got);

    var arg: hip.affine.Arg = .{
        .x = x.ptr,
        .words = words.ptr,
        .scale = .{ .p = scale.ptr, .kind = 1 },
        .bias = .{ .p = bias.ptr, .kind = 1 },
        .out = out.ptr,
        .m = 1,
        .n = @intCast(c.n),
        .k = @intCast(c.k),
        .bits = c.bits,
        .group = c.group,
        .fp16 = @intFromBool(t.fp16),
    };
    // each row alone
    for (0..most_rows) |i| {
        arg.x = x.ptr + i * c.k * 2;
        arg.m = 1;
        try out.fill8(0xA5, null);
        try t.kernels.run(t.gpu.d, arg, 4, t.stream.handle, 0, 1, half);
        try t.stream.synchronize();
        try out.download(0, alone[i * c.n * size ..][0 .. c.n * size]);
    }
    // the rows together
    var bad: usize = 0;
    arg.x = x.ptr;
    for (together) |m| {
        if (t.kernels.matrix and m > 15) continue; // the matrix cores take 16 rows and more
        arg.m = @intCast(m);
        try out.fill8(0xA5, null);
        try t.kernels.run(t.gpu.d, arg, 4, t.stream.handle, 0, 1, half);
        try t.stream.synchronize();
        try out.download(0, got[0 .. m * c.n * size]);
        for (0..m) |i| {
            const row = i * c.n * size;
            if (!std.mem.eql(u8, got[row..][0 .. c.n * size], alone[row..][0 .. c.n * size])) {
                if (bad < 6) std.debug.print("  {s} {s}: row {d} of {d} differs from the row alone\n", .{ c.name, if (half) "rounded" else "fp32", i, m });
                bad += 1;
            }
        }
    }
    return bad;
}

const Plan = struct { counts: [4]c_int };
const plans = [_]Plan{ .{ .counts = .{ 5, 3, 6, 2 } }, .{ .counts = .{ 15, 9, 1, 4 } }, .{ .counts = .{ 2, 2, 2, 2 } }, .{ .counts = .{ 1, 4, 1, 3 } } };
const routed_cases = [_]Case{
    .{ .name = "moe gate_up", .n = 1024, .k = 2048 },
    .{ .name = "moe down", .n = 2048, .k = 512 },
    .{ .name = "ragged", .n = 1002, .k = 2112 },
    .{ .name = "3-bit", .n = 1024, .k = 2048, .bits = 3 },
    .{ .name = "8-bit", .n = 1024, .k = 2048, .bits = 8, .group = 128 },
};

fn launchRouted(t: *Rig, arg: hip.affine.Arg, items: c_int, pair: bool) !void {
    if (pair) {
        if (!try t.kernels.pairRun(t.gpu.d, arg, 0, items, t.stream.handle)) return error.NoTile;
    } else {
        try t.kernels.routed(t.gpu.d, arg, items, t.stream.handle);
    }
    try t.stream.synchronize();
}

/// A routed plan's pairs together, against every pair as an item of its own: the number of output rows that differ.
fn routedProduct(t: *Rig, rng: *bench.Rng, c: Case, pair: bool) !usize {
    const gpa = t.gpu.gpa;
    const experts = 4;
    const words_row = c.k * @as(usize, @intCast(c.bits)) / 32;
    const groups = c.k / @as(usize, @intCast(c.group));
    const hx = try bench.fill(gpa, u16, 32 * c.k, rng, if (t.fp16) bench.makeX16 else bench.makeXB);
    defer gpa.free(hx);
    const hw = try bench.fill(gpa, u32, experts * c.n * words_row, rng, bench.makeWord);
    defer gpa.free(hw);
    const hs = try bench.fill(gpa, u16, experts * c.n * groups, rng, bench.makeScale);
    defer gpa.free(hs);
    const hb = try bench.fill(gpa, u16, experts * c.n * groups, rng, bench.makeBias);
    defer gpa.free(hb);
    var x = try bench.toDevice(t.gpu, hx);
    defer x.free();
    var words = try bench.toDevice(t.gpu, hw);
    defer words.free();
    var scale = try bench.toDevice(t.gpu, hs);
    defer scale.free();
    var bias = try bench.toDevice(t.gpu, hb);
    defer bias.free();
    const cols = if (pair) c.n / 2 else c.n;
    const size: usize = if (pair) 2 else 4;
    var out = try hip.DeviceBuffer.alloc(t.gpu.d, 32 * cols * size);
    defer out.free();
    const alone = try gpa.alloc(u8, 32 * cols * size);
    defer gpa.free(alone);
    const got = try gpa.alloc(u8, 32 * cols * size);
    defer gpa.free(got);
    var members: [32]c_int = undefined;
    for (&members, 0..) |*mb, i| mb.* = @intCast(i);
    var mem = try bench.toDevice(t.gpu, @as([]const c_int, &members));
    defer mem.free();

    var bad: usize = 0;
    for (plans) |plan| {
        var rows: usize = 0;
        var item_alone: [96]c_int = undefined;
        var item_all: [12]c_int = undefined;
        var most: c_int = 0;
        for (plan.counts, 0..) |n, e| {
            item_all[3 * e] = @intCast(e);
            item_all[3 * e + 1] = @intCast(rows);
            item_all[3 * e + 2] = n;
            most = @max(most, n);
            for (0..@as(usize, @intCast(n))) |_| {
                item_alone[3 * rows] = @intCast(e);
                item_alone[3 * rows + 1] = @intCast(rows);
                item_alone[3 * rows + 2] = 1;
                rows += 1;
            }
        }
        var plan_alone = try bench.toDevice(t.gpu, @as([]const c_int, item_alone[0 .. 3 * rows]));
        defer plan_alone.free();
        var plan_all = try bench.toDevice(t.gpu, @as([]const c_int, &item_all));
        defer plan_all.free();
        var arg: hip.affine.Arg = .{
            .x = x.ptr,
            .words = words.ptr,
            .scale = .{ .p = scale.ptr, .kind = 1 },
            .bias = .{ .p = bias.ptr, .kind = 1 },
            .out = if (pair) 0 else out.ptr,
            .out16 = if (pair) out.ptr else 0,
            .m = 1,
            .n = @intCast(c.n),
            .k = @intCast(c.k),
            .bits = c.bits,
            .group = c.group,
            .fp16 = @intFromBool(t.fp16),
            .route = .{ .items = plan_alone.ptr, .members = mem.ptr, .x_div = 1 },
        };
        try out.fill8(0xA5, null);
        try launchRouted(t, arg, @intCast(rows), pair);
        try out.download(0, alone[0 .. rows * cols * size]);
        arg.m = most;
        arg.route.items = plan_all.ptr;
        try out.fill8(0xA5, null);
        try launchRouted(t, arg, 4, pair);
        try out.download(0, got[0 .. rows * cols * size]);
        for (0..rows) |i| {
            const row = i * cols * size;
            if (!std.mem.eql(u8, got[row..][0 .. cols * size], alone[row..][0 .. cols * size])) {
                if (bad < 6) std.debug.print("  routed {s} {s}: pair {d} of {d} (most {d} rows) differs from the pair alone\n", .{ c.name, if (pair) "act" else "fp32", i, rows, most });
                bad += 1;
            }
        }
    }
    return bad;
}

pub fn run(gpu: Gpu) !void {
    const caps = try gpu.ctx.caps();
    var lib = try hip.rocm.Library.open(gpu.d, caps, try check.policyOf(gpu));
    defer lib.close();
    const launcher = &(lib.zig orelse return error.LibraryUnavailable);
    var t: Rig = .{ .gpu = gpu, .kernels = &launcher.affine, .stream = try hip.Stream.init(gpu.d, true), .fp16 = caps.family == .rdna2 };
    defer t.stream.deinit();
    var rng: bench.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    var bad: usize = 0;
    for (cases) |c| for ([_]bool{ false, true }) |half| {
        bad += try product(&t, &rng, c, half);
    };
    for (routed_cases) |c| for ([_]bool{ false, true }) |pair| {
        bad += try routedProduct(&t, &rng, c, pair);
    };
    try check.expect(bad == 0, "exact: {d} rows differ from the same row alone", .{bad});
    check.pass("exact: {d} products at {d} row counts, fp32 and rounded outputs, each row the same bytes alone and together", .{ cases.len, together.len });
}
