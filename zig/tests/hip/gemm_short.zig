//! The prefill tiles of a few rows against the 128-row one on the engine's products: `tf-hip-test gemm tiers` runs each
//! on the same random packed products (dense and routed, every width and group, rows 1 to 200) and
//! compares their bytes; `gemm short [reps]` times each of them on a short prompt's shapes.

const std = @import("std");
const hip = @import("hip");
const check = @import("check.zig");
const bench = @import("gemm_bench.zig");
const Gpu = check.Gpu;
const Rng = bench.Rng;
const fill = bench.fill;
const toDevice = bench.toDevice;
const makeWord = bench.makeWord;
const makeScale = bench.makeScale;
const makeBias = bench.makeBias;
const makeX16 = bench.makeX16;
const makeXB = bench.makeXB;
const fine = bench.fine;
const sameOnDevice = bench.sameOnDevice;
const time = bench.time;

const Rig = struct {
    gpu: Gpu,
    kernels: *const hip.affine.Kernels,
    stream: hip.Stream,
    fp16: bool,
    rng: Rng,
};

/// Row counts the block shapes are compared at: every short one's edges, ragged ones and a few blocks.
const tier_rows = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 33, 63, 64, 65, 100, 128, 200 };

/// One product (`m` rows dense, or `m` pairs over a few experts in items of 1 to 40 rows, and an empty item) through
/// every block shape: the words of each short block's output (and of the engine's own pick) that differ from the
/// 128-row block's.
fn tierProduct(t: *Rig, m: usize, n: usize, k: usize, bits: c_int, group: c_int, routed: bool) !usize {
    const gpa = t.gpu.gpa;
    const experts: usize = if (routed) 5 else 1;
    const words_row = k * @as(usize, @intCast(bits)) / 32;
    const groups = k / @as(usize, @intCast(group));
    var items: [3 * 256]i32 = undefined;
    var item_count: usize = 1;
    var max_rows = m;
    if (routed) {
        item_count = 0;
        max_rows = 1;
        var first: usize = 0;
        while (first < m) {
            const take = @min(m - first, 1 + t.rng.next() % 40);
            items[item_count * 3 ..][0..3].* = .{ @intCast(item_count % experts), @intCast(first), @intCast(take) };
            item_count += 1;
            first += take;
            max_rows = @max(max_rows, take);
        }
        items[item_count * 3 ..][0..3].* = .{ 0, @intCast(m), 0 };
        item_count += 1;
    }
    const pairs = try gpa.alloc(i32, m);
    defer gpa.free(pairs);
    for (pairs, 0..) |*p, i| p.* = @intCast(i);
    const x_div: usize = if (routed) 2 else 1;
    const hx = try fill(gpa, u16, (m / x_div + 1) * k, &t.rng, if (t.fp16) makeX16 else makeXB);
    defer gpa.free(hx);
    const hw = try fill(gpa, u32, experts * n * words_row, &t.rng, makeWord);
    defer gpa.free(hw);
    const hs = try fill(gpa, u16, experts * n * groups, &t.rng, makeScale);
    defer gpa.free(hs);
    const hb = try fill(gpa, u16, experts * n * groups, &t.rng, makeBias);
    defer gpa.free(hb);
    var x = try toDevice(t.gpu, hx);
    defer x.free();
    var words = try toDevice(t.gpu, hw);
    defer words.free();
    var scale = try toDevice(t.gpu, hs);
    defer scale.free();
    var bias = try toDevice(t.gpu, hb);
    defer bias.free();
    var plan = try toDevice(t.gpu, items[0 .. item_count * 3]);
    defer plan.free();
    var members = try toDevice(t.gpu, pairs);
    defer members.free();
    const bytes = m * n * 4;
    var arg: hip.affine.Arg = .{
        .x = x.ptr,
        .words = words.ptr,
        .scale = .{ .p = scale.ptr, .kind = 1 },
        .bias = .{ .p = bias.ptr, .kind = 1 },
        .out = 0,
        .m = @intCast(max_rows),
        .n = @intCast(n),
        .k = @intCast(k),
        .bits = bits,
        .group = group,
        .fp16 = @intFromBool(t.fp16),
        .route = if (routed) .{ .items = plan.ptr, .members = members.ptr, .x_div = @intCast(x_div) } else .{},
    };
    // every block shape (the K-parallel tiles, then the 128-row one) twice, and the engine's own pick
    const Kn = hip.affine.Kernels;
    var ids: [Kn.tier_count]usize = undefined;
    var n_ids: usize = 0;
    for (0..Kn.tier_count) |i| {
        if (!Kn.tierTakes(i, @intCast(max_rows))) continue;
        ids[n_ids] = i;
        n_ids += 1;
    }
    const big = n_ids - 1;
    var outs: [2 * Kn.tier_count + 1]hip.DeviceBuffer = undefined;
    const n_outs = 2 * n_ids + 1;
    for (outs[0..n_outs], 0..) |*o, i| {
        o.* = try hip.DeviceBuffer.alloc(t.gpu.d, bytes);
        errdefer for (outs[0..i]) |*f| f.free();
        try o.fill8(0xA5, null);
    }
    defer for (outs[0..n_outs]) |*o| o.free();
    for (0..2 * n_ids) |i| {
        arg.out = outs[i].ptr;
        try t.kernels.prefillTier(t.gpu.d, arg, t.stream.handle, @intCast(item_count), ids[i % n_ids]);
    }
    arg.out = outs[2 * n_ids].ptr;
    try t.kernels.prefillLaunch(t.gpu.d, arg, t.stream.handle, @intCast(item_count));
    try t.stream.synchronize();
    var diff: usize = 0;
    for (0..n_outs) |i| {
        if (i == big) continue;
        const d = try sameOnDevice(t.gpu, outs[big], outs[i], bytes);
        if (d != 0) {
            const what: usize = if (i < 2 * n_ids) ids[i % n_ids] else 99;
            std.debug.print("block shape {d} (99: the engine's pick) differs in {d} words\n", .{ what, d });
            const ha = try gpa.alloc(f32, m * n);
            defer gpa.free(ha);
            const hb2 = try gpa.alloc(f32, m * n);
            defer gpa.free(hb2);
            try outs[big].download(0, std.mem.sliceAsBytes(ha));
            try outs[i].download(0, std.mem.sliceAsBytes(hb2));
            var shown: usize = 0;
            for (ha, hb2, 0..) |u, v, at| {
                if (@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)) and shown < 12) {
                    std.debug.print("  row {d} col {d}: {e} vs {e}\n", .{ at / n, at % n, u, v });
                    shown += 1;
                }
            }
        }
        diff += d;
    }
    return diff;
}

/// Prefill's short blocks against the 128-row one at every row count of `tier_rows`, dense and routed, every width and
/// group: any differing word is an error.
fn tiers(t: *Rig) !usize {
    var ran: usize = 0;
    const widths = [_]c_int{ 2, 3, 4, 5, 6, 8 };
    const groups = [_]c_int{ 32, 64, 128 };
    for (widths) |bits| for (groups) |group| for (tier_rows) |m| for ([_]bool{ false, true }) |routed| {
        // n off every block width, k a few stages or a long odd run
        const ks = [_]usize{ 3 * @as(usize, @intCast(group)), if (group == 32) 2080 else 1024 };
        for (ks, 0..) |k, j| {
            const n: usize = if (j == 0) 97 else 288;
            const diff = try tierProduct(t, m, n, k, bits, group, routed);
            try check.expect(diff == 0, "gemm tiers m{d} n{d} k{d} b{d} g{d} routed {}: {d} words differ from the 128-row block", .{ m, n, k, bits, group, routed, diff });
            ran += 1;
        }
    };
    return ran;
}

const ShortJob = struct { k: *const hip.affine.Kernels, d: *const hip.Driver, arg: hip.affine.Arg, stream: hip.abi.Stream, items: c_int, tier: usize };

fn launchShort(j: ShortJob) hip.Error!void {
    return j.k.prefillTier(j.d, j.arg, j.stream, j.items, j.tier);
}

/// Best milliseconds of each block shape on a prompt's few rows: dense projections and a routed one (`m` tokens, 8 of
/// 256 experts each), with the weights' GB/s.
fn shortBench(t: *Rig, reps: usize) !void {
    const gpa = t.gpu.gpa;
    const dense = [_]struct { name: []const u8, n: usize, k: usize }{
        .{ .name = "9b qkv", .n = 8192, .k = 4096 },
        .{ .name = "9b gate_up", .n = 24576, .k = 4096 },
        .{ .name = "9b down", .n = 4096, .k = 12288 },
        .{ .name = "35b o", .n = 2048, .k = 4096 },
    };
    const rows = [_]usize{ 1, 2, 4, 8, 13, 24, 32, 64 };
    const group = 64;
    var rng = Rng{ .state = 0x1234567 };
    for (dense) |c| for (rows) |m| {
        const wr = c.k * 4 / 32;
        const hx = try fill(gpa, u16, m * c.k, &rng, if (t.fp16) makeX16 else makeXB);
        defer gpa.free(hx);
        const hw = try fill(gpa, u32, c.n * wr, &rng, makeWord);
        defer gpa.free(hw);
        const hs = try fill(gpa, u16, c.n * c.k / group, &rng, makeScale);
        defer gpa.free(hs);
        const hb = try fill(gpa, u16, c.n * c.k / group, &rng, makeBias);
        defer gpa.free(hb);
        var x = try toDevice(t.gpu, hx);
        defer x.free();
        var words = try toDevice(t.gpu, hw);
        defer words.free();
        var scale = try toDevice(t.gpu, hs);
        defer scale.free();
        var bias = try toDevice(t.gpu, hb);
        defer bias.free();
        var out = try hip.DeviceBuffer.alloc(t.gpu.d, m * c.n * 4);
        defer out.free();
        const arg: hip.affine.Arg = .{ .x = x.ptr, .words = words.ptr, .scale = .{ .p = scale.ptr, .kind = 1 }, .bias = .{ .p = bias.ptr, .kind = 1 }, .out = out.ptr, .m = @intCast(m), .n = @intCast(c.n), .k = @intCast(c.k), .bits = 4, .group = group, .fp16 = @intFromBool(t.fp16) };
        const bytes: f64 = @floatFromInt(c.n * (wr * 4 + c.k / group * 4));
        std.debug.print("RESULT short {s} m{d}:", .{ c.name, m });
        for (0..hip.affine.Kernels.tier_count) |tier| {
            if (!hip.affine.Kernels.tierTakes(tier, @intCast(m))) continue;
            const tm = try time(t.gpu, t.stream, reps, ShortJob{ .k = t.kernels, .d = t.gpu.d, .arg = arg, .stream = t.stream.handle, .items = 1, .tier = tier }, launchShort);
            std.debug.print(" [{d}] {d:.3} ms {d:.0} GB/s", .{ tier, tm.best, bytes / tm.best / 1e6 });
        }
        std.debug.print("\n", .{});
    };
    // routed: 35B gate_up (n 1024, k 2048) and down (n 2048, k 512) over 256 experts
    const routed = [_]struct { name: []const u8, n: usize, k: usize, x_div: usize }{
        .{ .name = "35b routed gate_up", .n = 1024, .k = 2048, .x_div = 8 },
        .{ .name = "35b routed down", .n = 2048, .k = 512, .x_div = 1 },
    };
    const experts = 256;
    for (routed) |c| for ([_]usize{ 1, 13, 32, 64, 128, 256 }) |tokens| {
        const wr = c.k * 4 / 32;
        const pairs = tokens * 8;
        const hw = try fill(gpa, u32, experts * c.n * wr, &rng, makeWord);
        defer gpa.free(hw);
        const hs = try fill(gpa, u16, experts * c.n * c.k / group, &rng, makeScale);
        defer gpa.free(hs);
        const hb = try fill(gpa, u16, experts * c.n * c.k / group, &rng, makeBias);
        defer gpa.free(hb);
        const hx = try fill(gpa, u16, (pairs / c.x_div + 1) * c.k, &rng, if (t.fp16) makeX16 else makeXB);
        defer gpa.free(hx);
        // pair i of token i / 8 goes to expert (7 * token + 31 * slot) % 256: each token's eight are distinct
        const counts = try gpa.alloc(usize, experts);
        defer gpa.free(counts);
        @memset(counts, 0);
        for (0..pairs) |i| counts[(7 * (i / 8) + 31 * (i % 8)) % experts] += 1;
        var items: std.ArrayList(i32) = .empty;
        defer items.deinit(gpa);
        var first: usize = 0;
        var max_rows: usize = 1;
        for (counts, 0..) |n, e| {
            if (n == 0) continue;
            try items.appendSlice(gpa, &.{ @intCast(e), @intCast(first), @intCast(n) });
            first += n;
            max_rows = @max(max_rows, n);
        }
        const hm = try gpa.alloc(i32, pairs);
        defer gpa.free(hm);
        for (hm, 0..) |*v, i| v.* = @intCast(i);
        var x = try toDevice(t.gpu, hx);
        defer x.free();
        var words = try toDevice(t.gpu, hw);
        defer words.free();
        var scale = try toDevice(t.gpu, hs);
        defer scale.free();
        var bias = try toDevice(t.gpu, hb);
        defer bias.free();
        var plan = try toDevice(t.gpu, items.items);
        defer plan.free();
        var members = try toDevice(t.gpu, hm);
        defer members.free();
        var out = try hip.DeviceBuffer.alloc(t.gpu.d, pairs * c.n * 4);
        defer out.free();
        const arg: hip.affine.Arg = .{ .x = x.ptr, .words = words.ptr, .scale = .{ .p = scale.ptr, .kind = 1 }, .bias = .{ .p = bias.ptr, .kind = 1 }, .out = out.ptr, .m = @intCast(max_rows), .n = @intCast(c.n), .k = @intCast(c.k), .bits = 4, .group = group, .fp16 = @intFromBool(t.fp16), .route = .{ .items = plan.ptr, .members = members.ptr, .x_div = @intCast(c.x_div) } };
        const item_count: c_int = @intCast(items.items.len / 3);
        const bytes: f64 = @floatFromInt(@as(usize, @intCast(item_count)) * c.n * (wr * 4 + c.k / group * 4));
        std.debug.print("RESULT short {s} tokens {d} ({d} items, up to {d} rows):", .{ c.name, tokens, item_count, max_rows });
        for (0..hip.affine.Kernels.tier_count) |tier| {
            if (!hip.affine.Kernels.tierTakes(tier, @intCast(max_rows))) continue;
            const tm = try time(t.gpu, t.stream, reps, ShortJob{ .k = t.kernels, .d = t.gpu.d, .arg = arg, .stream = t.stream.handle, .items = item_count, .tier = tier }, launchShort);
            std.debug.print(" [{d}] {d:.3} ms {d:.0} GB/s", .{ tier, tm.best, bytes / tm.best / 1e6 });
        }
        std.debug.print("\n", .{});
    };
}

pub fn run(gpu: Gpu, args: []const [:0]const u8) !void {
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
    if (std.mem.eql(u8, args[0], "short")) return shortBench(&t, if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 10);
    const ran = try tiers(&t);
    check.pass("gemm tiers: {d} products (rows 1..200, dense and routed, every width and group) with the same bytes in every block shape", .{ran});
}
