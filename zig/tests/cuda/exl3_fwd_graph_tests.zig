//! Gates for the decode-step CUDA graphs (exl3_fwd_graph.zig), run with `exl3-forward <pack> graph [<pool lines> ...]`.
//! For each pool size: an eager baseline (`hc+pN`), a second eager run (determinism), the segmented path launched eagerly
//! (`gref`, the enqueue-time reference) and the graphed run (`graph`), all on the same 16-token prompt and the same 15
//! teacher-forced decode rows. Gates: graphed logits byte-identical to eager at all 31 positions; each segment captured exactly once
//! (96 captures for 48 layers x A/B) and replayed on every decode row (15 x 96), with pool misses (changed ids / swapped lines)
//! in the run; ms per decode step eager vs graphed (a test measurement, not a published number).

const std = @import("std");
const check = @import("check.zig");
const real = @import("exl3_real_tests.zig");
const fwd = @import("exl3_fwd.zig");
const wire = @import("exl3_fwd_wire.zig");

const Gpu = check.Gpu;
const vocab = fwd.vocab;

fn meanOf(xs: []const f64) f64 {
    var s: f64 = 0;
    for (xs) |x| s += x;
    return s / @as(f64, @floatFromInt(xs.len));
}

fn minOf(xs: []const f64) f64 {
    var m = xs[0];
    for (xs) |x| m = @min(m, x);
    return m;
}

fn free(gpu: Gpu, r: wire.Result) void {
    gpu.gpa.free(r.logits);
    gpu.gpa.free(r.picks);
    gpu.gpa.free(r.gaps);
}

fn compare(what: []const u8, a: wire.Result, b: wire.Result) !void {
    const eq = std.mem.eql(u8, std.mem.sliceAsBytes(a.logits), std.mem.sliceAsBytes(b.logits));
    var agree: usize = 0;
    for (a.top1, b.top1) |x, y| agree += @intFromBool(x == y);
    var lmax: f64 = 0;
    for (a.logits, b.logits) |x, y| lmax = @max(lmax, @abs(@as(f64, y) - x));
    std.debug.print("  {s}: logits {s} ({d} positions x {d}), top-1 equal {d}/{d}, max |diff| {d:.5}\n", .{ what, if (eq) "BYTE-IDENTICAL" else "DIFFER", a.top1.len, vocab, agree, a.top1.len, lmax });
    try check.expect(eq, "{s}: logits are not byte-identical", .{what});
}

fn onePool(gpu: Gpu, pack: *const real.Pack, dir: []const u8, pool: usize) !void {
    var name: [32]u8 = undefined;
    const base_name = try std.fmt.bufPrint(&name, "hc+p{d}", .{pool});
    var n2: [32]u8 = undefined;
    const ref_name = try std.fmt.bufPrint(&n2, "hc+p{d}+gref", .{pool});
    var n3: [32]u8 = undefined;
    const g_name = try std.fmt.bufPrint(&n3, "hc+p{d}+graph", .{pool});
    std.debug.print("== graph gates, pool {d} lines/layer\n", .{pool});

    const base_cfg = try wire.Cfg.parse(base_name);
    const base = try wire.runCfg(gpu, pack, dir, base_cfg, null);
    defer free(gpu, base);
    wire.describe(base_cfg, base);
    const again = try wire.runCfg(gpu, pack, dir, base_cfg, &base.cont);
    defer free(gpu, again);
    try compare("eager vs eager (determinism)", base, again);

    const ref_cfg = try wire.Cfg.parse(ref_name);
    const ref = try wire.runCfg(gpu, pack, dir, ref_cfg, &base.cont);
    defer free(gpu, ref);
    wire.describe(ref_cfg, ref);
    try compare("segmented-eager vs eager", base, ref);

    const g_cfg = try wire.Cfg.parse(g_name);
    const g = try wire.runCfg(gpu, pack, dir, g_cfg, &base.cont);
    defer free(gpu, g);
    wire.describe(g_cfg, g);
    try compare("graphed vs eager", base, g);

    const decode_rows: u32 = 15;
    const segs: u32 = fwd.layers_total * 2;
    try check.expect(g.graph.captures == segs, "expected {d} captures (48 layers x 2 segments), got {d}", .{ segs, g.graph.captures });
    try check.expect(g.graph.replays == decode_rows * segs, "expected {d} replays, got {d}", .{ decode_rows * segs, g.graph.replays });
    try check.expect(g.stats.misses > 0, "pool {d} produced no misses, ids/lines were never changed under replay", .{pool});
    const em = ref.step_ms[0..];
    const gm = g.step_ms[0..];
    // step 0 of the graphed run includes the 96 captures; the steady state is steps 1..14
    std.debug.print("  decode step (forward + logits + 248k-wide copy), a test measurement:\n", .{});
    std.debug.print("    eager baseline        mean {d:.0} ms, min {d:.0} ms\n", .{ meanOf(&base.step_ms), minOf(&base.step_ms) });
    std.debug.print("    segmented, eager      mean {d:.0} ms, min {d:.0} ms; host enqueue {d:.2} ms/step\n", .{ meanOf(em), minOf(em), @as(f64, @floatFromInt(ref.graph.enqueue_ns)) / 1e6 / decode_rows });
    std.debug.print("    segmented, graphed    mean {d:.0} ms, min {d:.0} ms (step 0 {d:.0} ms incl. captures; steps 1.. mean {d:.0}); host enqueue {d:.2} ms/step\n", .{ meanOf(gm), minOf(gm), gm[0], meanOf(gm[1..]), @as(f64, @floatFromInt(g.graph.enqueue_ns)) / 1e6 / decode_rows });
    std.debug.print("    launch overhead saved: {d:.2} ms/step of host enqueue ({d:.0} -> {d:.0} us per segment)\n", .{
        (@as(f64, @floatFromInt(ref.graph.enqueue_ns)) - @as(f64, @floatFromInt(g.graph.enqueue_ns))) / 1e6 / decode_rows,
        @as(f64, @floatFromInt(ref.graph.enqueue_ns)) / 1e3 / @as(f64, @floatFromInt(ref.graph.eager_segments)),
        @as(f64, @floatFromInt(g.graph.enqueue_ns)) / 1e3 / @as(f64, @floatFromInt(g.graph.replays)),
    });
    check.pass("pool {d}: graphed decode logits byte-identical to eager (31 positions: 16 prompt + 15 decode rows); {d} captures, {d} replays, {d} pool misses staged ({d} hits) under replay", .{
        pool, g.graph.captures, g.graph.replays, g.stats.misses, g.stats.hits,
    });
}

pub fn run(gpu: Gpu, pack: *const real.Pack, dir: []const u8, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        try onePool(gpu, pack, dir, 12);
        try onePool(gpu, pack, dir, 64);
        return;
    }
    for (args) |a| try onePool(gpu, pack, dir, try std.fmt.parseInt(usize, a, 10));
}
