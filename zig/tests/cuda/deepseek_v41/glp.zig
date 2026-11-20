//! `tf-dsv41-test glp FATBIN_DIR`: GLP steering kernels (dsv41_glp, the production image through Kernels.loadGlp) on the
//! device against an independent FP64 formula and a host fp32 model of the documented order: the fold at the served and
//! small widths, strided and dense, with and without the tap; the writer at worlds 1-8; the final (also against mHC's
//! final); repeats on dirty memory; non-finite propagation; tails; row and slot separation; the alpha 0 bypass; the
//! launch refusals; the planted controls loaded from FATBIN_DIR/dsv41_glp_controls.fatbin (their absence fails the
//! command); program-level controls; and the CUDA graph lifetime.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");
const ref = @import("glp_ref.zig");
const rig = @import("glp_rig.zig");
const fold = @import("glp_fold.zig");
const final = @import("glp_final.zig");
const controls = @import("glp_controls.zig");
const graph = @import("glp_graph.zig");

const glp = rig.glp;

var counter: u64 = 0x1000;

fn nextSeed() u64 {
    counter += 1;
    return counter;
}

fn foldSuite(c: rig.Ctx) !void {
    var name: [64]u8 = undefined;
    for ([_]usize{ 5120, 128, 1032 }) |w| for ([_]usize{ 1, 3, 16, 17, 64, 255 }) |rows| {
        for ([_]ref.Mode{ .add, .project }) |mode| for ([_]bool{ false, true }) |strided| for ([_]bool{ false, true }) |tap| {
            if (w == 5120 and rows == 255 and !(strided and tap)) continue; // one layout at the widest case
            const label = try std.fmt.bufPrint(&name, "{s} w{d} R{d} {s}{s}", .{ @tagName(mode), w, rows, if (strided) "strided" else "dense", if (tap) " tap" else "" });
            try fold.foldCase(c, .{
                .name = label,
                .width = w,
                .rows = rows,
                .xs = if (strided) 4 * w + 64 else 0,
                .tap = tap,
                .ts = if (strided and tap) w + 64 else 0,
                .mode = mode,
                .alpha = if (mode == .add) 1.5 else 0.75,
                .seed = nextSeed(),
            });
        };
    };
}

/// Inputs of other kinds: bf16 ties (exact in FP64 too: no flip allowed), mirrored halves, integer terms, the largest
/// finite copies (the fp32 mean must not overflow), the last column's edit, NaN / infinity in a copy or in the direction.
fn fillSuite(c: rig.Ctx) !void {
    for ([_]usize{ 5120, 1032 }) |w| {
        const seed = nextSeed();
        const o = try fold.foldRun(c, c.ops, .{ .name = "ties", .width = w, .rows = 17, .mode = .add, .alpha = 1, .fill = .ties, .seed = seed });
        if (o.t.flips != 0) return rig.fail("ties w{d}: {d} bf16 ties rounded differently from FP64", .{ w, o.t.flips });
        try fold.report("fold", "add ties (every element on a bf16 tie)", o, .add, seed, false, false);
    }
    for ([_]ref.Mode{ .add, .project }) |mode| {
        const a = try fold.foldRun(c, c.ops, .{ .name = "extreme", .width = 128, .rows = 4, .mode = mode, .alpha = 0.5, .fill = .extreme, .tap = true, .seed = nextSeed() });
        if (a.nonfinite != 0 or !a.ok()) return rig.fail("extreme copies (0x7f7f / 0xff7f) {s}: not finite and equal to FP64", .{@tagName(mode)});
        check.pass("glp fold {s} extreme: four copies of the largest finite bf16 in both signs stay finite, {d} elements equal the FP64 math (no fp32 overflow of the mean)", .{ @tagName(mode), a.t.n });
        for ([_]ref.Fill{ .cancel, .stress }) |fill| try fold.foldCase(c, .{ .name = @tagName(fill), .width = 5120, .rows = 8, .mode = mode, .alpha = if (fill == .cancel) 64 else 1, .fill = fill, .tap = true, .seed = nextSeed() });
        for ([_]usize{ 1032, 5120 }) |w| try fold.foldCase(c, .{ .name = "last column edited", .width = w, .rows = 3, .mode = mode, .tap = true, .witness = true, .seed = nextSeed() });
        for ([_]usize{ 5120, 1032 }) |w| try fold.foldCase(c, .{ .name = "nonfinite copies", .width = w, .rows = 6, .mode = mode, .fill = .nonfinite, .tap = true, .seed = nextSeed() });
        try fold.foldCase(c, .{ .name = "NaN in the direction", .width = 1032, .rows = 4, .mode = mode, .fault = .nan, .seed = nextSeed() });
        try fold.foldCase(c, .{ .name = "infinity in the direction", .width = 1032, .rows = 4, .mode = mode, .fault = .inf, .seed = nextSeed() });
    }
}

fn writerSuite(c: rig.Ctx) !void {
    var name: [64]u8 = undefined;
    for ([_]usize{ 1, 2, 3, 8 }) |world| for ([_]ref.Mode{ .add, .project }) |mode| for ([_]usize{ 5120, 1032 }) |w| for ([_]usize{ 1, 5, 17 }) |rows| for ([_]bool{ false, true }) |strided| {
        const label = try std.fmt.bufPrint(&name, "w{d} R{d} {s}", .{ w, rows, if (strided) "strided" else "dense" });
        try fold.writerCase(c, .{
            .name = label,
            .width = w,
            .rows = rows,
            .total = if (strided) rows + 3 else 0,
            .world = world,
            .gs = if (strided) w + 64 else 0,
            .gr_extra = if (strided) 64 else 0,
            .mode = mode,
            .alpha = if (mode == .add) 1.5 else 0.75,
            .seed = nextSeed(),
        });
    };
    for ([_]ref.Mode{ .add, .project }) |mode| {
        for ([_]usize{ 1, 2, 3, 8 }) |world| try fold.writerCase(c, .{ .name = "nonfinite partials", .width = 1032, .rows = 6, .world = world, .mode = mode, .fill = .nonfinite, .seed = nextSeed() });
        for ([_]ref.Fill{ .cancel, .stress }) |fill| try fold.writerCase(c, .{ .name = @tagName(fill), .width = 5120, .rows = 8, .world = 2, .mode = mode, .alpha = if (fill == .cancel) 64 else 1, .fill = fill, .seed = nextSeed() });
    }
}

fn finalSuite(c: rig.Ctx, k: *const dsv41.Kernels) !void {
    var name: [64]u8 = undefined;
    for ([_]bool{ false, true }) |nw16| for ([_]usize{ 1, 3, 17 }) |rows| for ([_]usize{ 5120, 128 }) |w| for ([_]bool{ false, true }) |strided| {
        const label = try std.fmt.bufPrint(&name, "w{d} R{d} {s}", .{ w, rows, if (strided) "strided" else "dense" });
        try final.finalCase(c, .{ .name = label, .width = w, .rows = rows, .xs = if (strided) 4 * w + 64 else 0, .nw16 = nw16, .seed = nextSeed() });
    };
    for ([_]bool{ false, true }) |nw16| for ([_]usize{ 1, 3, 16 }) |rows| try final.mhcEqual(c, k, rows, nw16, nextSeed());
}

fn expectError(what: []const u8, want: anyerror, result: anytype) !void {
    if (result) |_| {
        return rig.fail("{s}: expected {t}, the call was accepted", .{ what, want });
    } else |e| {
        if (e != want) return rig.fail("{s}: expected {t}, got {t}", .{ what, want, e });
    }
}

/// Launches the checks refuse, with real buffers: each is refused with its error and no byte of any buffer changes.
/// Widths with a partial last group of eight columns are refused for the fold and the writer (and, for the final, any
/// width that is not whole 128-column chains).
fn refusals(c: rig.Ctx) !void {
    const a = c.gpu.gpa;
    const W = 5120;
    const zeros = try a.alloc(u8, 2 * 4 * W * 2 + 256);
    defer a.free(zeros);
    @memset(zeros, 0x3c);
    var xb = try rig.Buf.init(c.gpu, zeros);
    defer xb.free();
    var db = try rig.Buf.init(c.gpu, zeros[0 .. W * 4]);
    defer db.free();
    const ops = c.ops;
    const x = xb.ptr();
    const d = db.ptr();
    var refused: usize = 0;
    for ([_]usize{ 0, 4, 12, 1028, 5121, 5124, 10248 }) |w| {
        try expectError("fold width", error.BadWidth, rig.fold(ops, x, 4 * W, 0, 0, d, 1, .add, 2, w));
        try expectError("writer width", error.BadWidth, rig.writer(ops, x, W, W, d, 1, .add, 2, w, 2));
        refused += 2;
    }
    for (1..8) |extra| { // a partial last group of any size
        try expectError("fold partial group", error.BadWidth, rig.fold(ops, x, 4 * W, 0, 0, d, 1, .project, 2, 1024 + extra));
        try expectError("writer partial group", error.BadWidth, rig.writer(ops, x, W, W, d, 1, .project, 2, 1024 + extra, 2));
        refused += 2;
    }
    for ([_]usize{ 0, 5121, 5124, 5128, 130, 64, 10368 }) |w| {
        try expectError("final width", error.BadWidth, c.ops.final(.{ .x = x, .xs = 4 * W, .pre = d, .nw = d, .out = x, .eps = 1e-6, .nwbf16 = 0, .R = 2, .width = @intCast(w) }));
        refused += 1;
    }
    try expectError("fold stride below 4 width", error.BadStride, rig.fold(ops, x, 4 * W - 8, 0, 0, d, 1, .add, 2, W));
    try expectError("fold stride not a multiple of 8", error.BadStride, rig.fold(ops, x, 4 * W + 4, 0, 0, d, 1, .add, 2, W));
    try expectError("tap stride not a multiple of 8", error.Misaligned, rig.fold(ops, x, 4 * W, x, W + 4, d, 1, .add, 2, W));
    try expectError("fold x misaligned", error.Misaligned, rig.fold(ops, x + 8, 4 * W, 0, 0, d, 1, .add, 2, W));
    try expectError("fold dir misaligned", error.Misaligned, rig.fold(ops, x, 4 * W, 0, 0, d + 8, 1, .add, 2, W));
    try expectError("writer g misaligned", error.Misaligned, rig.writer(ops, x + 8, W, W, d, 1, .add, 2, W, 2));
    try expectError("writer row stride", error.BadStride, rig.writer(ops, x, W, W + 4, d, 1, .add, 2, W, 2));
    try expectError("writer rank stride", error.BadStride, rig.writer(ops, x, W + 4, W, d, 1, .add, 2, W, 2));
    try expectError("world 0", error.BadRanks, rig.writer(ops, x, W, W, d, 1, .add, 2, W, 0));
    try expectError("world 9", error.BadRanks, rig.writer(ops, x, W, W, d, 1, .add, 2, W, 9));
    try expectError("final pre misaligned", error.Misaligned, c.ops.final(.{ .x = x, .xs = 4 * W, .pre = d + 4, .nw = d, .out = x, .eps = 1e-6, .nwbf16 = 0, .R = 2, .width = W }));
    try expectError("fold mode 2", error.BadMode, ops.stream(.{ .x = x, .xs = 4 * W, .dir = d, .alpha = 1, .mode = 2, .R = 2, .width = W }));
    try expectError("writer mode -1", error.BadMode, ops.writer(.{ .g = x, .gr = W, .gs = W, .dir = d, .alpha = 1, .mode = -1, .R = 2, .width = W, .world = 2 }));
    // a negative or non-finite strength never reaches the kernels (zero, either sign, is the kernel's own bypass)
    for ([_]f32{ -0.5, -std.math.inf(f32), std.math.inf(f32), std.math.nan(f32), -1e-30 }) |bad| {
        try expectError("fold strength", error.BadStrength, rig.fold(ops, x, 4 * W, 0, 0, d, bad, .project, 2, W));
        try expectError("writer strength", error.BadStrength, rig.writer(ops, x, W, W, d, bad, .add, 2, W, 2));
        refused += 2;
    }
    try c.sync();
    const after = try xb.read(a);
    defer a.free(after);
    const dafter = try db.read(a);
    defer a.free(dafter);
    if (!std.mem.eql(u8, after, zeros) or !std.mem.eql(u8, dafter, zeros[0 .. W * 4])) return rig.fail("a refused call changed a buffer", .{});
    if (!try xb.guards(a) or !try db.guards(a)) return rig.fail("a refused call wrote a canary band", .{});
    check.pass("glp refusals: {d} launches refused (partial last groups of 1-7 columns, widths, strides, alignments, ranks, modes, strengths), no buffer byte changed", .{refused + 13});
}

/// alpha 0 is the kernels' own bypass: over rows full of NaN, infinity and random payload bits, with a tap buffer, every
/// byte of x, g, the tap and the direction is unchanged (the kernels return before reading or writing anything).
fn alphaZero(c: rig.Ctx) !void {
    const a = c.gpu.gpa;
    const w = 1032;
    const xs = 4 * w + 16;
    const ts = w + 8;
    var rng = ref.Rng.init(0x7101);
    const x = try a.alloc(u16, 6 * xs);
    defer a.free(x);
    const tap = try a.alloc(u16, 6 * ts);
    defer a.free(tap);
    const g = try a.alloc(u16, 3 * 5 * (w + 8));
    defer a.free(g);
    const d = try a.alloc(f32, w);
    defer a.free(d);
    for ([_][]u16{ x, tap, g }) |s| for (s) |*v| {
        v.* = @truncate(rng.next());
    };
    for (0..6) |r| {
        x[r * xs + r] = ref.nan16 | @as(u16, @intCast(r)); // NaN payloads, infinities of both signs
        x[r * xs + 2 * w + r] = if (r % 2 == 0) ref.inf16 else ref.ninf16;
        tap[r * ts + r] = 0xffc1;
    }
    for (d, 0..) |*v, i| v.* = if (i % 97 == 0) std.math.nan(f32) else ref.dirEntry(&rng);
    var xb = try rig.Buf.init(c.gpu, rig.bytes(x));
    defer xb.free();
    var tb = try rig.Buf.init(c.gpu, rig.bytes(tap));
    defer tb.free();
    var gb = try rig.Buf.init(c.gpu, rig.bytes(g));
    defer gb.free();
    var db = try rig.Buf.init(c.gpu, rig.bytes(d));
    defer db.free();
    const neg_zero: f32 = -0.0;
    var launches: usize = 0;
    for ([_]f32{ 0.0, neg_zero }) |alpha| for ([_]ref.Mode{ .add, .project }) |mode| {
        try rig.fold(c.ops, xb.ptr(), xs, tb.ptr(), ts, db.ptr(), alpha, mode, 6, w);
        try rig.writer(c.ops, gb.ptr(), 5 * (w + 8), w + 8, db.ptr(), alpha, mode, 5, w, 3);
        try c.sync();
        launches += 2;
        for ([_]struct { b: rig.Buf, s: []const u8 }{ .{ .b = xb, .s = rig.bytes(x) }, .{ .b = tb, .s = rig.bytes(tap) }, .{ .b = gb, .s = rig.bytes(g) }, .{ .b = db, .s = rig.bytes(d) } }) |p| {
            const got = try p.b.read(a);
            defer a.free(got);
            if (!std.mem.eql(u8, got, p.s)) return rig.fail("alpha {e} {s}: the bypass changed a byte (first at {d})", .{ alpha, @tagName(mode), std.mem.indexOfDiff(u8, got, p.s).? });
            if (!try p.b.guards(a)) return rig.fail("alpha {e} {s}: the bypass wrote a canary band", .{ alpha, @tagName(mode) });
        }
    };
    check.pass("glp alpha 0: {d} launches (+0 and -0, add and project, fold with tap and writer) over NaN / Inf / random payload rows left every byte of x, g, tap and the direction unchanged", .{launches});
}

/// Rows of other slots in one buffer stay untouched, and two launches on adjacent row ranges equal one launch on both.
fn separation(c: rig.Ctx) !void {
    const a = c.gpu.gpa;
    const w = 1032;
    const xs = 4 * w + 64;
    const total = 14;
    const first = 4; // the steered slot's rows: [4, 10)
    const n = 6;
    var rng = ref.Rng.init(0x7201);
    const d = try a.alloc(f32, w);
    defer a.free(d);
    ref.fillDir(.random, &rng, d);
    const x = try a.alloc(u16, total * xs);
    defer a.free(x);
    for (x, 0..) |*v, i| v.* = if (i / xs < first) 0x4a00 | @as(u16, @truncate(i)) & 0xff else ref.padWord(i) ^ 0x2000; // slots with their own canaries
    for (first..first + n) |r| ref.fillRow(.random, &rng, r, x[r * xs ..][0 .. 4 * w], d);
    var db = try rig.Buf.init(c.gpu, rig.bytes(d));
    defer db.free();
    var xb = try rig.Buf.init(c.gpu, rig.bytes(x));
    defer xb.free();
    const row_bytes = xs * 2;
    try xb.write(rig.bytes(x));
    try rig.fold(c.ops, xb.ptr() + first * row_bytes, xs, 0, 0, db.ptr(), 0.75, .project, n, w);
    try c.sync();
    const single = try xb.read(a);
    defer a.free(single);
    try xb.write(rig.bytes(x));
    try rig.fold(c.ops, xb.ptr() + first * row_bytes, xs, 0, 0, db.ptr(), 0.75, .project, 2, w);
    try rig.fold(c.ops, xb.ptr() + (first + 2) * row_bytes, xs, 0, 0, db.ptr(), 0.75, .project, n - 2, w);
    try c.sync();
    const split = try xb.read(a);
    defer a.free(split);
    if (!std.mem.eql(u8, single, split)) return rig.fail("separation: two launches on adjacent rows differ from one launch", .{});
    const ym = try a.alloc(u16, 4 * w);
    defer a.free(ym);
    var edited: usize = 0;
    for (0..total) |r| {
        const got = rig.words(single)[r * xs ..][0..xs];
        if (r < first or r >= first + n) {
            if (!std.mem.eql(u16, got, x[r * xs ..][0..xs])) return rig.fail("separation: row {d} of another slot changed", .{r});
            continue;
        }
        ref.foldRow(x[r * xs ..][0 .. 4 * w], d, 0.75, .project, .{}, ym, null);
        for (ym, got[0 .. 4 * w]) |m, v| {
            if (!ref.same(v, m)) return rig.fail("separation: steered row {d} is off the model", .{r});
            edited += 1;
        }
        if (!std.mem.eql(u16, got[4 * w ..], x[r * xs + 4 * w ..][0 .. xs - 4 * w])) return rig.fail("separation: pad columns of row {d} changed", .{r});
    }
    // the writer: the same over the rows of a gathered buffer [world, total, gs] with a steered row range
    const world = 2;
    const gs = w + 8;
    const g = try a.alloc(u16, world * total * gs);
    defer a.free(g);
    for (g, 0..) |*v, i| v.* = if ((i / gs) % total < first) 0x4b00 | @as(u16, @truncate(i)) & 0xff else ref.spread(&rng);
    var gb = try rig.Buf.init(c.gpu, rig.bytes(g));
    defer gb.free();
    const gbytes = gs * 2;
    try gb.write(rig.bytes(g));
    try rig.writer(c.ops, gb.ptr() + first * gbytes, total * gs, gs, db.ptr(), 0.75, .project, n, w, world);
    try c.sync();
    const wsingle = try gb.read(a);
    defer a.free(wsingle);
    try gb.write(rig.bytes(g));
    try rig.writer(c.ops, gb.ptr() + first * gbytes, total * gs, gs, db.ptr(), 0.75, .project, 3, w, world);
    try rig.writer(c.ops, gb.ptr() + (first + 3) * gbytes, total * gs, gs, db.ptr(), 0.75, .project, n - 3, w, world);
    try c.sync();
    const wsplit = try gb.read(a);
    defer a.free(wsplit);
    if (!std.mem.eql(u8, wsingle, wsplit)) return rig.fail("separation: two writer launches on adjacent rows differ from one launch", .{});
    var changed_other: usize = 0;
    for (0..world) |rk| for (0..total) |r| {
        if (r >= first and r < first + n) continue;
        const at = rk * total * gs + r * gs;
        if (!std.mem.eql(u16, rig.words(wsingle)[at..][0..gs], g[at..][0..gs])) changed_other += 1;
    };
    if (changed_other != 0) return rig.fail("separation: {d} writer rows of other slots changed", .{changed_other});
    check.pass("glp separation: a steered row range inside buffers holding other slots' rows (canaries of their own) leaves every other row and pad untouched;" ++
        " two launches on adjacent ranges equal one ({d} fold elements on the model; writer too)", .{edited});
}

pub fn run(gpu: check.Gpu, k: *dsv41.Kernels, dir: []const u8) !void {
    try k.loadGlp(gpu.d);
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads on the default stream
    defer stream.deinit();
    const ops = k.glpOps(stream) orelse return error.GlpKernelNotBuilt;
    const c: rig.Ctx = .{ .gpu = gpu, .stream = stream, .ops = ops };
    var ctl = try controls.Controls.load(c, dir, k.glp_fns.?); // required: a missing controls image fails the command
    defer ctl.unload();
    try foldSuite(c);
    try fillSuite(c);
    try writerSuite(c);
    try finalSuite(c, k);
    try refusals(c);
    try alphaZero(c);
    try separation(c);
    try controls.run(c, &ctl);
    try controls.programs(c);
    try graph.run(c);
    check.pass("glp: fold, writer, final (and mHC's final), repeats, nonfinite, tails, separation, alpha 0, refusals, 6 planted controls, 3 program controls and the graph lifetime all green", .{});
}
