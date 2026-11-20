//! Planted-fault controls for the GLP kernels. The controls image (dsv41_glp_controls.fatbin, never embedded in the
//! production module) holds each fault of glp.cuh: every control must go red at its own gate on data built to show it,
//! while the production kernel on the very same inputs stays green. The program-level controls (a missing companion
//! half, a layer shift, reduced precision) run the paired fold + writer program on the device and show each wrong
//! program's final bits differ from the paired result and from the host end-to-end model.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../check.zig");
const ref = @import("glp_ref.zig");
const rig = @import("glp_rig.zig");
const fold = @import("glp_fold.zig");

const glp = rig.glp;

pub const Control = enum { sum, half, drop_tail, drop_nonfinite, atomic, truncate };

/// The controls module and, per control, the stream and writer entry points (the production final beside them).
pub const Controls = struct {
    mod: cuda.Module,
    fns: [6]glp.Functions,

    pub fn load(c: rig.Ctx, dir: []const u8, prod: glp.Functions) !Controls {
        const a = c.gpu.gpa;
        const path = try std.fs.path.join(a, &.{ dir, "dsv41_glp_controls.fatbin" });
        defer a.free(path);
        const raw = std.Io.Dir.cwd().readFileAlloc(c.gpu.io, path, a, .limited(1 << 28)) catch |e| {
            std.debug.print("FAIL glp: the controls image {s} is required and cannot be read: {t}\n", .{ path, e });
            return e;
        };
        defer a.free(raw);
        const image = try a.alignedAlloc(u8, .@"16", raw.len);
        defer a.free(image);
        @memcpy(image, raw);
        var mod = try cuda.Module.load(c.gpu.d, image);
        errdefer mod.unload();
        var fns: [6]glp.Functions = undefined;
        inline for (std.meta.fields(Control), 0..) |f, i| fns[i] = .{
            .stream = try mod.function(std.fmt.comptimePrint("dsv41_glp_stream_{s}", .{f.name})),
            .writer = try mod.function(std.fmt.comptimePrint("dsv41_glp_writer_{s}", .{f.name})),
            .final = prod.final,
        };
        return .{ .mod = mod, .fns = fns };
    }

    pub fn unload(s: *Controls) void {
        s.mod.unload();
    }

    pub fn ops(s: *const Controls, which: Control, c: rig.Ctx) glp.Ops {
        return .{ .f = &s.fns[@intFromEnum(which)], .s = c.stream };
    }
};

/// What a control must break: the FP64 bound, the FP64 non-finite classes, or the fixed fp32 order (the host model).
const Gate = enum { fp64_bound, nonfinite, order };

fn red(g: Gate, t: ref.Tally) bool {
    return switch (g) {
        .fp64_bound => t.over > 0,
        .nonfinite => t.class > 0,
        .order => t.model_diff > 0,
    };
}

const Plan = struct { control: Control, gate: Gate, stream: fold.FoldCase, writer: ?fold.WriterCase };

fn plans() [6]Plan {
    return .{
        // mean vs sum: x' = x - 4 s d, past the bound
        .{ .control = .sum, .gate = .fp64_bound, .stream = .{ .name = "mean vs sum", .width = 5120, .rows = 16, .mode = .project, .seed = 0x5001 }, .writer = null },
        // fp16 accumulation of the dot on integer terms: every order of fp32 is exact, fp16 is not
        .{ .control = .half, .gate = .fp64_bound, .stream = .{ .name = "fp16 dot", .width = 5120, .rows = 4, .mode = .project, .alpha = 1, .fill = .stress, .seed = 0x5002 }, .writer = .{ .name = "fp16 dot", .width = 5120, .rows = 4, .world = 2, .mode = .project, .alpha = 1, .fill = .stress, .seed = 0x5003 } },
        // the last column's term dropped from the dot: its copies are 64 and its direction 1
        .{ .control = .drop_tail, .gate = .fp64_bound, .stream = .{ .name = "dropped tail", .width = 1032, .rows = 8, .mode = .project, .witness = true, .seed = 0x5004 }, .writer = .{ .name = "dropped tail", .width = 1032, .rows = 8, .world = 3, .mode = .project, .seed = 0x5005 } },
        // a NaN or infinity product dropped from the dot: a finite row where FP64 says NaN
        .{ .control = .drop_nonfinite, .gate = .nonfinite, .stream = .{ .name = "dropped nonfinite", .width = 1032, .rows = 6, .mode = .project, .fill = .nonfinite, .seed = 0x5006 }, .writer = .{ .name = "dropped nonfinite", .width = 1032, .rows = 6, .world = 2, .mode = .project, .fill = .nonfinite, .seed = 0x5007 } },
        // an unordered shared-memory add of the 320 thread partials against the fixed order, on a dot that cancels to
        // rounding noise so every change of order moves the result
        .{ .control = .atomic, .gate = .order, .stream = .{ .name = "unordered add", .width = 5120, .rows = 64, .mode = .project, .alpha = 64, .fill = .cancel, .seed = 0x5008 }, .writer = .{ .name = "unordered add", .width = 5120, .rows = 64, .world = 2, .mode = .project, .alpha = 64, .fill = .cancel, .seed = 0x5009 } },
        // truncation instead of round to nearest even
        .{ .control = .truncate, .gate = .fp64_bound, .stream = .{ .name = "truncation", .width = 1032, .rows = 8, .mode = .add, .seed = 0x500a }, .writer = .{ .name = "truncation", .width = 1032, .rows = 8, .world = 2, .mode = .add, .seed = 0x500b } },
    };
}

fn controlLine(p: Plan, half: []const u8, n: usize, bad: ref.Tally, repeats: usize) void {
    check.pass("glp control {s} ({s}): production green on the same {d} elements; control red at the {t} gate: " ++
        "{d} off the model, {d} past the FP64 bound, {d} class errors; repeats differing {d} of 4", .{ @tagName(p.control), half, n, p.gate, bad.model_diff, bad.over, bad.class, repeats });
}

pub fn run(c: rig.Ctx, controls: *const Controls) !void {
    for (plans()) |p| {
        const ops = controls.ops(p.control, c);
        const prod = try fold.foldRun(c, c.ops, p.stream);
        if (!prod.ok()) return rig.fail("control {s}: the production fold is not green on its inputs ({d} off the model, {d} past the bound)", .{ @tagName(p.control), prod.t.model_diff, prod.t.over });
        const bad = try fold.foldRun(c, ops, p.stream);
        if (!red(p.gate, bad.t)) return rig.fail("control {s} stayed green at the {t} gate on the fold: {d} off the model, {d} past the bound, {d} class", .{ @tagName(p.control), p.gate, bad.t.model_diff, bad.t.over, bad.t.class });
        controlLine(p, "fold", prod.t.n, bad.t, bad.repeat_diff);
        const wc = p.writer orelse continue;
        const wprod = try fold.writerRun(c, c.ops, wc);
        if (!wprod.ok()) return rig.fail("control {s}: the production writer is not green on its inputs", .{@tagName(p.control)});
        const wbad = try fold.writerRun(c, ops, wc);
        // the writer's fault is seen at the fixed order whichever gate the fold's names: any deviation is red
        if (wbad.t.model_diff + wbad.t.over + wbad.t.class == 0) return rig.fail("control {s} stayed green on the writer", .{@tagName(p.control)});
        controlLine(p, "writer", wprod.t.n, wbad.t, wbad.repeat_diff);
    }
}

// ---- program-level controls ---------------------------------------------------------------------------------------

const W = 1032;
const R = 4;
const world = 2;
const alpha: f32 = 0.75;
const eps: f32 = 1e-6;

const Prog = struct {
    steer: bool = true, // false: the unsteered program
    writer: bool = true,
    fold: bool = true,
    shift: bool = false, // layer 1 takes layer 0's direction
    arith: ref.Arith = .{}, // a reduced-precision model (host only)
};

const Inputs = struct {
    dirs: [2][]f32,
    x0: []u16,
    g: [2][]u16, // [world, R, W] a layer
    pre: []f32,
    nw: []f32,
    a: std.mem.Allocator,

    fn init(a: std.mem.Allocator, seed: u64) !Inputs {
        var rng = ref.Rng.init(seed);
        var s: Inputs = undefined;
        s.a = a;
        for (&s.dirs) |*d| {
            d.* = try a.alloc(f32, W);
            ref.fillDir(.random, &rng, d.*);
        }
        s.x0 = try a.alloc(u16, R * 4 * W);
        for (s.x0) |*v| v.* = ref.spread(&rng);
        for (&s.g) |*g| {
            g.* = try a.alloc(u16, world * R * W);
            for (g.*) |*v| v.* = ref.spread(&rng);
        }
        s.pre = try a.alloc(f32, R * 4);
        for (s.pre) |*q| q.* = @floatCast(rng.unit() * 1.5);
        s.nw = try a.alloc(f32, W);
        for (s.nw) |*v| v.* = @floatCast(0.5 + rng.unit());
        return s;
    }

    fn deinit(s: Inputs) void {
        for (s.dirs) |d| s.a.free(d);
        for (s.g) |g| s.a.free(g);
        s.a.free(s.x0);
        s.a.free(s.pre);
        s.a.free(s.nw);
    }
};

/// The program's final normed rows, on the device (`dev`) or all on the host model. Between the halves a plain residual
/// add stands for mHC's hc post: x_c = bf16(x_c + o), o the attention output after the writer.
fn program(c: rig.Ctx, in: Inputs, p: Prog, dev: bool) ![]u16 {
    const a = c.gpu.gpa;
    const x = try a.dupe(u16, in.x0);
    defer a.free(x);
    const o = try a.alloc(u16, R * W);
    defer a.free(o);
    const ym = try a.alloc(u16, 4 * W);
    defer a.free(ym);
    for (0..2) |l| {
        const d = if (p.shift and l == 1) in.dirs[0] else in.dirs[l];
        if (p.steer and p.writer) {
            if (dev) {
                var gb = try rig.Buf.init(c.gpu, rig.bytes(in.g[l]));
                defer gb.free();
                var db = try rig.Buf.init(c.gpu, rig.bytes(d));
                defer db.free();
                try rig.writer(c.ops, gb.ptr(), R * W, W, db.ptr(), alpha, .project, R, W, world);
                try c.sync();
                const got = try gb.read(a);
                defer a.free(got);
                for (0..R * W) |i| o[i] = rankSum(rig.words(got), i);
            } else for (0..R) |r| {
                var ranks: [world][]const u16 = undefined;
                for (0..world) |rk| ranks[rk] = in.g[l][rk * R * W + r * W ..][0..W];
                ref.writerRow(world, &ranks, d, alpha, .project, p.arith, o[r * W ..][0..W]);
            }
        } else for (0..R * W) |i| o[i] = rankSum(in.g[l], i);
        for (0..R) |r| for (0..4) |cc| for (0..W) |k| {
            const i = r * 4 * W + cc * W + k;
            x[i] = ref.rne32(ref.widen(x[i]) + ref.widen(o[r * W + k]));
        };
        if (!(p.steer and p.fold)) continue;
        if (dev) {
            var xb = try rig.Buf.init(c.gpu, rig.bytes(x));
            defer xb.free();
            var db = try rig.Buf.init(c.gpu, rig.bytes(d));
            defer db.free();
            try rig.fold(c.ops, xb.ptr(), 4 * W, 0, 0, db.ptr(), alpha, .project, R, W);
            try c.sync();
            const got = try xb.read(a);
            defer a.free(got);
            @memcpy(x, rig.words(got));
        } else for (0..R) |r| {
            ref.foldRow(x[r * 4 * W ..][0 .. 4 * W], d, alpha, .project, p.arith, ym, null);
            @memcpy(x[r * 4 * W ..][0 .. 4 * W], ym);
        }
    }
    const out = try a.alloc(u16, R * W);
    errdefer a.free(out);
    if (dev) {
        var xb = try rig.Buf.init(c.gpu, rig.bytes(x));
        defer xb.free();
        var pb = try rig.Buf.init(c.gpu, rig.bytes(in.pre));
        defer pb.free();
        var nb = try rig.Buf.init(c.gpu, rig.bytes(in.nw));
        defer nb.free();
        var ob = try rig.Buf.init(c.gpu, rig.bytes(out));
        defer ob.free();
        try c.ops.final(.{ .x = xb.ptr(), .xs = 4 * W, .pre = pb.ptr(), .nw = nb.ptr(), .out = ob.ptr(), .eps = eps, .nwbf16 = 0, .R = R, .width = W });
        try c.sync();
        const got = try ob.read(a);
        defer a.free(got);
        @memcpy(out, rig.words(got));
    } else for (0..R) |r| {
        const q: [4]f32 = in.pre[r * 4 ..][0..4].*;
        ref.finalRow(x[r * 4 * W ..][0 .. 4 * W], q, rig.bytes(in.nw), false, eps, out[r * W ..][0..W]);
    }
    return out;
}

/// The rank-order sum of element `i` of every rank's slot, rounded once: what mHC's post reads.
fn rankSum(g: []const u16, i: usize) u16 {
    var s = ref.widen(g[i]);
    for (1..world) |rk| s += ref.widen(g[rk * R * W + i]);
    return ref.rne32(s);
}

fn differ(a: []const u16, b: []const u16) usize {
    var n: usize = 0;
    for (a, b) |p, q| {
        if (!ref.same(p, q)) n += 1;
    }
    return n;
}

pub fn programs(c: rig.Ctx) !void {
    const a = c.gpu.gpa;
    const in = try Inputs.init(a, 0x7001);
    defer in.deinit();
    const paired = try program(c, in, .{}, true);
    defer a.free(paired);
    const model = try program(c, in, .{}, false);
    defer a.free(model);
    if (differ(paired, model) != 0) return rig.fail("program: the paired device program is not the host end-to-end model ({d} words)", .{differ(paired, model)});
    check.pass("glp program: the paired fold + writer program over 2 layers equals the host end-to-end model bit for bit ({d} words)", .{paired.len});
    const wrong = [_]struct { name: []const u8, p: Prog }{
        .{ .name = "no steering at all", .p = .{ .steer = false } },
        .{ .name = "missing companion: the fold half only", .p = .{ .writer = false } },
        .{ .name = "missing companion: the writer half only", .p = .{ .fold = false } },
        .{ .name = "layer shift: layer 0's direction at layer 1", .p = .{ .shift = true } },
    };
    for (wrong) |w| {
        const got = try program(c, in, w.p, true);
        defer a.free(got);
        const n = differ(got, paired);
        if (n == 0) return rig.fail("program control '{s}' left the final bits equal to the paired result", .{w.name});
        check.pass("glp program control '{s}': {d} of {d} final words differ from the paired result, as planted", .{ w.name, n, paired.len });
    }
    const variants = [_]struct { name: []const u8, arith: ref.Arith }{
        .{ .name = "bf16 direction", .arith = .{ .dir_bf16 = true } },
        .{ .name = "fp16 mean accumulation", .arith = .{ .mean_acc = .f16 } },
        .{ .name = "bf16 mean accumulation", .arith = .{ .mean_acc = .bf16 } },
        .{ .name = "fp16 dot accumulation", .arith = .{ .dot_acc = .f16 } },
        .{ .name = "bf16 dot accumulation", .arith = .{ .dot_acc = .bf16 } },
    };
    for (variants) |v| {
        const got = try program(c, in, .{ .arith = v.arith }, false);
        defer a.free(got);
        const n = differ(got, paired);
        if (n == 0) return rig.fail("reduced-precision control '{s}' equals the kernel's final bits: the kernel would not be told from it", .{v.name});
        check.pass("glp reduced-precision control '{s}': the host model of it differs from the kernel in {d} of {d} final words, as planted", .{ v.name, n, paired.len });
    }
}
