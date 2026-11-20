//! DeepSeek V4.1's GLP steering kernels (dsv41_glp: fold, writer, final) under the contract, with the planted-fault
//! controls of the dsv41_glp_controls image each red at the gate it breaks.
//!
//! The registration follows the earlier steering records of this tree (git ref c5947a8d2e: core/contract/steer_records.zig
//! and steer.zig): one set of cases on the heap, each a kernel with its own hooks. The control-vector idea is credited
//! in kernels/cuda/deepseek_v41/glp.cuh. The fold and the writer edit their buffers in place, so each run re-uploads the
//! prepared bytes into the output slot first; the direction (and the final's inputs) are immutable input slots, and the
//! gates' input and guard checks cover them. The layouts here are dense; strides, pads and tails are the direct test's
//! (`tf-dsv41-test glp`).
//!
//! The reference is independent FP64 math (glp_ref.zig). `check` answers the non-finite and shapes gates with both
//! tests: every output equal to a host fp32 model of the documented order (the dot's order is fixed, so equality is
//! exact), and within the stated FP64 bound. Gate 4 runs the generic path: integer terms make every fp32 order exact.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const images = @import("contract_deepseek_images.zig");
const ref = @import("deepseek_v41/glp_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows_ = core.contract.rows;

// Mirrors of glp_ops.zig's extern argument structs (this module cannot import the family's kernels): the kernels take
// each by value, 56 bytes.
const StreamArgs = extern struct { x: u64, xs: i64, tap: u64 = 0, ts: i64 = 0, dir: u64, alpha: f32, mode: c_int, R: c_int, width: c_int };
const WriterArgs = extern struct { g: u64, gr: i64, gs: i64, dir: u64, alpha: f32, mode: c_int, R: c_int, width: c_int, world: c_int };
const FinalArgs = extern struct { x: u64, xs: i64, pre: u64, nw: u64, out: u64, eps: f32, nwbf16: c_int, R: c_int, width: c_int };

comptime {
    std.debug.assert(@sizeOf(StreamArgs) == 56 and @offsetOf(StreamArgs, "dir") == 32 and @offsetOf(StreamArgs, "alpha") == 40);
    std.debug.assert(@sizeOf(WriterArgs) == 56 and @offsetOf(WriterArgs, "dir") == 24 and @offsetOf(WriterArgs, "alpha") == 32);
    std.debug.assert(@sizeOf(FinalArgs) == 56 and @offsetOf(FinalArgs, "eps") == 40 and @offsetOf(FinalArgs, "R") == 48);
}

const eps: f32 = 1e-6;

const Kind = enum { stream, writer, final };

/// The planted faults and the gate each is registered red at.
const Ctl = enum {
    sum,
    half,
    drop_tail,
    drop_nonfinite,
    atomic,
    truncate,

    fn gate(k: Ctl) c.Gate {
        return switch (k) {
            .sum, .drop_tail, .atomic, .truncate => .shapes,
            .half => .sums,
            .drop_nonfinite => .nonfinite,
        };
    }
};

/// Rows: the steered rows; n and k: the row's width. The served widths are the production 5120 and two small ones.
const edit_served = [_]c.Shape{ .{ .rows = 4, .n = 5120, .k = 5120 }, .{ .rows = 17, .n = 1032, .k = 1032 }, .{ .rows = 3, .n = 128, .k = 128 } };
const final_served = [_]c.Shape{ .{ .rows = 3, .n = 5120, .k = 5120 }, .{ .rows = 17, .n = 128, .k = 128 } };

/// The deepest chain of fp32 roundings of the dot: a thread's 32 fmas, the butterfly's 5 adds, the 10 warps, the mean's 3.
fn depth(_: c.Shape) usize {
    return 56;
}

const stress_why = "integer terms: every fp32 order is exact, a 16-bit accumulation of 2560 terms of up to 2000 is not";
const none_why = "no dot: the edit is one rounding of an exact sum of at most eight terms, or the final's positive squares";

pub const Case = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    kind: Kind,
    mode: ref.Mode = .project,
    /// The writer's ranks.
    world: usize = 1,
    /// The final's weights in bf16 (else fp32).
    nw16: bool = false,
    /// Random inputs are mirrored halves whose dot is rounding noise (an order change shows in the bits).
    cancel: bool = false,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    alpha: f32 = 1,
    /// The prepared input words: the fold's copies, the writer's partials, the final's streams.
    words: []u16 = &.{},
    d: []f32 = &.{},
    pre: []f32 = &.{},
    nw: []u8 = &.{},
    rowinfo: []ref.Row = &.{},
    o: []u16 = &.{},
    x: u64 = 0,
    dptr: u64 = 0,
    pre_ptr: u64 = 0,
    nw_ptr: u64 = 0,
    out: u64 = 0,
    msg: [192]u8 = undefined,

    pub fn spec(t: *const Case) c.Spec {
        const project = t.kind != .final and t.mode == .project;
        return .{
            .name = t.name,
            .accum = .f32,
            .depth = depth,
            .ops = 4,
            .mids = 2,
            .out = .bf16,
            .in_place = t.kind != .final,
            .served = if (t.kind == .final) &final_served else &edit_served,
            .sums = if (project) .{ .f16 = 10, .bf16 = 10, .why = stress_why } else .{ .f16 = 0, .bf16 = 0, .why = none_why },
        };
    }

    pub fn kernel(t: *Case) g.Kernel {
        return .{ .spec = t.spec(), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Case);

    pub fn deinit(t: *Case) void {
        t.free();
        t.rig.deinit();
    }

    fn free(t: *Case) void {
        const a = t.rig.gpu.gpa;
        a.free(t.words);
        a.free(t.d);
        a.free(t.pre);
        a.free(t.nw);
        a.free(t.rowinfo);
        a.free(t.o);
        t.words = &.{};
        t.d = &.{};
        t.pre = &.{};
        t.nw = &.{};
        t.rowinfo = &.{};
        t.o = &.{};
    }

    fn q(t: *const Case, r: usize) [4]f32 {
        return .{ t.pre[4 * r], t.pre[4 * r + 1], t.pre[4 * r + 2], t.pre[4 * r + 3] };
    }

    fn nwAt(t: *const Case, k: usize) f64 {
        return if (t.nw16) ref.widen(std.mem.readInt(u16, t.nw[2 * k ..][0..2], .little)) else @as(f32, @bitCast(std.mem.readInt(u32, t.nw[4 * k ..][0..4], .little)));
    }

    pub fn prepare(t: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const w = shape.n;
        const R = shape.rows;
        if (R == 0 or w == 0 or w % 8 != 0 or w > ref.max_width or (t.kind == .final and w % ref.chain != 0)) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        var rng = ref.Rng.init(seed);
        var kf: ref.Fill = switch (fill) {
            .random => if (t.cancel) .cancel else .random,
            .nonfinite => .nonfinite,
            .stress => .stress,
        };
        if (t.kind == .final and (kf == .stress or kf == .cancel)) kf = .random;
        t.alpha = if (t.mode == .add) 1.5 else switch (kf) {
            .stress => 1,
            .cancel => 64,
            else => 0.75,
        };
        t.d = try a.alloc(f32, w);
        ref.fillDir(kf, &rng, t.d);
        const tmp = try a.alloc(u16, 4 * w);
        defer a.free(tmp);
        switch (t.kind) {
            .stream, .final => {
                t.words = try a.alloc(u16, R * 4 * w);
                for (0..R) |r| ref.fillRow(kf, &rng, r, t.words[r * 4 * w ..][0 .. 4 * w], t.d);
            },
            .writer => {
                t.words = try a.alloc(u16, t.world * R * w);
                for (0..R) |r| {
                    ref.fillRow(kf, &rng, r, tmp, t.d);
                    for (0..t.world) |rk| {
                        const dst = t.words[rk * R * w + r * w ..][0..w];
                        if (kf == .stress) {
                            if (rk == 0) @memcpy(dst, tmp[0..w]) else @memset(dst, 0);
                        } else @memcpy(dst, tmp[(rk % 4) * w ..][0..w]);
                    }
                }
                t.o = try a.alloc(u16, R * w);
            },
        }
        t.rowinfo = try a.alloc(ref.Row, R);
        if (t.kind == .final) {
            t.pre = try a.alloc(f32, 4 * R);
            for (t.pre) |*v| v.* = 0.25 * (1 + @as(f32, @floatCast(rng.unit())));
            t.nw = try a.alloc(u8, w * if (t.nw16) @as(usize, 2) else 4);
            for (0..w) |k| {
                const v = ref.dirEntry(&rng);
                if (t.nw16) std.mem.writeInt(u16, t.nw[2 * k ..][0..2], ref.rne32(v), .little) else std.mem.writeInt(u32, t.nw[4 * k ..][0..4], @bitCast(v), .little);
            }
            t.x = try t.rig.input(std.mem.sliceAsBytes(t.words));
            t.pre_ptr = try t.rig.input(std.mem.sliceAsBytes(t.pre));
            t.nw_ptr = try t.rig.input(t.nw);
            t.out = try t.rig.output(R * w * 2);
        } else {
            t.dptr = try t.rig.input(std.mem.sliceAsBytes(t.d));
            t.x = try t.rig.output(t.words.len * 2);
            try t.upload();
        }
        return true;
    }

    fn upload(t: *Case) !void {
        const b: cuda.DeviceBuffer = .{ .d = t.rig.gpu.d, .ptr = t.x, .len = t.words.len * 2 };
        try b.upload(0, std.mem.sliceAsBytes(t.words));
    }

    pub fn run(t: *Case) !void {
        if (t.kind != .final) try t.upload(); // in place: every run starts from the prepared bytes
        const w: c_int = @intCast(t.shape.n);
        const R: c_int = @intCast(t.shape.rows);
        const mode: c_int = @intFromEnum(t.mode);
        var args: cuda.Args = .{};
        switch (t.kind) {
            .stream => args.add(StreamArgs{ .x = t.x, .xs = 4 * w, .dir = t.dptr, .alpha = t.alpha, .mode = mode, .R = R, .width = w }),
            .writer => args.add(WriterArgs{ .g = t.x, .gr = @as(i64, R) * w, .gs = w, .dir = t.dptr, .alpha = t.alpha, .mode = mode, .R = R, .width = w, .world = @intCast(t.world) }),
            .final => args.add(FinalArgs{ .x = t.x, .xs = 4 * w, .pre = t.pre_ptr, .nw = t.nw_ptr, .out = t.out, .eps = eps, .nwbf16 = @intFromBool(t.nw16), .R = R, .width = w }),
        }
        const shared: u32 = if (t.kind == .final) @intCast(2 * t.shape.n) else 0;
        const block: u32 = @intCast(ref.nt);
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast(t.shape.rows) }, .block = .{ .x = block }, .shared = shared }, t.rig.s, &args);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Case) usize {
        return switch (t.kind) {
            .stream => t.shape.rows * 4 * t.shape.n,
            .writer => t.world * t.shape.rows * t.shape.n,
            .final => t.shape.rows * t.shape.n,
        };
    }

    /// FP64 outputs of the prepared inputs (the fold's and the writer's unrounded edit, the writer's other slots -0.0).
    pub fn reference(t: *Case, want: []f64, mag: []f64) !void {
        const w = t.shape.n;
        const R = t.shape.rows;
        @memset(mag, 0);
        for (0..R) |r| switch (t.kind) {
            .stream => {
                const row = ref.foldRef(t.words[r * 4 * w ..][0 .. 4 * w], t.d, t.alpha, t.mode, want[r * 4 * w ..][0 .. 4 * w]);
                t.rowinfo[r] = row;
                @memset(mag[r * 4 * w ..][0 .. 4 * w], row.mag);
            },
            .writer => {
                var ranks: [ref.max_world][]const u16 = undefined;
                for (0..t.world) |rk| ranks[rk] = t.words[rk * R * w + r * w ..][0..w];
                const row = ref.writerRef(t.world, ranks[0..t.world], t.d, t.alpha, t.mode, t.o[r * w ..][0..w], want[r * w ..][0..w]);
                t.rowinfo[r] = row;
                @memset(mag[r * w ..][0..w], row.mag);
                for (1..t.world) |rk| @memset(want[rk * R * w + r * w ..][0..w], -0.0);
            },
            .final => ref.finalRef(t.words[r * 4 * w ..][0 .. 4 * w], t.q(r), t.nw, t.nw16, eps, want[r * w ..][0..w]),
        };
    }

    /// The stated agreement with FP64 (glp_ref.allow); the writer adds one bf16 ulp of its partial sums' rounding; the
    /// final's is two bf16 ulps.
    pub fn bounds(t: *Case, want: []const f64, _: []const f64, allow: []f64) !void {
        const w = t.shape.n;
        const R = t.shape.rows;
        for (allow, want, 0..) |*l, v, i| {
            var b: f64 = 1;
            switch (t.kind) {
                .stream => b = ref.allow(v, ref.widen(t.words[i]), t.d[i % w], t.alpha, t.mode, t.rowinfo[i / (4 * w)], 0),
                .writer => {
                    const rem = i % (R * w);
                    if (i / (R * w) == 0) {
                        const o = ref.widen(t.o[rem]);
                        const flip: f64 = if (t.world > 1) 0x1p-7 * @abs(o) else 0;
                        b = ref.allow(v, o, t.d[rem % w], t.alpha, t.mode, t.rowinfo[rem / w], flip) + flip;
                    } else b = 0x1p-133;
                },
                .final => b = 0x1p-6 * @abs(v) + 0x1p-133,
            }
            l.* = if (std.math.isFinite(b)) b else 1;
        }
    }

    /// The outputs if the dot's (the final's squares') sum ran in an accumulator, in an order; each element's own edit in
    /// FP64 and rounded once to bf16.
    pub fn simulate(t: *Case, acc: c.Accum, order: rows_.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const w = t.shape.n;
        const R = t.shape.rows;
        const m = try a.alloc(f64, 3 * w);
        defer a.free(m);
        const terms = m[w .. 2 * w];
        const base = m[2 * w ..];
        for (0..R) |r| switch (t.kind) {
            .stream, .writer => {
                for (0..w) |k| {
                    if (t.kind == .stream) {
                        const x = t.words[r * 4 * w ..];
                        m[k] = 0.25 * (@as(f64, ref.widen(x[k])) + ref.widen(x[w + k]) + ref.widen(x[2 * w + k]) + ref.widen(x[3 * w + k]));
                    } else {
                        var s: f64 = 0;
                        for (0..t.world) |rk| s += ref.widen(t.words[rk * R * w + r * w + k]);
                        m[k] = ref.widen(ref.rne64(s));
                    }
                    terms[k] = m[k] * t.d[k];
                }
                const s = if (t.mode == .project) @as(f64, t.alpha) * rows_.sumIn(acc, order, terms) else 0;
                const copies: usize = if (t.kind == .stream) 4 else 1;
                for (0..copies) |cc| for (0..w) |k| {
                    const x: f64 = if (t.kind == .stream) ref.widen(t.words[r * 4 * w + cc * w + k]) else m[k];
                    const y = if (t.mode == .project) x - s * t.d[k] else x + @as(f64, t.alpha) * t.d[k];
                    const at = if (t.kind == .stream) r * 4 * w + cc * w + k else r * w + k;
                    out[at] = ref.widen(ref.rne64(y));
                };
                if (t.kind == .writer) for (1..t.world) |rk| @memset(out[rk * R * w + r * w ..][0..w], -0.0);
            },
            .final => {
                const x = t.words[r * 4 * w ..];
                const qr = t.q(r);
                for (0..w) |k| {
                    base[k] = ref.widen(ref.rne64(@as(f64, qr[0]) * ref.widen(x[k]) + @as(f64, qr[1]) * ref.widen(x[w + k]) + @as(f64, qr[2]) * ref.widen(x[2 * w + k]) + @as(f64, qr[3]) * ref.widen(x[3 * w + k])));
                    terms[k] = base[k] * base[k];
                }
                const rn = 1.0 / @sqrt(rows_.sumIn(acc, order, terms) / @as(f64, @floatFromInt(w)) + eps);
                for (0..w) |k| out[r * w + k] = ref.widen(ref.rne64(base[k] * rn * t.nwAt(k)));
            },
        };
    }

    /// The independent fp32 model's bits for the whole output (glp_ref.zig; the writer's other slots read -0.0).
    fn model(t: *Case, exp: []u16) void {
        const w = t.shape.n;
        const R = t.shape.rows;
        for (0..R) |r| switch (t.kind) {
            .stream => ref.foldRow(t.words[r * 4 * w ..][0 .. 4 * w], t.d, t.alpha, t.mode, .{}, exp[r * 4 * w ..][0 .. 4 * w], null),
            .writer => {
                var ranks: [ref.max_world][]const u16 = undefined;
                for (0..t.world) |rk| ranks[rk] = t.words[rk * R * w + r * w ..][0..w];
                ref.writerRow(t.world, ranks[0..t.world], t.d, t.alpha, t.mode, .{}, exp[r * w ..][0..w]);
                for (1..t.world) |rk| @memset(exp[rk * R * w + r * w ..][0..w], 0x8000);
            },
            .final => ref.finalRow(t.words[r * 4 * w ..][0 .. 4 * w], t.q(r), t.nw, t.nw16, eps, exp[r * w ..][0..w]),
        };
    }

    /// Both tests at once: equal to the model's bits, and within the FP64 bound (classes equal where FP64 is not finite).
    pub fn check(t: *Case, at: g.Gate) !?[]const u8 {
        const a = t.rig.gpu.gpa;
        const n = t.count();
        const out = try t.rig.outputBytes();
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        const exp = try a.alloc(u16, n);
        defer a.free(exp);
        const want = buf[0..n];
        const allow = buf[2 * n ..];
        try t.reference(want, buf[n .. 2 * n]);
        try t.bounds(want, buf[n .. 2 * n], allow);
        t.model(exp);
        var planted: usize = 0;
        var off_model: usize = 0;
        var off_fp64: usize = 0;
        var first: ?usize = null;
        for (want, allow, exp, 0..) |wv, l, e, i| {
            const got = std.mem.readInt(u16, out[2 * i ..][0..2], .little);
            if (!std.math.isFinite(wv)) planted += 1;
            var bad = !ref.same(got, e);
            if (bad) off_model += 1;
            const gv: f64 = ref.widen(got);
            const ok = if (std.math.isFinite(wv)) (gv == wv or @abs(gv - wv) <= l) else c.classOf(gv) == c.classOf(wv);
            if (!ok) {
                off_fp64 += 1;
                bad = true;
            }
            if (bad and first == null) first = i;
        }
        if (at == .nonfinite and (planted == 0 or planted == n))
            return std.fmt.bufPrint(&t.msg, "the planted values make {d} of {d} reference outputs non-finite: no proof", .{ planted, n }) catch "no proof";
        if (off_model + off_fp64 == 0) return null;
        return std.fmt.bufPrint(&t.msg, "{d} of {d} words differ from the fp32 order model and {d} leave the FP64 bound, first at {?d}", .{ off_model, n, off_fp64, first }) catch "outputs differ";
    }
};

const cases_total = 3 + 2 + 2 + 6 + 5;

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    prod: cuda.Module,
    ctl: cuda.Module,
    cases: [cases_total]Case = undefined,
    made: usize = 0,

    fn add(set: *Set, case: Case) *Case {
        set.cases[set.made] = case;
        set.made += 1;
        return &set.cases[set.made - 1];
    }

    /// The set, its kernels appended to `list` (each control after the kernels it breaks).
    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var prod = try cuda.Module.load(gpu.d, images.image("dsv41_glp") orelse return error.UnknownContractImage);
        errdefer prod.unload();
        var ctl = try cuda.Module.load(gpu.d, images.image("dsv41_glp_controls") orelse return error.UnknownContractImage);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .prod = prod, .ctl = ctl };
        errdefer set.release();
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const stream = try prod.function("dsv41_glp_stream");
        const writer = try prod.function("dsv41_glp_writer");
        const final = try prod.function("dsv41_glp_final");
        // production: the fold (add; project; project on mirrored halves), the writer (add; project, three ranks), the final
        const fold_add = set.add(.{ .rig = rig, .f = stream, .kind = .stream, .mode = .add, .name = "cuda dsv41 glp fold add" });
        const fold_project = set.add(.{ .rig = rig, .f = stream, .kind = .stream, .name = "cuda dsv41 glp fold project" });
        const fold_cancel = set.add(.{ .rig = rig, .f = stream, .kind = .stream, .cancel = true, .name = "cuda dsv41 glp fold project, mirrored halves" });
        const writer_add = set.add(.{ .rig = rig, .f = writer, .kind = .writer, .mode = .add, .world = 3, .name = "cuda dsv41 glp writer add, 3 ranks" });
        const writer_project = set.add(.{ .rig = rig, .f = writer, .kind = .writer, .world = 3, .name = "cuda dsv41 glp writer project, 3 ranks" });
        const final_f32 = set.add(.{ .rig = rig, .f = final, .kind = .final, .name = "cuda dsv41 glp final, fp32 weights" });
        const final_bf16 = set.add(.{ .rig = rig, .f = final, .kind = .final, .nw16 = true, .name = "cuda dsv41 glp final, bf16 weights" });
        for ([_]*Case{ fold_add, fold_project, fold_cancel, writer_add, writer_project, final_f32, final_bf16 }) |p| try list.append(a, p.kernel());
        // each control: the fold, then the writer (the sum control breaks only the fold's mean)
        inline for (std.enums.values(Ctl)) |v| {
            const edit_only = v == .truncate; // the rounding fault is in the edit's store; the others live in the dot
            const mirrored = v == .atomic;
            const sf = try ctl.function(std.fmt.comptimePrint("dsv41_glp_stream_{s}", .{@tagName(v)}));
            const sc = set.add(.{ .rig = rig, .f = sf, .kind = .stream, .mode = if (edit_only) .add else .project, .cancel = mirrored, .name = "control: dsv41 glp fold " ++ @tagName(v), .control = v.gate() });
            try list.append(a, sc.kernel());
            if (v != .sum) {
                const wf = try ctl.function(std.fmt.comptimePrint("dsv41_glp_writer_{s}", .{@tagName(v)}));
                const wc = set.add(.{ .rig = rig, .f = wf, .kind = .writer, .mode = if (edit_only) .add else .project, .world = 2, .cancel = mirrored, .name = "control: dsv41 glp writer " ++ @tagName(v), .control = v.gate() });
                try list.append(a, wc.kernel());
            }
        }
        std.debug.assert(set.made == cases_total);
        return set;
    }

    fn release(set: *Set) void {
        for (set.cases[0..set.made]) |*x| x.deinit();
    }

    pub fn deinit(set: *Set) void {
        set.release();
        set.ctl.unload();
        set.prod.unload();
        set.gpa.destroy(set);
    }
};

/// The GLP kernels' gates 1-4 and their controls alone.
pub fn run(gpu: check_.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x676c70, &t);
    std.debug.print("SUMMARY dsv41 glp: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check_.expect(t.problems() == 0, "dsv41-glp contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
