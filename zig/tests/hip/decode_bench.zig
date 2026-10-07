//! The decode-step kernels of decode.hip and the merged affine launches against the launches they replace
//! (TF_DECODE_FUSE=old): the router's logits, the residual tails, products that share x in one launch and the routed
//! activation as an epilogue, each against a float64 reference on random data, with microseconds a launch. Weights
//! rotate over many copies where it matters so each launch reads them from DRAM.
//! `tf-hip-test decode [reps] [router|tail|group|pair|chain]`.

const std = @import("std");
const hip = @import("hip");
const check = @import("check.zig");
const ref = @import("gemm_ref.zig");
const chain = @import("decode_chain.zig");
const Gpu = check.Gpu;
pub const L = @typeInfo(@FieldType(hip.rocm.Library, "zig")).optional.child;
const P = ?*anyopaque;

pub const Rng = struct {
    state: u64,

    pub fn next(r: *Rng) u64 {
        r.state ^= r.state >> 12;
        r.state ^= r.state << 25;
        r.state ^= r.state >> 27;
        return r.state *% 0x2545F4914F6CDD1D;
    }

    pub fn unit(r: *Rng) f32 {
        const v: i32 = @intCast(r.next() >> 40 & 0x7ff);
        return @as(f32, @floatFromInt(v - 1024)) / 1024.0;
    }
};

fn f16Bits(v: f32) u16 {
    const h: f16 = @floatCast(v);
    return @bitCast(h);
}

fn f16Value(b: u16) f64 {
    const h: f16 = @bitCast(b);
    return h;
}

fn bf16Bits(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16);
}

fn bf16Value(b: u16) f64 {
    return @as(f32, @bitCast(@as(u32, b) << 16));
}

pub const Rig = struct {
    gpu: Gpu,
    fast: L,
    old: L,
    stream: hip.Stream,
    fp16: bool,
    rng: Rng,
    reps: usize,
    start: hip.Event,
    stop: hip.Event,

    pub fn bits(t: *const Rig, v: f32) u16 {
        return if (t.fp16) f16Bits(v) else bf16Bits(v);
    }

    pub fn value(t: *const Rig, b: u16) f64 {
        return if (t.fp16) f16Value(b) else bf16Value(b);
    }

    /// Microseconds a call of `go`, best of rounds after one warm-up; `go` takes the rig and the iteration.
    pub fn time(t: *Rig, iters: usize, ctx: anytype, comptime go: fn (@TypeOf(ctx), usize) hip.Error!void) !f64 {
        var best: f64 = std.math.inf(f64);
        for (0..3) |round| {
            try t.start.record(t.stream);
            for (0..iters) |i| try go(ctx, i);
            try t.stop.record(t.stream);
            try t.stop.synchronize();
            const us = @as(f64, try hip.Event.elapsedMs(t.start, t.stop)) * 1000.0 / @as(f64, @floatFromInt(iters));
            if (round > 0) best = @min(best, us);
        }
        return best;
    }
};

/// The router's logits for `rows` rows over (experts, d) fp32 weights.
fn router(t: *Rig, rows: usize, d: usize, experts: usize) !void {
    const gpa = t.gpu.gpa;
    const copies = 96;
    const hx = try gpa.alloc(u16, rows * d);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(t.rng.unit());
    const hw = try gpa.alloc(f32, experts * d);
    defer gpa.free(hw);
    for (hw) |*v| v.* = t.rng.unit() / 8.0;
    var x = try hip.DeviceBuffer.fromHost(t.gpu.d, std.mem.sliceAsBytes(hx));
    defer x.free();
    var w = try hip.DeviceBuffer.alloc(t.gpu.d, copies * hw.len * 4);
    defer w.free();
    for (0..copies) |i| try w.upload(i * hw.len * 4, std.mem.sliceAsBytes(hw));
    var out: [2]hip.DeviceBuffer = undefined;
    for (&out) |*o| o.* = try hip.DeviceBuffer.alloc(t.gpu.d, rows * experts * 4);
    defer for (&out) |*o| o.free();
    const Ctx = struct {
        t: *Rig,
        l: *const L,
        x: hip.DeviceBuffer,
        w: hip.DeviceBuffer,
        out: hip.DeviceBuffer,
        rows: usize,
        d: usize,
        e: usize,
        fn go(c: @This(), i: usize) hip.Error!void {
            const kind: c_int = if (c.t.fp16) 1 else 2;
            try c.l.tf_moe_router(@ptrFromInt(c.x.ptr), kind, @ptrFromInt(c.w.ptr + (i % copies) * c.d * c.e * 4), @ptrFromInt(c.out.ptr), @intCast(c.rows), @intCast(c.d), @intCast(c.e), c.t.stream.handle);
        }
    };
    var us: [2]f64 = undefined;
    var max_err: [2]f64 = .{ 0, 0 };
    for ([2]*const L{ &t.old, &t.fast }, 0..) |l, v| {
        const ctx: Ctx = .{ .t = t, .l = l, .x = x, .w = w, .out = out[v], .rows = rows, .d = d, .e = experts };
        us[v] = try t.time(copies * t.reps, ctx, Ctx.go);
        const got = try gpa.alloc(f32, rows * experts);
        defer gpa.free(got);
        try out[v].download(0, std.mem.sliceAsBytes(got));
        for (0..rows) |r| for (0..experts) |e| {
            var y: f64 = 0;
            var norm: f64 = 0;
            for (0..d) |i| {
                const term = t.value(hx[r * d + i]) * @as(f64, hw[e * d + i]);
                y += term;
                norm += @abs(term);
            }
            max_err[v] = @max(max_err[v], @abs(@as(f64, got[r * experts + e]) - y) / norm);
        };
    }
    std.debug.print("RESULT decode router r{d} d{d} e{d}: old {d:.1} us, new {d:.1} us, x{d:.2}; max|y-ref|/sum|terms| old {e:.1} new {e:.1}\n", .{ rows, d, experts, us[0], us[1], us[0] / us[1], max_err[0], max_err[1] });
    try check.expect(max_err[1] <= 2 * max_err[0] + 1e-9, "decode router r{d}: further from the float64 reference than twice the previous kernel", .{rows});
}

/// The residual tails: x = x + y (or the slots' weighted sum), then the next norm. Old is the combine, add and rms
/// launches, new is one. Errors of x and of the normed rows against a float64 reference, relative to the row's largest.
fn tails(t: *Rig, rows: usize, width: usize, slots: usize) !void {
    const gpa = t.gpu.gpa;
    const kind: c_int = if (t.fp16) 1 else 2;
    const n = rows * width;
    const eps: f32 = 1e-6;
    const hx = try gpa.alloc(u16, n);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(t.rng.unit());
    const hy16 = try gpa.alloc(u16, if (slots > 0) 0 else n);
    defer gpa.free(hy16);
    const hy32 = try gpa.alloc(f32, if (slots > 0) rows * slots * width else 0);
    defer gpa.free(hy32);
    for (hy16) |*v| v.* = t.bits(t.rng.unit() / 8.0);
    for (hy32) |*v| v.* = t.rng.unit() / 8.0;
    const hw = try gpa.alloc(f32, rows * @max(slots, 1));
    defer gpa.free(hw);
    for (hw) |*v| v.* = (t.rng.unit() + 1.0) / @as(f32, @floatFromInt(@max(slots, 1)));
    const hn = try gpa.alloc(f32, width);
    defer gpa.free(hn);
    for (hn) |*v| v.* = 1.0 + t.rng.unit() / 4.0;
    var y = try hip.DeviceBuffer.fromHost(t.gpu.d, if (slots > 0) std.mem.sliceAsBytes(hy32) else std.mem.sliceAsBytes(hy16));
    defer y.free();
    var wts = try hip.DeviceBuffer.fromHost(t.gpu.d, std.mem.sliceAsBytes(hw));
    defer wts.free();
    var weight = try hip.DeviceBuffer.fromHost(t.gpu.d, std.mem.sliceAsBytes(hn));
    defer weight.free();
    var tmp = try hip.DeviceBuffer.alloc(t.gpu.d, n * 2);
    defer tmp.free();
    var xs: [2]hip.DeviceBuffer = undefined;
    var normed: [2]hip.DeviceBuffer = undefined;
    for (&xs) |*b| b.* = try hip.DeviceBuffer.fromHost(t.gpu.d, std.mem.sliceAsBytes(hx));
    for (&normed) |*b| b.* = try hip.DeviceBuffer.alloc(t.gpu.d, n * 2);
    defer for (&xs) |*b| b.free();
    defer for (&normed) |*b| b.free();
    const Ctx = struct {
        t: *Rig,
        old: bool,
        x: hip.DeviceBuffer,
        y: hip.DeviceBuffer,
        wts: hip.DeviceBuffer,
        weight: hip.DeviceBuffer,
        tmp: hip.DeviceBuffer,
        normed: hip.DeviceBuffer,
        rows: usize,
        width: usize,
        slots: usize,
        kind: c_int,
        eps: f32,
        fn go(c: @This(), _: usize) hip.Error!void {
            const s = c.t.stream.handle;
            const n_all: i64 = @intCast(c.rows * c.width);
            if (c.old) {
                const sum: P = if (c.slots > 0) @ptrFromInt(c.tmp.ptr) else @ptrFromInt(c.y.ptr);
                const l = &c.t.old;
                if (c.slots > 0) try l.tf_moe_combine(@ptrFromInt(c.y.ptr), @ptrFromInt(c.wts.ptr), sum, c.kind, @intCast(c.rows), @intCast(c.slots), @intCast(c.width), s);
                try l.tf_add(@ptrFromInt(c.x.ptr), sum, @ptrFromInt(c.x.ptr), c.kind, n_all, s);
                try l.tf_rms(@ptrFromInt(c.x.ptr), @ptrFromInt(c.weight.ptr), @ptrFromInt(c.normed.ptr), c.kind, @intCast(c.rows), @intCast(c.width), c.eps, s);
            } else {
                try c.t.fast.tf_tail(@ptrFromInt(c.x.ptr), @ptrFromInt(c.y.ptr), if (c.slots > 0) @ptrFromInt(c.wts.ptr) else null, @ptrFromInt(c.weight.ptr), @ptrFromInt(c.normed.ptr), c.kind, @intCast(c.rows), @intCast(c.slots), @intCast(c.width), c.eps, s);
            }
        }
    };
    var us: [2]f64 = undefined;
    var errs: [2][2]f64 = .{ .{ 0, 0 }, .{ 0, 0 } };
    for (0..2) |v| {
        const ctx: Ctx = .{ .t = t, .old = v == 0, .x = xs[v], .y = y, .wts = wts, .weight = weight, .tmp = tmp, .normed = normed[v], .rows = rows, .width = width, .slots = slots, .kind = kind, .eps = eps };
        // one launch from the initial x is what the error is measured on; the timing loop then runs on the same buffers
        try Ctx.go(ctx, 0);
        try t.stream.synchronize();
        const gx = try gpa.alloc(u16, n);
        defer gpa.free(gx);
        const gn = try gpa.alloc(u16, n);
        defer gpa.free(gn);
        try xs[v].download(0, std.mem.sliceAsBytes(gx));
        try normed[v].download(0, std.mem.sliceAsBytes(gn));
        const want = try gpa.alloc(f64, width);
        defer gpa.free(want);
        for (0..rows) |r| {
            var sq: f64 = 0;
            var top: f64 = 0;
            for (0..width) |c| {
                var add: f64 = 0;
                if (slots > 0) {
                    for (0..slots) |sl| add += @as(f64, hy32[(r * slots + sl) * width + c]) * @as(f64, hw[r * slots + sl]);
                } else add = t.value(hy16[r * width + c]);
                want[c] = t.value(hx[r * width + c]) + add;
                sq += want[c] * want[c];
                top = @max(top, @abs(want[c]));
            }
            const inv = 1.0 / @sqrt(sq / @as(f64, @floatFromInt(width)) + eps);
            for (0..width) |c| {
                errs[v][0] = @max(errs[v][0], @abs(t.value(gx[r * width + c]) - want[c]) / top);
                errs[v][1] = @max(errs[v][1], @abs(t.value(gn[r * width + c]) - want[c] * inv * hn[c]) / (top * inv * 1.25));
            }
        }
        us[v] = try t.time(20 * t.reps, ctx, Ctx.go);
    }
    std.debug.print("RESULT decode tail rows{d} width{d} slots{d}: old {d:.1} us, new {d:.1} us, x{d:.2}; max|got-ref|/max|ref| of x old {e:.1} new {e:.1}, of the norm old {e:.1} new {e:.1}\n", .{ rows, width, slots, us[0], us[1], us[0] / us[1], errs[0][0], errs[1][0], errs[0][1], errs[1][1] });
    try check.expect(errs[1][0] <= 2 * errs[0][0] + 1e-7 and errs[1][1] <= 2 * errs[0][1] + 1e-7, "decode tail rows{d} slots{d}: further from the float64 reference than twice the previous launches", .{ rows, slots });
}

/// A random packed product's data on the host and the device: (n, k) at `bits` and `group`, bf16 tables, `experts` stacked.
const Matrix = struct {
    n: usize,
    k: usize,
    bits: usize,
    group: usize,
    words: []u32,
    scale: []u16,
    bias: []u16,
    dev: [3]hip.DeviceBuffer,

    fn init(t: *Rig, n: usize, k: usize, bits: usize, group_size: usize, experts: usize) !Matrix {
        const gpa = t.gpu.gpa;
        var m: Matrix = .{ .n = n, .k = k, .bits = bits, .group = group_size, .words = undefined, .scale = undefined, .bias = undefined, .dev = undefined };
        m.words = try gpa.alloc(u32, experts * n * k * bits / 32);
        m.scale = try gpa.alloc(u16, experts * n * k / group_size);
        m.bias = try gpa.alloc(u16, m.scale.len);
        for (m.words) |*w| w.* = @truncate(t.rng.next() >> 16);
        for (m.scale) |*v| v.* = bf16Bits((t.rng.unit() + 1.0) / 16.0);
        for (m.bias) |*v| v.* = bf16Bits(t.rng.unit() / 2.0);
        m.dev = .{
            try hip.DeviceBuffer.fromHost(t.gpu.d, std.mem.sliceAsBytes(m.words)),
            try hip.DeviceBuffer.fromHost(t.gpu.d, std.mem.sliceAsBytes(m.scale)),
            try hip.DeviceBuffer.fromHost(t.gpu.d, std.mem.sliceAsBytes(m.bias)),
        };
        return m;
    }

    fn deinit(m: *Matrix, gpa: std.mem.Allocator) void {
        gpa.free(m.words);
        gpa.free(m.scale);
        gpa.free(m.bias);
        for (&m.dev) |*b| b.free();
    }

    fn problem(m: Matrix, t: *const Rig, x: []const u16) ref.Problem {
        return .{ .fp16 = t.fp16, .n = m.n, .k = m.k, .bits = m.bits, .group = m.group, .x = x, .words = m.words, .scale = m.scale, .bias = m.bias };
    }

    fn arg(m: Matrix, t: *const Rig, x: u64, rows: usize, out: u64) hip.affine.Arg {
        return .{
            .x = x,
            .words = m.dev[0].ptr,
            .scale = .{ .p = m.dev[1].ptr, .kind = 1 },
            .bias = .{ .p = m.dev[2].ptr, .kind = 1 },
            .out = out,
            .m = @intCast(rows),
            .n = @intCast(m.n),
            .k = @intCast(m.k),
            .bits = @intCast(m.bits),
            .group = @intCast(m.group),
            .fp16 = @intFromBool(t.fp16),
        };
    }
};

/// Products that share x in one launch against one launch each: each side's rows rounded to the activation type, errors
/// against a float64 dequant(W) . x on sampled outputs.
fn group(t: *Rig, rows: usize, ns: []const usize, k: usize, bits: usize, group_size: usize) !void {
    const gpa = t.gpu.gpa;
    const hx = try gpa.alloc(u16, rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(t.rng.unit());
    var x = try hip.DeviceBuffer.fromHost(t.gpu.d, std.mem.sliceAsBytes(hx));
    defer x.free();
    var ms: [4]Matrix = undefined;
    var outs: [2][4]hip.DeviceBuffer = undefined;
    for (ns, 0..) |n, i| {
        ms[i] = try Matrix.init(t, n, k, bits, group_size, 1);
        for (&outs) |*o| o[i] = try hip.DeviceBuffer.alloc(t.gpu.d, rows * n * 2);
    }
    defer for (ns, 0..) |_, i| {
        ms[i].deinit(gpa);
        for (&outs) |*o| o[i].free();
    };
    const Ctx = struct {
        t: *Rig,
        one: bool,
        x: u64,
        ms: []const Matrix,
        outs: *const [4]hip.DeviceBuffer,
        rows: usize,
        fn go(c: @This(), _: usize) hip.Error!void {
            const kernels = &c.t.fast.affine;
            if (!c.one) {
                for (c.ms, 0..) |m, i| try kernels.run(c.t.gpu.d, m.arg(c.t, c.x, c.rows, c.outs[i].ptr), 0, c.t.stream.handle, 0, 1, true);
            } else {
                var sides: [4]hip.affine.Side = undefined;
                for (c.ms, 0..) |m, i| sides[i] = .{ .words = m.dev[0].ptr, .scale = m.dev[1].ptr, .bias = m.dev[2].ptr, .n = @intCast(m.n), .out = c.outs[i].ptr };
                try kernels.groupRun(c.t.gpu.d, c.ms[0].arg(c.t, c.x, c.rows, 0), sides[0..c.ms.len], true, c.t.stream.handle);
            }
        }
    };
    var us: [2]f64 = undefined;
    var errs: [2]f64 = .{ 0, 0 };
    for (0..2) |v| {
        const ctx: Ctx = .{ .t = t, .one = v == 1, .x = x.ptr, .ms = ms[0..ns.len], .outs = &outs[v], .rows = rows };
        try Ctx.go(ctx, 0);
        try t.stream.synchronize();
        for (ns, 0..) |n, i| {
            const got = try gpa.alloc(u16, rows * n);
            defer gpa.free(got);
            try outs[v][i].download(0, std.mem.sliceAsBytes(got));
            const p = ms[i].problem(t, hx);
            for (0..rows) |r| for (0..@min(n, 12)) |ci| {
                const col = ci * (n - 1) / @max(@min(n, 12) - 1, 1);
                const want = ref.reference(p, r, 0, col);
                errs[v] = @max(errs[v], @abs(t.value(got[r * n + col]) - want.y) / want.norm);
            };
        }
        us[v] = try t.time(100 * t.reps, ctx, Ctx.go);
    }
    std.debug.print("RESULT decode group rows{d} sides{d} n{d} k{d} b{d} g{d}: one launch each {d:.1} us, one launch {d:.1} us, x{d:.2}; max|y-ref|/sum|terms| each {e:.1} one {e:.1}\n", .{ rows, ns.len, ns[0], k, bits, group_size, us[0], us[1], us[0] / us[1], errs[0], errs[1] });
    try check.expect(errs[1] <= 2 * errs[0] + 1e-9, "decode group rows{d}: further from the float64 reference than twice one launch each", .{rows});
}

/// The routed gate and up of one token's slots (9 experts of 64): old is the stacked product to fp32 and the activation
/// launch, new is one launch with the activation as its epilogue. Errors of the activation against a float64 reference.
fn pair(t: *Rig, rows: usize, width: usize, k: usize, bits: usize, group_size: usize) !void {
    const gpa = t.gpu.gpa;
    const experts = 64;
    const slots = 9;
    const pairs = rows * slots;
    const m = try Matrix.init(t, 2 * width, k, bits, group_size, experts);
    var mm = m;
    defer mm.deinit(gpa);
    const hx = try gpa.alloc(u16, rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(t.rng.unit());
    var x = try hip.DeviceBuffer.fromHost(t.gpu.d, std.mem.sliceAsBytes(hx));
    defer x.free();
    // pairs of a token are its slots; sorted by expert in items of one pair each
    const pick = try gpa.alloc(u32, pairs);
    defer gpa.free(pick);
    for (pick) |*e| e.* = @intCast(t.rng.next() % experts);
    const items = try gpa.alloc(i32, pairs * 3);
    defer gpa.free(items);
    const members = try gpa.alloc(i32, pairs);
    defer gpa.free(members);
    for (0..pairs) |i| {
        items[3 * i ..][0..3].* = .{ @intCast(pick[i]), @intCast(i), 1 };
        members[i] = @intCast(i);
    }
    var dev_items = try hip.DeviceBuffer.fromHost(t.gpu.d, std.mem.sliceAsBytes(items));
    defer dev_items.free();
    var dev_members = try hip.DeviceBuffer.fromHost(t.gpu.d, std.mem.sliceAsBytes(members));
    defer dev_members.free();
    var both = try hip.DeviceBuffer.alloc(t.gpu.d, pairs * 2 * width * 4);
    defer both.free();
    var acts: [2]hip.DeviceBuffer = undefined;
    for (&acts) |*a| a.* = try hip.DeviceBuffer.alloc(t.gpu.d, pairs * width * 2);
    defer for (&acts) |*a| a.free();
    const Ctx = struct {
        t: *Rig,
        one: bool,
        m: Matrix,
        x: u64,
        items: u64,
        members: u64,
        both: u64,
        act: u64,
        rows: usize,
        pairs: usize,
        fn go(c: @This(), _: usize) hip.Error!void {
            const kernels = &c.t.fast.affine;
            var a = c.m.arg(c.t, c.x, c.rows, c.both);
            a.route = .{ .items = c.items, .members = c.members, .x_div = 9 };
            if (c.one) {
                a.out = 0;
                a.out16 = c.act;
                if (!try kernels.pairRun(c.t.gpu.d, a, 0, @intCast(c.pairs), c.t.stream.handle)) return error.Invalid;
            } else {
                try kernels.routedWith(c.t.gpu.d, a, @intCast(c.pairs), c.t.stream.handle, .gemm);
                try c.t.old.tf_moe_act(@ptrFromInt(c.both), @ptrFromInt(c.act), if (c.t.fp16) 1 else 2, @intCast(c.pairs), @intCast(c.m.n / 2), 0, c.t.stream.handle);
            }
        }
    };
    var us: [2]f64 = undefined;
    var errs: [2]f64 = .{ 0, 0 };
    for (0..2) |v| {
        const ctx: Ctx = .{ .t = t, .one = v == 1, .m = mm, .x = x.ptr, .items = dev_items.ptr, .members = dev_members.ptr, .both = both.ptr, .act = acts[v].ptr, .rows = rows, .pairs = pairs };
        try Ctx.go(ctx, 0);
        try t.stream.synchronize();
        const got = try gpa.alloc(u16, pairs * width);
        defer gpa.free(got);
        try acts[v].download(0, std.mem.sliceAsBytes(got));
        const p = mm.problem(t, hx);
        for (0..pairs) |pi| for (0..12) |ci| {
            const col = ci * (width - 1) / 11;
            const g = ref.reference(p, pi / slots, pick[pi], col);
            const u = ref.reference(p, pi / slots, pick[pi], col + width);
            const want = g.y / (1.0 + @exp(-g.y)) * u.y;
            errs[v] = @max(errs[v], @abs(t.value(got[pi * width + col]) - want) / @max(@abs(want), 0.01));
        };
        us[v] = try t.time(100 * t.reps, ctx, Ctx.go);
    }
    std.debug.print("RESULT decode pair rows{d} width{d} k{d} b{d} g{d}: gate_up and act {d:.1} us, one launch {d:.1} us, x{d:.2}; max|act-ref|/max(|ref|, 0.01) old {e:.1} new {e:.1}\n", .{ rows, width, k, bits, group_size, us[0], us[1], us[0] / us[1], errs[0], errs[1] });
    try check.expect(errs[1] <= 2 * errs[0] + 1e-9, "decode pair rows{d}: further from the float64 reference than twice the previous launches", .{rows});
}

pub fn run(gpu: Gpu, args: []const [:0]const u8) !void {
    const reps: usize = if (args.len > 0) try std.fmt.parseInt(usize, args[0], 10) else 3;
    const filter: []const u8 = if (args.len > 1) args[1] else "";
    const family = hip.rocm.familyOf(try gpu.ctx.capability()) orelse return error.UnsupportedGpu;
    var lib = try hip.rocm.Library.open(gpu.d, family, try check.policyOf(gpu));
    defer lib.close();
    const launcher = lib.zig orelse return error.LibraryUnavailable;
    var t: Rig = .{
        .gpu = gpu,
        .fast = launcher,
        .old = launcher,
        .stream = try hip.Stream.init(gpu.d, true),
        .fp16 = family == .rdna2,
        .rng = .{ .state = 0x9E3779B97F4A7C15 },
        .reps = reps,
        .start = try hip.Event.init(gpu.d, true),
        .stop = try hip.Event.init(gpu.d, true),
    };
    defer t.stream.deinit();
    defer t.start.deinit();
    defer t.stop.deinit();
    t.fast.fuse = true;
    t.old.fuse = false;
    var ran: usize = 0;
    const all = filter.len == 0;
    if (all or std.mem.eql(u8, filter, "router")) {
        for ([_]usize{ 1, 2, 4, 8 }) |rows| try router(&t, rows, 2048, 257);
        ran += 1;
    }
    if (all or std.mem.eql(u8, filter, "tail")) {
        for ([_]usize{ 1, 4, 16 }) |rows| {
            try tails(&t, rows, 2048, 0);
            try tails(&t, rows, 2048, 9);
        }
        try tails(&t, 1, 4096, 0);
        ran += 1;
    }
    if (all or std.mem.eql(u8, filter, "group")) {
        for ([_]usize{ 1, 2, 4, 8, 16 }) |rows| try group(&t, rows, &.{ 8192, 4096, 32, 32 }, 2048, 4, 64);
        try group(&t, 1, &.{ 8192, 512, 512 }, 2048, 4, 64);
        try group(&t, 1, &.{ 12288, 12288 }, 4096, 6, 64);
        try group(&t, 4, &.{ 4096, 4096 }, 4096, 8, 128);
        try group(&t, 2, &.{ 1024, 96, 96 }, 2048, 3, 32);
        ran += 1;
    }
    if (all or std.mem.eql(u8, filter, "pair")) {
        for ([_]usize{ 1, 2, 4 }) |rows| try pair(&t, rows, 512, 2048, 4, 64);
        try pair(&t, 1, 768, 2048, 6, 64);
        try pair(&t, 1, 512, 2048, 4, 128);
        ran += 1;
    }
    if (all or std.mem.eql(u8, filter, "chain")) {
        try chain.run(&t);
        ran += 1;
    }
    try check.expect(ran > 0, "decode: no case matches '{s}'", .{filter});
    check.pass("decode: {d} groups of kernels, each within twice the previous launches' error of the float64 reference", .{ran});
}
