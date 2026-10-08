//! The merged linear-attention launches against the chains they replace, byte for byte, and the MoE pick rule by rank.

const std = @import("std");
const hip = @import("hip");
const check = @import("../check.zig");
const bench = @import("decode.zig");
const Rig = bench.Rig;
const L = bench.L;
const P = ?*anyopaque;

fn upload(t: *Rig, host: anytype) !hip.DeviceBuffer {
    return hip.DeviceBuffer.fromHost(t.gpu.d, std.mem.sliceAsBytes(host));
}

fn same(t: *Rig, what: []const u8, a: hip.DeviceBuffer, b: hip.DeviceBuffer) !void {
    const ha = try check.download(t.gpu, a);
    defer t.gpu.gpa.free(ha);
    const hb = try check.download(t.gpu, b);
    defer t.gpu.gpa.free(hb);
    try check.sameBytes(what, hb, ha);
}

/// The pick rule over `rows` rows of (experts + 1) logits: picks, weights, items and members as the scanning kernel's.
pub fn select(t: *Rig, rows: usize, experts: usize, top_k: usize) !void {
    const gpa = t.gpu.gpa;
    const slots = top_k + 1;
    const capacity = slots + 7;
    const logits = try gpa.alloc(f32, rows * (experts + 1));
    defer gpa.free(logits);
    for (logits) |*v| v.* = t.rng.unit() * 4.0;
    // a few equal logits: the lower index wins
    for (0..rows) |r| logits[r * (experts + 1) + 5] = logits[r * (experts + 1) + 9];
    var dev_logits = try upload(t, logits);
    defer dev_logits.free();
    var picks: [2]hip.DeviceBuffer = undefined;
    var wts: [2]hip.DeviceBuffer = undefined;
    var items: [2]hip.DeviceBuffer = undefined;
    var members: [2]hip.DeviceBuffer = undefined;
    for (0..2) |v| {
        picks[v] = try hip.DeviceBuffer.alloc(t.gpu.d, rows * slots * 4);
        wts[v] = try hip.DeviceBuffer.alloc(t.gpu.d, rows * slots * 4);
        items[v] = try hip.DeviceBuffer.alloc(t.gpu.d, capacity * 12);
        members[v] = try hip.DeviceBuffer.alloc(t.gpu.d, capacity * 4);
        try members[v].fill8(0, null);
    }
    defer for (0..2) |v| {
        picks[v].free();
        wts[v].free();
        items[v].free();
        members[v].free();
    };
    const Ctx = struct {
        t: *Rig,
        old: bool,
        logits: u64,
        pick: u64,
        wts: u64,
        items: u64,
        members: u64,
        rows: usize,
        experts: usize,
        top_k: usize,
        capacity: usize,
        fn go(c: @This(), _: usize) hip.Error!void {
            const l = if (c.old) &c.t.old else &c.t.fast;
            const plan = c.rows == 1;
            try l.tf_moe_select(@ptrFromInt(c.logits), @ptrFromInt(c.pick), @ptrFromInt(c.wts), if (plan) @ptrFromInt(c.items) else null, if (plan) @ptrFromInt(c.members) else null, @intCast(c.capacity), @intCast(c.rows), @intCast(c.experts), @intCast(c.top_k), c.t.stream.handle);
        }
    };
    var us: [2]f64 = undefined;
    for (0..2) |v| {
        const ctx: Ctx = .{ .t = t, .old = v == 0, .logits = dev_logits.ptr, .pick = picks[v].ptr, .wts = wts[v].ptr, .items = items[v].ptr, .members = members[v].ptr, .rows = rows, .experts = experts, .top_k = top_k, .capacity = capacity };
        try Ctx.go(ctx, 0);
        try t.stream.synchronize();
        us[v] = try t.time(100 * t.reps, ctx, Ctx.go);
    }
    try same(t, "select picks", picks[0], picks[1]);
    try same(t, "select weights", wts[0], wts[1]);
    if (rows == 1) {
        try same(t, "select items", items[0], items[1]);
        try same(t, "select members", members[0], members[1]);
    }
    if (t.bench) std.debug.print("RESULT decode select rows{d} experts{d} top{d}: old {d:.1} us, new {d:.1} us, x{d:.2}; picks, weights and plan equal\n", .{ rows, experts, top_k, us[0], us[1], us[0] / us[1] });
}

/// A window's conv chain (cast, conv, column copies, q and k norms, gate) against one launch: equal bytes everywhere.
pub fn conv(t: *Rig, rows: usize) !void {
    const gpa = t.gpu.gpa;
    const kw: usize = 2048;
    const vw: usize = 4096;
    const ch = 2 * kw + vw;
    const kernel: usize = 4;
    const kind: c_int = if (t.fp16) 1 else 2;
    const eps: f32 = 1e-6;
    const hx = try gpa.alloc(u16, rows * ch);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(t.rng.unit() * 2.0);
    const hw = try gpa.alloc(f32, ch * kernel);
    defer gpa.free(hw);
    for (hw) |*v| v.* = t.rng.unit() / 2.0;
    const hs = try gpa.alloc(f32, (kernel - 1) * ch);
    defer gpa.free(hs);
    for (hs) |*v| v.* = t.rng.unit();
    const hn = try gpa.alloc(f32, 128 * 2);
    defer gpa.free(hn);
    for (hn) |*v| v.* = 1.0 + t.rng.unit() / 4.0;
    const vheads: usize = 32;
    const ha = try gpa.alloc(u16, rows * vheads);
    defer gpa.free(ha);
    for (ha) |*v| v.* = t.bits(t.rng.unit() * 3.0);
    const hb = try gpa.alloc(u16, rows * vheads);
    defer gpa.free(hb);
    for (hb) |*v| v.* = t.bits(t.rng.unit() * 3.0);
    const hlog = try gpa.alloc(f32, vheads * 2);
    defer gpa.free(hlog);
    for (hlog) |*v| v.* = t.rng.unit();
    var x = try upload(t, hx);
    defer x.free();
    var weight = try upload(t, hw);
    defer weight.free();
    var norm = try upload(t, hn);
    defer norm.free();
    var dev_a = try upload(t, ha);
    defer dev_a.free();
    var dev_b = try upload(t, hb);
    defer dev_b.free();
    var dev_log = try upload(t, hlog);
    defer dev_log.free();
    var gates: [2]hip.DeviceBuffer = undefined;
    var betas: [2]hip.DeviceBuffer = undefined;
    for (0..2) |v| {
        gates[v] = try hip.DeviceBuffer.alloc(t.gpu.d, rows * vheads * 4);
        betas[v] = try hip.DeviceBuffer.alloc(t.gpu.d, rows * vheads * 4);
    }
    defer for (0..2) |v| {
        gates[v].free();
        betas[v].free();
    };
    var state: [2]hip.DeviceBuffer = undefined;
    var snaps: [2]hip.DeviceBuffer = undefined;
    var qc: [2]hip.DeviceBuffer = undefined;
    var kc: [2]hip.DeviceBuffer = undefined;
    var vv: [2]hip.DeviceBuffer = undefined;
    var qn: [2]hip.DeviceBuffer = undefined;
    var kn: [2]hip.DeviceBuffer = undefined;
    for (0..2) |v| {
        state[v] = try upload(t, hs);
        snaps[v] = try hip.DeviceBuffer.alloc(t.gpu.d, rows * (kernel - 1) * ch * 4);
        qc[v] = try hip.DeviceBuffer.alloc(t.gpu.d, rows * kw * 4);
        kc[v] = try hip.DeviceBuffer.alloc(t.gpu.d, rows * kw * 4);
        vv[v] = try hip.DeviceBuffer.alloc(t.gpu.d, rows * vw * 4);
        qn[v] = try hip.DeviceBuffer.alloc(t.gpu.d, rows * kw * 4);
        kn[v] = try hip.DeviceBuffer.alloc(t.gpu.d, rows * kw * 4);
    }
    defer for (0..2) |v| {
        state[v].free();
        snaps[v].free();
        qc[v].free();
        kc[v].free();
        vv[v].free();
        qn[v].free();
        kn[v].free();
    };
    var xr = try hip.DeviceBuffer.alloc(t.gpu.d, rows * ch * 4);
    defer xr.free();
    var mixed = try hip.DeviceBuffer.alloc(t.gpu.d, rows * ch * 4);
    defer mixed.free();
    const Ctx = struct {
        t: *Rig,
        old: bool,
        v: usize,
        x: u64,
        weight: u64,
        norm: u64,
        state: u64,
        snaps: u64,
        qc: u64,
        kc: u64,
        vv: u64,
        qn: u64,
        kn: u64,
        a: u64,
        b: u64,
        a_log: u64,
        gate: u64,
        beta: u64,
        xr: u64,
        mixed: u64,
        rows: usize,
        ch: usize,
        kw: usize,
        vw: usize,
        kernel: usize,
        kind: c_int,
        eps: f32,
        fn go(c: @This(), _: usize) hip.Error!void {
            const s = c.t.stream.handle;
            const l = &c.t.fast;
            const heads = c.rows * (c.kw / 128);
            if (c.old) {
                try l.tf_cast(@ptrFromInt(c.x), c.kind, @ptrFromInt(c.xr), 0, @intCast(c.rows * c.ch), s);
                if (c.rows == 1) {
                    try l.tf_conv_decode(@ptrFromInt(c.xr), @ptrFromInt(c.weight), @ptrFromInt(c.state), @ptrFromInt(c.mixed), 1, @intCast(c.ch), @intCast(c.kernel), s);
                } else {
                    try l.tf_conv_rows(@ptrFromInt(c.xr), @ptrFromInt(c.weight), @ptrFromInt(c.state), @ptrFromInt(c.mixed), @ptrFromInt(c.snaps), @intCast(c.rows), @intCast(c.ch), @intCast(c.kernel), s);
                }
                try l.tf_copy_cols(@ptrFromInt(c.mixed), @intCast(c.ch), 0, @ptrFromInt(c.qc), 0, @intCast(c.rows), @intCast(c.kw), s);
                try l.tf_copy_cols(@ptrFromInt(c.mixed), @intCast(c.ch), @intCast(c.kw), @ptrFromInt(c.kc), 0, @intCast(c.rows), @intCast(c.kw), s);
                try l.tf_copy_cols(@ptrFromInt(c.mixed), @intCast(c.ch), @intCast(2 * c.kw), @ptrFromInt(c.vv), 0, @intCast(c.rows), @intCast(c.vw), s);
                try l.tf_rms(@ptrFromInt(c.qc), @ptrFromInt(c.norm), @ptrFromInt(c.qn), 0, @intCast(heads), 128, c.eps, s);
                try l.tf_rms(@ptrFromInt(c.kc), @ptrFromInt(c.norm + 512), @ptrFromInt(c.kn), 0, @intCast(heads), 128, c.eps, s);
                try l.tf_gdn_gate(@ptrFromInt(c.a), @ptrFromInt(c.b), c.kind, @ptrFromInt(c.a_log), @ptrFromInt(c.a_log + 128), @ptrFromInt(c.gate), @ptrFromInt(c.beta), @intCast(c.rows * 32), 32, s);
            } else {
                try l.tf_conv_split(.{
                    .x = c.x,
                    .kind = c.kind,
                    .weight = c.weight,
                    .state = c.state,
                    .states = if (c.rows > 1) c.snaps else 0,
                    .qn = c.qn,
                    .kn = c.kn,
                    .v = c.vv,
                    .channels = @intCast(c.ch),
                    .kernel = @intCast(c.kernel),
                    .rows = @intCast(c.rows),
                    .kw = @intCast(c.kw),
                    .vw = @intCast(c.vw),
                    .qw = c.norm,
                    .kw_w = c.norm + 512,
                    .eps = c.eps,
                    .norm = 128,
                    .ga = c.a,
                    .gb = c.b,
                    .a_log = c.a_log,
                    .dt_bias = c.a_log + 128,
                    .gate = c.gate,
                    .beta = c.beta,
                    .gcount = @intCast(c.rows * 32),
                    .heads = 32,
                }, s);
            }
        }
    };
    var us: [2]f64 = undefined;
    for (0..2) |v| {
        const ctx: Ctx = .{ .t = t, .old = v == 0, .v = v, .x = x.ptr, .weight = weight.ptr, .norm = norm.ptr, .state = state[v].ptr, .snaps = snaps[v].ptr, .qc = qc[v].ptr, .kc = kc[v].ptr, .vv = vv[v].ptr, .qn = qn[v].ptr, .kn = kn[v].ptr, .a = dev_a.ptr, .b = dev_b.ptr, .a_log = dev_log.ptr, .gate = gates[v].ptr, .beta = betas[v].ptr, .xr = xr.ptr, .mixed = mixed.ptr, .rows = rows, .ch = ch, .kw = kw, .vw = vw, .kernel = kernel, .kind = kind, .eps = eps };
        try Ctx.go(ctx, 0);
        try t.stream.synchronize();
    }
    try same(t, "conv state", state[0], state[1]);
    if (rows > 1) try same(t, "conv snapshots", snaps[0], snaps[1]);
    try same(t, "conv v", vv[0], vv[1]);
    try same(t, "q norm", qn[0], qn[1]);
    try same(t, "k norm", kn[0], kn[1]);
    try same(t, "gate", gates[0], gates[1]);
    try same(t, "beta", betas[0], betas[1]);
    for (0..2) |v| {
        const ctx: Ctx = .{ .t = t, .old = v == 0, .v = v, .x = x.ptr, .weight = weight.ptr, .norm = norm.ptr, .state = state[v].ptr, .snaps = snaps[v].ptr, .qc = qc[v].ptr, .kc = kc[v].ptr, .vv = vv[v].ptr, .qn = qn[v].ptr, .kn = kn[v].ptr, .a = dev_a.ptr, .b = dev_b.ptr, .a_log = dev_log.ptr, .gate = gates[v].ptr, .beta = betas[v].ptr, .xr = xr.ptr, .mixed = mixed.ptr, .rows = rows, .ch = ch, .kw = kw, .vw = vw, .kernel = kernel, .kind = kind, .eps = eps };
        us[v] = try t.time(100 * t.reps, ctx, Ctx.go);
    }
    if (t.bench) std.debug.print("RESULT decode conv rows{d}: cast, conv, 3 copies, 2 norms and the gate {d:.1} us, one launch {d:.1} us, x{d:.2}; state, snapshots, q, k, v, gate and beta equal bytes\n", .{ rows, us[0], us[1], us[0] / us[1] });
}

/// The gated norm of the linear attention: rms and gnorm_silu against one launch, equal bytes.
pub fn gnorm(t: *Rig, rows: usize) !void {
    const gpa = t.gpu.gpa;
    const heads: usize = 32;
    const dv: usize = 128;
    const n = rows * heads * dv;
    const kind: c_int = if (t.fp16) 1 else 2;
    const eps: f32 = 1e-6;
    const hy = try gpa.alloc(f32, n);
    defer gpa.free(hy);
    for (hy) |*v| v.* = t.rng.unit() * 3.0;
    const hz = try gpa.alloc(u16, n);
    defer gpa.free(hz);
    for (hz) |*v| v.* = t.bits(t.rng.unit() * 4.0);
    const hn = try gpa.alloc(f32, dv);
    defer gpa.free(hn);
    for (hn) |*v| v.* = 1.0 + t.rng.unit() / 4.0;
    var y = try upload(t, hy);
    defer y.free();
    var z = try upload(t, hz);
    defer z.free();
    var weight = try upload(t, hn);
    defer weight.free();
    var yn = try hip.DeviceBuffer.alloc(t.gpu.d, n * 4);
    defer yn.free();
    var outs: [2]hip.DeviceBuffer = undefined;
    for (&outs) |*o| o.* = try hip.DeviceBuffer.alloc(t.gpu.d, n * 2);
    defer for (&outs) |*o| o.free();
    const Ctx = struct {
        t: *Rig,
        old: bool,
        y: u64,
        z: u64,
        weight: u64,
        yn: u64,
        out: u64,
        rows: usize,
        n: usize,
        kind: c_int,
        eps: f32,
        fn go(c: @This(), _: usize) hip.Error!void {
            const s = c.t.stream.handle;
            const l = &c.t.fast;
            if (c.old) {
                try l.tf_rms(@ptrFromInt(c.y), @ptrFromInt(c.weight), @ptrFromInt(c.yn), 0, @intCast(c.rows * 32), 128, c.eps, s);
                try l.tf_gnorm_silu(@ptrFromInt(c.yn), @ptrFromInt(c.z), @ptrFromInt(c.out), c.kind, @intCast(c.n), s);
            } else {
                try l.tf_gnorm_out(@ptrFromInt(c.y), @ptrFromInt(c.weight), @ptrFromInt(c.z), @ptrFromInt(c.out), c.kind, @intCast(c.rows * 32), 128, c.eps, s);
            }
        }
    };
    var us: [2]f64 = undefined;
    for (0..2) |v| {
        const ctx: Ctx = .{ .t = t, .old = v == 0, .y = y.ptr, .z = z.ptr, .weight = weight.ptr, .yn = yn.ptr, .out = outs[v].ptr, .rows = rows, .n = n, .kind = kind, .eps = eps };
        try Ctx.go(ctx, 0);
        try t.stream.synchronize();
        us[v] = try t.time(100 * t.reps, ctx, Ctx.go);
    }
    try same(t, "gated norm", outs[0], outs[1]);
    if (t.bench) std.debug.print("RESULT decode gnorm rows{d}: rms and gnorm_silu {d:.1} us, one launch {d:.1} us, x{d:.2}; equal bytes\n", .{ rows, us[0], us[1], us[0] / us[1] });
}

pub fn run(t: *Rig) !void {
    for ([_]usize{ 1, 4, 16 }) |rows| try select(t, rows, 256, 8);
    try select(t, 1, 128, 6);
    for ([_]usize{ 1, 2, 4, 16 }) |rows| try conv(t, rows);
    for ([_]usize{ 1, 4, 16, 128 }) |rows| try gnorm(t, rows);
}
