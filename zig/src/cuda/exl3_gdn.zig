//! Gated DeltaNet layer for the EXL3 Flash Next path (tp=1): three EXL3 projections (`Exl3.linear`) around the kernels in
//! exl3_gdn.cu. State lives in a device table written in place, so a graph that captured these launches replays them.
//! The host oracle is families/flashnext/exl3_gdn_cpu.zig; the sizes below are cross-checked against it by tf-cuda-test.
//!
//! tp=2 cut (not implemented): value heads 0..23 / 24..47 per rank, key heads 0..7 / 8..15 (hv / 3 is unchanged), so a rank
//! owns q|k|v conv channels {hk range, hk range, hv range}, state table halves, `exl3_gdn_delta_f32` takes a head offset.
//! in_proj_qkv and in_proj_z are N-split at 128-column blocks (qkv as three slices q, k, v, since a rank's columns are not
//! contiguous), out_proj is K-split with the partial sums added by exl3_gdn.reduceOut.

const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const memory = @import("memory.zig");
const exl3_ = @import("exl3.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const DeviceBuffer = memory.DeviceBuffer;

pub const nk = 16;
pub const nv = 48;
pub const dk = 128;
pub const dv = 128;
pub const taps = 4;
pub const channels = 2 * nk * dk + nv * dv;
pub const z_width = nv * dv;
pub const keep = taps - 1;
/// bytes of one layer's conv state (fp16 rows) and recurrent state (fp32)
pub const conv_bytes = keep * channels * 2;
pub const rec_bytes = nv * dv * dk * 4;

pub const Gdn = struct {
    mod: Module,
    ab_fn: Function,
    conv_fn: Function,
    delta_fn: Function,
    norm_fn: Function,

    pub fn load(d: *const Driver) !Gdn {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var m = try Module.load(d, kernels.exl3_gdn);
        errdefer m.unload();
        return .{
            .mod = m,
            .ab_fn = try m.function(kernels.exl3_gdn_symbols.ab_f16),
            .conv_fn = try m.function(kernels.exl3_gdn_symbols.conv_f16),
            .delta_fn = try m.function(kernels.exl3_gdn_symbols.delta_f32),
            .norm_fn = try m.function(kernels.exl3_gdn_symbols.norm_f16),
        };
    }

    pub fn unload(g: *Gdn) void {
        g.mod.unload();
    }

    /// ab[t][0..48) = in_proj_a x[t], ab[t][48..96) = in_proj_b x[t] (fp32). Weights are fp16 [48][k].
    pub fn ab(g: *const Gdn, s: Stream, x: u64, wa: u64, wb: u64, out: u64, tokens: usize, k: usize) !void {
        if (tokens == 0) return;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(wa);
        a.add(wb);
        a.add(out);
        a.add(try cast(tokens));
        a.add(try cast(k));
        try launch_.launch(g.ab_fn, .{
            .grid = .{ .x = 12, .y = @intCast(tokens), .z = 1 },
            .block = .{ .x = 256, .y = 1, .z = 1 },
        }, s, &a);
    }

    /// conv + SiLU over `tokens` raw qkv rows in order; `state` is one layer's conv state, updated in place.
    pub fn conv(g: *const Gdn, s: Stream, qkv: u64, conv_w: u64, state: u64, act: u64, tokens: usize) !void {
        if (tokens == 0) return;
        var a: launch_.Args = .{};
        a.add(qkv);
        a.add(conv_w);
        a.add(state);
        a.add(act);
        a.add(try cast(tokens));
        try launch_.launch(g.conv_fn, .{
            .grid = .{ .x = (channels + 127) / 128, .y = 1, .z = 1 },
            .block = .{ .x = 128, .y = 1, .z = 1 },
        }, s, &a);
    }

    /// The delta rule over `tokens` tokens in order; `state` is one layer's recurrent state, updated in place.
    pub fn delta(g: *const Gdn, s: Stream, act: u64, ab_: u64, a_log: u64, dt_bias: u64, state: u64, y: u64, tokens: usize) !void {
        if (tokens == 0) return;
        var a: launch_.Args = .{};
        a.add(act);
        a.add(ab_);
        a.add(a_log);
        a.add(dt_bias);
        a.add(state);
        a.add(y);
        a.add(try cast(tokens));
        try launch_.launch(g.delta_fn, .{
            .grid = .{ .x = nv * dv / 4, .y = 1, .z = 1 },
            .block = .{ .x = 128, .y = 1, .z = 1 },
        }, s, &a);
    }

    /// Gated RMS norm per (token, value head) into the fp16 rows out_proj reads.
    pub fn norm(g: *const Gdn, s: Stream, y: u64, z: u64, w: u64, out: u64, tokens: usize, eps: f32) !void {
        if (tokens == 0) return;
        var a: launch_.Args = .{};
        a.add(y);
        a.add(z);
        a.add(w);
        a.add(out);
        a.add(try cast(tokens));
        a.add(eps);
        try launch_.launch(g.norm_fn, .{
            .grid = .{ .x = nv, .y = @intCast(tokens), .z = 1 },
            .block = .{ .x = 32, .y = 1, .z = 1 },
        }, s, &a);
    }

    /// One layer for `tokens` rows of `x` (fp16 [tokens][2560], the normed hidden), tokens in order, into `out` (fp16
    /// [tokens][2560]). Conv and recurrent state of (slot, layer) are `conv_state` and `rec_state`, updated in place.
    /// One token is decode; N tokens are a prefill that runs the same kernels over the N tokens in sequence.
    pub fn forward(g: *const Gdn, ex: *const exl3_.Exl3, s: Stream, w: Layer, conv_state: u64, rec_state: u64, x: u64, tokens: usize, sc: Scratch, out: u64) !void {
        if (tokens == 0) return;
        if (tokens > sc.tokens) return error.Exl3Shape;
        try ex.linear(s, x, w.qkv.trellis, w.qkv.suh, w.qkv.svh, sc.qkv.ptr, tokens, w.qkv.k, w.qkv.n, w.qkv.halves, 2);
        try ex.linear(s, x, w.z.trellis, w.z.suh, w.z.svh, sc.z.ptr, tokens, w.z.k, w.z.n, w.z.halves, 2);
        try g.ab(s, x, w.a_w, w.b_w, sc.ab.ptr, tokens, w.qkv.k);
        try g.conv(s, sc.qkv.ptr, w.conv_w, conv_state, sc.act.ptr, tokens);
        try g.delta(s, sc.act.ptr, sc.ab.ptr, w.a_log, w.dt_bias, rec_state, sc.y.ptr, tokens);
        try g.norm(s, sc.y.ptr, sc.z.ptr, w.norm_w, sc.gated.ptr, tokens, w.eps);
        try ex.linear(s, sc.gated.ptr, w.out.trellis, w.out.suh, w.out.svh, out, tokens, w.out.k, w.out.n, w.out.halves, 2);
    }
};

fn cast(v: usize) error{Exl3Shape}!i32 {
    return std.math.cast(i32, v) orelse error.Exl3Shape;
}

/// One strips-order EXL3 matrix on the device (device addresses).
pub const Proj = struct { trellis: u64, suh: u64, svh: u64, k: usize, n: usize, halves: u8 };

/// A GDN layer's device weights: in_proj_a/b fp16 [48][2560], conv1d bf16 [10240][4], A_log / dt_bias bf16 [48],
/// linear_attn.norm bf16 [128].
pub const Layer = struct {
    qkv: Proj,
    z: Proj,
    out: Proj,
    a_w: u64,
    b_w: u64,
    conv_w: u64,
    a_log: u64,
    dt_bias: u64,
    norm_w: u64,
    eps: f32 = 1e-6,
};

/// Per-call activations, sized for `tokens` rows. Reused across layers.
pub const Scratch = struct {
    tokens: usize,
    qkv: DeviceBuffer, // fp16 [tokens][10240]
    z: DeviceBuffer, // fp16 [tokens][6144]
    ab: DeviceBuffer, // fp32 [tokens][96]
    act: DeviceBuffer, // fp32 [tokens][10240]
    y: DeviceBuffer, // fp32 [tokens][6144]
    gated: DeviceBuffer, // fp16 [tokens][6144]

    pub fn alloc(d: *const Driver, tokens: usize) !Scratch {
        var qkv = try DeviceBuffer.alloc(d, tokens * channels * 2);
        errdefer qkv.free();
        var z = try DeviceBuffer.alloc(d, tokens * z_width * 2);
        errdefer z.free();
        var ab = try DeviceBuffer.alloc(d, tokens * 2 * nv * 4);
        errdefer ab.free();
        var act = try DeviceBuffer.alloc(d, tokens * channels * 4);
        errdefer act.free();
        var y = try DeviceBuffer.alloc(d, tokens * z_width * 4);
        errdefer y.free();
        const gated = try DeviceBuffer.alloc(d, tokens * z_width * 2);
        return .{ .tokens = tokens, .qkv = qkv, .z = z, .ab = ab, .act = act, .y = y, .gated = gated };
    }

    pub fn free(sc: *Scratch) void {
        sc.qkv.free();
        sc.z.free();
        sc.ab.free();
        sc.act.free();
        sc.y.free();
        sc.gated.free();
    }
};

/// Conv and recurrent state for every (slot, layer): [slot][layer] contiguous, so a sequence's layers are one span.
pub const Table = struct {
    layers: usize,
    slots: usize,
    conv: DeviceBuffer,
    rec: DeviceBuffer,

    pub fn alloc(d: *const Driver, layers: usize, slots: usize) !Table {
        var conv_ = try DeviceBuffer.alloc(d, conv_bytes * layers * slots);
        errdefer conv_.free();
        var rec = try DeviceBuffer.alloc(d, rec_bytes * layers * slots);
        errdefer rec.free();
        var t: Table = .{ .layers = layers, .slots = slots, .conv = conv_, .rec = rec };
        try t.conv.fill8(0, null);
        try t.rec.fill8(0, null);
        return t;
    }

    pub fn free(t: *Table) void {
        t.conv.free();
        t.rec.free();
    }

    pub fn convPtr(t: Table, slot: usize, layer: usize) !u64 {
        if (slot >= t.slots or layer >= t.layers) return error.Invalid;
        return t.conv.at((slot * t.layers + layer) * conv_bytes);
    }

    pub fn recPtr(t: Table, slot: usize, layer: usize) !u64 {
        if (slot >= t.slots or layer >= t.layers) return error.Invalid;
        return t.rec.at((slot * t.layers + layer) * rec_bytes);
    }

    /// Zero every layer of a slot (a new sequence), on `s`.
    pub fn reset(t: Table, s: Stream, slot: usize) !void {
        if (slot >= t.slots) return error.Invalid;
        const d = t.conv.d;
        const cb = conv_bytes * t.layers;
        const rb = rec_bytes * t.layers;
        try d.check(d.api.cuMemsetD8Async(try t.conv.at(slot * cb), 0, cb, s.handle), "cuMemsetD8Async");
        try d.check(d.api.cuMemsetD8Async(try t.rec.at(slot * rb), 0, rb, s.handle), "cuMemsetD8Async");
    }
};

test "symbols and sizes" {
    try std.testing.expectEqualStrings("exl3_gdn_delta_f32", kernels.exl3_gdn_symbols.delta_f32);
    try std.testing.expectEqual(@as(usize, 10240), channels);
    try std.testing.expectEqual(@as(usize, 61440), conv_bytes);
    try std.testing.expectEqual(@as(usize, 3145728), rec_bytes);
    var g: Gdn = undefined;
    const s = Stream{ .d = undefined, .handle = null };
    try g.conv(s, 1, 2, 3, 4, 0);
    try g.delta(s, 1, 2, 3, 4, 5, 6, 0);
}
