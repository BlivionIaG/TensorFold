//! Two-rank tensor parallel pieces of the EXL3 Flash Next layers (device side). Every rank owns its own CUDA context, module
//! instances, streams and buffers; the caller makes a rank's context current (`Context.makeCurrent`) before calling in here. Ranks
//! exchange nothing on the device (the two RTX 5060 Ti have no peer access): a K-split projection leaves one fp16 partial per rank
//! that the host adds in rank order (exl3_gdn.reduceOut / exl3_reduce.addHidden).
//!
//! GDN cut (rank r of 2): value heads 24r..24r+23, key heads 8r..8r+7 (a key head's three value heads stay together), so a rank's
//! conv channels are q (8 * 128) | k (8 * 128) | v (24 * 128) = 5120. in_proj_qkv is N-split as three 128-column-block slices
//! (q, k, v) gathered into ONE local matrix, in_proj_z is N-split, out_proj is K-split. Attention is in exl3_attn.zig with
//! `Dims{ .q_heads = 12, .kv_heads = 1 }`: no new kernel.
//!
//! The kernels are exl3_gdn_tp.cu: exl3_gdn.cu's arithmetic with the head counts as arguments, so a rank's rows, states and
//! gated outputs are byte for byte the tp=1 slice (the one difference between tp=1 and tp=2 is the out_proj reduction order).

const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const memory = @import("memory.zig");
const exl3_ = @import("exl3.zig");
const gdn_ = @import("exl3_gdn.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const DeviceBuffer = memory.DeviceBuffer;

pub const ranks = 2;
pub const hidden = 2560;
pub const nkl = gdn_.nk / ranks; // 8
pub const nvl = gdn_.nv / ranks; // 24
pub const channels = 2 * nkl * gdn_.dk + nvl * gdn_.dv; // 5120
pub const z_width = nvl * gdn_.dv; // 3072
pub const conv_bytes = gdn_.keep * channels * 2;
pub const rec_bytes = nvl * gdn_.dv * gdn_.dk * 4;

/// First column of rank `r`'s q, k and v slices inside the tp=1 in_proj_qkv output (and the slice widths).
pub fn qkvSlice(r: usize) [3]struct { lo: usize, width: usize } {
    return .{
        .{ .lo = r * nkl * gdn_.dk, .width = nkl * gdn_.dk },
        .{ .lo = gdn_.nk * gdn_.dk + r * nkl * gdn_.dk, .width = nkl * gdn_.dk },
        .{ .lo = 2 * gdn_.nk * gdn_.dk + r * z_width, .width = z_width },
    };
}

pub const GdnTp = struct {
    mod: Module,
    ab_fn: Function,
    conv_fn: Function,
    delta_fn: Function,
    norm_fn: Function,

    pub fn load(d: *const Driver) !GdnTp {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var m = try Module.load(d, kernels.exl3_gdn_tp);
        errdefer m.unload();
        return .{
            .mod = m,
            .ab_fn = try m.function(kernels.exl3_gdn_tp_symbols.ab_f16),
            .conv_fn = try m.function(kernels.exl3_gdn_tp_symbols.conv_f16),
            .delta_fn = try m.function(kernels.exl3_gdn_tp_symbols.delta_f32),
            .norm_fn = try m.function(kernels.exl3_gdn_tp_symbols.norm_f16),
        };
    }

    pub fn unload(g: *GdnTp) void {
        g.mod.unload();
    }

    pub fn ab(g: *const GdnTp, s: Stream, x: u64, wa: u64, wb: u64, out: u64, tokens: usize, k: usize) !void {
        if (tokens == 0) return;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(wa);
        a.add(wb);
        a.add(out);
        a.add(try cast(tokens));
        a.add(try cast(k));
        a.add(@as(i32, nvl));
        try launch_.launch(g.ab_fn, .{
            .grid = .{ .x = (2 * nvl + 7) / 8, .y = @intCast(tokens), .z = 1 },
            .block = .{ .x = 256, .y = 1, .z = 1 },
        }, s, &a);
    }

    pub fn conv(g: *const GdnTp, s: Stream, qkv: u64, conv_w: u64, state: u64, act: u64, tokens: usize) !void {
        if (tokens == 0) return;
        var a: launch_.Args = .{};
        a.add(qkv);
        a.add(conv_w);
        a.add(state);
        a.add(act);
        a.add(try cast(tokens));
        a.add(@as(i32, channels));
        try launch_.launch(g.conv_fn, .{
            .grid = .{ .x = (channels + 127) / 128, .y = 1, .z = 1 },
            .block = .{ .x = 128, .y = 1, .z = 1 },
        }, s, &a);
    }

    pub fn delta(g: *const GdnTp, s: Stream, act: u64, ab_: u64, a_log: u64, dt_bias: u64, state: u64, y: u64, tokens: usize) !void {
        if (tokens == 0) return;
        var a: launch_.Args = .{};
        a.add(act);
        a.add(ab_);
        a.add(a_log);
        a.add(dt_bias);
        a.add(state);
        a.add(y);
        a.add(try cast(tokens));
        a.add(@as(i32, nkl));
        a.add(@as(i32, nvl));
        try launch_.launch(g.delta_fn, .{
            .grid = .{ .x = nvl * gdn_.dv / 4, .y = 1, .z = 1 },
            .block = .{ .x = 128, .y = 1, .z = 1 },
        }, s, &a);
    }

    pub fn norm(g: *const GdnTp, s: Stream, y: u64, z: u64, w: u64, out: u64, tokens: usize, eps: f32) !void {
        if (tokens == 0) return;
        var a: launch_.Args = .{};
        a.add(y);
        a.add(z);
        a.add(w);
        a.add(out);
        a.add(try cast(tokens));
        a.add(eps);
        a.add(@as(i32, nvl));
        try launch_.launch(g.norm_fn, .{
            .grid = .{ .x = nvl, .y = @intCast(tokens), .z = 1 },
            .block = .{ .x = 32, .y = 1, .z = 1 },
        }, s, &a);
    }

    /// One rank's share of a GDN layer for `tokens` rows of `x` (fp16 [tokens][2560], the normed hidden, replicated on both ranks).
    /// `w` is the rank-local layer (qkv N = 5120, z N = 3072, out K = 3072; a_w / b_w / a_log / dt_bias already offset to the rank's
    /// first value head, conv_w gathered to the rank's 5120 channels). `out` receives this rank's fp16 [tokens][2560] PARTIAL of
    /// out_proj: the hidden is the host sum of the two ranks' partials, rank 0 first.
    pub fn forward(g: *const GdnTp, ex: *const exl3_.Exl3, s: Stream, w: gdn_.Layer, conv_state: u64, rec_state: u64, x: u64, tokens: usize, sc: Scratch, out: u64) !void {
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

/// One rank's activations for `tokens` rows.
pub const Scratch = struct {
    tokens: usize,
    qkv: DeviceBuffer, // fp16 [tokens][5120]
    z: DeviceBuffer, // fp16 [tokens][3072]
    ab: DeviceBuffer, // fp32 [tokens][48]
    act: DeviceBuffer, // fp32 [tokens][5120]
    y: DeviceBuffer, // fp32 [tokens][3072]
    gated: DeviceBuffer, // fp16 [tokens][3072]

    pub fn alloc(d: *const Driver, tokens: usize) !Scratch {
        var qkv = try DeviceBuffer.alloc(d, tokens * channels * 2);
        errdefer qkv.free();
        var z = try DeviceBuffer.alloc(d, tokens * z_width * 2);
        errdefer z.free();
        var ab = try DeviceBuffer.alloc(d, tokens * 2 * nvl * 4);
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

/// A rank's conv and recurrent state for every (slot, layer), the halves of exl3_gdn.Table.
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

    pub fn reset(t: Table, s: Stream, slot: usize) !void {
        if (slot >= t.slots) return error.Invalid;
        const d = t.conv.d;
        const cb = conv_bytes * t.layers;
        const rb = rec_bytes * t.layers;
        try d.check(d.api.cuMemsetD8Async(try t.conv.at(slot * cb), 0, cb, s.handle), "cuMemsetD8Async");
        try d.check(d.api.cuMemsetD8Async(try t.rec.at(slot * rb), 0, rb, s.handle), "cuMemsetD8Async");
    }
};

test "tp=2 GDN sizes are half of tp=1 and slices tile the tp=1 columns" {
    try std.testing.expectEqual(@as(usize, 5120), channels);
    try std.testing.expectEqual(gdn_.channels, 2 * channels);
    try std.testing.expectEqual(gdn_.z_width, 2 * z_width);
    try std.testing.expectEqual(gdn_.conv_bytes, 2 * conv_bytes);
    try std.testing.expectEqual(gdn_.rec_bytes, 2 * rec_bytes);
    var covered: [gdn_.channels]u8 = @splat(0);
    for (0..ranks) |r| {
        var width: usize = 0;
        for (qkvSlice(r)) |sl| {
            try std.testing.expect(sl.lo % 128 == 0 and sl.width % 128 == 0);
            for (covered[sl.lo..][0..sl.width]) |*c| c.* += 1;
            width += sl.width;
        }
        try std.testing.expectEqual(@as(usize, channels), width);
    }
    for (covered) |c| try std.testing.expectEqual(@as(u8, 1), c);
    try std.testing.expectEqual(@as(usize, 3), nvl / nkl);
}
