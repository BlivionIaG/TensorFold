//! Hyper-connection steps and the router dots on the device (kernels in zig/kernels/cuda/exl3_hc.cu). The host reference is
//! families/flashnext/exl3_hc.zig (`pre`, `post`, `expand`) and exl3_router.zig (the dots). Streams are f32 [token][4][2560];
//! the block input is the fp16 row every projection reads.

const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const memory = @import("memory.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const DeviceBuffer = memory.DeviceBuffer;

pub const streams = 4;
pub const hidden = 2560;
pub const wide = streams * hidden;
pub const lowrank = 320;
pub const gate_rows = 513;

/// Device weights of one hyper-connection (F16 as shipped): norm [10240], down [320][10240], up [10240][320], inject [4][10240] or null.
pub const Weights = struct {
    norm: u64,
    down: u64,
    up: u64,
    inject: ?u64,
};

pub const Scratch = struct {
    tokens: usize,
    normed: DeviceBuffer, // f32 [tokens][10240]
    low: DeviceBuffer, // f32 [tokens][320]
    mix: DeviceBuffer, // f32 [tokens][10240]

    pub fn alloc(d: *const Driver, tokens: usize) !Scratch {
        var normed = try DeviceBuffer.alloc(d, tokens * wide * 4);
        errdefer normed.free();
        var low = try DeviceBuffer.alloc(d, tokens * lowrank * 4);
        errdefer low.free();
        const mix = try DeviceBuffer.alloc(d, tokens * wide * 4);
        return .{ .tokens = tokens, .normed = normed, .low = low, .mix = mix };
    }

    pub fn free(s: *Scratch) void {
        s.normed.free();
        s.low.free();
        s.mix.free();
    }
};

pub const Hc = struct {
    mod: Module,
    expand_fn: Function,
    norm_fn: Function,
    low_fn: Function,
    mix_fn: Function,
    out_fn: Function,
    post_fn: Function,
    gate_fn: Function,

    pub fn load(d: *const Driver) !Hc {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var m = try Module.load(d, kernels.exl3_hc);
        errdefer m.unload();
        const sy = kernels.exl3_hc_symbols;
        return .{
            .mod = m,
            .expand_fn = try m.function(sy.expand_bf16),
            .norm_fn = try m.function(sy.norm),
            .low_fn = try m.function(sy.low),
            .mix_fn = try m.function(sy.mix),
            .out_fn = try m.function(sy.out),
            .post_fn = try m.function(sy.post),
            .gate_fn = try m.function(sy.gate),
        };
    }

    pub fn unload(g: *Hc) void {
        g.mod.unload();
    }

    /// h[t] = the bf16 embedding row t tiled over the 4 streams. `emb` is bf16 [tokens][2560].
    pub fn expand(g: *const Hc, s: Stream, emb: u64, h: u64, tokens: usize) !void {
        if (tokens == 0) return;
        var a: launch_.Args = .{};
        a.add(emb);
        a.add(h);
        a.add(try cast(tokens));
        try launch_.launch(g.expand_fn, .{ .grid = .{ .x = wide / 256, .y = @intCast(tokens), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }

    /// Block input `x` (fp16 [tokens][2560]) from the streams `h`, and, when `w.inject` and `inject` are given, the 4 gates a row
    /// (f32 [tokens][4]). `offset` is the centring offset (1 for gamma - 1 storage).
    pub fn pre(g: *const Hc, s: Stream, w: Weights, offset: f32, eps: f32, h: u64, sc: Scratch, x: u64, inject: ?u64, tokens: usize) !void {
        if (tokens == 0) return;
        if (tokens > sc.tokens) return error.Exl3Shape;
        const t = try cast(tokens);
        const ty: u32 = @intCast(tokens);
        var a: launch_.Args = .{};
        a.add(h);
        a.add(w.norm);
        a.add(offset);
        a.add(eps);
        a.add(sc.normed.ptr);
        a.add(t);
        try launch_.launch(g.norm_fn, .{ .grid = .{ .x = streams, .y = ty, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);

        const inj: u64 = if (inject != null and w.inject != null) w.inject.? else 0;
        const inj_out: u64 = if (inj != 0) inject.? else 0;
        var b: launch_.Args = .{};
        b.add(sc.normed.ptr);
        b.add(w.down);
        b.add(inj);
        b.add(sc.low.ptr);
        b.add(inj_out);
        b.add(t);
        try launch_.launch(g.low_fn, .{ .grid = .{ .x = (lowrank + 4 + 127) / 128, .y = ty, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &b);

        var c: launch_.Args = .{};
        c.add(sc.low.ptr);
        c.add(w.up);
        c.add(sc.mix.ptr);
        c.add(t);
        try launch_.launch(g.mix_fn, .{ .grid = .{ .x = wide / 128, .y = ty, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &c);

        var o: launch_.Args = .{};
        o.add(sc.mix.ptr);
        o.add(sc.normed.ptr);
        o.add(x);
        o.add(t);
        try launch_.launch(g.out_fn, .{ .grid = .{ .x = hidden / 256, .y = ty, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &o);
    }

    /// h[t][s][i] += branch[t][i] * inject[t][s]. `branch` is fp16 [tokens][2560].
    pub fn post(g: *const Hc, s: Stream, h: u64, branch: u64, inject: u64, tokens: usize) !void {
        if (tokens == 0) return;
        var a: launch_.Args = .{};
        a.add(h);
        a.add(branch);
        a.add(inject);
        a.add(try cast(tokens));
        try launch_.launch(g.post_fn, .{ .grid = .{ .x = wide / 256, .y = @intCast(tokens), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }

    /// logits f32 [tokens][513]: 512 routed gate dots, then the shared-expert gate dot, over the fp16 block input `x`.
    pub fn gate(g: *const Hc, s: Stream, x: u64, gate_w: u64, sgate_w: u64, logits: u64, tokens: usize) !void {
        if (tokens == 0) return;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(gate_w);
        a.add(sgate_w);
        a.add(logits);
        a.add(try cast(tokens));
        try launch_.launch(g.gate_fn, .{ .grid = .{ .x = (gate_rows + 127) / 128, .y = @intCast(tokens), .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }
};

fn cast(v: usize) error{Exl3Shape}!i32 {
    return std.math.cast(i32, v) orelse error.Exl3Shape;
}

test "symbols" {
    try std.testing.expectEqualStrings("exl3_hc_gate", kernels.exl3_hc_symbols.gate);
    var g: Hc = undefined;
    const s = Stream{ .d = undefined, .handle = null };
    try g.expand(s, 1, 2, 0);
    try g.post(s, 1, 2, 3, 0);
    try g.gate(s, 1, 2, 3, 4, 0);
}
