//! EXL3 strips projections. The device decode matches zig/src/core/exl3_format.zig.
//! Expert launches read ids from a pointer so a graph replay can swap that buffer.
//! Gate/up and down are the same projection. The epilogue is identity: no SiLU, no clamp.
//! Combine sums slot outputs with fp32 weights, then rounds to fp16.

const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const Exl3 = struct {
    mod_linear: Module,
    mod_experts: Module,
    linear_fn: Function,
    gate_up_fn: Function,
    down_fn: Function,
    combine_fn: Function,

    pub fn load(d: *const Driver) !Exl3 {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var linear_mod = try Module.load(d, kernels.exl3_linear);
        errdefer linear_mod.unload();
        var experts_mod = try Module.load(d, kernels.exl3_experts);
        errdefer experts_mod.unload();
        return .{
            .mod_linear = linear_mod,
            .mod_experts = experts_mod,
            .linear_fn = try linear_mod.function(kernels.exl3_symbols.linear_f16),
            .gate_up_fn = try experts_mod.function(kernels.exl3_symbols.gate_up_f16),
            .down_fn = try experts_mod.function(kernels.exl3_symbols.down_f16),
            .combine_fn = try experts_mod.function(kernels.exl3_symbols.combine_f16),
        };
    }

    pub fn unload(g: *Exl3) void {
        g.mod_experts.unload();
        g.mod_linear.unload();
    }

    /// y[rows, N] = x[rows, K] against one strips-order matrix. Grid (N/128, rows), block 32.
    pub fn linear(g: *const Exl3, s: Stream, x: u64, trellis: u64, suh: u64, svh: u64, y: u64, rows: usize, k: usize, n: usize, halves: u8, codebook: u8) !void {
        if (rows == 0) return;
        const dims = try projection(rows, k, n, halves, codebook);
        var a: launch_.Args = .{};
        a.add(x);
        a.add(trellis);
        a.add(suh);
        a.add(svh);
        a.add(y);
        a.add(dims.rows);
        a.add(dims.k);
        a.add(dims.n);
        a.add(dims.halves);
        a.add(dims.codebook);
        try launch_.launch(g.linear_fn, .{
            .grid = .{ .x = dims.n_blocks, .y = dims.rows_u, .z = 1 },
            .block = .{ .x = 32, .y = 1, .z = 1 },
        }, s, &a);
    }

    /// One expert matrix for every (token, slot). ids is int32[tokens * topk], read on the device.
    /// y is [tokens * topk, N]. Trellis, suh, and svh are packed per expert.
    pub fn experts(g: *const Exl3, s: Stream, which: Expert, x: u64, ids: u64, trellis: u64, suh: u64, svh: u64, y: u64, tokens: usize, k: usize, n: usize, n_experts: usize, topk: usize, halves: u8, codebook: u8) !void {
        if (tokens == 0) return;
        if (n_experts == 0 or topk == 0) return error.Exl3Shape;
        const slots = std.math.mul(usize, tokens, topk) catch return error.Exl3Shape;
        const dims = try projection(slots, k, n, halves, codebook);
        const experts_i = std.math.cast(i32, n_experts) orelse return error.Exl3Shape;
        const topk_i = std.math.cast(i32, topk) orelse return error.Exl3Shape;
        const tokens_i = std.math.cast(i32, tokens) orelse return error.Exl3Shape;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(ids);
        a.add(trellis);
        a.add(suh);
        a.add(svh);
        a.add(y);
        a.add(tokens_i);
        a.add(dims.k);
        a.add(dims.n);
        a.add(experts_i);
        a.add(topk_i);
        a.add(dims.halves);
        a.add(dims.codebook);
        const fn_ = switch (which) {
            .gate_up => g.gate_up_fn,
            .down => g.down_fn,
        };
        try launch_.launch(fn_, .{
            .grid = .{ .x = dims.n_blocks, .y = dims.rows_u, .z = 1 },
            .block = .{ .x = 32, .y = 1, .z = 1 },
        }, s, &a);
    }

    /// out[token, n] = sum_s fp32(weights[token, s]) * fp32(y[token, s, n]), then fp16.
    pub fn combine(g: *const Exl3, s: Stream, y: u64, weights: u64, out: u64, tokens: usize, n: usize, topk: usize) !void {
        if (tokens == 0) return;
        if (n == 0 or topk == 0) return error.Exl3Shape;
        const tokens_i = std.math.cast(i32, tokens) orelse return error.Exl3Shape;
        const n_i = std.math.cast(i32, n) orelse return error.Exl3Shape;
        const topk_i = std.math.cast(i32, topk) orelse return error.Exl3Shape;
        const n_blocks: u32 = @intCast((n + 127) / 128);
        var a: launch_.Args = .{};
        a.add(y);
        a.add(weights);
        a.add(out);
        a.add(tokens_i);
        a.add(n_i);
        a.add(topk_i);
        try launch_.launch(g.combine_fn, .{
            .grid = .{ .x = n_blocks, .y = @intCast(tokens_i), .z = 1 },
            .block = .{ .x = 128, .y = 1, .z = 1 },
        }, s, &a);
    }
};

pub const Expert = enum { gate_up, down };

const Dims = struct {
    rows: i32,
    rows_u: u32,
    k: i32,
    n: i32,
    n_blocks: u32,
    halves: i32,
    codebook: i32,
};

fn projection(rows: usize, k: usize, n: usize, halves: u8, codebook: u8) error{Exl3Shape}!Dims {
    if (k == 0 or n == 0 or k % 128 != 0 or n % 128 != 0) return error.Exl3Shape;
    if (halves < 2 or halves > 16 or codebook > 2) return error.Exl3Shape;
    return .{
        .rows = std.math.cast(i32, rows) orelse return error.Exl3Shape,
        .rows_u = std.math.cast(u32, rows) orelse return error.Exl3Shape,
        .k = std.math.cast(i32, k) orelse return error.Exl3Shape,
        .n = std.math.cast(i32, n) orelse return error.Exl3Shape,
        .n_blocks = std.math.cast(u32, n / 128) orelse return error.Exl3Shape,
        .halves = halves,
        .codebook = codebook,
    };
}
