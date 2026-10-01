//! Original TensorFold GPU sampler: fp32 arithmetic and 24-bit hash uniforms.
//! Selectable separately from the CPU f64 sampler.
const std = @import("std");
const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
const Sampling = @import("sampling.zig").Sampling;

pub fn sample(k: *mx.Kernels, s: *mx.Scope, logits: mx.Array, positions: []const i32, settings: Sampling, ids: ?mx.Array) !mx.Array {
    return sampleSettings(k, s, logits, positions, &.{settings}, ids);
}

pub fn sampleRows(k: *mx.Kernels, s: *mx.Scope, logits: mx.Array, positions: []const i32, settings: []const Sampling, ids: ?mx.Array) !mx.Array {
    if (settings.len != positions.len) return error.InvalidSamplingShape;
    return sampleSettings(k, s, logits, positions, settings, ids);
}

fn sampleSettings(k: *mx.Kernels, s: *mx.Scope, logits: mx.Array, positions: []const i32, settings: []const Sampling, ids: ?mx.Array) !mx.Array {
    var any_greedy = false;
    var all_greedy = true;
    for (settings) |cfg| {
        try cfg.validate();
        any_greedy = any_greedy or cfg.temperature == 0;
        all_greedy = all_greedy and cfg.temperature == 0;
    }
    const vocab = mx.dim(logits, -1);
    if (vocab <= 0 or positions.len == 0 or mx.c.mlx_array_size(logits) != positions.len * @as(usize, @intCast(vocab))) return error.InvalidSamplingShape;
    if (ids) |mapping| if (mx.dtype(mapping) != mx.c.MLX_UINT32 or mx.c.mlx_array_size(mapping) != @as(usize, @intCast(vocab))) return error.InvalidSamplingMapping;
    for (positions) |position| if (position < 0) return error.InvalidSamplingPosition;
    const rows: i32 = @intCast(positions.len);
    const x = try s.reshape(logits, &.{ rows, vocab });
    const picked = if (any_greedy) blk: {
        const indices = try s.argmax(x);
        break :blk if (ids) |mapping| try s.take(mapping, indices, 0) else indices;
    } else null;
    if (all_greedy) return picked.?;
    const seeds = try mx.allocator.alloc(u32, positions.len * 2);
    defer mx.allocator.free(seeds);
    const configs = try mx.allocator.alloc(f32, positions.len * 4);
    defer mx.allocator.free(configs);
    const caps = try mx.allocator.alloc(u32, positions.len);
    defer mx.allocator.free(caps);
    for (caps, 0..) |*cap, i| {
        const cfg = settings[if (settings.len == 1) 0 else i];
        const greedy = cfg.temperature == 0;
        const seed: u64 = if (greedy) 0 else cfg.seed;
        seeds[2 * i] = @truncate(seed);
        seeds[2 * i + 1] = @truncate(seed >> 32);
        const values = [_]f32{ if (greedy) 1 else @floatCast(1 / @max(cfg.temperature, 1e-6)), if (greedy) 1 else @floatCast(cfg.top_p), 20, if (greedy) -std.math.inf(f32) else @floatCast(cfg.minLog()) };
        @memcpy(configs[i * 4 ..][0..4], &values);
        cap.* = if (greedy) 1 else @intCast(@min(cfg.top_k, @as(usize, @intCast(vocab))));
    }
    var inputs = [_]mx.Array{
        x,                                                  try s.data(seeds.ptr, &.{rows * 2}, mx.c.MLX_UINT32),
        try s.cast(try s.ints(positions), mx.c.MLX_UINT32), try s.data(configs.ptr, &.{ rows, 4 }, mx.f32t),
        try s.data(caps.ptr, &.{rows}, mx.c.MLX_UINT32),    ids orelse mx.empty,
    };
    const sampled = (try k.run(s, if (ids != null) src.gpu_sample_ids else src.gpu_sample, inputs[0..if (ids != null) @as(usize, 6) else 5], &.{ mx.ti("V", vocab), mx.ti("C", 1024) }, .{ 1024 * rows, 1, 1 }, .{ 1024, 1, 1 }, &.{.{ .shape = &.{rows}, .dtype = mx.c.MLX_UINT32 }}))[0];
    if (picked) |greedy_ids| {
        const greedy = try mx.allocator.alloc(bool, positions.len);
        defer mx.allocator.free(greedy);
        for (greedy, settings) |*value, cfg| value.* = cfg.temperature == 0;
        const mask = try s.data(greedy.ptr, &.{rows}, mx.c.MLX_BOOL);
        var out = mx.c.mlx_array_new();
        const rc = mx.c.mlx_where(&out, mask, greedy_ids, sampled, mx.stream);
        return s.result(rc, out);
    }
    return sampled;
}

pub fn topk(k: *mx.Kernels, s: *mx.Scope, x: mx.Array, count: i32) ![2]mx.Array {
    const vocab = mx.dim(x, -1);
    if (count < 1 or count > 64 or count > vocab) return error.InvalidTopK;
    const rows: i32 = @intCast(mx.c.mlx_array_size(x) / @as(usize, @intCast(vocab)));
    const data = try s.contiguous(try s.reshape(try s.cast(x, mx.bf16), &.{ rows, vocab }));
    const out = try k.run(s, src.radix_topk, &.{ data, try s.ints(&.{ vocab, count }) }, &.{ mx.ti("TPG", 1024), mx.ti("MAXK", 64), mx.ti("MAXT", 2048) }, .{ rows * 1024, 1, 1 }, .{ 1024, 1, 1 }, &.{ .{ .shape = &.{ rows, count }, .dtype = mx.i32t }, .{ .shape = &.{ rows, count }, .dtype = mx.f32t } });
    return .{ out[0], out[1] };
}
