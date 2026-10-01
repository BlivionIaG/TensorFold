const std = @import("std");
const mx = @import("mlx.zig");
pub const Candidate = struct { id: i32, value: f64 };
/// Same prompt-derived default as engine/exact_sampling.py, salt zero.
pub fn seedFor(tokens: []const i32) u64 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var buffer: [16]u8 = undefined;
    for (tokens, 0..) |token, i| {
        if (i != 0) hash.update(",");
        hash.update(std.fmt.bufPrint(&buffer, "{d}", .{token}) catch unreachable);
    }
    hash.update("|0");
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return std.mem.readInt(u64, digest[0..8], .little) & 0x7fffffffffffffff;
}
pub const Sampling = struct {
    metal: bool = false,
    seed: u64 = 0,
    temperature: f64 = 1,
    top_k: usize = 20,
    top_p: f64 = 0.95,
    min_p: f64 = 0,
    pub fn validate(s: Sampling) !void {
        if (!std.math.isFinite(s.temperature) or s.temperature < 0 or !std.math.isFinite(s.top_p) or s.top_p <= 0 or s.top_p > 1) return error.InvalidSampling;
        if (!std.math.isFinite(s.min_p) or s.min_p < 0 or s.min_p > 1) return error.InvalidSampling;
    }
    pub fn minLog(s: Sampling) f64 {
        return if (s.min_p > 0) @log(s.min_p) else -std.math.inf(f64);
    }
    pub fn choose(s: Sampling, sorted: []const Candidate, position: u64) i32 {
        if (s.temperature == 0) return sorted[0].id;
        const n = if (s.top_k == 0) sorted.len else @min(s.top_k, sorted.len);
        const temp = @max(s.temperature, 1e-6);
        const max = sorted[0].value / temp;
        const floor = max + s.minLog();
        var total: f64 = 0;
        for (sorted[0..n]) |v| total += @exp(v.value / temp - max);
        var cumulative: f64 = 0;
        var best: f64 = -std.math.inf(f64);
        var chosen = sorted[0].id;
        for (sorted[0..n]) |v| {
            if (v.value / temp < floor) break;
            const score = v.value / temp + noise(s.seed, position, @intCast(v.id));
            if (score > best) {
                best = score;
                chosen = v.id;
            }
            cumulative += @exp(v.value / temp - max) / total;
            if (cumulative >= s.top_p) break;
        }
        return chosen;
    }
};
fn mix(v: u64) u64 {
    var x = v;
    x ^= x >> 30;
    x *%= 0xbf58476d1ce4e5b9;
    x ^= x >> 27;
    x *%= 0x94d049bb133111eb;
    return x ^ (x >> 31);
}
pub fn uniform(seed: u64, position: u64, id: u64) f64 {
    var x = mix(seed +% 0x9e3779b97f4a7c15);
    x = mix(x ^ (position *% 0xd1b54a32d192ed03));
    x = mix(x ^ id);
    return @as(f64, @floatFromInt(x >> 11)) * 0x1p-53 + 0x1p-54;
}
pub fn noise(seed: u64, position: u64, id: u64) f64 {
    return -@log(-@log(uniform(seed, position, id)));
}
pub fn less(_: void, a: Candidate, b: Candidate) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}
pub fn top(allocator: std.mem.Allocator, values: []const f32, n: usize) ![]Candidate {
    // Keep a sorted bounded set. k=20 is the model default; k=0 uses a full sort.
    const count = if (n == 0) values.len else @min(n, values.len);
    const result = try allocator.alloc(Candidate, count);
    if (n == 0) {
        for (values, 0..) |v, i| result[i] = .{ .id = @intCast(i), .value = v };
        std.mem.sort(Candidate, result, {}, less);
        return result;
    }
    var used: usize = 0;
    for (values, 0..) |v, i| {
        const candidate = Candidate{ .id = @intCast(i), .value = v };
        if (used == count and !less({}, candidate, result[count - 1])) continue;
        var at = @min(used, count - 1);
        while (at > 0 and less({}, candidate, result[at - 1])) : (at -= 1) {
            result[at] = result[at - 1];
        }
        result[at] = candidate;
        used = @min(used + 1, count);
    }
    return result;
}
pub fn rows(k: *mx.Kernels, s: *mx.Scope, logits: mx.Array, positions: []const i32, settings: Sampling) ![]i32 {
    return rowsMapped(k, s, logits, positions, settings, null);
}

pub fn streamRowsMapped(k: *mx.Kernels, s: *mx.Scope, logits: mx.Array, positions: []const i32, settings: []const Sampling, mapping: ?mx.Array) ![]i32 {
    if (positions.len == 0 or positions.len > 128 or positions.len != settings.len or mx.shape(logits).len != 2 or mx.dim(logits, 0) != positions.len) return error.InvalidSamplingRows;
    for (positions, settings) |position, cfg| {
        if (position < 0) return error.InvalidSamplingPosition;
        try cfg.validate();
    }
    if (mapping) |ids| {
        if (mx.dtype(ids) != mx.c.MLX_UINT32 or mx.c.mlx_array_size(ids) != @as(usize, @intCast(mx.dim(logits, -1)))) return error.InvalidSamplingMapping;
    }
    var gpu_only = true;
    for (settings) |cfg| gpu_only = gpu_only and (cfg.metal or cfg.temperature == 0);
    if (gpu_only) {
        const selected = try @import("gpu_sampling.zig").sampleRows(k, s, logits, positions, settings, mapping);
        try mx.eval(selected);
        const output = try mx.allocator.alloc(i32, positions.len);
        for (output, mx.c.mlx_array_data_uint32(selected)[0..positions.len]) |*token, id| token.* = @intCast(id);
        return output;
    }
    var arrays: [129]mx.Array = undefined;
    var starts: [128]usize = undefined;
    var ends: [128]usize = undefined;
    var count: usize = 0;
    var cpu = false;
    var begin: usize = 0;
    while (begin < positions.len) {
        var end = begin + 1;
        const cfg = settings[begin];
        while (end < positions.len and (if (cfg.metal) settings[end].metal else std.meta.eql(cfg, settings[end]))) : (end += 1) {}
        const part = if (begin == 0 and end == positions.len) logits else try s.slice(logits, 0, @intCast(begin), @intCast(end));
        arrays[count] = if (cfg.metal) try @import("gpu_sampling.zig").sampleRows(k, s, part, positions[begin..end], settings[begin..end], mapping) else if (cfg.temperature == 0) blk: {
            const picked = try s.argmax(part);
            break :blk if (mapping) |ids| try s.take(ids, picked, 0) else picked;
        } else blk: {
            cpu = true;
            break :blk try s.contiguous(try s.cast(part, mx.f32t));
        };
        starts[count] = begin;
        ends[count] = end;
        count += 1;
        begin = end;
    }
    var evaluated = count;
    if (cpu) if (mapping) |ids| {
        arrays[evaluated] = ids;
        evaluated += 1;
    };
    try mx.evalMany(arrays[0..evaluated], false);
    const width: usize = @intCast(mx.dim(logits, -1));
    const id_map: ?[]const u32 = if (cpu and mapping != null) mx.c.mlx_array_data_uint32(mapping.?)[0..width] else null;
    const output = try mx.allocator.alloc(i32, positions.len);
    errdefer mx.allocator.free(output);
    for (arrays[0..count], starts[0..count], ends[0..count]) |array, start, end| {
        const cfg = settings[start];
        if (cfg.metal or cfg.temperature == 0) {
            for (output[start..end], mx.c.mlx_array_data_uint32(array)[0 .. end - start]) |*token, id| token.* = @intCast(id);
        } else {
            const values = mx.c.mlx_array_data_float32(array);
            for (output[start..end], positions[start..end], 0..) |*token, position, row| {
                const candidates = try top(mx.allocator, values[row * width ..][0..width], cfg.top_k);
                defer mx.allocator.free(candidates);
                if (id_map) |ids| for (candidates) |*candidate| {
                    candidate.id = @intCast(ids[@intCast(candidate.id)]);
                };
                token.* = cfg.choose(candidates, @intCast(position));
            }
        }
    }
    return output;
}
/// Mapping must be sorted ascending, preserving token-ID tie ordering.
pub fn rowsMapped(k: *mx.Kernels, s: *mx.Scope, logits: mx.Array, positions: []const i32, settings: Sampling, mapping: ?mx.Array) ![]i32 {
    if (mapping) |ids| {
        if (mx.dtype(ids) != mx.c.MLX_UINT32 or mx.c.mlx_array_size(ids) != @as(usize, @intCast(mx.dim(logits, -1)))) return error.InvalidSamplingMapping;
    }
    const out = try mx.allocator.alloc(i32, positions.len);
    errdefer mx.allocator.free(out);
    if (settings.metal) {
        const ids = try @import("gpu_sampling.zig").sample(k, s, logits, positions, settings, mapping);
        try mx.eval(ids);
        for (out, 0..) |*v, i| v.* = @intCast(mx.c.mlx_array_data_uint32(ids)[i]);
        return out;
    }
    if (settings.temperature == 0) {
        const picked = try s.argmax(logits);
        const ids = if (mapping) |ids| try s.take(ids, picked, 0) else picked;
        try mx.eval(ids);
        for (out, 0..) |*v, i| v.* = @intCast(mx.c.mlx_array_data_uint32(ids)[i]);
        return out;
    }
    const f = try s.cast(logits, mx.f32t);
    try mx.eval(f);
    const width: usize = @intCast(mx.dim(f, -1));
    const ptr = mx.c.mlx_array_data_float32(f);
    const id_map = if (mapping) |ids| blk: {
        try mx.eval(ids);
        break :blk mx.c.mlx_array_data_uint32(ids)[0..width];
    } else null;
    for (positions, 0..) |pos, i| {
        const candidates = try top(mx.allocator, ptr[i * width ..][0..width], settings.top_k);
        defer mx.allocator.free(candidates);
        if (id_map) |ids| for (candidates) |*candidate| {
            candidate.id = @intCast(ids[@intCast(candidate.id)]);
        };
        out[i] = settings.choose(candidates, @intCast(pos));
    }
    return out;
}

pub fn checkStreams(k: *mx.Kernels) !void {
    var scope = mx.Scope{};
    defer scope.deinit();
    const count = 12;
    const width = 2053;
    var values: [count * width]f32 = undefined;
    var mapped: [width]u32 = undefined;
    for (&mapped, 0..) |*id, i| id.* = @intCast(101 + 7 * i);
    for (0..count) |row| {
        for (0..width) |col| values[row * width + col] = @as(f32, @floatFromInt(@as(i32, @intCast((37 * col + 17 * row) % 257)) - 128)) / 23;
        values[row * width + 1] = 12;
        values[row * width + 1027] = 12;
    }
    const cpu_greedy = Sampling{ .temperature = 0 };
    const metal_greedy = Sampling{ .metal = true, .temperature = 0 };
    const metal_sample = Sampling{ .metal = true, .seed = 1234, .temperature = 0.8, .top_k = 17, .top_p = 0.87, .min_p = 0.03 };
    const metal_other = Sampling{ .metal = true, .seed = 0xffffeeee12345678, .temperature = 1.2, .top_k = 0, .top_p = 0.96, .min_p = 0.02 };
    const cpu_sample = Sampling{ .seed = 0xabcdef0123456789, .temperature = 1.1, .top_k = 7, .top_p = 0.8, .min_p = 0.05 };
    const cpu_other = Sampling{ .seed = 991, .temperature = 0.35, .top_k = 0, .top_p = 1 };
    const settings = [_]Sampling{ cpu_greedy, cpu_greedy, metal_greedy, metal_greedy, metal_sample, metal_sample, metal_other, cpu_sample, cpu_sample, cpu_other, metal_sample, cpu_greedy };
    const positions = [_]i32{ 7, 17, 4, 99, 3, 1, 1007, 1007, 262143, 9, 43, 14 };
    const original = try scope.data(&values, &.{ count, width }, mx.f32t);
    const mapping = try scope.data(&mapped, &.{width}, mx.c.MLX_UINT32);
    const lazy_mapping = try scope.cast(try scope.cast(mapping, mx.i32t), mx.c.MLX_UINT32);
    for ([_]mx.c.mlx_dtype{ mx.f32t, mx.bf16 }) |dtype| {
        const logits = try scope.cast(original, dtype);
        for ([_]?mx.Array{ null, mapping, lazy_mapping }) |ids| {
            const actual = try streamRowsMapped(k, &scope, logits, &positions, &settings, ids);
            defer mx.allocator.free(actual);
            for (settings, 0..) |cfg, row| {
                const expected = try rowsMapped(k, &scope, try scope.slice(logits, 0, @intCast(row), @intCast(row + 1)), positions[row..][0..1], cfg, ids);
                defer mx.allocator.free(expected);
                try std.testing.expectEqual(expected[0], actual[row]);
                if (cfg.temperature == 0) try std.testing.expectEqual(@as(i32, if (ids != null) @intCast(mapped[1]) else 1), actual[row]);
            }
            var reversed_settings: [count]Sampling = undefined;
            var reversed_positions: [count]i32 = undefined;
            var reversed_rows: [count]i32 = undefined;
            for (0..count) |i| {
                reversed_settings[i] = settings[count - i - 1];
                reversed_positions[i] = positions[count - i - 1];
                reversed_rows[i] = @intCast(count - i - 1);
            }
            const reversed = try streamRowsMapped(k, &scope, try scope.take(logits, try scope.ints(&reversed_rows), 0), &reversed_positions, &reversed_settings, ids);
            defer mx.allocator.free(reversed);
            for (reversed, 0..) |token, i| try std.testing.expectEqual(actual[count - i - 1], token);

            var cpu_settings: [count]Sampling = undefined;
            for (&cpu_settings, 0..) |*cfg, row| cfg.* = if (row % 3 == 2) cpu_other else cpu_sample;
            const strided = try scope.slice(try scope.contiguous(try scope.cat(&.{ logits, try scope.zeros(&.{ count, 3 }, dtype) }, 1)), 1, 0, width);
            const cpu_actual = try streamRowsMapped(k, &scope, strided, &positions, &cpu_settings, ids);
            defer mx.allocator.free(cpu_actual);
            for (cpu_settings, 0..) |cfg, row| {
                const expected = try rowsMapped(k, &scope, try scope.slice(strided, 0, @intCast(row), @intCast(row + 1)), positions[row..][0..1], cfg, ids);
                defer mx.allocator.free(expected);
                try std.testing.expectEqual(expected[0], cpu_actual[row]);
            }
            for ([_]bool{ false, true }) |all_greedy| {
                var gpu_settings: [count]Sampling = undefined;
                for (&gpu_settings, settings, 0..) |*cfg, original_cfg, row| {
                    cfg.* = original_cfg;
                    cfg.seed +%= row;
                    if (all_greedy) cfg.temperature = 0 else if (cfg.temperature != 0) cfg.metal = true;
                }
                const gpu_actual = try streamRowsMapped(k, &scope, strided, &positions, &gpu_settings, ids);
                defer mx.allocator.free(gpu_actual);
                for (gpu_settings, 0..) |cfg, row| {
                    const expected = try rowsMapped(k, &scope, try scope.slice(strided, 0, @intCast(row), @intCast(row + 1)), positions[row..][0..1], cfg, ids);
                    defer mx.allocator.free(expected);
                    try std.testing.expectEqual(expected[0], gpu_actual[row]);
                }
            }
        }
    }
    try std.testing.expectError(error.InvalidSamplingRows, streamRowsMapped(k, &scope, original, &.{}, &.{}, null));
    try std.testing.expectError(error.InvalidSamplingRows, streamRowsMapped(k, &scope, original, &positions, settings[0 .. count - 1], null));
    try std.testing.expectError(error.InvalidSamplingRows, streamRowsMapped(k, &scope, original, positions[0 .. count - 1], settings[0 .. count - 1], null));
    const oversized_positions: [129]i32 = @splat(1);
    const oversized_settings: [129]Sampling = @splat(cpu_greedy);
    try std.testing.expectError(error.InvalidSamplingRows, streamRowsMapped(k, &scope, original, &oversized_positions, &oversized_settings, null));
    try std.testing.expectError(error.InvalidSamplingRows, streamRowsMapped(k, &scope, try scope.zeros(&.{count}, mx.f32t), &positions, &settings, null));
    try std.testing.expectError(error.InvalidSamplingRows, streamRowsMapped(k, &scope, try scope.reshape(original, &.{ count, 1, width }), &positions, &settings, null));
    var invalid_positions = positions;
    invalid_positions[0] = -1;
    try std.testing.expectError(error.InvalidSamplingPosition, streamRowsMapped(k, &scope, original, &invalid_positions, &settings, null));
    var invalid_settings = settings;
    invalid_settings[0].top_p = 0;
    try std.testing.expectError(error.InvalidSampling, streamRowsMapped(k, &scope, original, &positions, &invalid_settings, null));
    try std.testing.expectError(error.InvalidSamplingMapping, streamRowsMapped(k, &scope, original, &positions, &settings, try scope.cast(mapping, mx.i32t)));
    try std.testing.expectError(error.InvalidSamplingMapping, streamRowsMapped(k, &scope, original, &positions, &settings, try scope.slice(mapping, 0, 0, width - 1)));
    std.debug.print("PASS: CPU-only and mixed CPU/Metal shared sampling, strided rows, mapped IDs, reordered positions and invalid geometry.\n", .{});
}
test "sampling is position keyed, greedy ties use token id" {
    const values = [_]f32{ 1, 3, 3, -1 };
    const candidates = try top(std.testing.allocator, &values, 3);
    defer std.testing.allocator.free(candidates);
    try std.testing.expectEqual(@as(i32, 1), (Sampling{ .temperature = 0 }).choose(candidates, 3));
    const s = Sampling{ .seed = 1234 };
    try std.testing.expectEqual(s.choose(candidates, 4), s.choose(candidates, 4));
    try std.testing.expect(uniform(1234, 4, 2) > 0 and uniform(1234, 4, 2) < 1);
}
