const std = @import("std");
const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
const A = mx.Array;
const ti = mx.ti;

pub const Geometry = struct {
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    values_are_keys: bool,
    pub fn validate(g: Geometry) !void {
        if (g.heads <= 0 or g.kv_heads <= 0 or g.head_dim <= 0 or g.head_dim > 512 or @mod(g.head_dim, 64) != 0 or @mod(g.heads, g.kv_heads) != 0) return error.InvalidGemmaGeometry;
    }
    fn outputs(g: Geometry, rows: i32, shapes: *[3][3]i32) [3]mx.Output {
        shapes.* = .{ .{ rows, g.heads, g.head_dim }, .{ g.kv_heads, rows, g.head_dim }, .{ g.kv_heads, rows, g.head_dim } };
        return .{ .{ .shape = &shapes[0] }, .{ .shape = &shapes[1] }, .{ .shape = &shapes[2] } };
    }
};

pub fn qkv(k: *mx.Kernels, s: *mx.Scope, g: Geometry, x: A, weight: [3]A, qw: A, kw: A, inv: A, positions: A, eps: A, group: i32) ![3]A {
    try g.validate();
    const rows = mx.dim(x, 0);
    const width = mx.dim(x, -1);
    if (rows < 1 or rows > 128 or @mod(width, 64) != 0 or (group != 32 and group != 64 and group != 128)) return error.InvalidGemmaProjection;
    const slots = g.heads + g.kv_heads * @as(i32, if (g.values_are_keys) 1 else 2);
    if (!std.mem.eql(i32, mx.shape(weight[0]), &.{ slots * g.head_dim, @divExact(width, 8) }) or !std.mem.eql(i32, mx.shape(weight[1]), &.{ slots * g.head_dim, @divExact(width, group) }) or !std.mem.eql(i32, mx.shape(weight[1]), mx.shape(weight[2]))) return error.InvalidGemmaProjection;
    if (mx.c.mlx_array_size(qw) != g.head_dim or mx.c.mlx_array_size(kw) != g.head_dim or mx.c.mlx_array_size(inv) != @divExact(g.head_dim, 2) or mx.c.mlx_array_size(positions) < rows) return error.InvalidGemmaGeometry;
    var shapes: [3][3]i32 = undefined;
    const outputs = g.outputs(rows, &shapes);
    const result = try k.run(s, src.gemma_qkv_rows, &.{ x, weight[0], weight[1], weight[2], qw, kw, inv, positions, eps }, &.{ ti("DH", g.head_dim), ti("NQ", g.heads), ti("NK", g.kv_heads), ti("VK", @intFromBool(g.values_are_keys)), ti("KD", width), ti("GS", group), ti("RPS", 4), ti("SG", 32) }, .{ 1024 * slots, rows, 1 }, .{ 1024, 1, 1 }, &outputs);
    return result[0..3].*;
}

pub const Rows = struct {
    positions: A,
    lows: A,
    meta: A,
    count: i32,
    dims: i32,
    first_position: i32,
    ring: i32,
    chunks: i32,

    pub fn init(s: *mx.Scope, positions: []const i32, window: i32, ring: i32, dims: i32) !Rows {
        if (positions.len < 1 or positions.len > 128 or window < 0 or ring < 0 or (ring > 0 and ring <= window)) return error.InvalidGemmaAttention;
        if (dims < 32 or dims > 512 or @mod(dims, 32) != 0) return error.InvalidGemmaGeometry;
        for (positions, 0..) |position, i| if (position < 0 or position > 262144 or position != positions[0] + @as(i32, @intCast(i))) return error.InvalidGemmaPositions;
        const chunk: i32 = if (dims == 256) 128 else 64;
        var lows: [128]i32 = undefined;
        for (positions, 0..) |position, i| lows[i] = if (window > 0) @max(0, position - window + 1) else 0;
        const first = @divTrunc(lows[0], chunk);
        const chunks = @divTrunc(positions[positions.len - 1], chunk) - first + 1;
        const count: i32 = @intCast(positions.len);
        return .{
            .positions = try paddedInts(s, positions),
            .lows = try paddedInts(s, lows[0..positions.len]),
            .meta = try paddedInts(s, &.{ first, chunks, ring, count }),
            .count = count,
            .dims = dims,
            .first_position = positions[0],
            .ring = ring,
            .chunks = chunks,
        };
    }
};

pub fn attention(k: *mx.Kernels, s: *mx.Scope, q: A, keys: A, values: A, new_keys: A, new_values: A, positions: []const i32, window: i32, ring: i32, scale: f32) !A {
    const rows = try Rows.init(s, positions, window, ring, mx.dim(q, 2));
    return attentionRows(k, s, q, keys, values, new_keys, new_values, rows, scale);
}

pub fn attentionRows(k: *mx.Kernels, s: *mx.Scope, q: A, keys: A, values: A, new_keys: A, new_values: A, prepared: Rows, scale: f32) !A {
    const plan = try AttentionPlan.init(q, keys, values, new_keys, new_values, prepared, scale);
    return plan.apply(k, s, .{ q, keys, values, new_keys, new_values, prepared.positions, prepared.lows, prepared.meta });
}

// One exact shape per closure bounds MLX's internal specialization cache.
pub const Attention = struct {
    closure: mx.c.mlx_closure = .{ .ctx = null },
    signature: ?Signature = null,

    const Input = struct {
        dims: [4]i32 = @splat(0),
        rank: usize,
        dtype: mx.c.mlx_dtype,

        fn shape(input: *const Input) []const i32 {
            return input.dims[0..input.rank];
        }
    };
    const Signature = struct {
        inputs: [8]Input,
        chunks: i32,
        scale: u32,

        fn init(inputs: [8]A, chunks: i32, scale: f32) !Signature {
            var key = Signature{ .inputs = undefined, .chunks = chunks, .scale = @bitCast(scale) };
            for (inputs, &key.inputs) |value, *input| {
                const shape = mx.shape(value);
                if (shape.len > 4) return error.InvalidGemmaAttention;
                input.* = .{ .rank = shape.len, .dtype = mx.dtype(value) };
                @memcpy(input.dims[0..shape.len], shape);
            }
            return key;
        }
    };
    const Payload = struct {
        kernels: mx.Kernels,
        plan: AttentionPlan,

        fn destroy(raw: ?*anyopaque) callconv(.c) void {
            const p: *Payload = @ptrCast(@alignCast(raw.?));
            p.kernels.deinit();
            mx.allocator.destroy(p);
        }
        fn callback(out: [*c]mx.c.mlx_vector_array, ins: mx.c.mlx_vector_array, raw: ?*anyopaque) callconv(.c) c_int {
            const p: *Payload = @ptrCast(@alignCast(raw.?));
            return p.graph(out, ins) catch -1;
        }
        fn graph(p: *Payload, out: [*c]mx.c.mlx_vector_array, ins: mx.c.mlx_vector_array) !c_int {
            var scope = mx.Scope{};
            defer scope.deinit();
            var inputs: [8]A = undefined;
            for (&inputs, 0..) |*input, i| {
                var value = mx.c.mlx_array_new();
                const rc = mx.c.mlx_vector_array_get(&value, ins, i);
                input.* = try scope.result(rc, value);
            }
            const result = [_]A{try p.plan.apply(&p.kernels, &scope, inputs)};
            return mx.c.mlx_vector_array_set_data(out, &result, result.len);
        }
    };

    pub fn deinit(a: *Attention) void {
        if (a.closure.ctx != null) _ = mx.c.mlx_closure_free(a.closure);
        a.* = .{};
    }

    pub fn apply(a: *Attention, k: *mx.Kernels, s: *mx.Scope, q: A, keys: A, values: A, new_keys: A, new_values: A, prepared: Rows, scale: f32) !A {
        const inputs = [_]A{ q, keys, values, new_keys, new_values, prepared.positions, prepared.lows, prepared.meta };
        const signature = try Signature.init(inputs, prepared.chunks, scale);
        var result: [1]A = undefined;
        if (a.signature) |key| {
            if (std.meta.eql(key, signature)) {
                if (prepared.count != key.inputs[0].dims[0]) return error.InvalidGemmaAttention;
                if (prepared.dims != key.inputs[0].dims[2]) return error.InvalidGemmaGeometry;
                if (key.inputs[1].dims[2] < (if (prepared.ring > 0) prepared.ring else prepared.first_position)) return error.InvalidGemmaAttention;
                try k.call(s, a.closure, &inputs, &result);
                return result[0];
            }
        }
        const plan = try AttentionPlan.fromShapes(signature.inputs[0].shape(), signature.inputs[1].shape(), signature.inputs[2].shape(), signature.inputs[3].shape(), signature.inputs[4].shape(), prepared, scale);
        const payload = try mx.allocator.create(Payload);
        payload.* = .{ .kernels = mx.Kernels.init(), .plan = plan };
        const fun = mx.c.mlx_closure_new_func_payload(Payload.callback, payload, Payload.destroy);
        if (fun.ctx == null) {
            Payload.destroy(payload);
            return error.MlxFailure;
        }
        defer _ = mx.c.mlx_closure_free(fun);
        var replacement = Attention{};
        errdefer replacement.deinit();
        try mx.check(mx.c.mlx_compile(&replacement.closure, fun, false));
        try k.call(s, replacement.closure, &inputs, &result);
        replacement.signature = signature;
        a.deinit();
        a.* = replacement;
        return result[0];
    }
};

const AttentionPlan = struct {
    rows: i32,
    heads: i32,
    dims: i32,
    kv_heads: i32,
    chunks: i32,
    scale: f32,

    fn init(q: A, keys: A, values: A, new_keys: A, new_values: A, prepared: Rows, scale: f32) !AttentionPlan {
        return fromShapes(mx.shape(q), mx.shape(keys), mx.shape(values), mx.shape(new_keys), mx.shape(new_values), prepared, scale);
    }

    fn fromShapes(q: []const i32, keys: []const i32, values: []const i32, new_keys: []const i32, new_values: []const i32, prepared: Rows, scale: f32) !AttentionPlan {
        if (q.len < 3 or keys.len < 4) return error.InvalidGemmaAttention;
        const rows = q[0];
        const heads = q[1];
        const dims = q[2];
        const kv_heads = keys[1];
        if (prepared.count != rows or rows < 1 or rows > 128 or !std.math.isFinite(scale)) return error.InvalidGemmaAttention;
        if (dims != prepared.dims or dims < 32 or dims > 512 or @mod(dims, 32) != 0 or kv_heads <= 0 or @mod(heads, kv_heads) != 0) return error.InvalidGemmaGeometry;
        if (!std.mem.eql(i32, keys, values) or !std.mem.eql(i32, new_keys, &.{ kv_heads, rows, dims }) or !std.mem.eql(i32, new_values, new_keys)) return error.InvalidGemmaAttention;
        if (keys[0] != 1 or keys[3] != dims or keys[2] < (if (prepared.ring > 0) prepared.ring else prepared.first_position)) return error.InvalidGemmaAttention;
        const split: i32 = if (dims == 512) 1 else 4;
        const group = @divExact(heads, kv_heads);
        if (32 * group * split > 1024) return error.InvalidGemmaGeometry;
        return .{ .rows = rows, .heads = heads, .dims = dims, .kv_heads = kv_heads, .chunks = prepared.chunks, .scale = scale };
    }

    fn apply(p: AttentionPlan, k: *mx.Kernels, s: *mx.Scope, inputs: [8]A) !A {
        const chunk: i32 = if (p.dims == 256) 128 else 64;
        const split: i32 = if (p.dims == 512) 1 else 4;
        const group = @divExact(p.heads, p.kv_heads);
        const slots = p.heads * p.rows * p.chunks;
        const partials = try k.run(s, src.gemma_attention_partial, &inputs, &.{ ti("D", p.dims), ti("G", group), ti("HK", p.kv_heads), ti("CK", chunk), ti("S", split), ti("BLK", 4), ti("SCALE_BITS", @bitCast(p.scale)) }, .{ 32 * group * split * p.chunks, p.kv_heads, p.rows }, .{ 32 * group * split, 1, 1 }, &.{ .{ .shape = &.{@max(slots, 8)}, .dtype = mx.f32t }, .{ .shape = &.{@max(slots, 8)}, .dtype = mx.f32t }, .{ .shape = &.{ slots, p.dims }, .dtype = mx.f32t } });
        return (try k.run(s, src.gemma_attention_merge, &.{ partials[0], partials[1], partials[2], inputs[7] }, &.{ ti("D", p.dims), ti("H", p.heads) }, .{ 32, p.heads, p.rows }, .{ 32, 1, 1 }, &.{.{ .shape = &.{ p.rows, p.heads, p.dims } }}))[0];
    }
};

pub fn route(k: *mx.Kernels, s: *mx.Scope, logits: A, per_expert_scale: A, top: i32) ![2]A {
    const rows = mx.dim(logits, 0);
    const experts = mx.dim(logits, 1);
    if (rows < 1 or rows > 128 or experts < 1 or experts > 1024 or top < 1 or top > experts or mx.c.mlx_array_size(per_expert_scale) != experts) return error.InvalidGemmaRouting;
    const out = try k.run(s, src.gemma_route, &.{ logits, per_expert_scale }, &.{ ti("NE", experts), ti("K", top) }, .{ 32 * rows, 1, 1 }, .{ 32, 1, 1 }, &.{ .{ .shape = &.{@max(rows * top, 8)}, .dtype = mx.c.MLX_UINT32 }, .{ .shape = &.{@max(rows * top, 8)} } });
    return out[0..2].*;
}

pub fn router(k: *mx.Kernels, s: *mx.Scope, x: A, weight: [3]A, group: i32) !A {
    const rows = mx.dim(x, 0);
    const width = mx.dim(x, 1);
    const experts = mx.dim(weight[0], 0);
    if ((group != 32 and group != 64 and group != 128) or rows < 1 or rows > 128 or @mod(experts, 4) != 0 or @mod(width, group) != 0) return error.InvalidGemmaProjection;
    if (!std.mem.eql(i32, mx.shape(weight[0]), &.{ experts, @divExact(width, 4) }) or !std.mem.eql(i32, mx.shape(weight[1]), &.{ experts, @divExact(width, group) }) or !std.mem.eql(i32, mx.shape(weight[1]), mx.shape(weight[2]))) return error.InvalidGemmaProjection;
    return (try k.run(s, src.gemma_router, &.{ x, weight[0], weight[1], weight[2] }, &.{ ti("K", width), ti("N", experts), ti("GS", group), ti("SG", 4), ti("RPS", 1) }, .{ @divExact(experts, 4) * 128, rows, 1 }, .{ 128, 1, 1 }, &.{.{ .shape = &.{ rows, experts } }}))[0];
}

pub fn gateUp(k: *mx.Kernels, s: *mx.Scope, x: A, ids: A, top: i32, gate: [3]A, up: [3]A, group: i32) !A {
    const rows = mx.dim(x, 0);
    const width = mx.dim(x, 1);
    const hidden = mx.dim(gate[0], 1);
    try expertGeometry(gate, width, hidden, group);
    try expertGeometry(up, width, hidden, group);
    if (mx.dim(gate[0], 0) != mx.dim(up[0], 0) or rows < 1 or rows > 128 or top < 1 or top > mx.dim(gate[0], 0) or mx.c.mlx_array_size(ids) < rows * top) return error.InvalidGemmaRouting;
    return (try k.run(s, src.gemma_expert_gateup, &.{ x, ids, gate[0], gate[1], gate[2], up[0], up[1], up[2] }, &.{ ti("K", width), ti("N", hidden), ti("TOPK", top), ti("GS", group), ti("SG", 2), ti("RPS", 4) }, .{ 64, @divExact(hidden, 8), rows * top }, .{ 64, 1, 1 }, &.{.{ .shape = &.{ rows * top, hidden } }}))[0];
}

pub fn down(k: *mx.Kernels, s: *mx.Scope, act: A, ids: A, weights: A, top: i32, projection: [3]A, group: i32) !A {
    const hidden = mx.dim(act, 1);
    const width = mx.dim(projection[0], 1);
    if (top < 1 or top > mx.dim(projection[0], 0) or @mod(mx.dim(act, 0), top) != 0 or mx.c.mlx_array_size(ids) < mx.dim(act, 0) or mx.c.mlx_array_size(weights) < mx.dim(act, 0)) return error.InvalidGemmaRouting;
    try expertGeometry(projection, hidden, width, group);
    const rows = @divExact(mx.dim(act, 0), top);
    if (rows < 1 or rows > 128) return error.InvalidGemmaRouting;
    return (try k.run(s, src.gemma_expert_down, &.{ act, ids, weights, projection[0], projection[1], projection[2] }, &.{ ti("NI", hidden), ti("D", width), ti("TOPK", top), ti("GS", group) }, .{ 256, @divExact(width, 8), rows }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ rows, width } }}))[0];
}

fn expertGeometry(weight: [3]A, input: i32, output: i32, group: i32) !void {
    if ((group != 32 and group != 64 and group != 128) or input <= 0 or output <= 0 or @mod(input, 64) != 0 or @mod(input, group) != 0 or @mod(output, 8) != 0) return error.InvalidGemmaProjection;
    const experts = mx.dim(weight[0], 0);
    if (experts < 1 or !std.mem.eql(i32, mx.shape(weight[0]), &.{ experts, output, @divExact(input, 8) }) or !std.mem.eql(i32, mx.shape(weight[1]), &.{ experts, output, @divExact(input, group) }) or !std.mem.eql(i32, mx.shape(weight[1]), mx.shape(weight[2]))) return error.InvalidGemmaProjection;
}

pub fn paddedInts(s: *mx.Scope, values: []const i32) !A {
    if (values.len >= 8) return s.ints(values);
    var data: [8]i32 = @splat(0);
    @memcpy(data[0..values.len], values);
    return s.ints(&data);
}

test "Gemma geometry rejects unsupported heads before a GPU launch" {
    try (Geometry{ .heads = 16, .kv_heads = 8, .head_dim = 256, .values_are_keys = false }).validate();
    try (Geometry{ .heads = 16, .kv_heads = 2, .head_dim = 512, .values_are_keys = true }).validate();
    try std.testing.expectError(error.InvalidGemmaGeometry, (Geometry{ .heads = 16, .kv_heads = 3, .head_dim = 256, .values_are_keys = false }).validate());
    try std.testing.expectError(error.InvalidGemmaGeometry, (Geometry{ .heads = 16, .kv_heads = 0, .head_dim = 256, .values_are_keys = false }).validate());
}
