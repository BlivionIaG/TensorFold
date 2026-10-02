const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const gemma = @import("gemma.zig");
const c = mx.c;
const A = mx.Array;
const activations = @import("prefill_ops.zig");

pub const Ops = struct {
    layers: [30]?Layer = @splat(null),
    fronts: [30]c.mlx_closure = @splat(.{ .ctx = null }),
    backs: [30]c.mlx_closure = @splat(.{ .ctx = null }),

    pub fn deinit(o: *Ops) void {
        for (o.fronts) |fun| if (fun.ctx != null) {
            _ = c.mlx_closure_free(fun);
        };
        for (o.backs) |fun| if (fun.ctx != null) {
            _ = c.mlx_closure_free(fun);
        };
        o.* = .{};
    }

    fn layer(o: *Ops, m: *gemma.Model, i: usize) !Layer {
        if (o.layers[i]) |prepared| return prepared;
        const prepared = try Layer.init(m, i);
        o.layers[i] = prepared;
        return prepared;
    }

    fn closure(o: *Ops, m: *gemma.Model, i: usize, kind: Payload.Kind) !c.mlx_closure {
        const slot = if (kind == .front) &o.fronts[i] else &o.backs[i];
        if (slot.ctx != null) return slot.*;
        const prepared = try o.layer(m, i);
        const payload = try mx.allocator.create(Payload);
        payload.* = .{ .layer = prepared, .kind = kind };
        const fun = c.mlx_closure_new_func_payload(Payload.callback, payload, Payload.destroy);
        if (fun.ctx == null) {
            Payload.destroy(payload);
            return error.MlxFailure;
        }
        defer _ = c.mlx_closure_free(fun);
        var compiled: c.mlx_closure = .{ .ctx = null };
        errdefer if (compiled.ctx != null) {
            _ = c.mlx_closure_free(compiled);
        };
        try mx.check(c.mlx_compile(&compiled, fun, false));
        slot.* = compiled;
        return compiled;
    }

    fn front(o: *Ops, m: *gemma.Model, s: *mx.Scope, i: usize, h: A) ![3]A {
        if (mx.dim(h, 1) > gemma.Model.max_decode_rows) return (try o.layer(m, i)).front(s, h);
        var result: [3]A = undefined;
        try m.kernels.call(s, try o.closure(m, i, .front), &.{h}, &result);
        return result;
    }

    fn back(o: *Ops, m: *gemma.Model, s: *mx.Scope, i: usize, h: A, attended: A) !A {
        if (mx.dim(h, 1) > gemma.Model.max_decode_rows) return (try o.layer(m, i)).back(false, s, h, attended, &m.activations);
        var result: [1]A = undefined;
        try m.kernels.call(s, try o.closure(m, i, .back), &.{ h, attended }, &result);
        return result[0];
    }
};

const Payload = struct {
    layer: Layer,
    kind: Kind,
    const Kind = enum { front, back };

    fn destroy(raw: ?*anyopaque) callconv(.c) void {
        mx.allocator.destroy(@as(*Payload, @ptrCast(@alignCast(raw.?))));
    }
    fn callback(out: [*c]c.mlx_vector_array, ins: c.mlx_vector_array, raw: ?*anyopaque) callconv(.c) c_int {
        const p: *Payload = @ptrCast(@alignCast(raw.?));
        return p.graph(out, ins) catch -1;
    }
    fn graph(p: *Payload, out: [*c]c.mlx_vector_array, ins: c.mlx_vector_array) !c_int {
        var scope = mx.Scope{};
        defer scope.deinit();
        var args: [2]A = undefined;
        for (args[0..if (p.kind == .front) @as(usize, 1) else 2], 0..) |*arg, i| {
            var value = c.mlx_array_new();
            const rc = c.mlx_vector_array_get(&value, ins, i);
            arg.* = try scope.result(rc, value);
        }
        if (p.kind == .front) {
            const result = try p.layer.front(&scope, args[0]);
            return c.mlx_vector_array_set_data(out, &result, result.len);
        }
        const result = try p.layer.back(true, &scope, args[0], args[1], null);
        return c.mlx_vector_array_set_data(out, &result, 1);
    }
};

// Borrow immutable, realized checkpoint arrays; the model outlives these closures.
const Layer = struct {
    projections: [11][3]A,
    norms: [11]A,
    expert_scale: A,
    local: bool,

    fn init(m: *gemma.Model, i: usize) !Layer {
        var p: Layer = undefined;
        p.local = gemma.Model.sliding(i);
        const names = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj", "experts.switch_glu.gate_proj", "experts.switch_glu.up_proj", "experts.switch_glu.down_proj" };
        for (names, &p.projections, 0..) |name, *projection, j| projection.* = try m.triple(i, if (j == 2 and !p.local) "self_attn.k_proj" else name);
        const norms = [_][]const u8{ "input_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "pre_feedforward_layernorm_2.weight", "router_norm", "post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_2.weight", "post_feedforward_layernorm.weight", "layer_scalar" };
        for (norms, &p.norms) |name, *norm| norm.* = try m.weight(i, name);
        p.expert_scale = try m.weight(i, "router.per_expert_scale");
        return p;
    }

    fn front(p: Layer, s: *mx.Scope, h: A) ![3]A {
        const rows = mx.dim(h, 1);
        const heads: i32 = if (p.local) 8 else 2;
        const dims: i32 = if (p.local) 256 else 512;
        const x = try s.rms(h, p.norms[0]);
        const q = try s.rms(try s.reshape(try linear(s, x, p.projections[0], 4), &.{ 1, rows, 16, dims }), p.norms[1]);
        const projected = try s.reshape(try linear(s, x, p.projections[1], 4), &.{ 1, rows, heads, dims });
        const keys = try s.rms(projected, p.norms[2]);
        const value = if (p.local) try s.reshape(try linear(s, x, p.projections[2], 4), &.{ 1, rows, heads, dims }) else projected;
        const values = try cp.norm(s, value, mx.empty, 1e-6);
        return .{ try s.transpose(q, &.{ 0, 2, 1, 3 }), try s.transpose(keys, &.{ 0, 2, 1, 3 }), try s.transpose(values, &.{ 0, 2, 1, 3 }) };
    }

    fn geglu(comptime compiled: bool, s: *mx.Scope, gate: A, up: A, ops: ?*activations.Ops) !A {
        return if (compiled) activations.uncompiled(s, .geglu, &.{ gate, up }) else ops.?.call(s, .geglu, &.{ gate, up });
    }

    fn back(p: Layer, comptime compiled: bool, s: *mx.Scope, input: A, attended: A, ops: ?*activations.Ops) !A {
        const rows = mx.dim(input, 1);
        const dims: i32 = if (p.local) 256 else 512;
        const out = try linear(s, try s.reshape(try s.transpose(attended, &.{ 0, 2, 1, 3 }), &.{ 1, rows, 16 * dims }), p.projections[3], 4);
        const h = try s.binary(c.mlx_add, input, try s.rms(out, p.norms[3]));
        const dense_input = try s.rms(h, p.norms[4]);
        const gate = try linear(s, dense_input, p.projections[4], 4);
        const up = try linear(s, dense_input, p.projections[5], 4);
        const dense = try linear(s, try geglu(compiled, s, gate, up, ops), p.projections[6], 4);
        const combined = try s.binary(c.mlx_add, try s.rms(dense, p.norms[7]), try s.rms(try p.moe(compiled, s, h, ops), p.norms[8]));
        return s.binary(c.mlx_multiply, try s.binary(c.mlx_add, h, try s.rms(combined, p.norms[9])), p.norms[10]);
    }

    fn moe(p: Layer, comptime compiled: bool, s: *mx.Scope, h: A, ops: ?*activations.Ops) !A {
        const rows = mx.dim(h, 1);
        const scores = try linear(s, try s.rms(h, p.norms[6]), p.projections[7], 8);
        var partition = c.mlx_array_new();
        const rc = c.mlx_argpartition_axis(&partition, scores, -8, -1, mx.stream);
        partition = try s.result(rc, partition);
        const ids = try s.slice(partition, 2, 120, 128);
        var weights = c.mlx_array_new();
        const take_rc = c.mlx_take_along_axis(&weights, scores, ids, -1, mx.stream);
        weights = try s.result(take_rc, weights);
        var probabilities = c.mlx_array_new();
        const prob_rc = c.mlx_softmax_axis(&probabilities, weights, -1, false, mx.stream);
        weights = try s.binary(c.mlx_multiply, try s.result(prob_rc, probabilities), try s.take(p.expert_scale, ids, 0));
        const normalized = try s.rms(h, p.norms[5]);
        const sorted = rows * 8 >= 64;
        var x = try s.reshape(normalized, &.{ 1, rows, 1, 1, 2816 });
        var selected = ids;
        var inverse = mx.empty;
        if (sorted) {
            const flat = try s.reshape(ids, &.{rows * 8});
            const order = try s.unary(c.mlx_argsort, flat);
            inverse = try s.unary(c.mlx_argsort, order);
            const row_ids = try s.binary(c.mlx_floor_divide, order, try s.cast(try s.ints(&.{8}), mx.dtype(order)));
            x = try s.take(try s.reshape(normalized, &.{ rows, 1, 2816 }), row_ids, 0);
            selected = try s.take(flat, order, 0);
        }
        const act = try geglu(compiled, s, try expert(s, p.projections[8], x, selected, sorted), try expert(s, p.projections[9], x, selected, sorted), ops);
        var y = try expert(s, p.projections[10], act, selected, sorted);
        if (sorted) y = try s.take(y, inverse, 0);
        y = try s.reshape(y, &.{ 1, rows, 8, 2816 });
        const weighted = try s.binary(c.mlx_multiply, y, try s.reshape(weights, &.{ 1, rows, 8, 1 }));
        var out = c.mlx_array_new();
        const sum_rc = c.mlx_sum_axis(&out, weighted, -2, false, mx.stream);
        return s.result(sum_rc, out);
    }
};

fn linear(s: *mx.Scope, x: A, weights: [3]A, bits: i32) !A {
    var out = c.mlx_array_new();
    const group = @divExact(mx.dim(x, -1), mx.dim(weights[1], -1));
    const rc = c.mlx_quantized_matmul(&out, x, weights[0], weights[1], weights[2], true, mx.opt(group), mx.opt(bits), "affine", mx.stream);
    return s.result(rc, out);
}
fn expert(s: *mx.Scope, weights: [3]A, x: A, ids: A, sorted: bool) !A {
    const group = @divExact(mx.dim(x, -1), mx.dim(weights[1], -1));
    var out = c.mlx_array_new();
    const rc = c.mlx_gather_qmm(&out, x, weights[0], weights[1], weights[2], mx.empty, ids, true, mx.opt(group), mx.opt(4), "affine", sorted, mx.stream);
    return s.result(rc, out);
}
fn rope(m: *gemma.Model, s: *mx.Scope, x: A, local: bool) !A {
    var out = c.mlx_array_new();
    const freqs = if (local) mx.empty else try m.weights.get("freq_global");
    const rc = c.mlx_fast_rope(&out, x, if (local) 256 else 512, false, .{ .value = 10000, .has_value = local }, 1, m.position, freqs, mx.stream);
    return s.result(rc, out);
}
fn ordered(s: *mx.Scope, buffer: A, begin: i32, end: i32) !A {
    const slot = @mod(begin, 1152);
    const count = end - begin;
    if (slot + count <= 1152) return s.slice(buffer, 2, slot, slot + count);
    return s.cat(&.{ try s.slice(buffer, 2, slot, 1152), try s.slice(buffer, 2, 0, slot + count - 1152) }, 2);
}
pub fn forward(m: *gemma.Model, tokens: []const i32) !gemma.Pass {
    if (tokens.len == 0 or tokens.len > 2048 or tokens.len > 262144 - m.position) return error.ContextLimitExceeded;
    for (tokens) |id| if (id < 0 or id >= gemma.Model.vocab) return error.InvalidToken;
    var pass = gemma.Pass{ .position = m.position, .generation = m.generation, .rows = tokens.len };
    errdefer pass.deinit();
    const s = &pass.scope;
    const rows: i32 = @intCast(tokens.len);
    var h = try s.reshape(try s.binary(c.mlx_multiply, try m.weights.embed(s, "model.embed_tokens", tokens), try s.cast(try s.scalar(@floatCast(@sqrt(@as(f64, 2816)))), mx.bf16)), &.{ 1, rows, 2816 });
    var masks: [2]A = @splat(mx.empty);
    if (rows > 1 and @min(1023, m.position) + rows > 1024) {
        const earlier = @min(1023, m.position);
        const total = earlier + rows;
        const values = try mx.allocator.alloc(u8, @intCast(rows * total));
        defer mx.allocator.free(values);
        for (0..tokens.len) |r| for (0..@intCast(total)) |j| {
            const query = earlier + @as(i32, @intCast(r));
            const key: i32 = @intCast(j);
            values[r * @as(usize, @intCast(total)) + j] = @intFromBool(key <= query and query - key < 1024);
        };
        masks[0] = try s.data(values.ptr, &.{ rows, total }, c.MLX_BOOL);
    }
    var taps: [32]A = undefined;
    var tap_count: usize = 0;
    var carried = mx.empty;
    defer mx.free(carried);
    for (0..30) |i| {
        var layer = mx.Scope{};
        defer layer.deinit();
        const local = i % 6 != 5;
        const qkv = try m.prefills.front(m, &layer, i, h);
        const q = try rope(m, &layer, qkv[0], local);
        const keys = try rope(m, &layer, qkv[1], local);
        const values = qkv[2];
        pass.records[i] = .{ .keys = try s.own(try mx.retain(keys)), .values = try s.own(try mx.retain(values)) };
        try gemma.Model.stageCacheLayer(&m.cache, &pass, i);
        var all_keys = keys;
        var all_values = values;
        if (m.position > 0) {
            const previous_keys = if (local) try ordered(&layer, m.cache[i].keys, @max(0, m.position - 1023), m.position) else m.cache[i].keys;
            const previous_values = if (local) try ordered(&layer, m.cache[i].values, @max(0, m.position - 1023), m.position) else m.cache[i].values;
            all_keys = try layer.cat(&.{ previous_keys, keys }, 2);
            all_values = try layer.cat(&.{ previous_values, values }, 2);
        }
        const mask = masks[if (local) @as(usize, 0) else 1];
        var out = c.mlx_array_new();
        const rc = c.mlx_fast_scaled_dot_product_attention(&out, q, all_keys, all_values, 1, if (rows > 1 and mask.ctx == null) "causal" else "", mask, mx.empty, false, mx.stream);
        out = try layer.result(rc, out);
        h = try m.prefills.back(m, &layer, i, h, out);
        if (m.draft) |d| for (d.parsed.value.dflash_config.target_layer_ids) |id| if (id == i) {
            taps[tap_count] = try s.own(try mx.retain(try layer.reshape(h, &.{ rows, 2816 })));
            tap_count += 1;
        };
        if ((i + 1) % 8 == 0) {
            var pending: [17]A = undefined;
            pending[0] = h;
            for (pass.staged[i - 7 .. i + 1], 0..) |cache, j| {
                pending[1 + 2 * j] = cache.keys;
                pending[2 + 2 * j] = cache.values;
            }
            try mx.evalMany(&pending, true);
        }
        const next = try mx.retain(h);
        mx.free(carried);
        carried = next;
        h = carried;
    }
    pass.hidden = try s.reshape(try s.rms(h, try m.weights.get("model.norm.weight")), &.{ rows, 2816 });
    if (tap_count > 0) pass.taps = try s.cat(taps[0..tap_count], -1);
    pass.logits = try m.activations.call(s, .softcap, &.{ try m.project(s, try s.slice(pass.hidden, 0, rows - 1, rows), try m.weights.triple("model.embed_tokens")), try s.scalar(30) });
    var writes: [60]A = undefined;
    for (pass.staged, 0..) |cache, i| {
        writes[2 * i] = cache.keys;
        writes[2 * i + 1] = cache.values;
    }
    pass.logits = try gemma.Model.cacheDependency(s, pass.logits, &writes);
    try mx.eval(pass.logits);
    return pass;
}

pub fn check(io: std.Io, dir: []const u8, output: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var m = try gemma.Model.init(io, dir);
    defer m.deinit();
    try checkGraphs(&m);
    try std.Io.Dir.cwd().createDirPath(io, output);
    var buf: [256]u8 = undefined;
    for ([_]usize{ 1, 7, 129, 1024, 2048, 3 }, 0..) |count, step| {
        var tokens: [2048]i32 = undefined;
        for (tokens[0..count], 0..) |*id, j| id.* = 1000 + @mod(m.position + @as(i32, @intCast(j)), 37);
        var pass = try m.prefill(tokens[0..count]);
        defer pass.deinit();
        try ready(&pass);
        try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "hidden-{d}", .{step}), pass.hidden);
        try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "logits-{d}", .{step}), pass.logits);
        try m.commit(&pass, count);
        for (m.cache, 0..) |cache, i| {
            try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "keys-{d}-{d}", .{ step, i }), cache.keys);
            try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "values-{d}-{d}", .{ step, i }), cache.values);
        }
        std.debug.print("Gemma prefill verified cache commit at {d} tokens.\n", .{m.position});
    }
    for (0..4) |step| {
        var pass = try m.forward(&.{@as(i32, @intCast(step)) + 2000});
        defer pass.deinit();
        try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "continuation-{d}", .{step}), pass.logits);
        try m.commit(&pass, 1);
    }
    try checkTransactions(&m);
}

fn checkGraphs(m: *gemma.Model) !void {
    var retained: u64 = 0;
    var workspace: u64 = 0;
    const equal = @import("variant_checks.zig").equalBits;
    for ([_]i32{ 16, 1, 2, 3, 5, 7, 8, 9 }, 0..) |rows, step| {
        const before = try @import("memory_runtime.zig").activeBytes();
        if (step == 0) try mx.check(c.mlx_reset_peak_memory());
        {
            var scope = mx.Scope{};
            defer scope.deinit();
            var values: [16 * 2816]f32 = undefined;
            for (values[0..@intCast(rows * 2816)], 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 97)) - 48)) * 0.03125;
            const h = try scope.cast(try scope.data(&values, &.{ 1, rows, 2816 }, mx.f32t), mx.bf16);
            for ([_]usize{ 0, 5, 29 }) |i| {
                const prepared = try m.prefills.layer(m, i);
                const reference_front = try prepared.front(&scope, h);
                const actual_front = try m.prefills.front(m, &scope, i, h);
                for (actual_front, reference_front) |actual, reference| try equal(&scope, actual, reference);
                const reference_back = try prepared.back(false, &scope, h, reference_front[0], &m.activations);
                const actual_back = try m.prefills.back(m, &scope, i, h, reference_front[0]);
                try equal(&scope, actual_back, reference_back);
            }
            if (step == 0) {
                var peak: usize = 0;
                try mx.check(c.mlx_get_peak_memory(&peak));
                workspace = peak -| before;
            }
        }
        const after = try @import("memory_runtime.zig").activeBytes();
        if (step == 0) retained = after else if (after > retained +| workspace) return error.CompiledWeightsDuplicated;
    }
    std.debug.print("PASS: Gemma short-prefill compiled graphs preserve local/global layer bits, sorted expert branches and shared weights across eight row shapes.\n", .{});
}

fn ready(pass: *const gemma.Pass) !void {
    try std.testing.expect(pass.staged_ready);
    for (pass.staged) |cache| for ([_]A{ cache.keys, cache.values }) |array| {
        var available = false;
        try mx.check(c._mlx_array_is_available(&available, array));
        try std.testing.expect(available);
    };
}

fn checkTransactions(m: *gemma.Model) !void {
    const kv = @import("kv_buffer.zig");
    const equal = @import("sampling_checks.zig").equal;
    const enabled = kv.enabled;
    defer kv.enabled = enabled;
    for ([_]bool{ false, true }) |buffered| {
        kv.enabled = buffered;
        const position = m.position;
        const generation = m.generation;
        var previous: [30]gemma.Cache = undefined;
        var cloned: usize = 0;
        defer for (previous[0..cloned]) |*cache| cache.deinit();
        for (m.cache, &previous) |cache, *copy| {
            copy.* = try cache.clone();
            cloned += 1;
        }
        {
            var cancelled = try m.prefill(&.{ 2000, 2001, 2002 });
            defer cancelled.deinit();
            try ready(&cancelled);
            try std.testing.expectEqual(position, m.position);
            try std.testing.expectEqual(generation, m.generation);
        }
        var scope = mx.Scope{};
        defer scope.deinit();
        for (m.cache, previous) |actual, expected| {
            try equal(&scope, actual.keys, expected.keys);
            try equal(&scope, actual.values, expected.values);
        }
        var tokens: [1154]i32 = undefined;
        for (&tokens, 0..) |*token, i| token.* = @intCast(2000 + i % 37);
        for ([_]usize{ 3, 1154 }) |rows| {
            var pass = try m.prefill(tokens[0..rows]);
            defer pass.deinit();
            try ready(&pass);
            const keep: usize = 2;
            var expected = try gemma.Model.prepareCache(&scope, &m.cache, &pass, keep);
            defer for (&expected) |*cache| cache.deinit();
            var arrays: [60]A = undefined;
            for (expected, 0..) |cache, i| {
                arrays[2 * i] = cache.keys;
                arrays[2 * i + 1] = cache.values;
            }
            try mx.evalMany(&arrays, false);
            try m.commit(&pass, keep);
            try std.testing.expectError(error.InvalidCommit, m.commit(&pass, keep));
            for (m.cache, expected) |actual, reference| {
                try equal(&scope, actual.keys, reference.keys);
                try equal(&scope, actual.values, reference.values);
            }
        }
    }
    std.debug.print("PASS: Gemma prefill candidates are evaluated before commit; cancellation and partial writes beyond ring capacity preserve exact caches with/without buffers.\n", .{});
}
fn save(s: *mx.Scope, dir: []const u8, name: []const u8, value: A) !void {
    const path = try std.fmt.allocPrintSentinel(mx.allocator, "{s}/{s}.npy", .{ dir, name }, 0);
    defer mx.allocator.free(path);
    const out = try s.cast(value, mx.f32t);
    try mx.eval(out);
    try mx.saveArray(path, out);
}
