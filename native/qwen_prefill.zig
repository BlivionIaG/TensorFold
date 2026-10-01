//! Regular Qwen prompt arithmetic, separate from row-exact decode/verification.
const mx = @import("mlx.zig");
const model = @import("model.zig");
const lanes = @import("lanes.zig");
const src = @import("kernel_sources.zig");
const A = mx.Array;
const c = mx.c;
pub var evaluation_stride: usize = 4;

pub fn forward(m: *model.Model, tokens: []const i32) !model.Pass {
    return forwardImage(m, tokens, mx.empty, mx.empty, m.rope_delta);
}

pub fn checkCommit(io: @import("std").Io, directory: []const u8) !void {
    const std = @import("std");
    try mx.init();
    defer mx.shutdown();
    var m = try model.Model.init(io, directory);
    defer m.deinit();
    var tokens: [193]i32 = undefined;
    var rows: [193]i32 = undefined;
    for (&tokens, &rows, 0..) |*token, *row, i| {
        token.* = @intCast(1000 + i * 37);
        row.* = @intCast(i);
    }
    for ([_]usize{ 1, 17, 129, tokens.len }) |keep| {
        var expected: [64]model.Cache = @splat(.{});
        defer for (&expected) |*cache| cache.deinit();
        var scope = mx.Scope{};
        defer scope.deinit();
        var logits = mx.empty;
        for (0..2) |run| {
            m.reset();
            {
                var pass = try forward(&m, &tokens);
                defer pass.deinit();
                if (run == 0) for (&pass.records) |*record| {
                    record.values[7] = mx.empty;
                };
                try m.commit(&pass, rows[0..keep]);
            }
            if (run == 0) {
                for (m.cache, &expected) |cache, *saved| saved.* = try cache.clone();
            } else {
                try std.testing.expectEqual(@as(i32, @intCast(keep)), m.position);
                for (m.cache, expected) |actual, saved| {
                    try @import("variant_checks.zig").equalBits(&scope, actual.a, saved.a);
                    try @import("variant_checks.zig").equalBits(&scope, actual.b, saved.b);
                }
            }
            var next = try m.forward(&.{701}, &.{-1});
            defer next.deinit();
            if (run == 0) logits = try scope.own(try mx.retain(next.logits)) else try @import("variant_checks.zig").equalBits(&scope, next.logits, logits);
        }
    }
    std.debug.print("PASS: full prefill state reuse and 1/17/129-row partial commits match replay caches and continuation bit for bit.\n", .{});
}

pub fn forwardImage(m: *model.Model, tokens: []const i32, embeddings: A, positions: A, delta: i32) !model.Pass {
    if (tokens.len == 0 or tokens.len > 2048) return error.InvalidPrefillWidth;
    var p = model.Pass{ .count = tokens.len, .start = m.position };
    if (embeddings.ctx != null) {
        if (!@import("std").mem.eql(i32, mx.shape(embeddings), &.{ 1, @intCast(tokens.len), 5120 }) or !@import("std").mem.eql(i32, mx.shape(positions), &.{ 3, @intCast(tokens.len) })) return error.InvalidImageEmbeddings;
        p.vision_delta = delta;
    } else if (positions.ctx != null) return error.InvalidImageEmbeddings;
    errdefer p.deinit();
    const s = &p.scope;
    for (0..tokens.len) |i| p.parents[i] = @as(i32, @intCast(i)) - 1;
    var h = if (embeddings.ctx != null) embeddings else try m.weights.embedArray(s, try s.ints(tokens));
    var carried = mx.empty;
    defer mx.free(carried);
    for (0..64) |i| {
        var scope = mx.Scope{};
        defer scope.deinit();
        const work = &scope;
        const x = try work.rms(h, try m.weight(i, "input_layernorm.weight"));
        const residual = if (i % 4 == 3) try attention(m, work, i, x, positions, &p.records[i]) else try gdn(m, work, i, x, &p);
        h = try work.binary(c.mlx_add, h, residual);
        const norm = try work.rms(h, try m.weight(i, "post_attention_layernorm.weight"));
        const gate = try m.prefillProject(work, i, "mlp.gate_proj", norm);
        const up = try m.prefillProject(work, i, "mlp.up_proj", norm);
        const act = try m.prefill_ops.call(work, .swiglu, &.{ gate, up });
        h = try work.binary(c.mlx_add, h, try m.prefillProject(work, i, "mlp.down_proj", act));
        try m.trace(work, m.position, i, "hidden", h);
        for (&p.records[i].values) |*value| if (value.ctx != null) {
            value.* = try s.own(try mx.retain(value.*));
        };
        for ([_]usize{ 5, 19, 33, 47, 61 }, 0..) |layer, j| if (i == layer) {
            p.taps[j] = try s.own(try mx.retain(h));
        };
        if (evaluation_stride > 0 and (i == 0 or (i + 1) % evaluation_stride == 0)) try mx.evalMany(&.{h}, true);
        const next = try mx.retain(h);
        mx.free(carried);
        carried = next;
        h = carried;
    }
    const norm = try s.rms(h, try m.weights.get("model.norm.weight"));
    p.logits = try (try m.weights.linear("lm_head")).apply(&m.kernels, s, .{ .x = try s.slice(norm, 1, @intCast(tokens.len - 1), @intCast(tokens.len)) });
    try m.trace(s, m.position, 64, "logits", p.logits);
    try mx.eval(p.logits);
    return p;
}

fn gdn(m: *model.Model, s: *mx.Scope, i: usize, x: A, p: *model.Pass) !A {
    const n: i32 = @intCast(p.count);
    const qkv = try m.prefillProject(s, i, "linear_attn.in_proj_qkv", x);
    const z = try m.prefillProject(s, i, "linear_attn.in_proj_z", x);
    const a = try m.prefillProject(s, i, "linear_attn.in_proj_a", x);
    const b = try m.prefillProject(s, i, "linear_attn.in_proj_b", x);
    const cs = if (m.cache[i].a.ctx != null) m.cache[i].a else try s.zeros(&.{ 1, 3, 10240 }, mx.bf16);
    const state = if (m.cache[i].b.ctx != null) m.cache[i].b else try s.zeros(&.{ 1, 48, 128, 128 }, mx.f32t);
    const seq = try s.cat(&.{ cs, qkv }, 1);
    var conv = c.mlx_array_new();
    const rc = c.mlx_conv1d(&conv, seq, try m.weight(i, "linear_attn.conv1d.weight"), 1, 0, 1, 10240, mx.stream);
    const activated = try m.prefill_ops.call(s, .silu, &.{try s.result(rc, conv)});
    if (i == 16) {
        try m.trace(s, m.position, i, "qkv", qkv);
        try m.trace(s, m.position, i, "conv", conv);
    }
    const q0 = try s.reshape(try s.slice(activated, 2, 0, 2048), &.{ 1, n, 16, 128 });
    const k0 = try s.reshape(try s.slice(activated, 2, 2048, 4096), &.{ 1, n, 16, 128 });
    const v = try s.reshape(try s.slice(activated, 2, 4096, 10240), &.{ 1, n, 48, 128 });
    const q = try s.binary(c.mlx_multiply, try s.rmsEpsilon(q0, mx.empty, 1e-6 / 128.0), try s.cast(try s.scalar(1.0 / 128.0), mx.bf16));
    const k = try s.binary(c.mlx_multiply, try s.rmsEpsilon(k0, mx.empty, 1e-6 / 128.0), try s.cast(try s.scalar(0.08838834764831845), mx.bf16));
    const g = try m.prefill_ops.call(s, .decay, &.{ try m.weight(i, "linear_attn.A_log"), a, try m.weight(i, "linear_attn.dt_bias") });
    const beta = try s.unary(c.mlx_sigmoid, b);
    if (i == 16) {
        for ([_]A{ q, k, v, a, b, g, beta }, [_][]const u8{ "q", "k", "v", "a", "b", "g", "beta" }) |value, label| try m.trace(s, m.position, i, label, value);
    }
    const recurrence = try m.kernels.run(s, src.flash_prefill_gdn, &.{ q, k, v, g, beta, state, try s.reshape(try s.ints(&.{n}), &.{}) }, &.{ mx.td("InT", mx.bf16), mx.td("StT", mx.f32t), mx.ti("Dk", 128), mx.ti("Dv", 128), mx.ti("Hk", 16), mx.ti("Hv", 48) }, .{ 32, 128, 48 }, .{ 32, 4, 1 }, &.{ .{ .shape = &.{ 1, n, 48, 128 } }, .{ .shape = &.{ 1, 48, 128, 128 }, .dtype = mx.f32t } });
    const out = recurrence[0];
    p.records[i].values = .{ q, k, v, g, beta, try s.own(try mx.retain(state)), seq, recurrence[1] };
    const normalized = try s.rms(out, try m.weight(i, "linear_attn.norm.weight"));
    const gated = try m.prefill_ops.call(s, .gated, &.{ try s.reshape(z, &.{ 1, n, 48, 128 }), normalized });
    return m.prefillProject(s, i, "linear_attn.out_proj", try s.reshape(gated, &.{ 1, n, 6144 }));
}

fn attention(m: *model.Model, s: *mx.Scope, i: usize, x: A, positions: A, rec: *model.Record) !A {
    const n = mx.dim(x, 1);
    const qg = try s.reshape(try m.prefillProject(s, i, "self_attn.q_proj", x), &.{ 1, n, 24, 512 });
    var q = try s.rms(try s.slice(qg, 3, 0, 256), try m.weight(i, "self_attn.q_norm.weight"));
    const gate = try s.reshape(try s.slice(qg, 3, 256, 512), &.{ 1, n, 6144 });
    var k = try s.rms(try s.reshape(try m.prefillProject(s, i, "self_attn.k_proj", x), &.{ 1, n, 4, 256 }), try m.weight(i, "self_attn.k_norm.weight"));
    var v = try s.transpose(try s.reshape(try m.prefillProject(s, i, "self_attn.v_proj", x), &.{ 1, n, 4, 256 }), &.{ 0, 2, 1, 3 });
    // Regular MLX RoPE uses a scalar cache offset on the sequence axis.
    q = try s.transpose(q, &.{ 0, 2, 1, 3 });
    k = try s.transpose(k, &.{ 0, 2, 1, 3 });
    if (positions.ctx != null) {
        q = try @import("vision_positions.zig").rope(s, q, positions);
        k = try @import("vision_positions.zig").rope(s, k, positions);
    } else {
        q = try s.rope(q, try s.ints(&.{m.position + m.rope_delta}), 64);
        k = try s.rope(k, try s.ints(&.{m.position + m.rope_delta}), 64);
    }
    rec.values[0] = k;
    rec.values[1] = v;
    if (m.cache[i].a.ctx != null) {
        k = try s.cat(&.{ m.cache[i].a, k }, 2);
        v = try s.cat(&.{ m.cache[i].b, v }, 2);
    }
    const out = if (n <= 128) (if (mx.tensor_units) try lanes.sdpa(&m.kernels, s, q, k, v, 0.0625) else try exactAttention(s, q, k, v)) else try promptAttention(s, q, k, v);
    if (i == 3) {
        try m.trace(s, m.position, i, "q", q);
        try m.trace(s, m.position, i, "attention", out);
    }
    const rows = try s.reshape(try s.transpose(out, &.{ 0, 2, 1, 3 }), &.{ 1, n, 6144 });
    return m.prefillProject(s, i, "self_attn.o_proj", try s.binary(c.mlx_multiply, rows, try s.unary(c.mlx_sigmoid, gate)));
}

// The non-tensor production loader groups queries only within one MLX kernel
// regime. This is exact_attention.exact_sdpa's original grouping policy.
fn exactAttention(s: *mx.Scope, q: A, k: A, v: A) !A {
    const n = mx.dim(q, 2);
    const total = mx.dim(k, 2);
    var parts: [128]A = undefined;
    var count: usize = 0;
    var begin: i32 = 0;
    while (begin < n) {
        var width = @min(@divTrunc(32, @divExact(mx.dim(q, 1), mx.dim(k, 1))), n - begin);
        const first = total - n + begin + 1;
        const last = first + width - 1;
        for ([_]i32{ 1024, 1025, 4096, 8193, 16384, 32769, 65536, 65537 }) |edge| if (first < edge and edge <= last) {
            width = 1;
        };
        const visible = total - n + begin + width;
        var out = c.mlx_array_new();
        const rc = c.mlx_fast_scaled_dot_product_attention(&out, try s.slice(q, 2, begin, begin + width), try s.slice(k, 2, 0, visible), try s.slice(v, 2, 0, visible), 0.0625, if (width > 1) "causal" else "", mx.empty, mx.empty, false, mx.stream);
        parts[count] = try s.result(rc, out);
        count += 1;
        begin += width;
    }
    return if (count == 1) parts[0] else s.cat(parts[0..count], 2);
}

fn promptAttention(s: *mx.Scope, q: A, k: A, v: A) !A {
    const n = mx.dim(q, 2);
    const total = mx.dim(k, 2);
    var parts: [16]A = undefined;
    var count: usize = 0;
    var begin: i32 = 0;
    while (begin < n) {
        var end = if (total > 4096 and n > 128) @min(begin + 128, n) else n;
        if (n - end <= 16) end = n;
        const visible = total - n + end;
        var out = c.mlx_array_new();
        const rc = c.mlx_fast_scaled_dot_product_attention(&out, try s.slice(q, 2, begin, end), try s.slice(k, 2, 0, visible), try s.slice(v, 2, 0, visible), 0.0625, "causal", mx.empty, mx.empty, false, mx.stream);
        parts[count] = try s.result(rc, out);
        try mx.evalMany(&.{parts[count]}, true);
        if (count > 0) try mx.eval(parts[count - 1]);
        count += 1;
        begin = end;
    }
    return if (count == 1) parts[0] else s.cat(parts[0..count], 2);
}
