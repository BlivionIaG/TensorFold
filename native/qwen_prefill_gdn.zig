const std = @import("std");
const mx = @import("mlx.zig");
const ops = @import("prefill_ops.zig");
const A = mx.Array;
const Spec = @import("kernel_sources.zig").Spec;

pub const Fusion = struct {
    silu: A = mx.empty,
    gate: A = mx.empty,

    pub fn deinit(f: *Fusion) void {
        mx.free(f.silu);
        mx.free(f.gate);
        f.* = .{};
    }

    fn tables(f: *Fusion, compiled: *ops.Ops) !void {
        if (f.silu.ctx != null) return;
        var scope = mx.Scope{};
        defer scope.deinit();
        var bits: [65536]u16 = undefined;
        for (&bits, 0..) |*value, i| value.* = @intCast(i);
        const x = try scope.data(&bits, &.{65536}, mx.bf16);
        // Keep the compiled MLX activation's BF16 and FP32 rounding.
        const silu = try compiled.call(&scope, .silu, &.{x});
        const gate = try compiled.call(&scope, .silu, &.{try scope.cast(x, mx.f32t)});
        try mx.evalMany(&.{ silu, gate }, false);
        const held = try mx.retain(silu);
        errdefer mx.free(held);
        f.gate = try mx.retain(gate);
        f.silu = held;
    }

    pub fn prework(f: *Fusion, kernels: *mx.Kernels, compiled: *ops.Ops, s: *mx.Scope, seq: A, weights: A) ![3]A {
        try f.tables(compiled);
        const n = mx.dim(seq, 1) - 3;
        const result = try kernels.run(s, pre, &.{ seq, weights, f.silu }, &.{}, .{ 32, 80, n }, .{ 32, 1, 1 }, &.{
            .{ .shape = &.{ 1, n, 16, 128 } },
            .{ .shape = &.{ 1, n, 16, 128 } },
            .{ .shape = &.{ 1, n, 48, 128 } },
        });
        return .{ result[0], result[1], result[2] };
    }

    pub fn normGate(f: *Fusion, kernels: *mx.Kernels, compiled: *ops.Ops, s: *mx.Scope, y: A, z: A, weights: A) !A {
        try f.tables(compiled);
        const n = mx.dim(y, 1);
        const result = try kernels.run(s, post, &.{ y, z, weights, f.gate }, &.{}, .{ 32, 48, n }, .{ 32, 1, 1 }, &.{.{ .shape = &.{ 1, n, 48, 128 } }});
        return result[0];
    }
};

const pre = Spec{
    .name = "native_prefill_gdn_pre",
    .inputs = &.{ "X", "W", "SILU" },
    .outputs = &.{ "Q", "K", "V" },
    .header = "",
    .contiguous = true,
    .source =
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint head = threadgroup_position_in_grid.y;
    \\const uint row = threadgroup_position_in_grid.z;
    \\float values[4];
    \\float sum = 0.0f;
    \\for (uint j = 0; j < 4; ++j) {
    \\    const uint channel = head * 128 + lane * 4 + j;
    \\    float value = 0.0f;
    \\    for (uint tap = 0; tap < 4; ++tap)
    \\        value += float(X[(row + tap) * 10240 + channel]) * float(W[channel * 4 + tap]);
    \\    const bfloat activated = SILU[as_type<ushort>(bfloat(value))];
    \\    values[j] = float(activated);
    \\    sum += values[j] * values[j];
    \\}
    \\if (head < 32) {
    \\    sum = simd_sum(sum);
    \\    const float inv = metal::precise::rsqrt(sum / 128.0f + (1e-6f / 128.0f));
    \\    const bfloat scale = head < 16 ? bfloat(1.0f / 128.0f) : bfloat(0.08838834764831845f);
    \\    const uint start = (row * 16 + head % 16) * 128 + lane * 4;
    \\    for (uint j = 0; j < 4; ++j) {
    \\        const bfloat value = bfloat(values[j] * inv) * scale;
    \\        if (head < 16) Q[start + j] = value; else K[start + j] = value;
    \\    }
    \\} else {
    \\    const uint start = (row * 48 + head - 32) * 128 + lane * 4;
    \\    for (uint j = 0; j < 4; ++j) V[start + j] = bfloat(values[j]);
    \\}
    ,
};

const post = Spec{
    .name = "native_prefill_gdn_norm_gate",
    .inputs = &.{ "Y", "Z", "W", "SILU" },
    .outputs = &.{"OUT"},
    .header = "",
    .contiguous = true,
    .source =
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint head = threadgroup_position_in_grid.y;
    \\const uint row = threadgroup_position_in_grid.z;
    \\const uint start = (row * 48 + head) * 128 + lane * 4;
    \\float values[4];
    \\float sum = 0.0f;
    \\for (uint j = 0; j < 4; ++j) {
    \\    values[j] = float(Y[start + j]);
    \\    sum += values[j] * values[j];
    \\}
    \\sum = simd_sum(sum);
    \\const float inv = metal::precise::rsqrt(sum / 128.0f + 1e-6f);
    \\for (uint j = 0; j < 4; ++j) {
    \\    const bfloat normalized = W[lane * 4 + j] * bfloat(values[j] * inv);
    \\    OUT[start + j] = bfloat(SILU[as_type<ushort>(Z[start + j])] * float(normalized));
    \\}
    ,
};

pub fn check() !void {
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    var compiled = ops.Ops{};
    defer compiled.deinit();
    var fusion = Fusion{};
    defer fusion.deinit();
    var random = std.Random.DefaultPrng.init(382917);
    for ([_]i32{ 1, 2, 3, 17, 129, 2048 }) |n| {
        errdefer std.debug.print("GDN fusion rows={d}\n", .{n});
        var scope = mx.Scope{};
        defer scope.deinit();
        const x = if (n == 2) try scope.zeros(&.{ 1, n + 3, 10240 }, mx.bf16) else try noise(&scope, random.random(), &.{ 1, n + 3, 10240 }, 2);
        const w = try noise(&scope, random.random(), &.{ 10240, 4, 1 }, 0.25);
        const actual = try fusion.prework(&kernels, &compiled, &scope, x, w);
        var conv = mx.c.mlx_array_new();
        const rc = mx.c.mlx_conv1d(&conv, x, w, 1, 0, 1, 10240, mx.stream);
        const activated = try compiled.call(&scope, .silu, &.{try scope.result(rc, conv)});
        const q = try scope.reshape(try scope.slice(activated, 2, 0, 2048), &.{ 1, n, 16, 128 });
        const k = try scope.reshape(try scope.slice(activated, 2, 2048, 4096), &.{ 1, n, 16, 128 });
        const v = try scope.reshape(try scope.slice(activated, 2, 4096, 10240), &.{ 1, n, 48, 128 });
        const expected = [_]A{
            try scope.binary(mx.c.mlx_multiply, try scope.rmsEpsilon(q, mx.empty, 1e-6 / 128.0), try scope.cast(try scope.scalar(1.0 / 128.0), mx.bf16)),
            try scope.binary(mx.c.mlx_multiply, try scope.rmsEpsilon(k, mx.empty, 1e-6 / 128.0), try scope.cast(try scope.scalar(0.08838834764831845), mx.bf16)),
            v,
        };
        for (actual, expected, [_][]const u8{ "q", "k", "v" }) |a, e, name| {
            errdefer std.debug.print("GDN fusion {s}\n", .{name});
            try @import("sampling_checks.zig").equal(&scope, a, e);
        }
        const y = if (n == 2) try scope.zeros(&.{ 1, n, 48, 128 }, mx.bf16) else try noise(&scope, random.random(), &.{ 1, n, 48, 128 }, 4);
        const z = try noise(&scope, random.random(), mx.shape(y), 8);
        const nw = try noise(&scope, random.random(), &.{128}, 2);
        const gated = try fusion.normGate(&kernels, &compiled, &scope, y, z, nw);
        const reference = try compiled.call(&scope, .gated, &.{ z, try scope.rms(y, nw) });
        try @import("sampling_checks.zig").equal(&scope, gated, reference);
    }
    std.debug.print("PASS: exact GDN prework and norm-gate at six prefill widths.\n", .{});
}

fn noise(s: *mx.Scope, random: std.Random, shape: []const i32, scale: f32) !A {
    var count: usize = 1;
    for (shape) |dim| count *= @intCast(dim);
    const values = try mx.allocator.alloc(f32, count);
    defer mx.allocator.free(values);
    for (values) |*value| value.* = (random.float(f32) * 2 - 1) * scale;
    return s.cast(try s.data(values.ptr, shape, mx.f32t), mx.bf16);
}
