const std = @import("std");
const mx = @import("mlx.zig");

const probe = @import("kernel_sources.zig").Spec{
    .name = "native_launch_config_probe",
    .inputs = &.{"X"},
    .outputs = &.{ "Y0", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8" },
    .source =
    \\uint i = thread_position_in_grid.x;
    \\float value = X[i] + float(threads_per_threadgroup.x);
    \\native_config_store(Y0, i, value); native_config_store(Y1, i, value + 1); native_config_store(Y2, i, value + 2);
    \\native_config_store(Y3, i, value + 3); native_config_store(Y4, i, value + 4); native_config_store(Y5, i, value + 5);
    \\native_config_store(Y6, i, value + 6); native_config_store(Y7, i, value + 7); native_config_store(Y8, i, value + 8);
    ,
    .header =
    \\template <typename T> inline void native_config_store(device T* output, uint index, float value) {
    \\  output[index] = T(value);
    \\}
    ,
    .contiguous = true,
    .bake_templates = true,
};

const Case = struct {
    shape: []const c_int = &.{32},
    grid: c_int = 32,
    group: c_int = 8,
    dtype: mx.c.mlx_dtype = mx.f32t,
    init: ?f32 = null,
};

pub fn exercise(kernels: *mx.Kernels) !void {
    const cases = [_]Case{
        .{},
        .{},
        .{ .init = 7 },
        .{ .init = 7, .grid = 16 },
        .{ .init = 7, .grid = 16, .group = 16 },
        .{ .init = 7, .grid = 16, .group = 16, .shape = &.{ 4, 8 } },
        .{ .init = 7, .grid = 16, .group = 16, .shape = &.{ 4, 8 }, .dtype = mx.c.MLX_UINT32 },
        .{ .init = 0, .grid = 16 },
        .{ .init = -0.0, .grid = 16 },
        .{ .shape = &.{ 1, 2, 4, 4 }, .dtype = mx.bf16 },
        .{},
    };
    for (cases, 0..) |case, step| {
        var scope = mx.Scope{};
        defer scope.deinit();
        var values: [32]f32 = undefined;
        for (&values, 0..) |*value, i| value.* = @floatFromInt(i + step);
        const input = try scope.data(&values, &.{32}, mx.f32t);
        const outputs: [9]mx.Output = @splat(.{ .shape = case.shape, .dtype = case.dtype });
        var results: [9]mx.Array = undefined;
        try kernels.runInto(&scope, probe, &.{input}, &.{}, .{ case.grid, 1, 1 }, .{ case.group, 1, 1 }, &outputs, &results, case.init);
        try mx.evalMany(&results, false);
        try verify(&scope, case, &values, &results);
    }
    try pendingGraphs(kernels);
    try std.testing.expectEqual(@as(usize, 1), kernels.items.count());
    try keyIdentity();
}

const key_probe = @import("kernel_sources.zig").Spec{
    .name = "native_kernel_key_probe",
    .inputs = &.{"X"},
    .outputs = &.{"Y"},
    .source = "uint i = thread_position_in_grid.x; Y[i] = X[i] + float(A) + 10.0f * float(B);",
    .header = "",
    .contiguous = true,
    .bake_templates = true,
};

fn keyResult(kernels: *mx.Kernels, spec: @import("kernel_sources.zig").Spec, templates: []const mx.Template, expected: f32) !void {
    var scope = mx.Scope{};
    defer scope.deinit();
    const x = try scope.zeros(&.{32}, mx.f32t);
    const out = try kernels.run(&scope, spec, &.{x}, templates, .{ 32, 1, 1 }, .{ 32, 1, 1 }, &.{.{ .shape = &.{32}, .dtype = mx.f32t }});
    try mx.eval(out[0]);
    for (mx.c.mlx_array_data_float32(out[0])[0..32]) |value| try std.testing.expectEqual(expected, value);
}

fn keyIdentity() !void {
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    var name = key_probe.name[0..key_probe.name.len :0].*;
    var a: [1:0]u8 = .{'A'};
    var b: [1:0]u8 = .{'B'};
    var copied = key_probe;
    copied.name = &name;
    try keyResult(&kernels, copied, &.{ mx.ti(&a, 1), mx.ti(&b, 2) }, 21);
    @memset(&name, 'x');
    a[0] = 'C';
    b[0] = 'D';
    try keyResult(&kernels, key_probe, &.{ mx.ti("A", 1), mx.ti("B", 2) }, 21);
    try std.testing.expectEqual(@as(usize, 1), kernels.items.count());

    const cases = .{
        .{ &[_]mx.Template{ mx.ti("B", 1), mx.ti("A", 2) }, @as(f32, 12) },
        .{ &[_]mx.Template{ mx.ti("B", 2), mx.ti("A", 1) }, @as(f32, 21) },
        .{ &[_]mx.Template{ mx.tb("A", true), mx.ti("B", 2) }, @as(f32, 21) },
        .{ &[_]mx.Template{ mx.ti("A", 1), mx.ti("B", 3) }, @as(f32, 31) },
    };
    inline for (cases, 0..) |case, i| {
        try keyResult(&kernels, key_probe, case[0], case[1]);
        try std.testing.expectEqual(@as(usize, i + 2), kernels.items.count());
    }
    for ([_]u16{ 256, 512 }, 0..) |reserve, i| {
        var reserved = key_probe;
        reserved.reserve = reserve;
        try keyResult(&kernels, reserved, &.{ mx.ti("A", 1), mx.ti("B", 2) }, 21);
        try std.testing.expectEqual(@as(usize, i + 6), kernels.items.count());
    }
    var renamed = key_probe;
    renamed.name = "native_kernel_key_other";
    try keyResult(&kernels, renamed, &.{ mx.ti("A", 1), mx.ti("B", 2) }, 21);
    try std.testing.expectEqual(@as(usize, 8), kernels.items.count());
    var runtime = key_probe;
    runtime.bake_templates = false;
    try keyResult(&kernels, runtime, &.{ mx.ti("A", 1), mx.ti("B", 2) }, 21);
    try keyResult(&kernels, runtime, &.{ mx.ti("A", 2), mx.ti("B", 3) }, 32);
    try std.testing.expectEqual(@as(usize, 9), kernels.items.count());
}

fn verify(scope: *mx.Scope, case: Case, values: *const [32]f32, results: []const mx.Array) !void {
    for (results, 0..) |result, output| {
        try std.testing.expectEqual(case.dtype, mx.dtype(result));
        try std.testing.expectEqualSlices(c_int, case.shape, mx.shape(result));
        const floats = if (case.dtype == mx.f32t) result else try scope.cast(result, mx.f32t);
        if (case.dtype != mx.f32t) try mx.eval(floats);
        for (mx.c.mlx_array_data_float32(floats)[0..32], 0..) |actual, i| {
            const expected = if (i < @as(usize, @intCast(case.grid))) values[i] + @as(f32, @floatFromInt(case.group)) + @as(f32, @floatFromInt(output)) else case.init.?;
            try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(actual)));
        }
    }
}

fn pendingGraphs(kernels: *mx.Kernels) !void {
    var scope = mx.Scope{};
    defer scope.deinit();
    var values: [32]f32 = undefined;
    const cases = [_]Case{ .{}, .{ .shape = &.{ 4, 8 }, .grid = 16, .group = 16, .init = 3 } };
    var results: [18]mx.Array = undefined;
    for (cases, 0..) |case, i| {
        var inputs = mx.Scope{};
        defer inputs.deinit();
        for (&values, 0..) |*value, j| value.* = @floatFromInt(j + 32 * i);
        const input = try inputs.data(&values, &.{32}, mx.f32t);
        const outputs: [9]mx.Output = @splat(.{ .shape = case.shape, .dtype = case.dtype });
        try kernels.runInto(&scope, probe, &.{input}, &.{}, .{ case.grid, 1, 1 }, .{ case.group, 1, 1 }, &outputs, results[i * 9 ..][0..9], case.init);
    }
    try mx.evalMany(&results, false);
    for (cases, 0..) |case, i| {
        for (&values, 0..) |*value, j| value.* = @floatFromInt(j + 32 * i);
        try verify(&scope, case, &values, results[i * 9 ..][0..9]);
    }
}

fn failedOutputOwnership(kernels: *mx.Kernels) !void {
    const baseline = try active();
    const outputs: [9]mx.Output = @splat(.{ .shape = &.{32}, .dtype = mx.f32t });
    {
        var scope = mx.Scope{};
        defer scope.deinit();
        const input = try scope.zeros(&.{32}, mx.f32t);
        var results: [9]mx.Array = undefined;
        try kernels.runInto(&scope, probe, &.{input}, &.{}, .{ 32, 1, 1 }, .{ 8, 1, 1 }, &outputs, &results, null);
        try mx.evalMany(&results, false);
    }
    for ([_]usize{ 0, 1, 8 }) |held| {
        {
            var inputs = mx.Scope{};
            defer inputs.deinit();
            const input = try inputs.zeros(&.{32}, mx.f32t);
            try mx.eval(input);
            var scope = mx.Scope{ .arrays = try .initCapacity(mx.allocator, held) };
            defer scope.deinit();
            var results: [9]mx.Array = undefined;
            const previous = mx.allocator;
            var failing = std.testing.FailingAllocator.init(previous, .{ .fail_index = 0, .resize_fail_index = 0 });
            {
                mx.allocator = failing.allocator();
                defer mx.allocator = previous;
                try std.testing.expectError(error.OutOfMemory, kernels.runInto(&scope, probe, &.{input}, &.{}, .{ 32, 1, 1 }, .{ 8, 1, 1 }, &outputs, &results, null));
            }
            try std.testing.expectEqual(held, scope.arrays.items.len);
            try std.testing.expect(failing.has_induced_failure);
        }
        try std.testing.expectEqual(baseline, try active());
    }
    {
        var scope = mx.Scope{};
        defer scope.deinit();
        var results: [9]mx.Array = undefined;
        try std.testing.expectError(error.MlxFailure, kernels.runInto(&scope, probe, &.{mx.empty}, &.{}, .{ 32, 1, 1 }, .{ 8, 1, 1 }, &outputs, &results, null));
    }
    try std.testing.expectEqual(baseline, try active());
    try pendingGraphs(kernels);
}

const ReentryProbe = struct {
    kernels: *mx.Kernels,
    refused: bool = false,

    fn callback(out: [*c]mx.c.mlx_vector_array, ins: mx.c.mlx_vector_array, raw: ?*anyopaque) callconv(.c) c_int {
        const p: *ReentryProbe = @ptrCast(@alignCast(raw.?));
        var scope = mx.Scope{};
        defer scope.deinit();
        var results: [0]mx.Array = .{};
        if (p.kernels.call(&scope, .{ .ctx = null }, &.{}, &results)) {
            return -1;
        } else |err| {
            if (err != error.ReentrantKernelLaunch) return -1;
            p.refused = true;
        }
        return mx.c.mlx_vector_array_set(out, ins);
    }
};

fn closureOwnership(kernels: *mx.Kernels) !void {
    var payload = ReentryProbe{ .kernels = kernels };
    const raw = mx.c.mlx_closure_new_func_payload(ReentryProbe.callback, &payload, null);
    defer _ = mx.c.mlx_closure_free(raw);
    if (raw.ctx == null) return error.MlxFailure;
    var compiled = mx.c.mlx_closure_new();
    defer _ = mx.c.mlx_closure_free(compiled);
    try mx.check(mx.c.mlx_compile(&compiled, raw, false));
    var scope = mx.Scope{};
    defer scope.deinit();
    var results: [2]mx.Array = undefined;
    for (0..2) |i| {
        var inputs = mx.Scope{};
        defer inputs.deinit();
        const input = try inputs.scalar(@floatFromInt(i + 7));
        var wrong: [2]mx.Array = undefined;
        try std.testing.expectError(error.InvalidKernelArity, kernels.call(&scope, compiled, &.{input}, &wrong));
        try kernels.call(&scope, compiled, &.{input}, results[i..][0..1]);
    }
    try mx.evalMany(&results, false);
    for (results, 0..) |result, i| try std.testing.expectEqual(@as(f32, @floatFromInt(i + 7)), mx.c.mlx_array_data_float32(result)[0]);
    try std.testing.expect(payload.refused);
}

fn active() !usize {
    try mx.check(mx.c.mlx_synchronize(mx.stream));
    var bytes: usize = 0;
    try mx.check(mx.c.mlx_get_active_memory(&bytes));
    return bytes;
}

pub fn check() !void {
    const previous = mx.allocator;
    var tracking = std.testing.FailingAllocator.init(previous, .{});
    mx.allocator = tracking.allocator();
    defer mx.allocator = previous;
    const baseline = try active();
    {
        var kernels = mx.Kernels.init();
        defer kernels.deinit();
        try failedOutputOwnership(&kernels);
        try closureOwnership(&kernels);
        try std.testing.expectEqual(baseline, try active());
        try exercise(&kernels);
        try std.testing.expectEqual(baseline, try active());
        const retained = tracking.allocated_bytes - tracking.freed_bytes;
        for (0..8) |_| try exercise(&kernels);
        try std.testing.expectEqual(retained, tracking.allocated_bytes - tracking.freed_bytes);
        try std.testing.expectEqual(baseline, try active());
    }
    try std.testing.expectEqual(tracking.allocated_bytes, tracking.freed_bytes);
    std.debug.print("PASS: cached Metal configs preserve launch metadata and nine outputs without retaining tensors or shape history\n", .{});
}
