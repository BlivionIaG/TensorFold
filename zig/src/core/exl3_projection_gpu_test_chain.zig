//! The chain kernels against the unfused dispatches they replace: every output bit must agree.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ex = @import("exl3_projection.zig");
const support = @import("exl3_projection_gpu_test_support.zig");
const k = support.k;
const n = support.n;
const Owned = support.Owned;
const complete = support.complete;

/// A matrix with its scales as packed signs (the checkpoint's sign-only form), owning the sign buffers.
pub const Signed = struct {
    matrix: ex.Matrix,
    su: mtl.Buffer,
    sv: mtl.Buffer,

    pub fn init(device: mtl.Device, w: Owned, su_word: u16, sv_word: u16) !Signed {
        const su = try device.buffer(k / 8, mtl.ResourceOptions.shared);
        errdefer su.deinit();
        const sv = try device.buffer(n / 8, mtl.ResourceOptions.shared);
        errdefer sv.deinit();
        @memset(su.slice(i16, k / 16), @bitCast(su_word));
        @memset(sv.slice(i16, n / 16), @bitCast(sv_word));
        var m = w.matrix;
        m.suh = .{ .buffer = su };
        m.svh = .{ .buffer = sv };
        m.suh_format = .packed_signs;
        m.svh_format = .packed_signs;
        return .{ .matrix = m, .su = su, .sv = sv };
    }
    pub fn deinit(s: Signed) void {
        s.su.deinit();
        s.sv.deinit();
    }
};

/// wo_a (two indexed groups) into wo_b, unfused and fused.
fn woChain(first: ex.Matrix, second: ex.Matrix, next: ex.Matrix) !void {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    if (!ex.chain.rotateOutInReady(&p)) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    const groups = 2;
    const descriptors = try device.buffer(groups * @sizeOf(ex.ExpertDescriptor), mtl.ResourceOptions.shared);
    defer descriptors.deinit();
    descriptors.slice(ex.ExpertDescriptor, groups)[0] = try ex.descriptor(first);
    descriptors.slice(ex.ExpertDescriptor, groups)[1] = try ex.descriptor(second);
    const ids = try device.buffer(groups * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    @memcpy(ids.slice(u32, groups), &[_]u32{ 0, 1 });
    const x = try support.input(device, groups);
    defer x.deinit();
    const grouped_ref = try device.buffer(groups * n * 2, mtl.ResourceOptions.shared);
    defer grouped_ref.deinit();
    const grouped = try device.buffer(groups * n * 2, mtl.ResourceOptions.shared);
    defer grouped.deinit();
    const y_ref = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    defer y_ref.deinit();
    const y = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    defer y.deinit();
    @memset(grouped_ref.slice(u16, groups * n), 0);
    @memset(grouped.slice(u16, groups * n), 0);
    const scratch = try device.buffer(try ex.scratchBytes(k, n, groups) + try ex.scratchBytes(groups * n, n, 1), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const experts: ex.Experts = .{ .descriptors = .{ .buffer = descriptors }, .count = groups, .k = k, .n = n };
    const rows: ex.Rows = .{ .count = groups, .x_stride = k + 16, .y_stride = n, .routes = .{ .ids = .{ .buffer = ids }, .repeat = 1 }, .output_rows = groups };
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    ex.useMatrix(e, first);
    ex.useMatrix(e, second);
    // The unfused chain: wo_a, then wo_b over the stored bf16 rows.
    try ex.encodeIndexed(&p, e, experts, .{ .buffer = x, .offset = 32 }, .{ .buffer = grouped_ref }, .{ .buffer = scratch }, rows);
    try ex.encode(&p, e, next, .{ .buffer = grouped_ref }, .{ .buffer = y_ref }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = groups * n, .y_stride = n });
    // The fused chain.
    try ex.chain.encodeIndexedRotateNext(&p, e, experts, .{ .buffer = x, .offset = 32 }, .{ .buffer = grouped }, .{ .buffer = scratch }, rows, .{ .su = next.suh, .su_signs = next.suh_format == .packed_signs, .k = groups * n });
    try ex.encode(&p, e, next, .{ .buffer = scratch }, .{ .buffer = y }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = groups * n, .y_stride = n, .prerotated = true });
    try complete(cb, e);
    try std.testing.expectEqualSlices(u16, grouped_ref.slice(u16, groups * n), grouped.slice(u16, groups * n));
    try std.testing.expectEqualSlices(u16, y_ref.slice(u16, n), y.slice(u16, n));
}

test "wo_a epilogue fused with wo_b's rotate_in keeps every bit (f16 scales, biases)" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const w0 = try Owned.init(device, 6, .mul1, 21);
    defer w0.deinit();
    const w1 = try Owned.init(device, 6, .mul1, 22);
    defer w1.deinit();
    const wb = try Owned.init(device, 6, .mul1, 23);
    defer wb.deinit();
    try woChain(w0.matrix, w1.matrix, wb.matrix);
}

test "wo_a epilogue fused with wo_b's rotate_in keeps every bit (packed sign scales)" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const w0 = try Owned.init(device, 4, .inst3, 31);
    defer w0.deinit();
    const w1 = try Owned.init(device, 4, .inst3, 32);
    defer w1.deinit();
    const wb = try Owned.init(device, 4, .inst3, 33);
    defer wb.deinit();
    const s0 = try Signed.init(device, w0, 0x9635, 0x4a17);
    defer s0.deinit();
    const s1 = try Signed.init(device, w1, 0x1b2d, 0xc3a5);
    defer s1.deinit();
    const sb = try Signed.init(device, wb, 0x5f01, 0x8e3c);
    defer sb.deinit();
    try woChain(s0.matrix, s1.matrix, sb.matrix);
}

/// dsv41_rms in place, then the projection's own rotate_in, against the one-dispatch norm + rotate_in.
fn qNorm(matrix: ex.Matrix, rows: u32, weight_f32: bool, scale: f32) !void {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    if (ex.chain.rmsRotateInWidth(&p) < k) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    var opts = mtl.CompileOptions.mlx();
    opts.math_functions = .precise;
    const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_ops, opts);
    defer lib.deinit();
    const rms = try mtl.Pipeline.init(device, lib, "dsv41_rms", false);
    defer rms.deinit();
    const stride = k + 16;
    const x_ref = try device.buffer(rows * stride * 2, mtl.ResourceOptions.shared);
    defer x_ref.deinit();
    const x = try device.buffer(rows * stride * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    var prng = std.Random.DefaultPrng.init(0x51a7);
    for (x_ref.slice(u16, rows * stride)) |*v| v.* = support.bf16((prng.random().float(f32) - 0.5) * scale);
    @memcpy(x.slice(u16, rows * stride), x_ref.slice(u16, rows * stride));
    const weight = try device.buffer(k * 4, mtl.ResourceOptions.shared);
    defer weight.deinit();
    for (0..k) |i| {
        const value = 0.25 + prng.random().float(f32) * 2;
        if (weight_f32) weight.slice(f32, k)[i] = value else weight.slice(u16, 2 * k)[i] = support.bf16(value);
    }
    const y_ref = try device.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer y_ref.deinit();
    const y = try device.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer y.deinit();
    const scratch = try device.buffer(try ex.scratchBytes(k, n, rows), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const eps: f32 = 1e-6;
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    e.setPipeline(rms);
    e.setBuffer(x_ref, 0, 0);
    e.setBuffer(weight, 0, 1);
    e.setBuffer(x_ref, 0, 2);
    e.setValue(extern struct { rows: u32, width: u32, x_stride: u32, y_stride: u32, eps: f32, weight_f32: u32 }{ .rows = rows, .width = k, .x_stride = stride, .y_stride = stride, .eps = eps, .weight_f32 = @intFromBool(weight_f32) }, 3);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
    e.barrier();
    try ex.encode(&p, e, matrix, .{ .buffer = x_ref }, .{ .buffer = y_ref }, .{ .buffer = scratch }, .{ .count = rows, .x_stride = stride, .y_stride = n });
    try ex.chain.encodeRmsRotateIn(&p, e, .{ .buffer = x }, .{ .buffer = weight }, .{ .buffer = x }, matrix.suh, .{ .buffer = scratch }, .{ .rows = rows, .width = k, .x_stride = stride, .y_stride = stride, .eps = eps, .weight_f32 = weight_f32, .su_signs = matrix.suh_format == .packed_signs });
    try ex.encode(&p, e, matrix, .{ .buffer = scratch }, .{ .buffer = y }, .{ .buffer = scratch }, .{ .count = rows, .x_stride = k, .y_stride = n, .prerotated = true });
    try complete(cb, e);
    try std.testing.expectEqualSlices(u16, x_ref.slice(u16, rows * stride), x.slice(u16, rows * stride));
    try std.testing.expectEqualSlices(u16, y_ref.slice(u16, rows * n), y.slice(u16, rows * n));
}

test "norm fused with the next rotate_in keeps every bit (bf16 and f32 weights, f16 scales)" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const w = try Owned.init(device, 6, .mul1, 41);
    defer w.deinit();
    for ([_]u32{ 1, 3 }) |rows| for ([_]bool{ false, true }) |f32_weight| for ([_]f32{ 0.05, 2, 300 }) |scale|
        try qNorm(w.matrix, rows, f32_weight, scale);
}

test "norm fused with the next rotate_in keeps every bit (packed sign scales)" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const w = try Owned.init(device, 4, .inst3, 42);
    defer w.deinit();
    const s = try Signed.init(device, w, 0x6c35, 0x3a17);
    defer s.deinit();
    try qNorm(s.matrix, 1, false, 2);
    try qNorm(s.matrix, 3, true, 2);
}
