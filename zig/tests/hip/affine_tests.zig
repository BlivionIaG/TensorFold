//! The affine 4-bit product on the GPU: the host recipe's exact bits at every row count, and within fp32 of float64.

const std = @import("std");
const hip = @import("hip");
const check = @import("check.zig");
const ref = @import("affine_reference.zig");
const Gpu = check.Gpu;

/// Shapes: every MLX group size, an output count that ends mid-block, and a projection-sized k.
const shapes = [_]struct { n: usize, k: usize, group: usize }{
    .{ .n = 257, .k = 2048, .group = 32 },
    .{ .n = 257, .k = 4096, .group = 64 },
    .{ .n = 130, .k = 5120, .group = 128 },
};

/// The row counts each shape runs at; a row's bits may not depend on how many rows share the launch.
const row_counts = [_]u32{ 1, 3, 16 };

const Device = struct {
    x: hip.DeviceBuffer,
    words: hip.DeviceBuffer,
    scale: hip.DeviceBuffer,
    bias: hip.DeviceBuffer,
    out: hip.DeviceBuffer,

    fn init(d: *const hip.Driver, p: ref.Problem) !Device {
        var x = try hip.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(p.x));
        errdefer x.free();
        var words = try hip.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(p.words));
        errdefer words.free();
        var scale = try hip.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(p.scale));
        errdefer scale.free();
        var bias = try hip.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(p.bias));
        errdefer bias.free();
        return .{ .x = x, .words = words, .scale = scale, .bias = bias, .out = try hip.DeviceBuffer.alloc(d, p.m * p.n * 4) };
    }

    fn deinit(dev: *Device) void {
        inline for (.{ &dev.x, &dev.words, &dev.scale, &dev.bias, &dev.out }) |b| b.free();
    }

    /// The first `m` rows of the problem, into a poisoned output.
    fn run(dev: Device, f: hip.Function, stream: hip.Stream, p: ref.Problem, m: u32, out: []f32) !void {
        try dev.out.fill8(0xa5, null);
        try hip.affine.launch(f, stream, .{
            .x = dev.x.ptr,
            .words = dev.words.ptr,
            .scale = dev.scale.ptr,
            .bias = dev.bias.ptr,
            .out = dev.out.ptr,
            .m = m,
            .n = @intCast(p.n),
            .k = @intCast(p.k),
            .group = @intCast(p.group),
        });
        try stream.synchronize();
        try dev.out.download(0, std.mem.sliceAsBytes(out[0 .. m * p.n]));
    }
};

pub fn run(gpu: Gpu) !void {
    var module = try hip.Module.load(gpu.d, hip.kernels.affine);
    defer module.unload();
    const f = try module.function(hip.affine.symbol);
    var stream = try hip.Stream.init(gpu.d, true);
    defer stream.deinit();
    var worst: f64 = 0;
    var outputs: usize = 0;
    for (shapes, 0..) |s, i| {
        const p = try ref.Problem.init(gpu.gpa, 0x9e37 + i, row_counts[row_counts.len - 1], s.n, s.k, s.group);
        defer p.deinit(gpu.gpa);
        const want = try p.serialAll(gpu.gpa);
        defer gpu.gpa.free(want);
        const got = try gpu.gpa.alloc(f32, want.len);
        defer gpu.gpa.free(got);
        var dev = try Device.init(gpu.d, p);
        defer dev.deinit();
        for (row_counts) |m| {
            try dev.run(f, stream, p, m, got);
            var label_buf: [96]u8 = undefined;
            const label = try std.fmt.bufPrint(&label_buf, "affine m{d} n{d} k{d} g{d} against the host recipe", .{ m, s.n, s.k, s.group });
            try check.sameBytes(label, std.mem.sliceAsBytes(got[0 .. m * s.n]), std.mem.sliceAsBytes(want[0 .. m * s.n]));
        }
        for (0..p.m) |row| for (0..p.n) |col| {
            const e = ref.excess(got[row * p.n + col], p.reference(row, col), p.k);
            try check.expect(e <= 1, "affine n{d} k{d} g{d} output ({d}, {d}) is {d:.2} times the fp32 bound of float64", .{ s.n, s.k, s.group, row, col, e });
            worst = @max(worst, e);
        };
        outputs += p.m * p.n;
    }
    try pinned(gpu, f, stream);
    check.pass("affine: {d} outputs at {d} row counts equal the host recipe bit for bit, worst {d:.2} of the float64 bound", .{ outputs, row_counts.len, worst });
}

/// The pinned problem's outputs hash to the digest the host test pins.
fn pinned(gpu: Gpu, f: hip.Function, stream: hip.Stream) !void {
    const c = ref.pinned;
    const p = try ref.Problem.init(gpu.gpa, c.seed, c.m, c.n, c.k, c.group);
    defer p.deinit(gpu.gpa);
    var dev = try Device.init(gpu.d, p);
    defer dev.deinit();
    const got = try gpu.gpa.alloc(f32, p.m * p.n);
    defer gpu.gpa.free(got);
    try dev.run(f, stream, p, @intCast(p.m), got);
    const digest = ref.digest(got);
    try check.expect(digest == ref.pinned_digest, "affine pinned digest {x}, expected {x}", .{ digest, ref.pinned_digest });
}
