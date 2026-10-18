//! One real routed expert from the EXL3 pack on the device: gate, up, SwiGLU, down.
//! The Zig CPU decoder (`dequantize` then a matvec) is the oracle for the strips-order GPU kernels.
//! Usage: tf-cuda-test exl3-real <pack dir> [layer] [expert]

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const st = @import("safetensors.zig");
const format = @import("exl3_format.zig");
const strips = @import("exl3_strips.zig");
const cpu = @import("exl3_cpu.zig");

const Gpu = check.Gpu;
pub const shards = 7;

pub const Mat = struct {
    trellis: []const i16,
    suh: []const f16,
    svh: []const f16,
    tiles_k: usize,
    tiles_n: usize,
    halves: u8,
};

pub const Pack = struct {
    files: [shards]st.File,
    open: usize = 0,

    pub fn find(p: *const Pack, name: []const u8) ?st.Tensor {
        for (p.files[0..p.open]) |*f| if (f.get(name)) |t| return t;
        return null;
    }

    /// Maps all seven model shards of `dir`. Pair with `close`.
    pub fn openAll(p: *Pack, gpu: Gpu, dir: []const u8) !void {
        for (0..shards) |i| {
            var path: [512]u8 = undefined;
            const name = try std.fmt.bufPrint(&path, "{s}/model-{d:0>5}-of-{d:0>5}.safetensors", .{ dir, i + 1, shards });
            p.files[i] = try st.File.open(gpu.gpa, gpu.io, name);
            p.open += 1;
        }
    }

    pub fn close(p: *Pack, io: std.Io) void {
        for (p.files[0..p.open]) |*f| f.close(io);
        p.open = 0;
    }
};

pub fn words16(comptime T: type, bytes: []const u8) []const T {
    return @as([*]const T, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / @sizeOf(T)];
}

pub fn mat(p: *const Pack, gpa: std.mem.Allocator, base: []const u8) !Mat {
    var buf: [160]u8 = undefined;
    const tr = p.find(try std.fmt.bufPrint(&buf, "{s}.trellis", .{base})) orelse return error.MissingTensor;
    const suh = p.find(try std.fmt.bufPrint(&buf, "{s}.suh", .{base})) orelse return error.MissingTensor;
    const svh = p.find(try std.fmt.bufPrint(&buf, "{s}.svh", .{base})) orelse return error.MissingTensor;
    if (tr.rank != 3) return error.BadShape;
    const words = tr.dim(2);
    // safetensors maps are only byte aligned for the data start; copy so the typed views are aligned.
    const t = try gpa.alloc(i16, tr.bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(t), tr.bytes);
    const su = try gpa.alloc(f16, suh.bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(su), suh.bytes);
    const sv = try gpa.alloc(f16, svh.bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(sv), svh.bytes);
    return .{ .trellis = t, .suh = su, .svh = sv, .tiles_k = tr.dim(0), .tiles_n = tr.dim(1), .halves = @intCast(words / 8) };
}

pub const Dev = struct {
    trellis: cuda.DeviceBuffer,
    suh: cuda.DeviceBuffer,
    svh: cuda.DeviceBuffer,
    pub fn free(d: *Dev) void {
        d.trellis.free();
        d.suh.free();
        d.svh.free();
    }
};

pub fn upload(gpu: Gpu, m: Mat) !Dev {
    const words = @as(usize, m.halves) * 8;
    const ordered = try gpu.gpa.alloc(i16, m.trellis.len);
    defer gpu.gpa.free(ordered);
    try strips.reorder(m.trellis, m.tiles_k, m.tiles_n, words, ordered);
    var t = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ordered));
    errdefer t.free();
    var su = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(m.suh));
    errdefer su.free();
    const sv = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(m.svh));
    return .{ .trellis = t, .suh = su, .svh = sv };
}

pub const Diff = struct { max_abs: f64, peak: f64 };

pub fn compare(got: []const f16, want: []const f64) Diff {
    var d: Diff = .{ .max_abs = 0, .peak = 0 };
    for (got, want) |g, w| {
        d.max_abs = @max(d.max_abs, @abs(@as(f64, @floatCast(g)) - w));
        d.peak = @max(d.peak, @abs(w));
    }
    return d;
}

pub fn cpuLinear(gpa: std.mem.Allocator, m: Mat, x: []const f64, y: []f64) !void {
    const k = m.tiles_k * 16;
    const n = m.tiles_n * 16;
    const scratch = try gpa.alloc(f64, 3 * k * n);
    defer gpa.free(scratch);
    try cpu.gemv(.{ .halves = m.halves }, .mul1, m.trellis, m.tiles_k, m.tiles_n, m.suh, m.svh, x, y, scratch);
}

pub fn gpuLinear(gpu: Gpu, ex: *const cuda.exl3.Exl3, stream: cuda.Stream, m: Mat, d: Dev, x_dev: u64, y_len: usize) ![]f16 {
    var y = try cuda.DeviceBuffer.alloc(gpu.d, y_len * 2);
    defer y.free();
    try ex.linear(stream, x_dev, d.trellis.ptr, d.suh.ptr, d.svh.ptr, y.ptr, 1, m.tiles_k * 16, m.tiles_n * 16, m.halves, 2);
    try stream.synchronize();
    const bytes = try check.download(gpu, y);
    defer gpu.gpa.free(bytes);
    const out = try gpu.gpa.alloc(f16, y_len);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    const layer: usize = if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 0;
    const expert: usize = if (rest.len > 2) try std.fmt.parseInt(usize, rest[2], 10) else 0;

    var pack: Pack = .{ .files = undefined };
    defer pack.close(gpu.io);
    try pack.openAll(gpu, dir);

    var ex = try cuda.exl3.Exl3.load(gpu.d);
    defer ex.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();

    var base: [128]u8 = undefined;
    const stem = try std.fmt.bufPrint(&base, "model.language_model.layers.{d}.mlp.experts.{d}", .{ layer, expert });
    var names: [3][160]u8 = undefined;
    const gate_m = try mat(&pack, gpu.gpa, try std.fmt.bufPrint(&names[0], "{s}.gate_proj", .{stem}));
    const up_m = try mat(&pack, gpu.gpa, try std.fmt.bufPrint(&names[1], "{s}.up_proj", .{stem}));
    const down_m = try mat(&pack, gpu.gpa, try std.fmt.bufPrint(&names[2], "{s}.down_proj", .{stem}));
    const k = gate_m.tiles_k * 16;
    const inter = gate_m.tiles_n * 16;
    try check.expect(down_m.tiles_k * 16 == inter and down_m.tiles_n * 16 == k, "down shape {d}x{d} for {d}/{d}", .{ down_m.tiles_k * 16, down_m.tiles_n * 16, k, inter });
    std.debug.print("layer {d} expert {d}: K={d} inter={d} halves gate={d} up={d} down={d}\n", .{ layer, expert, k, inter, gate_m.halves, up_m.halves, down_m.halves });

    // deterministic input in a plausible range
    const x = try gpu.gpa.alloc(f16, k);
    defer gpu.gpa.free(x);
    var rng = std.Random.DefaultPrng.init(0x5eed);
    for (x) |*v| v.* = @floatCast((rng.random().float(f32) - 0.5) * 2.0);
    const x64 = try gpu.gpa.alloc(f64, k);
    defer gpu.gpa.free(x64);
    for (x64, x) |*d, s| d.* = @floatCast(s);
    var x_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer x_dev.free();

    var dg = try upload(gpu, gate_m);
    defer dg.free();
    var du = try upload(gpu, up_m);
    defer du.free();
    var dd = try upload(gpu, down_m);
    defer dd.free();

    // gate and up
    const gate_g = try gpuLinear(gpu, &ex, stream, gate_m, dg, x_dev.ptr, inter);
    defer gpu.gpa.free(gate_g);
    const up_g = try gpuLinear(gpu, &ex, stream, up_m, du, x_dev.ptr, inter);
    defer gpu.gpa.free(up_g);
    const gate_c = try gpu.gpa.alloc(f64, inter);
    defer gpu.gpa.free(gate_c);
    const up_c = try gpu.gpa.alloc(f64, inter);
    defer gpu.gpa.free(up_c);
    try cpuLinear(gpu.gpa, gate_m, x64, gate_c);
    try cpuLinear(gpu.gpa, up_m, x64, up_c);
    const dgate = compare(gate_g, gate_c);
    const dup = compare(up_g, up_c);
    std.debug.print("gate max_abs {e} peak {e}; up max_abs {e} peak {e}\n", .{ dgate.max_abs, dgate.peak, dup.max_abs, dup.peak });
    try check.expect(dgate.max_abs <= 0.02 * dgate.peak and dgate.peak > 0, "gate differs from the Zig decoder", .{});
    try check.expect(dup.max_abs <= 0.02 * dup.peak and dup.peak > 0, "up differs from the Zig decoder", .{});
    check.pass("real gate_proj and up_proj match the Zig decoder (layer {d} expert {d})", .{ layer, expert });

    // SwiGLU on the device from the device's own gate and up, versus the CPU on the CPU's
    var gate_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(gate_g));
    defer gate_dev.free();
    var up_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(up_g));
    defer up_dev.free();
    var glu_dev = try cuda.DeviceBuffer.alloc(gpu.d, inter * 2);
    defer glu_dev.free();
    try ex.swiglu(stream, gate_dev.ptr, up_dev.ptr, glu_dev.ptr, 1, inter);
    try stream.synchronize();
    const glu_bytes = try check.download(gpu, glu_dev);
    defer gpu.gpa.free(glu_bytes);
    const glu_g = try gpu.gpa.alloc(f16, inter);
    defer gpu.gpa.free(glu_g);
    @memcpy(std.mem.sliceAsBytes(glu_g), glu_bytes);
    const glu_c = try gpu.gpa.alloc(f64, inter);
    defer gpu.gpa.free(glu_c);
    try cpu.swiglu(gate_c, up_c, glu_c, std.math.inf(f64));

    // down: CPU from the CPU glu, GPU from the GPU glu
    const down_g = try gpuLinear(gpu, &ex, stream, down_m, dd, glu_dev.ptr, k);
    defer gpu.gpa.free(down_g);
    const down_c = try gpu.gpa.alloc(f64, k);
    defer gpu.gpa.free(down_c);
    try cpuLinear(gpu.gpa, down_m, glu_c, down_c);
    const dglu = compare(glu_g, glu_c);
    const ddown = compare(down_g, down_c);
    std.debug.print("swiglu max_abs {e} peak {e}; down max_abs {e} peak {e}\n", .{ dglu.max_abs, dglu.peak, ddown.max_abs, ddown.peak });
    try check.expect(dglu.max_abs <= 0.02 * dglu.peak and dglu.peak > 0, "swiglu differs from the CPU", .{});
    try check.expect(ddown.max_abs <= 0.03 * ddown.peak and ddown.peak > 0, "down differs from the Zig decoder", .{});
    check.pass("real expert sublayer (gate, up, swiglu, down) matches the Zig decoder (layer {d} expert {d})", .{ layer, expert });
}
