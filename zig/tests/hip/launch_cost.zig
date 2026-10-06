//! Host cost of a kernel launch: the library's C launchers against the Zig launches, the same kernels and arguments.

const std = @import("std");
const hip = @import("hip");
const check = @import("check.zig");
const Gpu = check.Gpu;

const Kernel = enum { rms, affine_row, affine_rows, conv, gated_delta, causal, mix };

const Buffers = struct { act: hip.DeviceBuffer, f32a: hip.DeviceBuffer, f32b: hip.DeviceBuffer, words: hip.DeviceBuffer, table: hip.DeviceBuffer, cache: hip.DeviceBuffer, pos: hip.DeviceBuffer };

fn alloc(gpu: Gpu, bytes: usize) !hip.DeviceBuffer {
    var b = try hip.DeviceBuffer.alloc(gpu.d, bytes);
    errdefer b.free();
    try b.fill8(0, null);
    return b;
}

fn one(lib: *const hip.rocm.Library, k: Kernel, b: Buffers, fp16: bool, s: hip.abi.Stream) hip.rocm.Error!void {
    const act: c_int = if (fp16) 1 else 2;
    const p = struct {
        fn of(x: hip.DeviceBuffer) ?*anyopaque {
            return @ptrFromInt(x.ptr);
        }
        fn f(x: hip.DeviceBuffer) ?[*]f32 {
            return @ptrFromInt(x.ptr);
        }
    };
    switch (k) {
        .rms => try lib.call("tf_rms", .{ p.of(b.act), p.f(b.f32a), p.of(b.act), act, 1, 1024, 1e-6, s }),
        .affine_row, .affine_rows => {
            const m: c_int = if (k == .affine_row) 1 else 4;
            try lib.call("tf_affine", .{ p.of(b.act), p.of(b.words), p.of(b.table), p.of(b.table), 1, p.of(b.f32b), m, 1024, 1024, 8, 64, 0, @intFromBool(fp16), s, null, 1, 0 });
        },
        .conv => try lib.call("tf_conv_decode", .{ p.f(b.f32a), p.f(b.f32b), p.f(b.f32a), p.f(b.f32b), 1, 6144, 4, s }),
        .gated_delta => try lib.call("tf_gated_delta", .{ p.f(b.f32a), p.f(b.f32a), p.f(b.f32a), p.f(b.f32a), p.f(b.f32a), p.f(b.f32b), p.f(b.f32a), 1, 1, 16, 16, 128, 128, s, null }),
        .causal => {
            const sh: c_longlong = 256 * 256;
            try lib.call("tf_causal", .{ p.f(b.f32a), p.of(b.cache), p.of(b.cache), p.f(b.f32b), 1, 1, 256, 8, 2, 256, 0.0625, 0, 0, sh, 256, 0, sh, 256, if (fp16) 0 else 1, p.f(b.f32b), p.f(b.f32b), p.f(b.f32b), s, @as(?[*]const i32, @ptrFromInt(b.pos.ptr)) });
        },
        .mix => {
            inline for (.{ Kernel.rms, .affine_row, .conv, .gated_delta, .rms, .affine_row, .causal }) |each| try one(lib, each, b, fp16, s);
        },
    }
}

pub fn run(gpu: Gpu, n: usize, reps: usize) !void {
    const family = hip.rocm.familyOf(try gpu.ctx.capability()) orelse return error.UnsupportedGpu;
    const fp16 = family == .rdna2;
    var stream = try hip.Stream.init(gpu.d, true);
    defer stream.deinit();
    var b: Buffers = undefined;
    b.act = try alloc(gpu, 1 << 20);
    defer b.act.free();
    b.f32a = try alloc(gpu, 8 << 20);
    defer b.f32a.free();
    b.f32b = try alloc(gpu, 8 << 20);
    defer b.f32b.free();
    b.words = try alloc(gpu, 1 << 20);
    defer b.words.free();
    b.table = try alloc(gpu, 1 << 20);
    defer b.table.free();
    b.cache = try alloc(gpu, 8 << 20);
    defer b.cache.free();
    b.pos = try alloc(gpu, 64);
    defer b.pos.free();
    const times = try gpu.gpa.alloc(f64, reps);
    defer gpu.gpa.free(times);
    const enqueue = try gpu.gpa.alloc(f64, reps);
    defer gpu.gpa.free(enqueue);
    const modes = [_]hip.rocm.Launch{ .library, .zig };
    for (std.enums.values(Kernel)) |k| {
        var per: [2][2]f64 = undefined;
        for (modes, &per) |mode, *out| {
            var lib = try hip.rocm.Library.openMode(gpu.d, family, mode);
            defer lib.close();
            for (times, enqueue) |*t, *e| {
                const t0 = check.now(gpu.io);
                for (0..n) |_| try one(&lib, k, b, fp16, stream.handle);
                e.* = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1000 / @as(f64, @floatFromInt(n));
                try stream.synchronize();
                t.* = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1000 / @as(f64, @floatFromInt(n));
            }
            out.* = .{ check.median(enqueue), check.median(times) };
        }
        std.debug.print("RESULT {s}: enqueue / done us a call: library {d:.2} / {d:.2}, Zig {d:.2} / {d:.2}\n", .{ @tagName(k), per[0][0], per[0][1], per[1][0], per[1][1] });
    }
}
