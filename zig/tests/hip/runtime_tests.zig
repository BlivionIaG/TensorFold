//! Runtime-level GPU tests: copies, fills, launches, argument packing, module globals, cooperative grids, refused images.

const std = @import("std");
const hip = @import("hip");
const check = @import("check.zig");
const Gpu = check.Gpu;
const expect = check.expect;

const ProbeView = extern struct { src: u64, n: c_int, scale: f32 };

/// The runtime, the device and its caps; a GPU outside the caps table fails here.
pub fn info(gpu: Gpu) !void {
    var name_buf: [256]u8 = undefined;
    var arch_buf: [64]u8 = undefined;
    const c = try gpu.ctx.caps();
    const mem = try gpu.ctx.memInfo();
    std.debug.print("INFO HIP {d}, device {s} ({s}), {d} CUs, wave {d}, matrix {t}, {d} MiB, kernels for {s}\n", .{
        try gpu.d.version(),               try gpu.ctx.name(&name_buf),
        try gpu.ctx.archName(&arch_buf),   try gpu.ctx.attribute(.multiprocessor_count),
        try gpu.ctx.attribute(.warp_size), c.matrix,
        mem.total >> 20,                   if (hip.kernels.available) hip.kernels.targets else "none",
    });
}

pub fn smoke(gpu: Gpu) !void {
    const d = gpu.d;
    const n: usize = 1 << 20;
    const pattern = try gpu.gpa.alloc(u8, n);
    defer gpu.gpa.free(pattern);
    for (pattern, 0..) |*p, i| p.* = @truncate(i *% 2654435761 >> 7);

    var a = try hip.DeviceBuffer.fromHost(d, pattern);
    defer a.free();
    var b = try hip.DeviceBuffer.alloc(d, n);
    defer b.free();
    try b.copyFrom(0, a.ptr, n, null);
    const back = try check.download(gpu, b);
    defer gpu.gpa.free(back);
    try check.sameBytes("H2D, D2D, D2H round trip (1 MiB)", back, pattern);
    check.pass("blocking copies: 1 MiB host -> device -> device -> host equal", .{});

    var stream = try hip.Stream.init(d, true);
    defer stream.deinit();
    var pinned = try hip.HostBuffer.alloc(d, n);
    defer pinned.free();
    @memcpy(pinned.bytes, pattern);
    var c = try hip.DeviceBuffer.alloc(d, n);
    defer c.free();
    try c.uploadAsync(0, pinned.bytes, stream.handle);
    @memset(back, 0);
    try c.downloadAsync(0, back, stream.handle);
    try stream.synchronize();
    try check.sameBytes("async copies through pinned memory", back, pattern);
    // again with other bytes over the same buffers: a copy that does nothing leaves the old bytes and fails
    for (pinned.bytes) |*p| p.* = ~p.*;
    try c.uploadAsync(0, pinned.bytes, stream.handle);
    try c.downloadAsync(0, back, stream.handle);
    try stream.synchronize();
    try check.sameBytes("async copies of new bytes over old ones", back, pinned.bytes);
    try c.fill8(0x5a, stream.handle);
    try stream.synchronize();
    try c.download(0, back);
    try expect(std.mem.allEqual(u8, back, 0x5a), "memset8", .{});
    try c.fill32(0xdeadbeef, null);
    try c.download(0, back);
    for (std.mem.bytesAsSlice(u32, back)) |w| try expect(w == 0xdeadbeef, "memset32 word {x}", .{w});
    check.pass("async copies, memset8 and memset32", .{});

    var mapped = try hip.HostBuffer.allocMapped(d, 4096);
    defer mapped.free();
    @memset(mapped.bytes, 0);
    var probe = try hip.Module.load(d, hip.kernels.probe);
    defer probe.unload();
    const fill = try probe.function("tf_probe_fill");
    var margs: hip.Args = .{};
    margs.add(try mapped.device());
    margs.add(@as(f32, 7));
    margs.add(@as(u32, 1024));
    try hip.launch.launch(fill, .{ .grid = .{ .x = 4 }, .block = .{ .x = 256 } }, stream, &margs);
    try stream.synchronize();
    for (mapped.slice(f32), 0..) |v, i| try expect(v == 7 + @as(f32, @floatFromInt(i)), "mapped host element {d}: {d}", .{ i, v });
    check.pass("mapped host memory: a kernel writes pinned host memory directly", .{});

    const axpy = try probe.function("tf_probe_axpy");
    const count: u32 = 100_000;
    var y = try hip.DeviceBuffer.alloc(d, count * 4);
    defer y.free();
    var x = try hip.DeviceBuffer.alloc(d, count * 4);
    defer x.free();
    const blocks = (count + 255) / 256;
    var args: hip.Args = .{};
    args.add(x.ptr);
    args.add(@as(f32, 0));
    args.add(count);
    try hip.launch.launch(fill, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, stream, &args);
    args = .{};
    args.add(y.ptr);
    args.add(@as(f32, 1));
    args.add(count);
    try hip.launch.launch(fill, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, stream, &args);
    args = .{};
    args.add(y.ptr);
    args.add(x.ptr);
    args.add(@as(f32, 2));
    args.add(count);
    try hip.launch.launch(axpy, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, stream, &args);
    try stream.synchronize();
    const ys = try check.download(gpu, y);
    defer gpu.gpa.free(ys);
    for (std.mem.bytesAsSlice(f32, ys), 0..) |v, i| {
        const want: f32 = 2 * @as(f32, @floatFromInt(i)) + (1 + @as(f32, @floatFromInt(i)));
        try expect(v == want, "fill+axpy element {d}: {d} != {d}", .{ i, v, want });
    }
    check.pass("launches: fill, fill, axpy over {d} elements exact", .{count});

    const table = try probe.global("tf_probe_table");
    try expect(table.len == 16, "module global size {d}", .{table.len});
    const vals = [4]i32{ 1, 2, 3, 4 };
    try d.check(d.api.hipMemcpyHtoD(table.ptr, &vals, 16), "write global");
    var out = try hip.DeviceBuffer.alloc(d, 64);
    defer out.free();
    args = .{};
    args.add(out.ptr);
    try hip.launch.launch(try probe.function("tf_probe_read_table"), .{ .grid = .{}, .block = .{ .x = 4 } }, stream, &args);
    try stream.synchronize();
    var got: [4]i32 = undefined;
    try out.download(0, std.mem.asBytes(&got));
    try expect(std.mem.eql(i32, &got, &.{ 2, 4, 6, 8 }), "module global read back {any}", .{got});
    check.pass("module global: hipModuleGetGlobal write, kernel read", .{});

    args = .{};
    args.add(out.ptr);
    try hip.launch.launch(try probe.function("tf_probe_wave"), .{ .grid = .{}, .block = .{ .x = 64 } }, stream, &args);
    try stream.synchronize();
    var wave: i32 = 0;
    try out.download(0, std.mem.asBytes(&wave));
    try expect(wave == 32, "the kernels run in wave32 (got {d})", .{wave});
    check.pass("wave32: the code object runs 32-lane wavefronts", .{});

    try argPacking(gpu, probe, stream);

    var bad: hip.Args = .{};
    bad.add(out.ptr);
    const refused = hip.launch.launch(fill, .{ .grid = .{}, .block = .{ .x = 2048 } }, stream, &bad);
    try expect(refused == error.Invalid, "a 2048-thread block is refused before HIP", .{});
    const missing = probe.function("tf_probe_missing");
    try expect(missing == error.NotFound, "an absent symbol is NOT_FOUND", .{});
    check.pass("refusals: oversized block, absent kernel symbol", .{});
}

/// A by-value struct, bool, int8, double and int64 reach the kernel exactly as packed.
fn argPacking(gpu: Gpu, probe: hip.Module, stream: hip.Stream) !void {
    const d = gpu.d;
    const src = [8]f32{ 1, -2, 3.5, 4, 5, 6.25, -7, 8 };
    var s = try hip.DeviceBuffer.fromHost(d, std.mem.asBytes(&src));
    defer s.free();
    var out = try hip.DeviceBuffer.alloc(d, 12 * 4);
    defer out.free();
    var args: hip.Args = .{};
    args.add(out.ptr);
    args.add(ProbeView{ .src = s.ptr, .n = 8, .scale = 0.5 });
    args.add(true);
    args.add(@as(i8, -5));
    args.add(@as(f64, 3.25));
    args.add(@as(i64, -123456789));
    try hip.launch.launch(try probe.function("tf_probe_args"), .{ .grid = .{}, .block = .{ .x = 32 } }, stream, &args);
    try stream.synchronize();
    var got: [12]f32 = undefined;
    try out.download(0, std.mem.asBytes(&got));
    for (0..8) |i| try expect(got[i] == src[i] * 0.5, "struct arg element {d}", .{i});
    const big: f32 = @floatFromInt(@as(i64, -123456789));
    try expect(got[8] == 1 and got[9] == -5 and got[10] == 3.25 and got[11] == big, "scalar args {any}", .{got[8..]});
    check.pass("argument packing: by-value struct, bool, int8, double, int64", .{});
}

/// hipModuleLaunchCooperativeKernel: one block per CU, every block's output exact.
pub fn cooperative(gpu: Gpu) !void {
    const d = gpu.d;
    try expect(try gpu.ctx.attribute(.cooperative_launch) != 0, "the device does not report cooperative launch", .{});
    var probe = try hip.Module.load(d, hip.kernels.probe);
    defer probe.unload();
    var stream = try hip.Stream.init(d, true);
    defer stream.deinit();
    const fill = try probe.function("tf_probe_fill");
    const cus: u32 = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var y = try hip.DeviceBuffer.alloc(d, cus * 256 * 4);
    defer y.free();
    var fa: hip.Args = .{};
    fa.add(y.ptr);
    fa.add(@as(f32, 3));
    fa.add(cus * 256);
    try hip.launch.launch(fill, .{ .grid = .{ .x = cus }, .block = .{ .x = 256 }, .cooperative = true }, stream, &fa);
    try stream.synchronize();
    const ys = try check.download(gpu, y);
    defer gpu.gpa.free(ys);
    for (std.mem.bytesAsSlice(f32, ys), 0..) |v, i| try expect(v == 3 + @as(f32, @floatFromInt(i)), "cooperative fill {d}", .{i});
    check.pass("cooperative grid of {d} blocks (one per CU) exact", .{cus});
}

/// Bytes that are not a code object are refused, and the error is HIP's, not a crash.
pub fn image(gpu: Gpu) !void {
    var junk: [64]u8 align(8) = @splat(0);
    junk[0..4].* = .{ 0x7f, 'E', 'L', 'F' };
    const refused = hip.Module.load(gpu.d, &junk);
    try expect(refused == error.HipFailed, "a broken image must be refused (got {any})", .{refused});
    const empty = hip.Module.load(gpu.d, &.{});
    try expect(empty == error.Invalid, "an empty image is refused before HIP", .{});
    check.pass("images: a broken code object refused by HIP, an empty one before it", .{});
}
