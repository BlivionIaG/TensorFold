//! Runtime-level GPU tests: copies, fills, launches, argument packing, module globals, graphs and refused images.

const std = @import("std");
const hip = @import("hip");
const check = @import("check.zig");
const Gpu = check.Gpu;
const expect = check.expect;

const ProbeView = extern struct { src: u64, n: c_int, scale: f32 };

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

pub fn graphs(gpu: Gpu) !void {
    const d = gpu.d;
    var probe = try hip.Module.load(d, hip.kernels.probe);
    defer probe.unload();
    const step = try probe.function("tf_probe_step");
    var stream = try hip.Stream.init(d, true);
    defer stream.deinit();
    var counter = try hip.DeviceBuffer.alloc(d, 8);
    defer counter.free();
    try counter.fill8(0, null);
    const one: hip.Config = .{ .grid = .{}, .block = .{} };

    try hip.graph.beginCapture(stream, .thread_local);
    try expect(try hip.graph.captureStatus(stream) == .active, "stream is capturing", .{});
    for (1..9) |i| {
        var args: hip.Args = .{};
        args.add(counter.ptr);
        args.add(@as(u64, i));
        try hip.launch.launch(step, one, stream, &args);
    }
    var captured = try hip.graph.endCapture(stream);
    defer captured.deinit();
    try expect(try captured.nodeCount() == 8, "captured graph has 8 nodes (got {d})", .{try captured.nodeCount()});
    try expect(try readCounter(counter) == 0, "capture runs nothing", .{});
    var exec = try captured.instantiate();
    defer exec.deinit();
    try exec.upload(stream);
    for (0..3) |_| try exec.launchOn(stream);
    try stream.synchronize();
    try expect(try readCounter(counter) == 108, "three replays add 3 * 36", .{});
    check.pass("stream capture: 8 launches captured, nothing run, 3 replays = 108", .{});

    var nodes_buf: [8]hip.graph.Node = undefined;
    const nodes = try captured.nodes(&nodes_buf);
    for (nodes) |node| {
        var args: hip.Args = .{};
        args.add(counter.ptr);
        args.add(@as(u64, 10));
        try exec.setKernel(node, step, one, &args);
    }
    try exec.launchOn(stream);
    try stream.synchronize();
    try expect(try readCounter(counter) == 188, "updated nodes add 8 * 10", .{});
    check.pass("exec kernel-node update: new arguments in place, replay adds 80", .{});

    try explicitGraphs(gpu, probe, stream);
}

fn readCounter(counter: hip.DeviceBuffer) !u64 {
    var v: u64 = 0;
    try counter.download(0, std.mem.asBytes(&v));
    return v;
}

fn fillAxpy(g: hip.graph.Graph, fill: hip.Function, axpy: hip.Function, y: u64, x: u64, base: f32, a: f32, n: u32) !void {
    const cfg: hip.Config = .{ .grid = .{ .x = (n + 255) / 256 }, .block = .{ .x = 256 } };
    var fa: hip.Args = .{};
    fa.add(y);
    fa.add(base);
    fa.add(n);
    const first = try g.addKernel(&.{}, fill, cfg, &fa);
    var aa: hip.Args = .{};
    aa.add(y);
    aa.add(x);
    aa.add(a);
    aa.add(n);
    _ = try g.addKernel(&.{first}, axpy, cfg, &aa);
}

/// Explicit nodes with an edge, then a whole-exec update from a same-topology graph, and a refused topology change.
fn explicitGraphs(gpu: Gpu, probe: hip.Module, stream: hip.Stream) !void {
    const d = gpu.d;
    const n: u32 = 4096;
    const fill = try probe.function("tf_probe_fill");
    const axpy = try probe.function("tf_probe_axpy");
    var x = try hip.DeviceBuffer.alloc(d, n * 4);
    defer x.free();
    var y = try hip.DeviceBuffer.alloc(d, n * 4);
    defer y.free();
    var args: hip.Args = .{};
    args.add(x.ptr);
    args.add(@as(f32, 0));
    args.add(n);
    try hip.launch.launch(fill, .{ .grid = .{ .x = n / 256 }, .block = .{ .x = 256 } }, stream, &args);

    var g1 = try hip.graph.Graph.init(d);
    defer g1.deinit();
    try fillAxpy(g1, fill, axpy, y.ptr, x.ptr, 1, 2, n);
    var exec = try g1.instantiate();
    defer exec.deinit();
    try exec.launchOn(stream);
    try stream.synchronize();
    try expectAxpy(gpu, y, 1, 2, n);

    var g2 = try hip.graph.Graph.init(d);
    defer g2.deinit();
    try fillAxpy(g2, fill, axpy, y.ptr, x.ptr, 5, 3, n);
    const r2 = try exec.update(g2);
    try expect(r2 == .success, "same-topology update result {t}", .{r2});
    try exec.launchOn(stream);
    try stream.synchronize();
    try expectAxpy(gpu, y, 5, 3, n);

    var g3 = try hip.graph.Graph.init(d);
    defer g3.deinit();
    try fillAxpy(g3, fill, axpy, y.ptr, x.ptr, 7, 4, n);
    var extra: hip.Args = .{};
    extra.add(x.ptr);
    extra.add(@as(f32, 0));
    extra.add(n);
    _ = try g3.addKernel(&.{}, fill, .{ .grid = .{ .x = n / 256 }, .block = .{ .x = 256 } }, &extra);
    const r3 = try exec.update(g3);
    try expect(r3 != .success, "a topology change must be refused", .{});
    try exec.launchOn(stream);
    try stream.synchronize();
    try expectAxpy(gpu, y, 5, 3, n);
    check.pass("explicit graph: fill->axpy edge exact; exec update from a same-topology graph; topology change refused ({t}) and the exec kept", .{r3});
}

fn expectAxpy(gpu: Gpu, y: hip.DeviceBuffer, base: f32, a: f32, n: u32) !void {
    const got = try check.download(gpu, y);
    defer gpu.gpa.free(got);
    for (std.mem.bytesAsSlice(f32, got)[0..n], 0..) |v, i| {
        const fi: f32 = @floatFromInt(i);
        try expect(v == a * fi + (base + fi), "graph axpy element {d}: {d}", .{ i, v });
    }
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

/// `n` dependent one-thread launches on a stream, then the same chain captured once and replayed: microseconds a launch.
pub fn overhead(gpu: Gpu, n: usize, reps: usize) !void {
    const d = gpu.d;
    var probe = try hip.Module.load(d, hip.kernels.probe);
    defer probe.unload();
    const step = try probe.function("tf_probe_step");
    var stream = try hip.Stream.init(d, true);
    defer stream.deinit();
    var counter = try hip.DeviceBuffer.alloc(d, 8);
    defer counter.free();
    try counter.fill8(0, null);
    const one: hip.Config = .{ .grid = .{}, .block = .{} };
    const times = try gpu.gpa.alloc(f64, reps);
    defer gpu.gpa.free(times);
    for (times) |*t| {
        const t0 = check.now(gpu.io);
        for (0..n) |_| {
            var args: hip.Args = .{};
            args.add(counter.ptr);
            args.add(@as(u64, 1));
            try hip.launch.launch(step, one, stream, &args);
        }
        try stream.synchronize();
        t.* = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1000 / @as(f64, @floatFromInt(n));
    }
    const plain = check.median(times);
    try hip.graph.beginCapture(stream, .thread_local);
    for (0..n) |_| {
        var args: hip.Args = .{};
        args.add(counter.ptr);
        args.add(@as(u64, 1));
        try hip.launch.launch(step, one, stream, &args);
    }
    var g = try hip.graph.endCapture(stream);
    defer g.deinit();
    var exec = try g.instantiate();
    defer exec.deinit();
    try exec.upload(stream);
    for (times) |*t| {
        const t0 = check.now(gpu.io);
        try exec.launchOn(stream);
        try stream.synchronize();
        t.* = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1000 / @as(f64, @floatFromInt(n));
    }
    const graphed = check.median(times);
    const want: u64 = @intCast(2 * n * reps);
    const ran = try readCounter(counter);
    try expect(ran == want, "every launch ran once: {d} steps, expected {d}", .{ ran, want });
    std.debug.print("RESULT {d} dependent launches: {d:.2} us each on a stream, {d:.2} us each in a graph\n", .{ n, plain, graphed });
}

/// The embedded kernel library of this GPU's family opens from memory, reports its build and runs a kernel.
pub fn library(gpu: Gpu) !void {
    const cap = try gpu.ctx.capability();
    const family = hip.rocm.familyOf(cap) orelse return error.UnsupportedGpu;
    var lib = try hip.rocm.Library.open(family);
    defer lib.close();
    try expect(lib.api.tf_wmma_build() == @intFromBool(family == .rdna3), "the {t} library's WMMA switch", .{family});
    const d = gpu.d;
    var stream = try hip.Stream.init(d, true);
    defer stream.deinit();
    const width = 64;
    var xs: [2 * width]f32 = undefined;
    for (&xs, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) - 3;
    var x = try hip.DeviceBuffer.fromHost(d, std.mem.asBytes(&xs));
    defer x.free();
    var y = try hip.DeviceBuffer.alloc(d, xs.len * 4);
    defer y.free();
    try lib.check(lib.api.tf_rms(@ptrFromInt(x.ptr), null, @ptrFromInt(y.ptr), 0, 2, width, 1e-6, stream.handle), "tf_rms");
    try stream.synchronize();
    var ys: [2 * width]f32 = undefined;
    try y.download(0, std.mem.asBytes(&ys));
    var ss: f64 = 0;
    for (xs[0..width]) |v| ss += v * v;
    const want = xs[1] / @sqrt(ss / width + 1e-6);
    try expect(@abs(ys[1] - want) < 1e-5, "rms row 0 element 1: {d} vs {d}", .{ ys[1], want });
    check.pass("kernel library: {t} opened from memory, WMMA {d}, rms runs", .{ family, lib.api.tf_wmma_build() });
}
