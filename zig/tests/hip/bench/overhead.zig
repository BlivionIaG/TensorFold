//! Host cost of a graph: dependent one-thread launches on a stream against the same chain replayed from one graph.

const std = @import("std");
const hip = @import("hip");
const check = @import("../check.zig");
const probes = @import("../runtime/probes.zig");
const Gpu = check.Gpu;
const expect = check.expect;
const readCounter = probes.readCounter;

/// `n` dependent one-thread launches on a stream, then the same chain captured once and replayed: microseconds a launch.
pub fn run(gpu: Gpu, n: usize, reps: usize) !void {
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

