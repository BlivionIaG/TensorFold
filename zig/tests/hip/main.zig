//! GPU test runner for the Zig HIP runtime: `tf-hip-test <command> [args]`, PASS/FAIL/RESULT lines, exit 1 on failure.

const std = @import("std");
const hip = @import("hip");
const check = @import("check.zig");
const runtime_tests = @import("runtime_tests.zig");

const usage =
    \\usage: tf-hip-test <command>
    \\  info                      runtime, device and embedded kernel targets
    \\  smoke                     copies, fills, launches, argument packing, module globals, refusals
    \\  graph                     stream capture, explicit graphs, node and whole-exec updates
    \\  cooperative               a cooperative grid of one block per CU
    \\  image                     an offload bundle without this GPU's target, and a broken image, refused
    \\  overhead [n] [reps]       dependent one-thread kernels: plain stream vs one graph
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var driver = try hip.Driver.open();
    defer driver.close();
    var ctx = try hip.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    const cmd = args[1];
    const rest = args[2..];

    run(gpu, cmd, rest) catch |e| {
        std.debug.print("FAIL {s}: {t}\n", .{ cmd, e });
        return 1;
    };
    return 0;
}

fn run(gpu: check.Gpu, cmd: []const u8, rest: []const [:0]const u8) !void {
    if (std.mem.eql(u8, cmd, "info")) return info(gpu);
    if (std.mem.eql(u8, cmd, "smoke")) return runtime_tests.smoke(gpu);
    if (std.mem.eql(u8, cmd, "graph")) return runtime_tests.graphs(gpu);
    if (std.mem.eql(u8, cmd, "cooperative")) return runtime_tests.cooperative(gpu);
    if (std.mem.eql(u8, cmd, "image")) return runtime_tests.image(gpu);
    if (std.mem.eql(u8, cmd, "overhead")) {
        const n = if (rest.len > 0) try std.fmt.parseInt(usize, rest[0], 10) else 1000;
        const reps = if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 20;
        return runtime_tests.overhead(gpu, n, reps);
    }
    std.debug.print("{s}", .{usage});
    return error.UnknownCommand;
}

fn info(gpu: check.Gpu) !void {
    var name_buf: [256]u8 = undefined;
    const name = try gpu.ctx.name(&name_buf);
    const mem = try gpu.ctx.memInfo();
    std.debug.print("RESULT HIP {d}, device {s}, gfx {d}, {d} CUs, wave {d}, {d} MiB total, {d} MiB free, kernels embedded {} ({s})\n", .{
        try gpu.d.version(),               name,
        try gpu.ctx.capability(),          try gpu.ctx.attribute(.multiprocessor_count),
        try gpu.ctx.attribute(.warp_size), mem.total >> 20,
        mem.free >> 20,                    hip.kernels.available,
        hip.kernels.targets,
    });
}
