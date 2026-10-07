//! GPU test runner for the Zig HIP runtime: `tf-hip-test <command>`, PASS/FAIL lines, exit 1 on failure.

const std = @import("std");
const hip = @import("hip");
const check = @import("check.zig");
const runtime_tests = @import("runtime_tests.zig");
const affine_tests = @import("affine_tests.zig");

const usage =
    \\usage: tf-hip-test <command>
    \\  info          runtime and device, its gfx name and caps, the embedded kernel targets
    \\  smoke         copies, async copies through pinned memory, fills, launches, argument packing, module globals
    \\  cooperative   a cooperative grid of one block per CU
    \\  image         a broken and an empty code object are refused
    \\  runtime       info, smoke, cooperative and image
    \\  affine        the affine 4-bit product: the host recipe's bits at 1, 3 and 16 rows, the float64 bound, a pinned digest
    \\  all           runtime, then affine
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
    run(gpu, cmd) catch |e| {
        std.debug.print("FAIL {s}: {t}\n", .{ cmd, e });
        return 1;
    };
    return 0;
}

fn run(gpu: check.Gpu, cmd: []const u8) !void {
    const eql = std.mem.eql;
    if (eql(u8, cmd, "info")) return runtime_tests.info(gpu);
    if (eql(u8, cmd, "smoke")) return runtime_tests.smoke(gpu);
    if (eql(u8, cmd, "cooperative")) return runtime_tests.cooperative(gpu);
    if (eql(u8, cmd, "image")) return runtime_tests.image(gpu);
    if (eql(u8, cmd, "affine")) return affine_tests.run(gpu);
    if (eql(u8, cmd, "runtime") or eql(u8, cmd, "all")) {
        try runtime_tests.info(gpu);
        try runtime_tests.smoke(gpu);
        try runtime_tests.cooperative(gpu);
        try runtime_tests.image(gpu);
        if (eql(u8, cmd, "all")) try affine_tests.run(gpu);
        return;
    }
    std.debug.print("{s}", .{usage});
    return error.UnknownCommand;
}
