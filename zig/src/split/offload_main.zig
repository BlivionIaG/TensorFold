//! tf-offload: probe, plan and demo over the CPU mock device and a CPU stand-in for the Mac's GPU.
const std = @import("std");
const split = @import("split.zig");

fn openMock(gpa: std.mem.Allocator, _: c_int) anyerror!split.device.ExternalDevice {
    const m = try split.mock.Mock.create(gpa, .{ .name = "mock", .act = .f16 });
    return m.external();
}

pub fn main(init: std.process.Init) !u8 {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    var cpu: split.cpu_host.CpuHost = .{};
    return split.cli.main(init, argv[1..], cpu.host(), &.{.{ .name = "mock", .open = openMock }});
}
