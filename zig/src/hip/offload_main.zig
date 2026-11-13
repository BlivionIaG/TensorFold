//! tf-offload-hip: tf-offload with a HIP card as an external device, over the runtime TENSORFOLD_HIP names.
const std = @import("std");
const hip = @import("hip");

const split = hip.split;

fn openMock(gpa: std.mem.Allocator, _: c_int) anyerror!split.device.ExternalDevice {
    const m = try split.mock.Mock.create(gpa, .{ .name = "mock", .act = .f16 });
    return m.external();
}

/// Datasheet memory bandwidth by architecture, bytes a second; the planner scales it by a measured efficiency.
fn peakBandwidth(arch: []const u8) ?u64 {
    if (std.mem.startsWith(u8, arch, "gfx900")) return 410_000_000_000;
    if (std.mem.startsWith(u8, arch, "gfx1100")) return 960_000_000_000;
    if (std.mem.startsWith(u8, arch, "gfx1151")) return 256_000_000_000;
    return null;
}

/// The HIP card at `index`: the runtime stays open for the life of the process (the device borrows it).
fn openHip(gpa: std.mem.Allocator, index: c_int) anyerror!split.device.ExternalDevice {
    const rt = try gpa.create(hip.Runtime);
    rt.* = try hip.Runtime.open();
    var arch_buf: [64]u8 = undefined;
    const arch = try hip.Device.archName(rt, index, &arch_buf);
    const peak = peakBandwidth(arch) orelse return error.UnknownPeakBandwidth;
    const card = try hip.offload.HipExternal.create(gpa, rt, index, .{ .peak_bandwidth = peak });
    return card.external();
}

pub fn main(init: std.process.Init) !u8 {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    var cpu: split.cpu_host.CpuHost = .{};
    const backends = [_]split.cli.Backend{ .{ .name = "mock", .open = openMock }, .{ .name = "hip", .open = openHip } };
    return split.cli.main(init, argv[1..], cpu.host(), &backends);
}
