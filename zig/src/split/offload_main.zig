//! tf-offload: `probe`, `plan` and `demo` on a Metal host. `tf-offload` opens only the CPU mock device; `tf-offload-hip`
//! (same file, `with_hip`) adds a HIP card through hipmac. `demo` runs the toy layer pipeline between the mock device
//! and Metal.
const std = @import("std");
const metal = @import("metal");
const split = @import("split");
const options = @import("offload_options");
const metal_probe = @import("metal_probe.zig");
const metal_toy = @import("metal_toy.zig");

const toy = split.toy;
const pipeline = split.pipeline;

fn openMock(gpa: std.mem.Allocator, _: c_int) anyerror!split.device.ExternalDevice {
    const m = try split.mock.Mock.create(gpa, .{ .name = "mock", .act = .f16 });
    return m.external();
}

/// The HIP card at `index`: the runtime stays open for the life of the process (the device borrows it).
fn openHip(gpa: std.mem.Allocator, index: c_int) anyerror!split.device.ExternalDevice {
    const hip = @import("hip");
    const rt = try gpa.create(hip.Runtime);
    rt.* = try hip.Runtime.open();
    var arch_buf: [64]u8 = undefined;
    const arch = try hip.Device.archName(rt, index, &arch_buf);
    const peak = peakBandwidth(arch) orelse return error.UnknownPeakBandwidth;
    const d = try hip.offload.HipExternal.create(gpa, rt, index, .{ .peak_bandwidth = peak, .pump_mock = hipmacMock() });
    return d.external();
}

/// hipmac's mock backend (no card) runs a stream only while it is synchronized, so the adapter pumps it.
fn hipmacMock() bool {
    const v = std.c.getenv("HIPMAC_BACKEND") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "mock");
}

/// Datasheet memory bandwidth by architecture, bytes a second; the planner scales it by a measured efficiency. gfx900 is
/// the Vega 56's 410 GB/s (the Vega 64 has 484). Cards not listed need a row here, with its datasheet number.
fn peakBandwidth(arch: []const u8) ?u64 {
    if (std.mem.startsWith(u8, arch, "gfx900")) return 410_000_000_000;
    if (std.mem.startsWith(u8, arch, "gfx1100")) return 960_000_000_000;
    if (std.mem.startsWith(u8, arch, "gfx1151")) return 256_000_000_000;
    return null;
}

pub const backends = if (options.with_hip)
    [_]split.cli.Backend{ .{ .name = "mock", .open = openMock }, .{ .name = "hip", .open = openHip } }
else
    [_]split.cli.Backend{.{ .name = "mock", .open = openMock }};

const demo_usage = "usage: tf-offload demo [--order external_first|metal_first] [--level a|b] [--split K] [--tokens N]\n";

/// Run the toy model split at layer K between the mock device and Metal and compare with the unsplit run.
fn demo(args: []const []const u8, out: *std.Io.Writer) !u8 {
    var order: pipeline.Order = .external_first;
    var level: pipeline.Level = .a;
    var k: usize = toy.layers / 2;
    var n: usize = 32;
    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) return usageErr(out);
        const v = args[i + 1];
        if (std.mem.eql(u8, args[i], "--order")) {
            order = std.meta.stringToEnum(pipeline.Order, v) orelse return usageErr(out);
        } else if (std.mem.eql(u8, args[i], "--level")) {
            level = std.meta.stringToEnum(pipeline.Level, v) orelse return usageErr(out);
        } else if (std.mem.eql(u8, args[i], "--split")) {
            k = std.fmt.parseInt(usize, v, 10) catch return usageErr(out);
        } else if (std.mem.eql(u8, args[i], "--tokens")) {
            n = std.fmt.parseInt(usize, v, 10) catch return usageErr(out);
        } else return usageErr(out);
    }
    if (k < 1 or k >= toy.layers or n == 0 or n > 4096 or order == .external_only) return usageErr(out);
    const gpa = std.heap.c_allocator;
    const got = try gpa.alloc(u32, n);
    defer gpa.free(got);
    const want = try gpa.alloc(u32, n);
    defer gpa.free(want);
    const prompt = [_]u32{ 5, 17, 3, 90 };
    const fixture = try metal_toy.Fixture.init(gpa);
    defer fixture.deinit(gpa);
    try toy.reference(fixture.params, &prompt, want);
    const t0 = split.device.nowNs();
    try metal_toy.runSplit(gpa, fixture, order, level, k, &prompt, got);
    const ms = @as(f64, @floatFromInt(split.device.nowNs() - t0)) / 1e6;
    const same = std.mem.eql(u32, got, want);
    try out.print("toy model, {d} layers split at {d} ({s}, level {s}): {d} tokens in {d:.1} ms, {s} the unsplit run\n", .{ toy.layers, k, @tagName(order), @tagName(level), n, ms, if (same) "equal to" else "DIFFERENT FROM" });
    return if (same) 0 else 1;
}

fn usageErr(out: *std.Io.Writer) !u8 {
    try out.writeAll(demo_usage);
    return 2;
}

pub fn main(init: std.process.Init) !u8 {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len >= 2 and std.mem.eql(u8, argv[1], "demo")) {
        var buf: [4096]u8 = undefined;
        var w = std.Io.File.stdout().writerStreaming(init.io, &buf);
        const args = try init.arena.allocator().alloc([]const u8, argv.len - 2);
        for (args, argv[2..]) |*x, y| x.* = y;
        const code = try demo(args, &w.interface);
        try w.interface.flush();
        return code;
    }
    var host: metal_probe.MetalHost = .{};
    return split.cli.main(init, argv[1..], host.host(), &backends);
}

test {
    _ = metal_toy;
    _ = metal_probe;
}
