//! The split core's host tests and `tf-offload`, over the CPU mock device with no GPU.
const std = @import("std");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    const split = b.createModule(.{ .root_source_file = b.path("zig/src/split/split.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = split }));
    b.step("test-split", "Split core: slot protocol, mock device, layer pipeline, planner, probe receipt, placement.json and tf-offload").dependOn(&run.step);
    test_step.dependOn(&run.step);
    const main = b.createModule(.{ .root_source_file = b.path("zig/src/split/offload_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tf-offload", .root_module = main });
    b.step("tf-offload", "tf-offload probe | plan | demo over the CPU mock device").dependOn(&b.addInstallArtifact(exe, .{}).step);
    if (target.result.os.tag == .macos) metalSteps(b, target, optimize, test_step, split);
}

/// The Metal runtime as its own module for the programs and tests that join it with the split core.
fn metalModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .optimize = optimize, .link_libc = true });
    for ([_][]const u8{ "Metal", "Foundation", "IOSurface" }) |framework| metal.linkFramework(framework, .{});
    metal.linkSystemLibrary("objc", .{});
    return metal;
}

/// `test-split-metal` (part of `test`): the toy model split at several layers, Metal on either side of the mock device.
fn metalSteps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step, split: *std.Build.Module) void {
    const toy = b.createModule(.{
        .root_source_file = b.path("zig/src/split/metal_toy.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metalModule(b, target, optimize) }, .{ .name = "split", .module = split } },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = toy }));
    b.step("test-split-metal", "Split core with Metal as the host stage: the toy model against its unsplit run, on this Mac's GPU").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
