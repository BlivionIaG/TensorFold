//! The split core's host tests and `tf-offload`: over the CPU mock device, and on a Mac with Metal as the host.
const std = @import("std");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    const split = b.createModule(.{ .root_source_file = b.path("zig/src/split/split.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = split }));
    b.step("test-split", "Split core: slot protocol, mock device, layer pipeline, planner, probe receipt, placement.json and tf-offload").dependOn(&run.step);
    test_step.dependOn(&run.step);
    if (target.result.os.tag == .macos) return metalSteps(b, target, optimize, test_step, split);
    const main = b.createModule(.{ .root_source_file = b.path("zig/src/split/offload_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tf-offload", .root_module = main });
    b.step("tf-offload", "tf-offload probe | plan | demo over the CPU mock device").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// The Metal runtime as its own module for the programs and tests that join it with the split core.
fn metalModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .optimize = optimize, .link_libc = true });
    for ([_][]const u8{ "Metal", "Foundation", "IOSurface" }) |framework| metal.linkFramework(framework, .{});
    metal.linkSystemLibrary("objc", .{});
    return metal;
}

/// `tf-offload` with this Mac's GPU as the host, and `test-split-metal` (part of `test`): the Metal stage and probe.
fn metalSteps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step, split: *std.Build.Module) void {
    const main = b.createModule(.{
        .root_source_file = b.path("zig/src/split/offload_metal_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metalModule(b, target, optimize) }, .{ .name = "split", .module = split } },
    });
    const exe = b.addExecutable(.{ .name = "tf-offload", .root_module = main });
    b.step("tf-offload", "tf-offload probe | plan | demo: this Mac's GPU through Metal, the mock device as the external one").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = main }));
    b.step("test-split-metal", "Split core with Metal as the host: the toy model against its unsplit run and the Mac's probe, on its GPU").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
