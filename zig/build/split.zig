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
}
