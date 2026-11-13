//! The split core's host tests: protocol, mock device and layer pipeline, with no GPU.
const std = @import("std");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    const split = b.createModule(.{ .root_source_file = b.path("zig/src/split/split.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = split }));
    b.step("test-split", "Split core: slot protocol, mock device and layer pipeline against the unsplit toy model").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
