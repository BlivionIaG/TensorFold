//! The HIP half of the root build: the runtime module and its host tests.

const std = @import("std");

/// The runtime module for `target`.
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/hip/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

/// Host unit tests of the HIP runtime (no GPU), on any host.
pub fn hostTests(b: *std.Build, step: *std.Build.Step) void {
    const hip = runtime(b, b.graph.host, .debug);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = hip })).step);
}
