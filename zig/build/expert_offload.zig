//! The expert offload's device-free tests (core/expert_*): policy, window layout, stage and the worker's handshake.
const std = @import("std");

pub fn hostTests(b: *std.Build, all: *std.Build.Step) void {
    const mod = b.createModule(.{ .root_source_file = b.path("zig/src/core/expert_offload_tests.zig"), .target = b.graph.host, .link_libc = true });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = mod }));
    all.dependOn(&run.step);
    b.step("test-expert-offload-host", "Expert offload policy, window, stage and worker handshake without a device").dependOn(&run.step);
}
