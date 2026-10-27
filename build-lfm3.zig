//! Standalone LFM3 checkpoint checks while native inference remains under development.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const checks = b.step("test", "LFM3 checkpoint admission without a GPU");
    @import("zig/build/lfm3_host.zig").steps(b, checks);
    const image = b.createModule(.{
        .root_source_file = b.path("zig/src/lfm3_image_tests.zig"),
        .target = b.graph.host,
        .link_libc = true,
    });
    const image_run = b.addRunArtifact(b.addTest(.{ .root_module = image }));
    b.step("test-lfm3-image-host", "LFM3 image geometry and shared decoder checks").dependOn(&image_run.step);
    checks.dependOn(&image_run.step);
    @import("zig/build/sys.zig").wire(b);
    b.default_step.dependOn(checks);
}
