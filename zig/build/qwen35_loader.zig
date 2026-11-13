//! The Qwen3.5 / 3.6 checkpoint loader tests: part of `test` on every build, HIP or not.
const std = @import("std");
const hip = @import("hip.zig");

pub fn tests(b: *std.Build, test_step: *std.Build.Step) void {
    const core = hip.coreModule(b, b.graph.host);
    const loader = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen3_5/loader.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = core }},
    }) }));
    loader.has_side_effects = true; // TF_QWEN_DIR is not a cached input
    test_step.dependOn(&loader.step);
    loader.step.dependOn(@import("mesh.zig").qwenDeclTest(b, core)); // its cuts as declarations
    b.step("test-qwen35", "Qwen3.5 / 3.6 checkpoint loader tests; TF_QWEN_DIR indexes a real checkpoint").dependOn(&loader.step);
}
