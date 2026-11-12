//! Build steps of the HIP side of a split model: host tests of the layer range. Kept out of hip.zig, which is full.
const std = @import("std");
const hip = @import("hip.zig");

/// The kernels module naming gfx1030 with eight-byte code objects, which the stand-in runtime loads: a launcher loads.
const split_stub = std.fmt.comptimePrint("pub const arch = \"gfx1030\";\nconst blob align(8) = [8]u8{{ 0, 1, 2, 3, 4, 5, 6, 7 }};\npub const images: [{d}][]align(8) const u8 = @splat(&blob);\n", .{hip.groups.len});

/// `test-qwen35-split-host`: layer ranges against a recording stand-in for the HIP runtime, no device.
pub fn hostSteps(b: *std.Build, test_step: *std.Build.Step) void {
    const kernels = b.createModule(.{ .root_source_file = b.addWriteFiles().add("kernels.zig", split_stub) });
    const host = hip.engineModules(b, b.graph.host, .Debug, kernels, null);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = host.family, .filters = &.{"layer range:"} }));
    test_step.dependOn(&run.step);
    b.step("test-qwen35-split-host", "Qwen HIP layer ranges: dry-run launches against a recording runtime, no device").dependOn(&run.step);
}
