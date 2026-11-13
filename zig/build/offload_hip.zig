//! Build steps of the HIP side of a split model: host tests of the layer range and the external stage.
const std = @import("std");
const hip = @import("hip.zig");

/// The kernels module naming gfx1030 with eight-byte code objects, which the stand-in runtime loads: a launcher loads.
const split_stub = std.fmt.comptimePrint("pub const arch = \"gfx1030\";\nconst blob align(8) = [8]u8{{ 0, 1, 2, 3, 4, 5, 6, 7 }};\npub const images: [{d}][]align(8) const u8 = @splat(&blob);\n", .{hip.groups.len});

/// `test-qwen35-split-host` and `test-hip-offload-host` (no device, a stand-in runtime), and `tf-offload-hip`.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    offloadExe(b, target, optimize);
    const kernels = b.createModule(.{ .root_source_file = b.addWriteFiles().add("kernels.zig", split_stub) });
    const host = hip.engineModules(b, b.graph.host, .Debug, kernels, null);
    const stage = b.addRunArtifact(b.addTest(.{ .root_module = host.family, .filters = &.{ "layer range:", "split stage:" } }));
    test_step.dependOn(&stage.step);
    b.step("test-qwen35-split-host", "Qwen HIP layer ranges and the split stage: dry-run launches against a recording runtime").dependOn(&stage.step);
    const adapter = b.addRunArtifact(b.addTest(.{ .root_module = host.hip, .filters = &.{ "HipExternal:", "split rows:" } }));
    test_step.dependOn(&adapter.step);
    b.step("test-hip-offload-host", "The HIP external device on a stand-in runtime, and the split rows' wire layout").dependOn(&adapter.step);
}

/// `tf-offload-hip`: tf-offload with a HIP card as an external device, built against inert kernels on any host.
fn offloadExe(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const inert = b.createModule(.{ .root_source_file = b.addWriteFiles().add("kernels.zig", hip.stub) });
    const e = hip.engineModules(b, target, optimize, inert, null);
    const module = b.createModule(.{
        .root_source_file = b.path("zig/src/hip/offload_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "hip", .module = e.hip }},
    });
    const exe = b.addExecutable(.{ .name = "tf-offload-hip", .root_module = module });
    b.step("tf-offload-hip", "tf-offload probe | plan | demo with a HIP card as an external device").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
