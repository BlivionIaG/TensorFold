//! TB4 offload: the device-agnostic `split` module (no GPU imports), its Metal half, `tf-offload` and `tf-offload-hip`.

const std = @import("std");
const hip_build = @import("hip.zig");


/// `test-split` (part of `test`): protocol, planner, mock device, layer pipeline, Metal window and relay. `tf-offload` and
/// `tf-offload-hip` are installable steps; the latter compiles on a macOS host against stub kernels and opens a card only
/// when run with `--device hip`.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step, metal: *std.Build.Module) void {
    const split = b.createModule(.{ .root_source_file = b.path("zig/src/split/split.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const options = b.addOptions();
    options.addOption(bool, "with_hip", false);
    const split_metal = b.createModule(.{
        .root_source_file = b.path("zig/src/split/offload_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "split", .module = split } },
    });
    split_metal.addOptions("offload_options", options);
    const exe = b.addExecutable(.{ .name = "tf-offload", .root_module = split_metal });
    b.step("tf-offload", "tf-offload probe | plan | demo: TB4 external-GPU offload (mock device on this build)").dependOn(&b.addInstallArtifact(exe, .{}).step);
    hip_build.offloadExe(b, target, optimize, metal);
    const tests = b.step("test-split", "TB4 offload: protocol, planner, mock device, layer pipeline, Metal window and relay");
    for ([_]*std.Build.Module{ split, split_metal }) |m| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
        tests.dependOn(&run.step);
        test_step.dependOn(&run.step);
    }
}
