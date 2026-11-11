//! Build steps of the split (offload) programs: the card half, the HIP external device program, and the host tests of the
//! split stage's plumbing. Kept out of hip.zig, which holds the engine's own steps.
const std = @import("std");
const hip = @import("hip.zig");

/// The layer-range and split-stage plumbing runs a recording stand-in for the HIP runtime on kernels that name gfx1030.
fn splitHostTests(b: *std.Build, test_step: *std.Build.Step) void {
    const split_kernels = b.createModule(.{ .root_source_file = b.addWriteFiles().add("kernels.zig", split_stub) });
    const split_host = hip.engineModules(b, b.graph.host, .Debug, split_kernels, null);
    const family_split = b.addRunArtifact(b.addTest(.{ .root_module = split_host.family, .filters = &.{"split stage:"} }));
    test_step.dependOn(&family_split.step);
    b.step("test-qwen35-split-host", "Qwen HIP layer ranges and the split stage: dry-run launches and plumbing, no device").dependOn(&family_split.step);
}

/// `tf-offload-card`: the HIP side of a model split between this Mac's GPU and a card, one process of two. The Mac's
/// side is `tf-offload-metal` (zig/build/qwen27.zig); they meet over a socket and the window's shared memory, because one
/// program cannot hold both engines (a source file belongs to one module of a program).
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, e: hip.Engine, test_step: *std.Build.Step) void {
    splitHostTests(b, test_step);
    const exe = b.addExecutable(.{ .name = "tf-offload-card", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/qwen27/split_card.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = e.hip }, .{ .name = "qwen3_5", .module = e.family }, .{ .name = "split", .module = e.hip.import_table.get("split").? } },
    }) });
    b.step("tf-offload-card", "The HIP half of the Qwen3.5-4B split between this Mac's GPU and a card (opens the card)").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// `tf-offload-hip`: tf-offload with a HIP card as an external device, over hipmac's runtime (TENSORFOLD_HIP or the search path).
/// Compiles on any macOS host against stub kernels; running it against a card is a step of its own.
pub fn offloadExe(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module) void {
    const stub_kernels = b.createModule(.{ .root_source_file = b.addWriteFiles().add("kernels.zig", hip.stub) });
    const e = hip.engineModules(b, target, optimize, stub_kernels, null);
    const offload = b.addOptions();
    offload.addOption(bool, "with_hip", true);
    const module = b.createModule(.{
        .root_source_file = b.path("zig/src/split/offload_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        // hip's own split instance: the device types must be the very ones the adapter returns
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "split", .module = e.hip.import_table.get("split").? }, .{ .name = "hip", .module = e.hip } },
    });
    module.addOptions("offload_options", offload);
    const exe = b.addExecutable(.{ .name = "tf-offload-hip", .root_module = module });
    b.step("tf-offload-hip", "tf-offload probe | plan | demo with the HIP card as an external device (macOS)").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// `stub` naming gfx1030 (RDNA2: the caps table row with fp16 activations, as the gfx900 target has) with a code object that is eight bytes (a stand-in runtime loads anything): the launcher loads.
const split_stub = std.fmt.comptimePrint("pub const arch = \"gfx1030\";\npub const g9_mix: u8 = 2;\nconst blob align(8) = [8]u8{{ 0, 1, 2, 3, 4, 5, 6, 7 }};\npub const images: [{d}][]align(8) const u8 = @splat(&blob);\n", .{hip.groups.len});
