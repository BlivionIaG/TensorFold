//! The DeepSeek-V4.1 GLP producer: the derive tool over the EXL3 Metal family, and the modules it shares.
const std = @import("std");

/// The tool is macOS-only (the family is Metal); its statistics are checked by `test-glp` instead.
pub fn steps(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module) void {
    const target = b.graph.host;
    const glp_mod = b.createModule(.{
        .root_source_file = b.path("zig/src/core/glp.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
    });
    const derive_mod = b.createModule(.{
        .root_source_file = b.path("zig/src/core/glp_derive.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    const prompt_mod = b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/glp_prompts.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    // The engine's own tokenizer module: one file belongs to one module, so the tool borrows it.
    const tok = engine.import_table.get("tokenizer").?;
    const tool = b.addExecutable(.{ .name = "tf-dsv41-glp-derive", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/dsv41_glp_derive.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{
            .{ .name = "tensorfold", .module = engine },
            .{ .name = "metal", .module = metal },
            .{ .name = "glp", .module = glp_mod },
            .{ .name = "glp_derive", .module = derive_mod },
            .{ .name = "glp_prompts", .module = prompt_mod },
            .{ .name = "tokenizer", .module = tok },
        },
    }) });
    b.step("tf-dsv41-glp-derive", "Derive a DeepSeek-V4.1 GLP vector at hc_fold_pre_engram and write the file")
        .dependOn(&b.addInstallArtifact(tool, .{}).step);

    // The producer's own device gate, on this GPU: the capture kernel against a float64 reduction.
    // The engine module is the root `tensorfold.zig` reaches `metal/glp_capture.zig` through, so the test
    // travels with it; the filter keeps the run to the producer's own tests.
    const gates = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/deepseek_v41_capture_check.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal },
            .{ .name = "kernel_sources", .module = engine.import_table.get("kernel_sources").? },
        },
    }) });
    const gates_run = b.addRunArtifact(gates);
    gates_run.has_side_effects = true; // a device run: never cached
    b.step("test-dsv41-glp-capture", "The GLP producer's capture: the fold's own four-copy mean, read-only").dependOn(&gates_run.step);
}
