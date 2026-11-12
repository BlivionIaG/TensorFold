//! expert_cache_sim (tools/vega-macos), the offline expert-cache policy replay: `tf-expert-cache-sim` builds the tool, `test-expert-cache-sim` tests it.
const std = @import("std");

fn module(b: *std.Build, file: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path(file), .target = target, .optimize = optimize });
    // the pool the production worker drives (hip/expert_store.zig), which the handshake replay runs: std only
    m.addImport("expert_offload", b.createModule(.{ .root_source_file = b.path("zig/src/core/expert_offload.zig"), .target = target, .optimize = optimize }));
    return m;
}

/// The executable, installed like the other tf-* tools, and the host tests that join `test`; no device, no libc.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    const tool = b.addExecutable(.{ .name = "tf-expert-cache-sim", .root_module = module(b, "tools/vega-macos/expert_cache_sim.zig", target, optimize) });
    b.step("tf-expert-cache-sim", "Build tf-expert-cache-sim, the offline replay of MoE routing traces against expert-cache policies").dependOn(&b.addInstallArtifact(tool, .{}).step);
    const check = b.step("test-expert-cache-sim", "expert_cache_sim's npz reader, random generators, policies and a whole-pipeline replay (offline)");
    check.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module(b, "tools/vega-macos/expert_cache_sim_test.zig", b.graph.host, .ReleaseSafe) })).step);
    test_step.dependOn(check);
}
