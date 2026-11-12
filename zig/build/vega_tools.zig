//! The macOS + Thunderbolt/eGPU support stack's developer tools (tools/vega-macos), all Zig:
//! `zig build tf-prof-report` (a hipprof table as a per-token breakdown), `zig build tf-vega-gate` (the Qwen3.5-4B token and logits gate and decode benchmark
//! on the HIP runtime), and `zig build test-vega-tools`: unit tests of the tools and of their shared modules, and the gate run as a process against a stand-in
//! for `tensorfold-native` and `tf-qwen35-test` (tools/vega-macos/gate_fake.zig), passing and failing. Offline; no card.
const std = @import("std");

const Tool = struct {
    /// the executable (installed to zig-out/bin) and its build step
    name: []const u8,
    /// its root source file; its tests run under `test-vega-tools`
    root: []const u8,
    help: []const u8,
};

const tools = [_]Tool{
    .{ .name = "tf-prof-report", .root = "tools/vega-macos/prof_report.zig", .help = "Build tf-prof-report, a hipprof table as a per-token breakdown" },
    .{ .name = "tf-vega-gate", .root = "tools/vega-macos/vega_gate.zig", .help = "Build tf-vega-gate, the gfx900 token / logits gate and decode benchmark against mlx_lm" },
};

// modules without an executable of their own whose tests also run
const extra_tests = [_][]const u8{
    "tools/vega-macos/npy.zig",
    "tools/vega-macos/pyfmt.zig",
    "tools/vega-macos/pyjson.zig",
    "tools/vega-macos/cli.zig",
    "tools/vega-macos/gate_lib.zig",
};

fn module(b: *std.Build, file: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path(file), .target = target, .optimize = optimize, .link_libc = true });
}

/// The tools, installed like the other tf-* tools, and their host tests that join `test`; not on Windows.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    if (target.result.os.tag == .windows or b.graph.host.result.os.tag == .windows) return;
    const host = b.graph.host;
    const check = b.step("test-vega-tools", "Tests of the tf-* tools of tools/vega-macos, the gates also as processes over a fake server (offline, no card)");
    for (tools) |tool| {
        const exe = b.addExecutable(.{ .name = tool.name, .root_module = module(b, tool.root, target, optimize) });
        b.step(tool.name, tool.help).dependOn(&b.addInstallArtifact(exe, .{}).step);
        check.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module(b, tool.root, host, .Debug) })).step);
    }
    for (extra_tests) |file| check.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module(b, file, host, .Debug) })).step);

    // the gate as a process: gate_check runs it against gate_fake
    const gate = b.addExecutable(.{ .name = "tf-vega-gate", .root_module = module(b, "tools/vega-macos/vega_gate.zig", host, .ReleaseSafe) });
    const fake = b.addExecutable(.{ .name = "gate_fake", .root_module = module(b, "tools/vega-macos/gate_fake.zig", host, .ReleaseSafe) });
    const driver = b.addExecutable(.{ .name = "gate_check", .root_module = module(b, "tools/vega-macos/gate_check.zig", host, .ReleaseSafe) });
    const run = b.addRunArtifact(driver);
    run.addArtifactArg(gate);
    run.addArg("-"); // no tf-moe-gate in this tree
    run.addArtifactArg(fake);
    _ = run.addOutputDirectoryArg("gate-check");
    check.dependOn(&run.step);
    test_step.dependOn(check);
}
