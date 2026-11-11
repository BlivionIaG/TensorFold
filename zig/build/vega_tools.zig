//! The macOS + Thunderbolt/eGPU support stack's developer tools (tools/vega-macos): `zig build tf-prof-report` and the host tests, `zig build test-vega-tools`.
const std = @import("std");

const Tool = struct {
    /// the executable (installed to zig-out/bin) and its build step
    name: []const u8,
    /// its root source file; the tests of the tool and of the modules it imports run under `test-vega-tools`
    root: []const u8,
    help: []const u8,
};

const tools = [_]Tool{
    .{ .name = "tf-prof-report", .root = "tools/vega-macos/prof_report.zig", .help = "Build tf-prof-report, a hipprof table as a per-token breakdown" },
};

// modules without an executable of their own whose tests also run
const extra_tests = [_][]const u8{ "tools/vega-macos/npy.zig", "tools/vega-macos/pyfmt.zig" };

fn module(b: *std.Build, file: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path(file), .target = target, .optimize = optimize, .link_libc = true });
}

/// The tools, installed like the other tf-* tools, and their host tests that join `test`; not on Windows.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    if (target.result.os.tag == .windows or b.graph.host.result.os.tag == .windows) return;
    const check = b.step("test-vega-tools", "Tests of the tf-* tools of tools/vega-macos (offline, no card)");
    for (tools) |tool| {
        const exe = b.addExecutable(.{ .name = tool.name, .root_module = module(b, tool.root, target, optimize) });
        b.step(tool.name, tool.help).dependOn(&b.addInstallArtifact(exe, .{}).step);
        check.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module(b, tool.root, b.graph.host, .Debug) })).step);
    }
    for (extra_tests) |file| check.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module(b, file, b.graph.host, .Debug) })).step);
    test_step.dependOn(check);
}
