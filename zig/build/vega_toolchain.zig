//! The macOS + Thunderbolt/eGPU support stack's toolchain tools (tools/vega-macos), all Zig: `zig build tf-hipcc` (the `hipcc --genco` wrapper over Homebrew LLVM,
//! meant to be symlinked as <rocm root>/bin/hipcc), `zig build tf-fetch-rocm` (installs the HIP headers and ROCm device bitcode from the pinned wheel into a ROCm
//! root and links the wrapper), and `zig build test-vega-toolchain`: unit tests of both, the wrapper as a process against a stand-in for clang / clang++
//! (tools/vega-macos/hipcc_fake_clang.zig) and the installer as a process on a synthetic wheel. Offline: no Homebrew, no network, no card.
const std = @import("std");

const Tool = struct {
    /// the executable (installed to zig-out/bin) and its build step
    name: []const u8,
    /// its root source file; its tests run under `test-vega-toolchain`
    root: []const u8,
    help: []const u8,
};

const tools = [_]Tool{
    .{ .name = "tf-hipcc", .root = "tools/vega-macos/hipcc.zig", .help = "Build tf-hipcc, the `hipcc --genco` wrapper over Homebrew LLVM (symlink it as <rocm root>/bin/hipcc)" },
    .{ .name = "tf-fetch-rocm", .root = "tools/vega-macos/fetch_rocm.zig", .help = "Build tf-fetch-rocm, which installs the HIP headers and ROCm device bitcode of the pinned wheel into a ROCm root" },
};

fn module(b: *std.Build, file: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path(file), .target = target, .optimize = optimize, .link_libc = true });
}

/// The tools, installed like the other tf-* tools, and their host tests that join `test`; not on Windows.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    if (target.result.os.tag == .windows or b.graph.host.result.os.tag == .windows) return;
    const host = b.graph.host;
    const check = b.step("test-vega-toolchain", "Tests of tf-hipcc and tf-fetch-rocm of tools/vega-macos, as processes over a fake clang and a synthetic wheel (offline, no Homebrew)");
    for (tools) |tool| {
        const exe = b.addExecutable(.{ .name = tool.name, .root_module = module(b, tool.root, target, optimize) });
        b.step(tool.name, tool.help).dependOn(&b.addInstallArtifact(exe, .{}).step);
        check.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module(b, tool.root, host, .Debug) })).step);
    }

    // the wrapper as a process: hipcc_check runs it through a bin/hipcc link against hipcc_fake_clang
    const hipcc = b.addExecutable(.{ .name = "tf-hipcc", .root_module = module(b, "tools/vega-macos/hipcc.zig", host, .ReleaseSafe) });
    const fake = b.addExecutable(.{ .name = "hipcc_fake_clang", .root_module = module(b, "tools/vega-macos/hipcc_fake_clang.zig", host, .ReleaseSafe) });
    const hipcc_check = b.addExecutable(.{ .name = "hipcc_check", .root_module = module(b, "tools/vega-macos/hipcc_check.zig", host, .ReleaseSafe) });
    const run_hipcc = b.addRunArtifact(hipcc_check);
    run_hipcc.addArtifactArg(hipcc);
    run_hipcc.addArtifactArg(fake);
    _ = run_hipcc.addOutputDirectoryArg("hipcc-check");
    check.dependOn(&run_hipcc.step);

    // the installer as a process: fetch_rocm_check runs it on a synthetic wheel
    const fetch = b.addExecutable(.{ .name = "tf-fetch-rocm", .root_module = module(b, "tools/vega-macos/fetch_rocm.zig", host, .ReleaseSafe) });
    const fetch_check = b.addExecutable(.{ .name = "fetch_rocm_check", .root_module = module(b, "tools/vega-macos/fetch_rocm_check.zig", host, .ReleaseSafe) });
    const run_fetch = b.addRunArtifact(fetch_check);
    run_fetch.addArtifactArg(fetch);
    _ = run_fetch.addOutputDirectoryArg("fetch-rocm-check");
    check.dependOn(&run_fetch.step);

    test_step.dependOn(check);
}
