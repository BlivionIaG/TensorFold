//! The native server's build, shared by the Linux backends: tensorfold-native over the engines built into it.

const std = @import("std");

/// The server executable over `engines`; the HTTP side keeps its safety checks, the engine below it runs at `optimize`.
pub fn server(b: *std.Build, target: std.Build.ResolvedTarget, api: *std.Build.Module, engines: *std.Build.Module, tokenizer: *std.Build.Module, build_options: *std.Build.Step.Options) *std.Build.Step.Compile {
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = engines }, .{ .name = "checkpoint_cli", .module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "native_engines", .module = engines }} }) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    return exe;
}

/// `zig build native` on Linux: tensorfold-native with the CUDA and HIP engines into zig-out/native/bin.
pub fn linux(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, api: *std.Build.Module, cuda_engines: *std.Build.Module, hip_engines: *std.Build.Module, tokenizer: *std.Build.Module, build_options: *std.Build.Step.Options, test_step: *std.Build.Step) void {
    const engines = b.createModule(.{
        .root_source_file = b.path("zig/src/native/linux.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "cuda_engines", .module = cuda_engines }, .{ .name = "hip_engines", .module = hip_engines } },
    });
    const exe = server(b, target, api, engines, tokenizer, build_options);
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    b.step("native", "tensorfold-native with the CUDA and HIP engines into zig-out/native/bin").dependOn(&install.step);
    if (b.graph.host.result.os.tag == .linux and target.result.os.tag == .linux) test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = engines })).step);
}
