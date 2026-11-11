//! hipprof (tools/vega-macos), the HIP timing shim: `tf-hipprof` builds libhipprof, `test-hipprof` tests it.
const std = @import("std");

const modes = [_][]const u8{ "each", "events", "graph", "start", "exit" };

fn module(b: *std.Build, file: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path(file), .target = target, .optimize = optimize, .link_libc = true });
}

/// The shim library, installed like the other tf-* tools, and the host tests that join `test`; not on Windows.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    if (target.result.os.tag == .windows or b.graph.host.result.os.tag == .windows) return;
    const shim = b.addLibrary(.{ .name = "hipprof", .linkage = .dynamic, .root_module = module(b, "tools/vega-macos/hipprof.zig", target, optimize) });
    b.step("tf-hipprof", "Build libhipprof, the HIP runtime timing shim (TENSORFOLD_HIP=<it>, HIPPROF_REAL=<the runtime>)").dependOn(&b.addInstallArtifact(shim, .{}).step);
    const host = b.graph.host;
    const abi = b.createModule(.{ .root_source_file = b.path("zig/src/hip/abi.zig") });
    const unit = module(b, "tools/vega-macos/hipprof_test.zig", host, .Debug);
    unit.addImport("hip_abi", abi);
    const check = b.step("test-hipprof", "hipprof's table and symbol list tests, and its four modes over a mock runtime");
    check.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = unit })).step);
    const host_shim = b.addLibrary(.{ .name = "hipprof", .linkage = .dynamic, .root_module = module(b, "tools/vega-macos/hipprof.zig", host, .ReleaseSafe) });
    const mock = b.addLibrary(.{ .name = "hipmock", .linkage = .dynamic, .root_module = module(b, "tools/vega-macos/hipprof_mock.zig", host, .ReleaseSafe) });
    const driver = b.addExecutable(.{ .name = "hipprof_check", .root_module = module(b, "tools/vega-macos/hipprof_check.zig", host, .ReleaseSafe) });
    for (modes) |mode| {
        const run = b.addRunArtifact(driver);
        run.addArgs(&.{mode});
        run.addArtifactArg(host_shim);
        run.addArtifactArg(mock);
        const out = run.addOutputDirectoryArg(b.fmt("hipprof-{s}", .{mode}));
        if (std.mem.eql(u8, mode, "each")) run.expectStdErrMatch("hipprof: #2 affine_dot2_stream k=2560 n=1536 m=1 x non-finite 1 of 2560, rough max 0\n");
        check.dependOn(&run.step);
        if (!std.mem.eql(u8, mode, "exit")) continue;
        // the table the "exit" process wrote without being asked: read by a second process
        const after = b.addRunArtifact(driver);
        after.addArgs(&.{"after-exit"});
        after.addArtifactArg(host_shim);
        after.addArtifactArg(mock);
        after.addDirectoryArg(out);
        check.dependOn(&after.step);
    }
    test_step.dependOn(check);
}
