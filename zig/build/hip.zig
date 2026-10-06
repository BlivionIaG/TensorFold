//! The HIP half of the root build: kernel offload bundles for each gfx target, the runtime and its GPU test program.

const std = @import("std");

/// Each .hip in zig/kernels/hip with its own hipcc flags and the headers it includes (hipcc --genco writes no dep file).
const Kernel = struct { name: []const u8, flags: []const []const u8 = &.{}, headers: []const []const u8 = &.{} };

/// The Python ROCm extension's flags: wave32 on RDNA, no contraction, so a kernel keeps its bits.
const shared_flags = [_][]const u8{ "-O3", "-mno-wavefrontsize64", "-ffp-contract=off", "-std=c++20" };

const kernels = [_]Kernel{
    .{ .name = "probe" },
};

/// The runtime module for `target`; without images it builds host-only (empty images).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const ?std.Build.LazyPath, gfx: []const u8) *std.Build.Module {
    const options = b.addOptions();
    var with = images.len > 0;
    for (images) |i| with = with and i != null;
    options.addOption(bool, "with_kernels", with);
    options.addOption([]const u8, "gfx", if (with) gfx else "");
    const hip = b.createModule(.{ .root_source_file = b.path("zig/src/hip/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    hip.addOptions("kernel_options", options);
    if (with) for (kernels, images) |k, image| hip.addAnonymousImport(b.fmt("hsaco_{s}", .{k.name}), .{ .root_source_file = image.? });
    return hip;
}

/// Linux targets: offload bundles (-Dhipcc builds them, -Dhsaco embeds prebuilt ones) and `tf-hip-test`.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const hipcc = b.option([]const u8, "hipcc", "hipcc that builds the HIP kernel offload bundles");
    const prebuilt = b.option([]const u8, "hsaco", "absolute directory of prebuilt <name>.hsaco bundles to embed");
    const gfx = b.option([]const u8, "gfx", "gfx targets, comma separated (default gfx1030,gfx1100,gfx1151)") orelse "gfx1030,gfx1100,gfx1151";
    // the compiler's version text is an input of every bundle, so a new hipcc rebuilds them all
    const version: ?std.Build.LazyPath = if (prebuilt == null and hipcc != null) blk: {
        const run = b.addSystemCommand(&.{ hipcc.?, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    const bundle_step = b.step("hsaco", "Build and install the HIP kernel offload bundles alone");
    for (kernels, &images) |k, *image| {
        if (prebuilt) |dir| {
            image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.hsaco", .{k.name}) }));
        } else if (hipcc) |tool| {
            image.* = bundle(b, tool, version.?, k, gfx);
        }
        if (image.*) |file| bundle_step.dependOn(&b.addInstallFile(file, b.fmt("hsaco/{s}.hsaco", .{k.name})).step);
    }
    const hip = runtime(b, target, optimize, if (hipcc != null or prebuilt != null) &images else &.{}, gfx);
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/hip/main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner.addImport("hip", hip);
    const exe = b.addExecutable(.{ .name = "tf-hip-test", .root_module = runner });
    b.installArtifact(exe);
    b.step("tf-hip-test", "The HIP runtime's GPU test program").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// Host unit tests of the HIP runtime (no GPU), on any host.
pub fn hostTests(b: *std.Build, step: *std.Build.Step) void {
    const hip = runtime(b, b.graph.host, .debug, &.{}, "");
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = hip })).step);
}

/// hipcc --genco with the shared flags, the kernel's own and one --offload-arch per gfx target: one offload bundle.
fn bundle(b: *std.Build, hipcc: []const u8, version: std.Build.LazyPath, k: Kernel, gfx: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ hipcc, "--genco" });
    run.addFileInput(version);
    run.addArgs(&shared_flags);
    run.addArgs(k.flags);
    var it = std.mem.tokenizeScalar(u8, gfx, ',');
    while (it.next()) |arch| run.addArg(b.fmt("--offload-arch={s}", .{arch}));
    for (k.headers) |h| run.addFileInput(b.path(b.fmt("zig/kernels/hip/{s}", .{h})));
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.hsaco", .{k.name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/hip/{s}.hip", .{k.name})));
    return out;
}
