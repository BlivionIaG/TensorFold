//! The HIP half of the root build: kernel bundles (-Dhipcc), the runtime, `tf-hip-test`, host and mock-library tests.

const std = @import("std");
const caps = @import("../src/hip/caps.zig");

/// Each .hip in zig/kernels/hip that the runtime embeds, by name.
const kernels = [_][]const u8{"probe"};

/// wave32 on RDNA, no contraction: the device code computes what the source says.
const flags = [_][]const u8{ "-O3", "-mno-wavefrontsize64", "-ffp-contract=off", "-std=c++20" };

/// The gfx targets a family name in -Dgfx stands for: every RDNA2, RDNA3 and RDNA3.5 part the caps table lists.
const gfx_families = [_]struct { name: []const u8, arches: []const u8 }{
    .{ .name = "rdna2", .arches = "gfx1030,gfx1031,gfx1032,gfx1033,gfx1034,gfx1035,gfx1036" },
    .{ .name = "rdna3", .arches = "gfx1100,gfx1101,gfx1102,gfx1103" },
    .{ .name = "rdna3.5", .arches = "gfx1150,gfx1151,gfx1152,gfx1153" },
};

/// -Dgfx with its family names expanded, each gfx target once; a target outside the caps table stops the build.
fn expandGfx(b: *std.Build, gfx: []const u8) []const u8 {
    var out: std.ArrayList(u8) = .empty;
    var it = std.mem.tokenizeScalar(u8, gfx, ',');
    while (it.next()) |name| {
        var arches = name;
        for (gfx_families) |fam| if (std.mem.eql(u8, name, fam.name)) {
            arches = fam.arches;
        };
        var each = std.mem.tokenizeScalar(u8, arches, ',');
        while (each.next()) |arch| {
            if (caps.Caps.of(arch) == null) std.debug.panic("-Dgfx: {s} is not in the caps table (zig/src/hip/caps.zig)", .{arch});
            var seen = false;
            var done = std.mem.tokenizeScalar(u8, out.items, ',');
            while (done.next()) |prev| seen = seen or std.mem.eql(u8, prev, arch);
            if (seen) continue;
            if (out.items.len > 0) out.append(b.allocator, ',') catch @panic("OOM");
            out.appendSlice(b.allocator, arch) catch @panic("OOM");
        }
    }
    return out.items;
}

/// The runtime module for `target`; without images it builds host-only (empty images).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: ?[]const std.Build.LazyPath, gfx: []const u8) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", images != null);
    options.addOption([]const u8, "gfx", if (images != null) gfx else "");
    const hip = b.createModule(.{ .root_source_file = b.path("zig/src/hip/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    hip.addOptions("kernel_options", options);
    if (images) |list| for (kernels, list) |name, image| hip.addAnonymousImport(b.fmt("hsaco_{s}", .{name}), .{ .root_source_file = image });
    return hip;
}

/// Linux targets: the kernel bundles (-Dhipcc builds them) and `tf-hip-test`.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const hipcc = b.option([]const u8, "hipcc", "hipcc that builds the HIP kernels");
    const gfx = expandGfx(b, b.option([]const u8, "gfx", "gfx targets or families (rdna2, rdna3, rdna3.5), comma separated (default gfx1030,gfx1100,gfx1151)") orelse "gfx1030,gfx1100,gfx1151");
    var images: [kernels.len]std.Build.LazyPath = undefined;
    if (hipcc) |tool| {
        // the compiler's version text is an input of every bundle, so a new hipcc rebuilds them all
        const run = b.addSystemCommand(&.{ tool, "--version" });
        run.has_side_effects = true;
        const version = run.captureStdOut(.{});
        const step = b.step("hsaco", "Build and install the HIP kernel bundles alone");
        for (kernels, &images) |name, *image| {
            image.* = bundle(b, tool, version, name, gfx);
            step.dependOn(&b.addInstallFile(image.*, b.fmt("hsaco/{s}.hsaco", .{name})).step);
        }
    }
    const hip = runtime(b, target, optimize, if (hipcc != null) &images else null, gfx);
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/hip/main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner.addImport("hip", hip);
    const exe = b.addExecutable(.{ .name = "tf-hip-test", .root_module = runner });
    b.installArtifact(exe);
    b.step("tf-hip-test", "The HIP runtime's GPU test program").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// Host unit tests of the HIP runtime (no GPU), on any host.
pub fn hostTests(b: *std.Build, step: *std.Build.Step) void {
    const hip = runtime(b, b.graph.host, .debug, null, "");
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = hip })).step);
    step.dependOn(mockTests(b, hip));
}

/// hipcc --genco with the shared flags and one --offload-arch per gfx target: one offload bundle.
fn bundle(b: *std.Build, hipcc: []const u8, version: std.Build.LazyPath, name: []const u8, gfx: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ hipcc, "--genco" });
    run.addFileInput(version);
    run.addArgs(&flags);
    var it = std.mem.tokenizeScalar(u8, gfx, ',');
    while (it.next()) |arch| run.addArg(b.fmt("--offload-arch={s}", .{arch}));
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.hsaco", .{name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/hip/{s}.hip", .{name})));
    return out;
}

/// The runtime against a stand-in libamdhip64 (zig/tests/hip/mock): complete, missing one entry point, an unknown GPU.
fn mockTests(b: *std.Build, hip: *std.Build.Module) *std.Build.Step {
    const abi = b.createModule(.{ .root_source_file = b.path("zig/src/hip/runtime/abi.zig"), .target = b.graph.host, .optimize = .debug });
    const variants = [_]struct { name: []const u8, arch: []const u8, omit: []const u8 }{
        .{ .name = "full", .arch = "gfx1100", .omit = "" },
        .{ .name = "missing", .arch = "gfx1100", .omit = "hipModuleLaunchKernel" },
        .{ .name = "unknown", .arch = "gfx803", .omit = "" },
    };
    const paths = b.addOptions();
    for (variants) |v| {
        const options = b.addOptions();
        options.addOption([]const u8, "arch", v.arch);
        options.addOption([]const u8, "omit", v.omit);
        const root = b.createModule(.{ .root_source_file = b.path("zig/tests/hip/mock/mock.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
        root.addImport("abi", abi);
        root.addOptions("mock_options", options);
        const lib = b.addLibrary(.{ .name = b.fmt("hipmock_{s}", .{v.name}), .linkage = .dynamic, .root_module = root });
        paths.addOptionPath(v.name, lib.getEmittedBin());
    }
    const tests = b.createModule(.{ .root_source_file = b.path("zig/tests/hip/mock/mock_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    tests.addImport("hip", hip);
    tests.addOptions("mock_libs", paths);
    return &b.addRunArtifact(b.addTest(.{ .root_module = tests })).step;
}
