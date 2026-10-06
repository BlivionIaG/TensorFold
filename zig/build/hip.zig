//! The HIP half of the root build: the probe's offload bundle, the ROCm kernel libraries a GPU family each, the runtime
//! and its GPU test program.

const std = @import("std");

/// Each .hip in zig/kernels/hip with its own hipcc flags and the headers it includes (hipcc --genco writes no dep file).
const Kernel = struct { name: []const u8, flags: []const []const u8 = &.{}, headers: []const []const u8 = &.{} };

/// The probe's flags: wave32 on RDNA, no contraction.
const shared_flags = [_][]const u8{ "-O3", "-mno-wavefrontsize64", "-ffp-contract=off", "-std=c++20" };

const kernels = [_]Kernel{
    .{ .name = "probe" },
};

/// torch.utils.cpp_extension's hipcc flags for the Python ROCm extensions (build.ninja), less its include paths:
/// the same compiler and flags give the same device code, so a kernel keeps the Python engine's bits.
const torch_flags = [_][]const u8{
    "-D__HIP_PLATFORM_AMD__=1", "-DUSE_ROCM=1",                      "-DHIPBLAS_V2", "-fPIC",
    "-DCUDA_HAS_FP16=1",        "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1", "-std=c++20",   "-fno-gpu-rdc",
    "-mno-wavefrontsize64",     "-ffp-contract=off",
};

/// One library's sources: the shim, the torch-op kernels and the ROCm kernels (attention.hip includes attention_fa.hip).
const lib_sources = [_][]const u8{
    "capi.hip",             "ops.hip",               "rocm/act.hip",         "rocm/attention.hip",
    "rocm/gated_delta.hip", "rocm/affine_gemv.hip",  "rocm/affine_wmma.hip", "rocm/affine_wmma_pair.hip",
    "rocm/affine_dot2.hip", "rocm/affine_tiles.hip",
};

/// What the library sources include, so an edit to one rebuilds the libraries.
const lib_headers = [_][]const u8{
    "rocm/act.hpp",         "rocm/affine.hpp", "rocm/affine_api.hpp", "rocm/affine_dot2.hpp",
    "rocm/affine_wmma.hpp", "rocm/arch.hpp",   "rocm/attention.hpp",  "rocm/attention_fa.hip",
    "rocm/gated_delta.hpp",
};

/// A GPU family's library: its gfx targets and whether its host dispatch takes the WMMA schedules.
const Family = struct { name: []const u8, prefixes: []const []const u8, wmma: bool };
const families = [_]Family{
    .{ .name = "rdna2", .prefixes = &.{"gfx103"}, .wmma = false },
    .{ .name = "rdna3", .prefixes = &.{ "gfx11", "gfx12" }, .wmma = true },
};

/// The runtime module for `target`; without images it builds host-only (empty images).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const ?std.Build.LazyPath, libs: []const ?std.Build.LazyPath, gfx: []const u8) *std.Build.Module {
    const options = b.addOptions();
    var with = images.len > 0;
    for (images) |i| with = with and i != null;
    options.addOption(bool, "with_kernels", with);
    options.addOption([]const u8, "gfx", if (with) gfx else "");
    var have: [families.len]bool = @splat(false);
    for (libs, 0..) |l, i| have[i] = l != null;
    inline for (families, 0..) |f, i| options.addOption(bool, "with_" ++ f.name, have[i]);
    const hip = b.createModule(.{ .root_source_file = b.path("zig/src/hip/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    hip.addOptions("kernel_options", options);
    if (with) for (kernels, images) |k, image| hip.addAnonymousImport(b.fmt("hsaco_{s}", .{k.name}), .{ .root_source_file = image.? });
    for (families, libs) |f, l| if (l) |file| hip.addAnonymousImport(b.fmt("lib_{s}", .{f.name}), .{ .root_source_file = file });
    return hip;
}

/// Linux targets: the probe bundle and the kernel libraries (-Dhipcc builds them, -Dhsaco embeds prebuilt ones) and
/// `tf-hip-test`.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const hipcc = b.option([]const u8, "hipcc", "hipcc that builds the HIP kernels");
    const prebuilt = b.option([]const u8, "hsaco", "absolute directory of prebuilt <name>.hsaco bundles and libtf_<family>.so");
    const gfx = b.option([]const u8, "gfx", "gfx targets, comma separated (default gfx1030,gfx1100,gfx1151)") orelse "gfx1030,gfx1100,gfx1151";
    // the compiler's version text is an input of every build, so a new hipcc rebuilds them all
    const version: ?std.Build.LazyPath = if (prebuilt == null and hipcc != null) blk: {
        const run = b.addSystemCommand(&.{ hipcc.?, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    const bundle_step = b.step("hsaco", "Build and install the HIP kernel bundles and libraries alone");
    for (kernels, &images) |k, *image| {
        if (prebuilt) |dir| {
            image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.hsaco", .{k.name}) }));
        } else if (hipcc) |tool| {
            image.* = bundle(b, tool, version.?, k, gfx);
        }
        if (image.*) |file| bundle_step.dependOn(&b.addInstallFile(file, b.fmt("hsaco/{s}.hsaco", .{k.name})).step);
    }
    var libs: [families.len]?std.Build.LazyPath = @splat(null);
    for (families, &libs) |f, *lib| {
        const arches = archesOf(b, gfx, f);
        if (arches.len == 0) continue;
        if (prebuilt) |dir| {
            lib.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("libtf_{s}.so", .{f.name}) }));
        } else if (hipcc) |tool| {
            lib.* = library(b, tool, version.?, f, arches);
        }
        if (lib.*) |file| bundle_step.dependOn(&b.addInstallFile(file, b.fmt("hsaco/libtf_{s}.so", .{f.name})).step);
    }
    const hip = runtime(b, target, optimize, if (hipcc != null or prebuilt != null) &images else &.{}, &libs, gfx);
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/hip/main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner.addImport("hip", hip);
    const exe = b.addExecutable(.{ .name = "tf-hip-test", .root_module = runner });
    b.installArtifact(exe);
    b.step("tf-hip-test", "The HIP runtime's GPU test program").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// Host unit tests of the HIP runtime (no GPU), on any host.
pub fn hostTests(b: *std.Build, step: *std.Build.Step) void {
    const hip = runtime(b, b.graph.host, .debug, &.{}, &.{ null, null }, "");
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = hip })).step);
}

/// The gfx targets of `gfx` that belong to family `f`.
fn archesOf(b: *std.Build, gfx: []const u8, f: Family) []const []const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeScalar(u8, gfx, ',');
    while (it.next()) |arch| for (f.prefixes) |p| if (std.mem.startsWith(u8, arch, p)) {
        list.append(b.allocator, arch) catch @panic("OOM");
        break;
    };
    return list.items;
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

/// hipcc -shared over every library source with torch's flags and the family's WMMA switch: libtf_<family>.so.
fn library(b: *std.Build, hipcc: []const u8, version: std.Build.LazyPath, f: Family, arches: []const []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ hipcc, "-shared" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    run.addArg(b.fmt("-DTENSORFOLD_RDNA_WMMA={d}", .{@intFromBool(f.wmma)}));
    // hipcc's own ROCm tree, with its device bitcode, as the Python build points at it
    const root = std.fs.path.dirname(std.fs.path.dirname(hipcc) orelse ".") orelse ".";
    run.addArg(b.fmt("--rocm-path={s}", .{root}));
    run.addArg(b.fmt("--rocm-device-lib-path={s}/lib/llvm/amdgcn/bitcode", .{root}));
    for (arches) |arch| run.addArg(b.fmt("--offload-arch={s}", .{arch}));
    run.addPrefixedDirectoryArg("-I", b.path("zig/kernels/hip/rocm"));
    for (lib_headers) |h| run.addFileInput(b.path(b.fmt("zig/kernels/hip/{s}", .{h})));
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("libtf_{s}.so", .{f.name}));
    for (lib_sources) |src| run.addFileArg(b.path(b.fmt("zig/kernels/hip/{s}", .{src})));
    return out;
}
