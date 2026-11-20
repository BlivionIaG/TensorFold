//! The control vector's serve checks need no GPU, no model download and no checkpoint.
const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const target = b.graph.host;
    const step = b.step("test-glp-serve", "Control vector flags, startup refusals, the checkpoint's numbers and the report line");
    all.dependOn(step);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    // the server tree over no native engines, with the checkpoint-config reader the startup checks wired in
    const module = @import("server.zig").plain(b, "zig/src/server/glp_startup_test.zig", target, .ReleaseSafe, lanes);
    const host = b.addRunArtifact(b.addTest(.{ .root_module = module }));
    // the file tests write beside the build cache and name their files from the repo root
    host.setCwd(b.path("."));
    step.dependOn(&host.step);
}
