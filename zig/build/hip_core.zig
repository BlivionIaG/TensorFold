//! The backend-neutral core (checkpoint formats, the kernel registry), as the HIP build imports it.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Module {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .link_libc = true }));
    return core;
}
