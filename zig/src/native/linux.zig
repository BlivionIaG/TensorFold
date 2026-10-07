//! The engines a native server opens on Linux: CUDA and HIP side by side; a checkpoint goes to its model's backend.
const std = @import("std");
const api = @import("engine_api");
const cuda = @import("cuda_engines");
const hip = @import("hip_engines");
const Allocator = std.mem.Allocator;

pub const backends: []const []const u8 = cuda.backends ++ hip.backends;
pub const families: []const api.Family = cuda.families ++ hip.families;

/// The chip class gate entries name: the CUDA device's when there is one, else the HIP card's.
pub fn chip(a: Allocator) ?[]const u8 {
    return cuda.chip(a) orelse hip.chip(a);
}

fn lists(comptime engines: type, model_type: []const u8) bool {
    for (engines.families) |f| if (std.mem.eql(u8, f.model_type, model_type)) return true;
    return false;
}

/// The engine for `o.dir`, or null with `problem` set when no backend reads the checkpoint.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (lists(cuda, o.model_type)) return cuda.open(a, gpa, io, o, problem);
    if (lists(hip, o.model_type)) return hip.open(a, gpa, io, o, problem);
    problem.* = try std.fmt.allocPrint(a, "the native engines have no backend for {s} checkpoints yet; serve with --engine python", .{o.model_type});
    return null;
}

/// A tensor-parallel rank above 0 (HIP only).
pub fn follow(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !bool {
    return hip.follow(a, gpa, io, o, problem);
}

test "each backend's families are listed once" {
    try std.testing.expectEqual(cuda.families.len + hip.families.len, families.len);
    try std.testing.expect(lists(cuda, "nemotron_h"));
    try std.testing.expect(lists(hip, "qwen3_5"));
    try std.testing.expect(!lists(cuda, "qwen3_5"));
}
