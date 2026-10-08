//! `tf-hip-test runtime`: the HIP runtime on this GPU, one subtest a line.

const std = @import("std");
const hip = @import("hip");
const check = @import("../check.zig");
const probes = @import("probes.zig");
const Gpu = check.Gpu;

pub fn info(gpu: Gpu) !void {
    var name_buf: [256]u8 = undefined;
    const name = try gpu.ctx.name(&name_buf);
    const mem = try gpu.ctx.memInfo();
    std.debug.print("INFO HIP {d}, device {s}, gfx {d}, {d} CUs, wave {d}, {d} MiB total, {d} MiB free, kernels embedded {} ({s})\n", .{
        try gpu.d.version(),               name,
        try gpu.ctx.capability(),          try gpu.ctx.attribute(.multiprocessor_count),
        try gpu.ctx.attribute(.warp_size), mem.total >> 20,
        mem.free >> 20,                    hip.kernels.available,
        hip.kernels.targets,
    });
}

fn smoke(gpu: Gpu) !void {
    try probes.smoke(gpu);
    check.pass("smoke: copies, fills, launches, argument packing, module globals and refusals", .{});
}

fn graph(gpu: Gpu) !void {
    try probes.graphs(gpu);
    check.pass("graph: stream capture, explicit graphs, node and whole-exec updates", .{});
}

fn cooperative(gpu: Gpu) !void {
    try probes.cooperative(gpu);
    check.pass("cooperative: a grid of one block per CU", .{});
}

fn library(gpu: Gpu) !void {
    try probes.library(gpu);
    check.pass("library: this GPU family's kernel library opens, builds as the caps say and runs a kernel", .{});
}

fn image(gpu: Gpu) !void {
    try probes.image(gpu);
    check.pass("image: an offload bundle without this GPU's target and a broken image are refused", .{});
}

/// Every subtest whose name holds `filter` (all when empty); the exit code is 0 when none failed.
pub fn run(gpu: Gpu, filter: []const u8) !u8 {
    try info(gpu);
    var t: check.Tally = .{ .filter = filter };
    t.group("smoke", gpu, smoke);
    t.group("graph", gpu, graph);
    t.group("cooperative", gpu, cooperative);
    t.group("library", gpu, library);
    t.group("image", gpu, image);
    return t.summary("runtime");
}
