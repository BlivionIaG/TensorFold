//! The model's side of --glp (cuda_model.zig boot): the vector's admission, the knobs it refuses, its one upload.
//! `m` is the `*Model`; nothing here runs for a model without a plan: an unsteered server loads no GLP image or buffer.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const glp_plan = @import("glp_plan.zig");
const pf = @import("forward_prefill.zig");

/// This model's own copy of the plan (its arena), then every rank's agreement on it, before any execution or graph.
pub fn open(m: anytype, a: std.mem.Allocator, given: ?*const glp_plan.Plan) !void {
    m.glp = if (given) |g| try g.clone(a) else null;
    try @import("glp_agree.zig").agree(&m.session.plan, if (m.glp) |*g| g else null);
}

/// Bytes the buffer plan allocates after the KV pool is sized (the direction table is a persistent role).
pub fn resident(m: anytype) u64 {
    const g = m.glp orelse return 0;
    return std.mem.alignForward(u64, g.tableBytes(), 256);
}

/// A vector this model cannot steer is refused before any weight loads: its width and layers, the whole backbone (a
/// layer prefix would leave a fold unsteered), and the kernel image (loaded only here, never for an unsteered model).
pub fn admit(m: anytype, g: *const glp_plan.Plan, layer_prefix: bool) !void {
    g.admit(m.cfg.layers, m.cfg.hidden) catch |e| {
        if (!quiet) std.log.err("glp: the vector does not fit this model ({t}): {d} wide against hidden {d}, {d} backbone layers", .{ e, g.width, m.cfg.hidden, m.cfg.layers });
        return e;
    };
    if (layer_prefix) {
        if (!quiet) std.log.err("glp: TF_DSV41_LAYERS runs a layer prefix, which leaves a fold unsteered; steer the whole backbone", .{});
        return error.GlpUnsupported;
    }
    try m.kernels.loadGlp(&m.driver);
}

/// Knobs whose programs would show or skip an unsteered fold: refused with --glp, never silently unsteered.
pub fn refuseKnobs(m: anytype) !void {
    const why: ?[]const u8 = if (pf.state(&m.f).mode == .replay)
        "TF_DSV41_PREFILL=replay enters decoder layers from a stash, past the fold"
    else if (m.f.opts.pf_tbo)
        "TF_DSV41_PF_TBO splits a layer's parts across micro-batches"
    else if (m.f.opts.trace)
        "the trace shows each block's unsteered fold"
    else
        null;
    if (why) |text| {
        if (!quiet) std.log.err("glp: {s}; unset it to serve a vector", .{text});
        return error.GlpUnsupported;
    }
}

/// The directions once, before any program runs or any graph is captured: never uploaded again.
pub fn upload(m: anytype, gpa: std.mem.Allocator, g: *const glp_plan.Plan) !void {
    const table = try g.table(gpa);
    defer gpa.free(table);
    try m.f.upload(m.runner.addressOf(glp_plan.dirs_role).?, table);
    std.log.scoped(.dsv41).info("glp: {t} at strength {d} on {d} layers, directions uploaded ({d} bytes)", .{ g.mode, g.alpha, g.layers.len, table.len });
}
