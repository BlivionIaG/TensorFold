//! Admit and upload CUDA GLP state once; absent or zero-strength policies create no GPU plan or image.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const glp_plan = @import("glp_plan.zig");
const pf = @import("forward_prefill.zig");

/// Clone active plans into the model arena and agree on every loaded policy before weights or graphs.
pub fn open(m: anytype, a: std.mem.Allocator, given: ?*const glp_plan.Plan, policy: [8]i64) !void {
    // a plan is always its own policy: a caller that built the two from different sources is a bug, never served
    if (given != null and !std.mem.eql(i64, &glp_plan.Plan.words(given), &policy)) return error.GlpMismatch;
    m.glp = if (given) |g| try g.clone(a) else null;
    try @import("glp_agree.zig").agree(&m.session.plan, policy);
}

/// Bytes the buffer plan allocates after the KV pool is sized (the direction table is a persistent role).
pub fn resident(m: anytype) u64 {
    const g = m.glp orelse return 0;
    return std.mem.alignForward(u64, g.tableBytes(), 256);
}

/// Refuse incompatible geometry, partial-backbone execution and missing active images before weights load.
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
