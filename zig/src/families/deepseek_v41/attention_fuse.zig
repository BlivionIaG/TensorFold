//! The decode attention chain's fused dispatches (one row): each replaces adjacent dispatches whose arithmetic is
//! block local, with bit-identical output. Every fusion has a kill switch, `TF_DSV41_NO_FUSE_<NAME>`, default on.
const std = @import("std");
const mtl = @import("metal");
const exl3 = @import("core").exl3_projection;
const weights = @import("weights.zig");
const config = @import("config.zig");
const common = @import("attention_common.zig");
const View = common.View;
const Workspace = common.Workspace;
const core = common.core;
const tensor = common.tensor;

/// The fusion kill switches, read once: a process does not toggle them.
pub const Switches = struct {
    /// wo_a's epilogue and wo_b's rotate_in in one dispatch.
    wo_chain: bool,
    /// The query latent's RMS norm and wq_b's rotate_in in one dispatch.
    qnorm: bool,
};

var cached: ?Switches = null;

pub fn switches() Switches {
    if (cached) |s| return s;
    const s: Switches = .{
        .wo_chain = std.c.getenv("TF_DSV41_NO_FUSE_WO_CHAIN") == null,
        .qnorm = std.c.getenv("TF_DSV41_NO_FUSE_QNORM") == null,
    };
    cached = s;
    return s;
}

/// The eight output groups' indexed wo_a call with its epilogue fused with wo_b's rotate_in (`encodeWoB` runs wo_b).
/// `grouped` still receives wo_a's bf16 output.
pub fn encodeWoA(ep: *const exl3.Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, work: *const Workspace, q: View, grouped: View, scratch: exl3.View, c: config.Config) !void {
    for (&w.wo_a) |*m| exl3.useMatrix(e, m.*);
    const group_input = c.heads * c.head_dim / c.o_groups;
    try exl3.chain.encodeIndexedRotateNext(ep, e, .{ .descriptors = .{ .buffer = w.wo_a_descriptors.?, .offset = 0 }, .count = c.o_groups, .k = group_input, .n = c.o_lora }, core(q), core(grouped), scratch, .{
        .count = c.o_groups,
        .x_stride = group_input,
        .y_stride = c.o_lora,
        .output = .bf16,
        .output_rows = c.o_groups,
        .routes = .{ .ids = core(.{ .buffer = work.ids, .offset = 0 }), .repeat = 1 },
    }, .{ .su = w.wo_b.suh, .su_signs = w.wo_b.suh_format == .packed_signs, .k = c.o_groups * c.o_lora });
}

/// wo_b over the rotated input `encodeWoA` left in the scratch base.
pub fn encodeWoB(ep: *const exl3.Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, y: View, scratch: exl3.View, c: config.Config) !void {
    try exl3.encode(ep, e, w.wo_b, scratch, core(y), scratch, .{ .count = 1, .x_stride = c.o_groups * c.o_lora, .y_stride = c.hidden, .prerotated = true });
}

/// Whether one decode row's `qr` norm and wq_b's rotate_in can run as one dispatch.
pub fn qNormReady(ep: *const exl3.Pipelines, w: *const weights.Attention, c: config.Config) bool {
    return switches().qnorm and w.wq_b.k == c.q_lora and c.q_lora <= exl3.chain.rmsRotateInWidth(ep) and
        std.c.getenv("TF_DSV41_DOUBLE_ATTN_NORM") == null;
}

/// `qr = q_norm(qr)` in place, and wq_b's rotated input in the scratch base (`encodeQb` runs wq_b).
pub fn encodeQNorm(ep: *const exl3.Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, qr: View, scratch: exl3.View, c: config.Config) !void {
    try exl3.chain.encodeRmsRotateIn(ep, e, core(qr), core(tensor(w.q_norm)), core(qr), w.wq_b.suh, scratch, .{
        .rows = 1,
        .width = c.q_lora,
        .x_stride = c.q_lora,
        .y_stride = c.q_lora,
        .eps = c.eps,
        .weight_f32 = w.q_norm.dtype == .f32,
        .su_signs = w.wq_b.suh_format == .packed_signs,
    });
}

/// wq_b over the rotated input `encodeQNorm` left in the scratch base.
pub fn encodeQb(ep: *const exl3.Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, q: View, scratch: exl3.View, c: config.Config) !void {
    try exl3.encode(ep, e, w.wq_b, scratch, core(q), scratch, .{ .count = 1, .x_stride = c.q_lora, .y_stride = c.heads * c.head_dim, .prerotated = true });
}
