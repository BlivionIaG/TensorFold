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
    /// wq_b's epilogue and the query's RoPE in one dispatch.
    qrope: bool,
    /// The output's inverse RoPE and wo_a's rotate_in in one dispatch.
    orope: bool,
};

var cached: ?Switches = null;

pub fn switches() Switches {
    if (cached) |s| return s;
    const s: Switches = .{
        .wo_chain = std.c.getenv("TF_DSV41_NO_FUSE_WO_CHAIN") == null,
        .qnorm = std.c.getenv("TF_DSV41_NO_FUSE_QNORM") == null,
        .qrope = std.c.getenv("TF_DSV41_NO_FUSE_QROPE") == null,
        .orope = std.c.getenv("TF_DSV41_NO_FUSE_OROPE") == null,
    };
    cached = s;
    return s;
}

/// The rope's rotation factors for a layer of compress ratio `ratio`: the compressed layers' follow the plain ones.
fn ropeFactors(ap: *const common.Pipelines, ratio: u32) exl3.View {
    const offset: usize = if (ratio == 0) 0 else @as(usize, ap.rotary.capacity) * ap.rotary.rotary * @sizeOf(f32);
    return .{ .buffer = ap.rotary.factors, .offset = offset };
}

/// The eight output groups' indexed wo_a call. `chained` fuses its epilogue with wo_b's rotate_in (`encodeWoB` then
/// runs wo_b over the scratch); `rope` fuses the output's inverse RoPE into the call's own rotate_in (the rope the
/// read stage left pending). `grouped` still receives wo_a's bf16 output.
pub fn encodeWoA(ep: *const exl3.Pipelines, ap: *const common.Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, work: *const Workspace, q: View, grouped: View, scratch: exl3.View, c: config.Config, chained: bool, rope: ?struct { ratio: u32, position: u32 }) !void {
    for (&w.wo_a) |*m| exl3.useMatrix(e, m.*);
    const q_width = c.heads * c.head_dim;
    const group_input = q_width / c.o_groups;
    var f: exl3.chain.Fused = .{};
    if (rope) |r| f.rope_in = .{ .factors = ropeFactors(ap, r.ratio), .width = c.head_dim, .rotary = c.rope_dim, .stride = q_width, .position = r.position, .inverse = true };
    if (chained) f.next = .{ .su = w.wo_b.suh, .su_signs = w.wo_b.suh_format == .packed_signs, .k = c.o_groups * c.o_lora };
    try exl3.chain.encodeIndexed(ep, e, .{ .descriptors = .{ .buffer = w.wo_a_descriptors.?, .offset = 0 }, .count = c.o_groups, .k = group_input, .n = c.o_lora }, core(q), core(grouped), scratch, .{
        .count = c.o_groups,
        .x_stride = group_input,
        .y_stride = c.o_lora,
        .output = .bf16,
        .output_rows = c.o_groups,
        .routes = .{ .ids = core(.{ .buffer = work.ids, .offset = 0 }), .repeat = 1 },
    }, f);
}

/// Whether the output's inverse RoPE can wait for wo_a's rotate_in (`encodeWoA` with `rope`), one decode row.
pub fn oRopeReady(ep: *const exl3.Pipelines, ap: *const common.Pipelines, w: *const weights.Attention, c: config.Config, position: u32) bool {
    return switches().orope and w.wo_a_descriptors != null and exl3.chain.ropeRotateInReady(ep) and
        ap.rotary.rotary == c.rope_dim and position < ap.rotary.capacity and c.head_dim % 128 == 0 and
        c.rope_dim % 32 == 0 and c.rope_dim <= 128 and (c.heads * c.head_dim / c.o_groups) % c.head_dim == 0 and
        std.c.getenv("TF_DSV41_NO_FUSE_WO_A") == null and std.c.getenv("TF_DSV41_DOUBLE_ATTN_WOA") == null and
        std.c.getenv("TF_DSV41_DOUBLE_ROPE") == null;
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

/// Whether wq_b's epilogue and the query's RoPE of one decode row can run as one dispatch.
pub fn qRopeReady(ep: *const exl3.Pipelines, ap: *const common.Pipelines, c: config.Config, position: u32) bool {
    return switches().qrope and ep.chain.rotate_out_rope != null and ap.rotary.rotary == c.rope_dim and
        position < ap.rotary.capacity and c.head_dim % 128 == 0 and c.rope_dim % 32 == 0 and c.rope_dim <= 128 and
        std.c.getenv("TF_DSV41_DOUBLE_ROPE") == null;
}

/// wq_b into `q`. `rotated`: the input is already rotated in the scratch base (`encodeQNorm` left it there), else it is
/// `qr`. `rope` (the layer's compress ratio and position) fuses the query's RoPE into the epilogue.
pub fn encodeQb(ep: *const exl3.Pipelines, ap: *const common.Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, qr: View, q: View, scratch: exl3.View, c: config.Config, rotated: bool, rope: ?struct { ratio: u32, position: u32 }) !void {
    const rows: exl3.Rows = .{ .count = 1, .x_stride = c.q_lora, .y_stride = c.heads * c.head_dim, .prerotated = rotated };
    const x = if (rotated) scratch else core(qr);
    const r = rope orelse return exl3.encode(ep, e, w.wq_b, x, core(q), scratch, rows);
    try exl3.chain.encodeRotateOutRope(ep, e, w.wq_b, x, core(q), scratch, rows, ropeFactors(ap, r.ratio), .{
        .width = c.head_dim,
        .rotary = c.rope_dim,
        .position = r.position,
    });
}
