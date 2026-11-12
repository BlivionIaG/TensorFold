//! Chain entries: an EXL3 projection's epilogue fused with the block-local work that follows it.
//! Every fused dispatch computes, statement for statement, what the dispatches it replaces compute, so the
//! chain is bit-identical to the unfused sequence (kernels/metal/core/exl3_projection_chain.metal).
const std = @import("std");
const mtl = @import("metal");
const core = @import("exl3_projection_core.zig");
const pipe = @import("exl3_projection_pipelines.zig");
const swi = @import("exl3_projection_swiglu.zig");
const enc = @import("exl3_projection_encode.zig");
const Pipelines = pipe.Pipelines;
const View = core.View;
const Rows = core.Rows;
const Experts = core.Experts;
const bind = core.bind;

/// The chain kernels' resident pipelines; a pipeline that failed to build leaves its entry null.
pub const Set = struct {
    rotate_out_in: ?mtl.Pipeline = null,
    rms_rotate_in: ?mtl.Pipeline = null,
    rotate_out_rope: ?mtl.Pipeline = null,

    pub fn load(device: mtl.Device, lib: mtl.Library) Set {
        return .{
            .rotate_out_in = pipeline(device, lib, "tf_exl3_rotate_out_in", 128),
            .rms_rotate_in = pipeline(device, lib, "tf_exl3_rms_rotate_in", 256),
            .rotate_out_rope = pipeline(device, lib, "tf_exl3_rotate_out_rope", 128),
        };
    }

    pub fn deinit(s: *Set) void {
        if (s.rotate_out_in) |pf| pf.deinit();
        if (s.rms_rotate_in) |pf| pf.deinit();
        if (s.rotate_out_rope) |pf| pf.deinit();
    }
};

fn pipeline(device: mtl.Device, lib: mtl.Library, name: [:0]const u8, threads: usize) ?mtl.Pipeline {
    const pf = mtl.Pipeline.init(device, lib, name, false) catch |err| {
        std.log.err("exl3: chain pipeline {s} failed: {s}", .{ name, @errorName(err) });
        return null;
    };
    if (pf.simdWidth() != 32 or pf.maxThreads() < threads) {
        pf.deinit();
        return null;
    }
    return pf;
}

/// The projection that consumes the fused epilogue's bf16 values: its rotate_in scales (`su`, `su_signs`) and its k.
pub const Next = struct { su: View, su_signs: bool, k: usize };

const NextArgs = extern struct { k: u32, su_signs: u32, reserved0: u32 = 0, reserved1: u32 = 0 };

/// Whether `encodeIndexedRotateNext` can run on these pipelines.
pub fn rotateOutInReady(p: *const Pipelines) bool {
    return p.chain.rotate_out_in != null;
}

/// `encodeIndexed` with its epilogue fused with `next`'s rotate_in: the call's bf16 output `y` is written as ever and
/// the next projection's rotated input lands in the scratch base, so that projection runs `prerotated`.
/// The call's rows must be bf16 and contiguous (`y_stride == n`, a multiple of 128, no output ids), and `next` must read
/// them as one flat row-major input of `next.k` columns.
pub fn encodeIndexedRotateNext(p: *const Pipelines, e: mtl.ComputeEncoder, experts: Experts, x: View, y: View, scratch: View, r: Rows, next: Next) !void {
    const kernel = p.chain.rotate_out_in orelse return error.UnsupportedExl3Chain;
    if (r.output != .bf16 or r.output_ids != null or r.grouped or r.y_stride != experts.n or experts.n % 128 != 0 or
        next.k % 128 != 0 or (r.count * experts.n) % next.k != 0 or experts.n > experts.k or
        next.k > std.math.maxInt(u32)) return error.InvalidExl3Chain;
    const a = try enc.indexedArgs(experts, x, y, scratch, r);
    var plain = r;
    plain.no_epilogue = true;
    swi.stages(p, e, a, scratch, scratch, scratch, scratch, experts.descriptors, x, y, scratch, plain);
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + try core.rotatedBytes(a.k, a.rows) };
    e.setPipeline(kernel);
    bind(e, sums, 0);
    bind(e, scratch, 1);
    bind(e, scratch, 2);
    bind(e, y, 3);
    e.setValue(a, 4);
    bind(e, scratch, 5);
    bind(e, r.routes.?.ids, 6);
    bind(e, experts.descriptors, 7);
    bind(e, next.su, 8);
    e.setValue(NextArgs{ .k = @intCast(next.k), .su_signs = @intFromBool(next.su_signs) }, 9);
    bind(e, scratch, 10);
    e.dispatchGroups(.{ .width = (@as(usize, a.n) / 128 + 3) / 4, .height = a.rows }, .{ .width = 128 });
    e.barrier();
}

/// A row-wise RMS norm (`dsv41_rms`) and the next projection's rotate_in: `y` receives the normalized bf16 rows (it may
/// be `x`) and the scratch base the prerotated input of a projection with `width` columns (run it `prerotated`).
pub const Rms = struct {
    rows: u32,
    width: u32,
    x_stride: u32,
    y_stride: u32,
    eps: f32,
    weight_f32: bool,
    su_signs: bool,
};

const RmsArgs = extern struct { rows: u32, width: u32, x_stride: u32, y_stride: u32, eps: f32, weight_f32: u32, su_signs: u32, reserved: u32 = 0 };

/// Threads a row's threadgroup runs: one SIMDgroup a 128-block, and at least the eight of the norm's 256 threads.
fn rmsThreads(width: u32) usize {
    return 32 * @max(8, width / 128);
}

/// The widest row `encodeRmsRotateIn` takes on these pipelines (0 when the kernel is unavailable).
pub fn rmsRotateInWidth(p: *const Pipelines) u32 {
    const kernel = p.chain.rms_rotate_in orelse return 0;
    return @intCast(kernel.maxThreads() / 32 * 128);
}

pub fn encodeRmsRotateIn(p: *const Pipelines, e: mtl.ComputeEncoder, x: View, weight: View, y: View, su: View, scratch: View, d: Rms) !void {
    const kernel = p.chain.rms_rotate_in orelse return error.UnsupportedExl3Chain;
    if (d.rows == 0 or d.width % 128 != 0 or d.width > rmsRotateInWidth(p) or d.x_stride < d.width or d.y_stride < d.width)
        return error.InvalidExl3Chain;
    const x_bytes = (@as(usize, d.rows - 1) * d.x_stride + d.width) * 2;
    const y_bytes = (@as(usize, d.rows - 1) * d.y_stride + d.width) * 2;
    const rotated_bytes = try core.rotatedBytes(d.width, d.rows);
    try core.check(x, x_bytes, 2);
    try core.check(y, y_bytes, 2);
    try core.check(scratch, rotated_bytes, 16);
    try core.disjoint(x, x_bytes, scratch, rotated_bytes);
    try core.disjoint(y, y_bytes, scratch, rotated_bytes);
    e.setPipeline(kernel);
    bind(e, x, 0);
    bind(e, weight, 1);
    bind(e, y, 2);
    e.setValue(RmsArgs{ .rows = d.rows, .width = d.width, .x_stride = d.x_stride, .y_stride = d.y_stride, .eps = d.eps, .weight_f32 = @intFromBool(d.weight_f32), .su_signs = @intFromBool(d.su_signs) }, 3);
    bind(e, su, 4);
    bind(e, scratch, 5);
    e.dispatchGroups(.{ .width = d.rows }, .{ .width = rmsThreads(d.width) });
    e.barrier();
}

/// A rope over each output row's heads: `width` columns a head, the last `rotary` of them rotated at
/// `position + row * step` with the factors the caller binds (`factors`: a float2 per position and pair).
pub const Rope = struct { width: u32, rotary: u32, position: u32, step: u32 = 1, inverse: bool = false };

const RopeArgs = extern struct { width: u32, rotary: u32, position: u32, step: u32, inverse: u32, reserved0: u32 = 0, reserved1: u32 = 0, reserved2: u32 = 0 };

/// `encode` of a dense matrix with its epilogue fused with the output's RoPE (`dsv41_rope` over the heads of `y`, in
/// place after the bf16 rounding): `y` receives the rotated rows. `r.prerotated` is honoured. The rotary columns must
/// be a multiple of 32, end the head, and the head must hold whole 128-column blocks; bias and sign scales both work.
pub fn encodeRotateOutRope(p: *const Pipelines, e: mtl.ComputeEncoder, m: core.Matrix, x: View, y: View, scratch: View, r: Rows, factors: View, rope: Rope) !void {
    const kernel = p.chain.rotate_out_rope orelse return error.UnsupportedExl3Chain;
    if (r.output != .bf16 or r.output_ids != null or r.routes != null or r.y_stride != m.n or rope.width % 128 != 0 or
        m.n % rope.width != 0 or rope.rotary == 0 or rope.rotary % 32 != 0 or rope.rotary > 128 or r.count == 0)
        return error.InvalidExl3Chain;
    const a = try enc.matrixArgs(m, x, y, scratch, r);
    var plain = r;
    plain.no_epilogue = true;
    swi.stages(p, e, a, m.trellis, m.suh, m.svh, m.bias orelse m.svh, scratch, x, y, scratch, plain);
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + try core.rotatedBytes(a.k, a.rows) };
    e.setPipeline(kernel);
    bind(e, sums, 0);
    bind(e, m.svh, 1);
    bind(e, m.bias orelse m.svh, 2);
    bind(e, y, 3);
    e.setValue(a, 4);
    bind(e, scratch, 5);
    bind(e, scratch, 6);
    bind(e, scratch, 7);
    bind(e, factors, 8);
    e.setValue(RopeArgs{ .width = rope.width, .rotary = rope.rotary, .position = rope.position, .step = rope.step, .inverse = @intFromBool(rope.inverse) }, 9);
    e.dispatchGroups(.{ .width = (@as(usize, a.n) / 128 + 3) / 4, .height = a.rows }, .{ .width = 128 });
    e.barrier();
}
