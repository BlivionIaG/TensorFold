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

    pub fn load(device: mtl.Device, lib: mtl.Library) Set {
        return .{ .rotate_out_in = pipeline(device, lib, "tf_exl3_rotate_out_in", 128) };
    }

    pub fn deinit(s: *Set) void {
        if (s.rotate_out_in) |pf| pf.deinit();
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
