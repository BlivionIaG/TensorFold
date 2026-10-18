//! The batched pass's EXL3 branches (cuda_batch.zig): `lane` for a Projection whose rows are EXL3 parts, `moe` for a
//! layer whose experts are EXL3. `p` is the pass (`*cuda_batch.Pass(Tri)`): its stream, kernels and scratch.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const load = @import("cuda_exl3_load.zig");

const dims = wt.dims;
const slots = wt.slots;

/// Rows a routed-experts call takes: its scratch grows with rows * 11 (fp16 inputs, fp32 split sums).
pub const moe_rows: usize = 256;

fn copy2d(s: cuda.Stream, dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize) !void {
    const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = spitch, .dst = dst, .dst_pitch = dpitch, .width = width, .height = height };
    try s.d.check(s.d.api.cuMemcpy2DAsync_v2(&m, s.handle), "cuMemcpy2DAsync");
}

/// One part's rows [m, part.n] into `y` (row pitch part.n), bf16 or fp32.
fn part(p: anytype, ops: *load.Ops, pt: load.Part, x: u64, k: usize, m: usize, y: u64, f32_out: bool) !void {
    switch (pt.kind) {
        .exl3 => |w| {
            const de = try ops.linear();
            try de.decode(p.stream, w, x, .bf16, m, y, if (f32_out) .f32 else .bf16);
        },
        .native => |w| {
            if (f32_out) return p.k.native_lane.matmulF32(p.stream, x, k, m, w, y, p.s.native_part);
            try p.k.native_lane.matmul(p.stream, x, k, m, w, y, p.s.native_part);
        },
    }
}

/// Pass.lane() for an EXL3 projection: out bf16 [m, n] (sk 1), or fp32 [m, n] in part's first slice with the other
/// sk - 1 slices zeroed (the slice-summing callers' contract, as the native lane keeps it).
pub fn lane(p: anytype, x: u64, e: *const load.Proj, out: u64, slices: u64, m: usize, sk: usize) !void {
    const s = p.stream;
    if (sk > 1) {
        if (e.parts.len != 1) return error.UnexpectedTensor;
        try part(p, e.ops, e.parts[0], x, e.k, m, slices, true);
        const bytes = m * e.n * 4;
        try s.d.check(s.d.api.cuMemsetD8Async(slices + bytes, 0, bytes * (sk - 1), s.handle), "cuMemsetD8Async");
        return;
    }
    for (e.parts) |pt| {
        // one part over the whole row, or a single row: the part writes its columns in place
        if (pt.n == e.n or m == 1) {
            try part(p, e.ops, pt, x, e.k, m, out + pt.col * 2, false);
            continue;
        }
        const tmp = try e.ops.scratch(s, &e.ops.tmp, m * pt.n * 2);
        try part(p, e.ops, pt, x, e.k, m, tmp, false);
        try copy2d(s, out + pt.col * 2, e.n * 2, tmp, pt.n * 2, pt.n * 2, m);
    }
}

/// Pass.moeBlock() for EXL3 experts: the router's 513 logits (the BF16 rows the Store widened exactly), each row's ten
/// picks and the shared slot (id 512), then every slot's expert output, fp32 y [rows, 11, dims], in blocks of rows.
pub fn moe(p: anytype, lw: anytype, x: *const load.Experts) !void {
    const s = p.s;
    const n = p.rows;
    try p.tri.routerMany(s.mixed, lw.router.ptr, s.router_l, dims, n);
    try p.tri.topkMany(s.router_l, s.picks, s.wts, n);
    const ex = try x.ops.grouped();
    const block = @min(n, moe_rows);
    const bytes = try cuda.exl3_experts.scratchBytes(x.layer, block, slots);
    const base = try x.ops.scratch(p.stream, &x.ops.moe, bytes);
    var r0: usize = 0;
    while (r0 < n) : (r0 += block) {
        const rows = @min(block, n - r0);
        var sc = try cuda.exl3_experts.scratch(base, x.layer, block, slots);
        // runSlots leaves each slot's fp32 output in sc.y [rows * slots, dims]: the pass's y rows are that layout
        sc.y = s.y + r0 * slots * dims * 4;
        try ex.runSlots(p.stream, x.layer, s.mixed + r0 * dims * 2, dims, .bf16, s.picks + r0 * slots * 4, rows, sc);
    }
}
