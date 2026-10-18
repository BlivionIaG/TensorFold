//! EXL3 strips projections. The device decode matches zig/src/core/exl3_format.zig.
//! Expert launches read ids from a pointer so a graph replay can swap that buffer.
//! Gate/up and down are the same identity projection. SwiGLU is a separate kernel between them.
//! A hit passes three packed bases. The id is a line index. Combine has no mask.
//! A CPU slot needs a finite fp16 zero in `y` before combine. A zero weight alone does not drop NaN.

const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const Exl3 = struct {
    mod_linear: Module,
    mod_experts: Module,
    linear_fn: Function,
    gate_up_fn: Function,
    swiglu_fn: Function,
    down_fn: Function,
    combine_fn: Function,
    linear_ref_fn: Function,
    gate_up_ref_fn: Function,
    down_ref_fn: Function,
    probe_fn: Function,

    /// Threads per block of the decode-path projections (must equal kFastThreads in exl3_decode.cuh).
    pub const fast_threads: u32 = 512;

    pub fn load(d: *const Driver) !Exl3 {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var linear_mod = try Module.load(d, kernels.exl3_linear);
        errdefer linear_mod.unload();
        var experts_mod = try Module.load(d, kernels.exl3_experts);
        errdefer experts_mod.unload();
        return .{
            .mod_linear = linear_mod,
            .mod_experts = experts_mod,
            .linear_fn = try linear_mod.function(kernels.exl3_symbols.linear_f16),
            .gate_up_fn = try experts_mod.function(kernels.exl3_symbols.gate_up_f16),
            .swiglu_fn = try experts_mod.function(kernels.exl3_symbols.swiglu_f16),
            .down_fn = try experts_mod.function(kernels.exl3_symbols.down_f16),
            .combine_fn = try experts_mod.function(kernels.exl3_symbols.combine_f16),
            .linear_ref_fn = try linear_mod.function(kernels.exl3_symbols.linear_f16_ref),
            .gate_up_ref_fn = try experts_mod.function(kernels.exl3_symbols.gate_up_f16_ref),
            .down_ref_fn = try experts_mod.function(kernels.exl3_symbols.down_f16_ref),
            .probe_fn = try linear_mod.function(kernels.exl3_symbols.mul1_probe),
        };
    }

    pub fn unload(g: *Exl3) void {
        g.mod_experts.unload();
        g.mod_linear.unload();
    }

    /// Counts the 16-bit states whose fp32 mul1 codebook value differs from the binary64 reference into `bad` (u32).
    pub fn probeMul1(g: *const Exl3, s: Stream, bad: u64) !void {
        var a: launch_.Args = .{};
        a.add(bad);
        try launch_.launch(g.probe_fn, .{ .grid = .{ .x = 256, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }

    /// y[rows, N] = x[rows, K] against one strips-order matrix. Grid (N/128, rows), block `fast_threads`.
    pub fn linear(g: *const Exl3, s: Stream, x: u64, trellis: u64, suh: u64, svh: u64, y: u64, rows: usize, k: usize, n: usize, halves: u8, codebook: u8) !void {
        return g.linearWith(g.linear_fn, fast_threads, s, x, trellis, suh, svh, y, rows, k, n, halves, codebook);
    }

    /// The original one-warp kernel (block 32). Bit-identical to `linear`; for the bench and equality checks.
    pub fn linearRef(g: *const Exl3, s: Stream, x: u64, trellis: u64, suh: u64, svh: u64, y: u64, rows: usize, k: usize, n: usize, halves: u8, codebook: u8) !void {
        return g.linearWith(g.linear_ref_fn, 32, s, x, trellis, suh, svh, y, rows, k, n, halves, codebook);
    }

    fn linearWith(g: *const Exl3, f: Function, block: u32, s: Stream, x: u64, trellis: u64, suh: u64, svh: u64, y: u64, rows: usize, k: usize, n: usize, halves: u8, codebook: u8) !void {
        _ = g;
        if (rows == 0) return;
        const dims = try projection(rows, k, n, halves, codebook);
        var a: launch_.Args = .{};
        a.add(x);
        a.add(trellis);
        a.add(suh);
        a.add(svh);
        a.add(y);
        a.add(dims.rows);
        a.add(dims.k);
        a.add(dims.n);
        a.add(dims.halves);
        a.add(dims.codebook);
        try launch_.launch(f, .{
            .grid = .{ .x = dims.n_blocks, .y = dims.rows_u, .z = 1 },
            .block = .{ .x = block, .y = 1, .z = 1 },
        }, s, &a);
    }

    /// One expert matrix for every (token, slot). `ids` is int32[tokens * topk] of line indices.
    /// y is [tokens * topk, N]. The pointers are that one matrix's packed base.
    /// The kernel adds `line * tileSpan(k, n, halves)` to trellis, `line * k` to suh, and `line * n` to svh.
    /// `tokens * topk` is `grid.y`. The slot is `blockIdx.y`, so a count above `grid_y_max` is refused
    /// before the driver. `hit` passes whole-token chunks that fit.
    pub fn experts(g: *const Exl3, s: Stream, which: Expert, x: u64, ids: u64, trellis: u64, suh: u64, svh: u64, y: u64, tokens: usize, k: usize, n: usize, n_experts: usize, topk: usize, halves: u8, codebook: u8) !void {
        const fn_ = switch (which) {
            .gate_up => g.gate_up_fn,
            .down => g.down_fn,
        };
        return g.expertsWith(fn_, fast_threads, s, x, ids, trellis, suh, svh, y, tokens, k, n, n_experts, topk, halves, codebook);
    }

    /// The original one-warp expert kernels (block 32); bit-identical to `experts`.
    pub fn expertsRef(g: *const Exl3, s: Stream, which: Expert, x: u64, ids: u64, trellis: u64, suh: u64, svh: u64, y: u64, tokens: usize, k: usize, n: usize, n_experts: usize, topk: usize, halves: u8, codebook: u8) !void {
        const fn_ = switch (which) {
            .gate_up => g.gate_up_ref_fn,
            .down => g.down_ref_fn,
        };
        return g.expertsWith(fn_, 32, s, x, ids, trellis, suh, svh, y, tokens, k, n, n_experts, topk, halves, codebook);
    }

    fn expertsWith(g: *const Exl3, fn_: Function, block: u32, s: Stream, x: u64, ids: u64, trellis: u64, suh: u64, svh: u64, y: u64, tokens: usize, k: usize, n: usize, n_experts: usize, topk: usize, halves: u8, codebook: u8) !void {
        _ = g;
        if (tokens == 0) return;
        if (n_experts == 0 or topk == 0) return error.Exl3Shape;
        const slots = std.math.mul(usize, tokens, topk) catch return error.Exl3Shape;
        if (slots > grid_y_max) return error.Exl3Shape;
        const dims = try projection(slots, k, n, halves, codebook);
        const experts_i = std.math.cast(i32, n_experts) orelse return error.Exl3Shape;
        const topk_i = std.math.cast(i32, topk) orelse return error.Exl3Shape;
        const tokens_i = std.math.cast(i32, tokens) orelse return error.Exl3Shape;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(ids);
        a.add(trellis);
        a.add(suh);
        a.add(svh);
        a.add(y);
        a.add(tokens_i);
        a.add(dims.k);
        a.add(dims.n);
        a.add(experts_i);
        a.add(topk_i);
        a.add(dims.halves);
        a.add(dims.codebook);
        try launch_.launch(fn_, .{
            .grid = .{ .x = dims.n_blocks, .y = dims.rows_u, .z = 1 },
            .block = .{ .x = block, .y = 1, .z = 1 },
        }, s, &a);
    }

    /// fp16 rows of `silu(fp32(gate)) * fp32(up)`, the v0.6.6 act_mode 1 formula with limit +inf.
    /// `rows` is `tokens * topk`. `n` is the gate/up N. Launched between the identity projections and down.
    /// The row is `blockIdx.y`, so each launch has at most `grid_y_max` rows. The next chunk advances
    /// gate, up, and y by that many rows of `n` fp16 elements.
    pub fn swiglu(g: *const Exl3, s: Stream, gate: u64, up: u64, y: u64, rows: usize, n: usize) !void {
        if (rows == 0) return;
        if (n == 0) return error.Exl3Shape;
        const n_i = std.math.cast(i32, n) orelse return error.Exl3Shape;
        const n_blocks: u32 = @intCast((n + 127) / 128);
        var left = rows;
        var gate_at = gate;
        var up_at = up;
        var y_at = y;
        while (left > 0) {
            const chunk = try fitRows(left, 1);
            const rows_i: i32 = @intCast(chunk);
            var a = swigluArgs(gate_at, up_at, y_at, rows_i, n_i);
            try launch_.launch(g.swiglu_fn, .{
                .grid = .{ .x = n_blocks, .y = @intCast(rows_i), .z = 1 },
                .block = .{ .x = 128, .y = 1, .z = 1 },
            }, s, &a);
            const cells = std.math.mul(usize, chunk, n) catch return error.Exl3Shape;
            gate_at = try bump(gate_at, cells, fp16_bytes);
            up_at = try bump(up_at, cells, fp16_bytes);
            y_at = try bump(y_at, cells, fp16_bytes);
            left -= chunk;
        }
    }

    /// Gate, up, SwiGLU, down. `gate`, `up`, and `down` are three packed bases, each with its own strides.
    /// `ids` are line indices into all three. `n_lines` is one past the highest line those arrays hold.
    /// Whole-token chunks keep `chunk * topk <= grid_y_max`, because the slot is `blockIdx.y`.
    /// Each chunk advances `ids` by `chunk * topk` int32s, gate/up `x` by `chunk * K` fp16s, and each `y`
    /// by `chunk * topk * N` fp16s. Down is one token per slot (`topk` 1) and its rows are capped at `grid_y_max`.
    /// `expert_slot` is unchanged. Combine stays separate. Before combine, the host writes a finite fp16 zero
    /// into a CPU slot's `y` and a 0 weight. A zero weight alone does not drop NaN. An out-of-range id does not
    /// store that zero. This does not pass `slot_ptrs` or `hit_mask`.
    pub fn hit(g: *const Exl3, s: Stream, x: u64, ids: u64, gate: Packed, up: Packed, down: Packed, y_gate: u64, y_up: u64, y_glu: u64, y_down: u64, tokens: usize, n_lines: usize, topk: usize) !void {
        if (tokens == 0) return;
        const p = try plan(x, gate, up, down, y_gate, y_up, y_glu, y_down, tokens, n_lines, topk);
        var left = tokens;
        var ids_at = ids;
        var x_at = x;
        var y_gate_at = y_gate;
        var y_up_at = y_up;
        var y_glu_at = y_glu;
        var y_down_at = y_down;
        while (left > 0) {
            const chunk = try fitRows(left, topk);
            const slots = std.math.mul(usize, chunk, topk) catch return error.Exl3Shape;
            try g.experts(s, .gate_up, x_at, ids_at, p.gate.trellis, p.gate.suh, p.gate.svh, y_gate_at, chunk, p.gate.k, p.gate.n, p.gate.n_lines, topk, p.gate.halves, p.gate.codebook);
            try g.experts(s, .gate_up, x_at, ids_at, p.up.trellis, p.up.suh, p.up.svh, y_up_at, chunk, p.up.k, p.up.n, p.up.n_lines, topk, p.up.halves, p.up.codebook);
            try g.swiglu(s, y_gate_at, y_up_at, y_glu_at, slots, p.glu_n);
            try g.experts(s, .down, y_glu_at, ids_at, p.down.trellis, p.down.suh, p.down.svh, y_down_at, slots, p.down.k, p.down.n, p.down.n_lines, 1, p.down.halves, p.down.codebook);
            ids_at = try bump(ids_at, slots, id_bytes);
            const x_elems = std.math.mul(usize, chunk, p.gate.k) catch return error.Exl3Shape;
            x_at = try bump(x_at, x_elems, fp16_bytes);
            const up_elems = std.math.mul(usize, slots, p.gate.n) catch return error.Exl3Shape;
            y_gate_at = try bump(y_gate_at, up_elems, fp16_bytes);
            y_up_at = try bump(y_up_at, up_elems, fp16_bytes);
            y_glu_at = try bump(y_glu_at, up_elems, fp16_bytes);
            const down_elems = std.math.mul(usize, slots, p.down.n) catch return error.Exl3Shape;
            y_down_at = try bump(y_down_at, down_elems, fp16_bytes);
            left -= chunk;
        }
    }

    /// out[token, n] = sum_s fp32(weights[token, s]) * fp32(y[token, s, n]), then fp16.
    /// The host writes a finite fp16 zero into a CPU slot's `y` before this launch, and a 0 weight.
    /// `fmaf(0, NaN, acc)` is NaN, so the weight alone does not drop an unwritten slot. There is no mask argument.
    pub fn combine(g: *const Exl3, s: Stream, y: u64, weights: u64, out: u64, tokens: usize, n: usize, topk: usize) !void {
        if (tokens == 0) return;
        if (n == 0 or topk == 0) return error.Exl3Shape;
        const tokens_i = std.math.cast(i32, tokens) orelse return error.Exl3Shape;
        const n_i = std.math.cast(i32, n) orelse return error.Exl3Shape;
        const topk_i = std.math.cast(i32, topk) orelse return error.Exl3Shape;
        const n_blocks: u32 = @intCast((n + 127) / 128);
        var a: launch_.Args = .{};
        a.add(y);
        a.add(weights);
        a.add(out);
        a.add(tokens_i);
        a.add(n_i);
        a.add(topk_i);
        try launch_.launch(g.combine_fn, .{
            .grid = .{ .x = n_blocks, .y = @intCast(tokens_i), .z = 1 },
            .block = .{ .x = 128, .y = 1, .z = 1 },
        }, s, &a);
    }
};

pub const Expert = enum { gate_up, down };

const Dims = struct {
    rows: i32,
    rows_u: u32,
    k: i32,
    n: i32,
    n_blocks: u32,
    halves: i32,
    codebook: i32,
};

fn projection(rows: usize, k: usize, n: usize, halves: u8, codebook: u8) error{Exl3Shape}!Dims {
    if (k == 0 or n == 0 or k % 128 != 0 or n % 128 != 0) return error.Exl3Shape;
    if (halves < 2 or halves > 16 or codebook > 2) return error.Exl3Shape;
    return .{
        .rows = std.math.cast(i32, rows) orelse return error.Exl3Shape,
        .rows_u = std.math.cast(u32, rows) orelse return error.Exl3Shape,
        .k = std.math.cast(i32, k) orelse return error.Exl3Shape,
        .n = std.math.cast(i32, n) orelse return error.Exl3Shape,
        .n_blocks = std.math.cast(u32, n / 128) orelse return error.Exl3Shape,
        .halves = halves,
        .codebook = codebook,
    };
}

/// One packed expert matrix. Line `i` is `trellis + i * trellis_stride` (int16 words),
/// `suh + i * suh_stride` and `svh + i * svh_stride` (fp16 elements).
/// `trellis_stride` is `tileSpan`. `suh_stride` is K. `svh_stride` is N.
/// That span is not the 1,901,213-byte residency line, which sums three regions.
pub const Packed = struct {
    trellis: u64,
    suh: u64,
    svh: u64,
    trellis_stride: usize,
    suh_stride: usize,
    svh_stride: usize,
    halves: u8,
    codebook: u8,
};

/// int16 words between consecutive lines. `expert_slot` multiplies the line index by this.
pub fn tileSpan(k: usize, n: usize, halves: u8) error{Exl3Shape}!usize {
    if (k == 0 or n == 0 or k % 128 != 0 or n % 128 != 0) return error.Exl3Shape;
    if (halves < 2 or halves > 16) return error.Exl3Shape;
    const tiles_k = k / 16;
    const tiles_n = n / 16;
    const words = @as(usize, halves) * 8;
    const tiles = std.math.mul(usize, tiles_k, tiles_n) catch return error.Exl3Shape;
    return std.math.mul(usize, tiles, words) catch return error.Exl3Shape;
}

/// Element strides the lane uses when it allocates one packed matrix. Not byte sizes.
pub const Strides = struct { trellis: usize, suh: usize, svh: usize };

pub fn strides(k: usize, n: usize, halves: u8) error{Exl3Shape}!Strides {
    return .{ .trellis = try tileSpan(k, n, halves), .suh = k, .svh = n };
}

const Step = struct {
    which: Expert,
    x: u64,
    trellis: u64,
    suh: u64,
    svh: u64,
    y: u64,
    tokens: usize,
    k: usize,
    n: usize,
    n_lines: usize,
    topk: usize,
    halves: u8,
    codebook: u8,
};

const Plan = struct {
    gate: Step,
    up: Step,
    glu_gate: u64,
    glu_up: u64,
    glu_y: u64,
    glu_rows: usize,
    glu_n: usize,
    down: Step,
};

fn swigluArgs(gate: u64, up: u64, y: u64, rows: i32, n: i32) launch_.Args {
    var a: launch_.Args = .{};
    a.add(gate);
    a.add(up);
    a.add(y);
    a.add(rows);
    a.add(n);
    return a;
}

fn matchSpan(m: Packed) error{Exl3Shape}!void {
    if (m.codebook > 2) return error.Exl3Shape;
    if (m.trellis_stride != try tileSpan(m.suh_stride, m.svh_stride, m.halves)) return error.Exl3Shape;
}

fn stepOf(which: Expert, x: u64, y: u64, m: Packed, tokens: usize, n_lines: usize, topk: usize) error{Exl3Shape}!Step {
    try matchSpan(m);
    return .{
        .which = which,
        .x = x,
        .trellis = m.trellis,
        .suh = m.suh,
        .svh = m.svh,
        .y = y,
        .tokens = tokens,
        .k = m.suh_stride,
        .n = m.svh_stride,
        .n_lines = n_lines,
        .topk = topk,
        .halves = m.halves,
        .codebook = m.codebook,
    };
}

fn plan(x: u64, gate: Packed, up: Packed, down: Packed, y_gate: u64, y_up: u64, y_glu: u64, y_down: u64, tokens: usize, n_lines: usize, topk: usize) error{Exl3Shape}!Plan {
    if (n_lines == 0 or topk == 0) return error.Exl3Shape;
    if (std.math.cast(i32, n_lines) == null) return error.Exl3Shape;
    if (gate.suh_stride != up.suh_stride or gate.svh_stride != up.svh_stride) return error.Exl3Shape;
    // Down's K is the SwiGLU width, which is the gate/up N. Its N is the hidden the gate read.
    if (down.suh_stride != gate.svh_stride or down.svh_stride != gate.suh_stride) return error.Exl3Shape;
    const rows = std.math.mul(usize, tokens, topk) catch return error.Exl3Shape;
    return .{
        .gate = try stepOf(.gate_up, x, y_gate, gate, tokens, n_lines, topk),
        .up = try stepOf(.gate_up, x, y_up, up, tokens, n_lines, topk),
        .glu_gate = y_gate,
        .glu_up = y_up,
        .glu_y = y_glu,
        .glu_rows = rows,
        .glu_n = gate.svh_stride,
        .down = try stepOf(.down, y_glu, y_down, down, rows, n_lines, 1),
    };
}

/// CUDA gridDim.y maximum. The expert slot and the SwiGLU row are blockIdx.y, so they cannot exceed this.
const grid_y_max: usize = 65535;
const fp16_bytes: usize = 2;
const id_bytes: usize = 4;

/// Whole rows that fit in grid.y when each row occupies `per` slots. `per` 1 is SwiGLU and down.
fn fitRows(rows: usize, per: usize) error{Exl3Shape}!usize {
    if (per == 0) return error.Exl3Shape;
    const room = grid_y_max / per;
    if (room == 0) return error.Exl3Shape;
    if (rows == 0) return 0;
    return @min(rows, room);
}

fn bump(ptr: u64, elems: usize, width: usize) error{Exl3Shape}!u64 {
    const bytes = std.math.mul(usize, elems, width) catch return error.Exl3Shape;
    const wide = std.math.cast(u64, bytes) orelse return error.Exl3Shape;
    return std.math.add(u64, ptr, wide) catch return error.Exl3Shape;
}

fn testPacked(trellis: u64, suh: u64, svh: u64, k: usize, n: usize, halves: u8, codebook: u8) !Packed {
    return .{
        .trellis = trellis,
        .suh = suh,
        .svh = svh,
        .trellis_stride = try tileSpan(k, n, halves),
        .suh_stride = k,
        .svh_stride = n,
        .halves = halves,
        .codebook = codebook,
    };
}

test "swiglu is the device symbol between the identity projections" {
    try std.testing.expectEqualStrings("exl3_experts_swiglu_f16", kernels.exl3_symbols.swiglu_f16);
    var a = swigluArgs(0x10, 0x20, 0x30, 7, 640);
    try std.testing.expectEqual(@as(usize, 5), a.count);
    try std.testing.expectEqual(@as(u64, 0x10), a.integer(0).?);
    try std.testing.expectEqual(@as(u64, 0x20), a.integer(1).?);
    try std.testing.expectEqual(@as(u64, 0x30), a.integer(2).?);
    try std.testing.expectEqual(@as(u64, 7), a.integer(3).?);
    try std.testing.expectEqual(@as(u64, 640), a.integer(4).?);

    var g: Exl3 = undefined;
    const s = Stream{ .d = undefined, .handle = null };
    try g.swiglu(s, 1, 2, 3, 0, 640);
    try std.testing.expectError(error.Exl3Shape, g.swiglu(s, 1, 2, 3, 4, 0));
}

test "three packed bases address a line by the kernel tile span, not 1901213" {
    const gate_span = try tileSpan(2560, 640, 4);
    const down_span = try tileSpan(640, 2560, 4);
    try std.testing.expect(gate_span != 1_901_213);
    try std.testing.expectEqual(gate_span, down_span);
    const wide = try strides(2560, 640, 4);
    const up_only = try strides(2560, 640, 6);
    try std.testing.expectEqual(@as(usize, 2560), wide.suh);
    try std.testing.expectEqual(@as(usize, 640), wide.svh);
    try std.testing.expect(up_only.trellis != wide.trellis);
    try std.testing.expectError(error.Exl3Shape, tileSpan(2560, 640, 1));
    try std.testing.expectError(error.Exl3Shape, tileSpan(2560, 600, 4));

    const gate = try testPacked(0x10, 0x11, 0x12, 2560, 640, 4, 2);
    const up = try testPacked(0x20, 0x21, 0x22, 2560, 640, 6, 1);
    const down = try testPacked(0x30, 0x31, 0x32, 640, 2560, 4, 2);
    const p = try plan(0x40, gate, up, down, 0x50, 0x51, 0x52, 0x53, 3, 9, 4);
    try std.testing.expectEqual(Expert.gate_up, p.gate.which);
    try std.testing.expectEqual(Expert.gate_up, p.up.which);
    try std.testing.expectEqual(Expert.down, p.down.which);
    try std.testing.expectEqual(@as(u64, 0x10), p.gate.trellis);
    try std.testing.expectEqual(@as(u64, 0x21), p.up.suh);
    try std.testing.expectEqual(@as(u64, 0x32), p.down.svh);
    try std.testing.expectEqual(@as(u64, 0x40), p.gate.x);
    try std.testing.expectEqual(@as(u64, 0x40), p.up.x);
    try std.testing.expectEqual(@as(u64, 0x52), p.down.x);
    try std.testing.expectEqual(@as(u64, 0x50), p.glu_gate);
    try std.testing.expectEqual(@as(u64, 0x51), p.glu_up);
    try std.testing.expectEqual(@as(u64, 0x52), p.glu_y);
    try std.testing.expectEqual(@as(usize, 3), p.gate.tokens);
    try std.testing.expectEqual(@as(usize, 4), p.gate.topk);
    try std.testing.expectEqual(@as(usize, 12), p.down.tokens);
    try std.testing.expectEqual(@as(usize, 1), p.down.topk);
    try std.testing.expectEqual(@as(usize, 12), p.glu_rows);
    try std.testing.expectEqual(@as(usize, 640), p.glu_n);
    try std.testing.expectEqual(@as(usize, 2560), p.gate.k);
    try std.testing.expectEqual(@as(usize, 640), p.down.k);
    try std.testing.expectEqual(@as(usize, 2560), p.down.n);
    try std.testing.expectEqual(@as(usize, 9), p.gate.n_lines);
    try std.testing.expectEqual(@as(u8, 6), p.up.halves);
    try std.testing.expectEqual(@as(u8, 1), p.up.codebook);

    var blob = gate;
    blob.trellis_stride = 1_901_213;
    try std.testing.expectError(error.Exl3Shape, plan(0x40, blob, up, down, 1, 2, 3, 4, 1, 1, 1));
    var same = down;
    same.suh_stride = 2560;
    same.svh_stride = 640;
    same.trellis_stride = gate_span;
    try std.testing.expectError(error.Exl3Shape, plan(0x40, gate, up, same, 1, 2, 3, 4, 1, 1, 1));
    try std.testing.expectError(error.Exl3Shape, plan(0x40, gate, up, down, 1, 2, 3, 4, 1, 0, 1));

    var g: Exl3 = undefined;
    const s = Stream{ .d = undefined, .handle = null };
    // An empty batch returns before any launch, even when a stride is the residency line.
    try g.hit(s, 0x40, 0x41, blob, up, down, 1, 2, 3, 4, 0, 1, 1);
    try std.testing.expectError(error.Exl3Shape, g.hit(s, 0x40, 0x41, blob, up, down, 1, 2, 3, 4, 2, 1, 1));
}

test "tokens times topk above grid.y is split and does not reach the driver" {
    const topk: usize = 10;
    const tokens: usize = 8192;
    try std.testing.expect(tokens * topk > grid_y_max);
    try std.testing.expectEqual(@as(usize, 6553), try fitRows(6554, topk));
    try std.testing.expect(6553 * topk <= grid_y_max);
    try std.testing.expect(6554 * topk > grid_y_max);
    try std.testing.expectError(error.Exl3Shape, fitRows(1, grid_y_max + 1));

    const k: usize = 2560;
    const n_up: usize = 640;
    const n_down: usize = 2560;
    var left = tokens;
    var ids_at: u64 = 0x1000;
    var x_at: u64 = 0x2000;
    var y_gate_at: u64 = 0x3000;
    var y_up_at: u64 = 0x4000;
    var y_glu_at: u64 = 0x5000;
    var y_down_at: u64 = 0x6000;
    const first = try fitRows(left, topk);
    try std.testing.expectEqual(@as(usize, 6553), first);
    try std.testing.expect(first * topk <= grid_y_max);
    const slots = first * topk;
    ids_at = try bump(ids_at, slots, id_bytes);
    x_at = try bump(x_at, first * k, fp16_bytes);
    const up_elems = slots * n_up;
    y_gate_at = try bump(y_gate_at, up_elems, fp16_bytes);
    y_up_at = try bump(y_up_at, up_elems, fp16_bytes);
    y_glu_at = try bump(y_glu_at, up_elems, fp16_bytes);
    y_down_at = try bump(y_down_at, slots * n_down, fp16_bytes);
    try std.testing.expectEqual(@as(u64, 0x1000) + slots * id_bytes, ids_at);
    try std.testing.expectEqual(@as(u64, 0x2000) + first * k * fp16_bytes, x_at);
    try std.testing.expectEqual(@as(u64, 0x3000) + slots * n_up * fp16_bytes, y_gate_at);
    try std.testing.expectEqual(@as(u64, 0x4000) + slots * n_up * fp16_bytes, y_up_at);
    try std.testing.expectEqual(@as(u64, 0x5000) + slots * n_up * fp16_bytes, y_glu_at);
    try std.testing.expectEqual(@as(u64, 0x6000) + slots * n_down * fp16_bytes, y_down_at);
    left -= first;
    const second = try fitRows(left, topk);
    try std.testing.expectEqual(@as(usize, 8192 - 6553), second);
    try std.testing.expect(second * topk <= grid_y_max);
    left -= second;
    try std.testing.expectEqual(@as(usize, 0), left);
    try std.testing.expectEqual(@as(usize, 0), try fitRows(left, topk));

    // Down is topk 1, so its rows chunk on 65535, not on the token width.
    var down_left: usize = 6554 * topk;
    var down_rows: usize = 0;
    var down_chunks: usize = 0;
    while (down_left > 0) {
        const chunk = try fitRows(down_left, 1);
        try std.testing.expect(chunk <= grid_y_max);
        down_rows += chunk;
        down_chunks += 1;
        down_left -= chunk;
    }
    try std.testing.expectEqual(@as(usize, 65540), down_rows);
    try std.testing.expectEqual(@as(usize, 2), down_chunks);

    var g: Exl3 = undefined;
    const s = Stream{ .d = undefined, .handle = null };
    try std.testing.expectError(error.Exl3Shape, g.experts(s, .gate_up, 0x40, 0x41, 0x10, 0x11, 0x12, 0x50, 6554, k, n_up, 8, topk, 4, 2));
}
