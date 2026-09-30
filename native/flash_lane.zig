const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
const Weight = @import("flash_ops.zig").Weight;
const A = mx.Array;

pub const Projection = struct {
    weight: A,
    sb: A,
    n: i32,
    k: i32,
    bits: i32,
    group: i32,
    tile: i32,

    pub fn init(s: *mx.Scope, w: Weight) !Projection {
        const shape = try w.geometry(2);
        const bits = w.format.bits;
        const group = if (w.format.group_size == 128) 64 else w.format.group_size;
        if (!mx.tensor_units or (group != 32 and group != 64) or @mod(shape.k, 64) != 0 or @mod(shape.n, 4) != 0) return error.UnsupportedProjectionGeometry;
        var scales = w.arrays[1];
        var biases = w.arrays[2];
        if (w.format.group_size == 128) {
            var value = mx.c.mlx_array_new();
            const sr = mx.c.mlx_repeat_axis(&value, scales, 2, 1, mx.stream);
            scales = try s.result(sr, value);
            value = mx.c.mlx_array_new();
            const br = mx.c.mlx_repeat_axis(&value, biases, 2, 1, mx.stream);
            biases = try s.result(br, value);
        }
        const tile: i32 = if (bits == 4 and @mod(shape.n, 64) == 0) 64 else if (@mod(shape.n, 32) == 0) 32 else 0;
        const sb = try s.cast(try s.stack(&.{ try s.transpose(scales, &.{ 1, 0 }), try s.transpose(biases, &.{ 1, 0 }) }, -1), mx.bf16);
        const words = @divExact(group * bits, 32);
        const weight = if (tile != 0) try s.contiguous(try s.reshape(try s.transpose(try s.reshape(w.arrays[0], &.{ @divExact(shape.n, tile), tile, @divExact(shape.k, group), words }), &.{ 0, 2, 1, 3 }), mx.shape(w.arrays[0]))) else w.arrays[0];
        try mx.evalMany(&.{ weight, sb }, false);
        const own = try mx.retain(weight);
        errdefer mx.free(own);
        return .{ .weight = own, .sb = try mx.retain(sb), .n = shape.n, .k = shape.k, .bits = bits, .group = group, .tile = tile };
    }

    pub fn deinit(p: *Projection) void {
        mx.free(p.weight);
        mx.free(p.sb);
    }

    pub fn apply(p: Projection, kernels: *mx.Kernels, s: *mx.Scope, x: A) !A {
        if (mx.dtype(x) != mx.bf16 or mx.dim(x, -1) != p.k) return error.InvalidProjectionInput;
        const rows: i32 = @intCast(mx.c.mlx_array_size(x) / @as(usize, @intCast(p.k)));
        if (rows < 1 or rows > 128) return error.InvalidLaneWidth;
        const mp = @divTrunc(rows + 15, 16) * 16;
        const block = @min(mp, 32);
        const dims = try s.ints(&.{ rows, mp });
        const flat = try s.reshape(x, &.{ rows, p.k });
        const kg = @divExact(p.k, p.group);
        const sums = (try kernels.run(s, src.lane_qmm_xsum, &.{ flat, dims }, &.{ mx.ti("K", p.k), mx.ti("GS", p.group) }, .{ kg, mp, 1 }, .{ @min(kg, 256), 1, 1 }, &.{.{ .shape = &.{ kg, mp }, .dtype = mx.f32t }}))[0];
        // Upstream split_k counts 32-column tiles even on the cooperative 64-column path.
        var split: i32 = 1;
        while (split < 8 and @divTrunc(p.n + 31, 32) * split < 1024 and @divTrunc(@divExact(p.k, 64), split * 2) >= 8) split *= 2;
        const wide = p.tile == 64;
        const grouped = p.bits != 4 and p.group != 64;
        const spec = if (wide) src.lane_qmm_coop else if (p.bits == 4) (if (p.tile != 0) src.lane_qmm_main_tiled else src.lane_qmm_main) else if (p.bits < 4) (if (grouped) src.lane_qmm_lowbit_grouped else src.lane_qmm_lowbit) else (if (grouped) src.lane_qmm_bytes_grouped else src.lane_qmm_bytes);
        var params: [8]mx.Template = undefined;
        params[0..4].* = .{ mx.ti("TMR", @divExact(block, 16)), mx.ti("N", p.n), mx.ti("K", p.k), mx.ti(if (wide) "SK" else "NT", if (wide) split else 32) };
        params[4] = mx.ti(if (wide) "GS" else "SK", if (wide) p.group else split);
        params[5] = mx.ti(if (wide) "EDGE" else if (p.bits == 4) "GS" else "BITS", if (wide) @intFromBool(@mod(mp, block) != 0) else if (p.bits == 4) p.group else p.bits);
        params[6] = mx.ti(if (p.bits == 4) "EDGE" else "TILED", if (p.bits == 4) @intFromBool(@mod(mp, block) != 0) else @intFromBool(p.tile != 0));
        params[7] = mx.ti("GS", p.group);
        const threads: i32 = (if (wide) @as(i32, 64) else 32) * split;
        const tile: i32 = if (wide) 64 else 32;
        const out = (try kernels.run(s, spec, &.{ flat, sums, p.weight, p.sb, dims }, params[0..if (wide) @as(usize, 6) else if (grouped) 8 else 7], .{ @divTrunc(p.n + tile - 1, tile) * threads, @divTrunc(mp + block - 1, block), 1 }, .{ threads, 1, 1 }, &.{.{ .shape = &.{ rows, p.n } }}))[0];
        var shape: [8]i32 = undefined;
        const input_shape = mx.shape(x);
        if (input_shape.len > shape.len) return error.InvalidTensorShape;
        @memcpy(shape[0..input_shape.len], input_shape);
        shape[input_shape.len - 1] = p.n;
        return s.reshape(out, shape[0..input_shape.len]);
    }
};

pub fn hyper(kernels: *mx.Kernels, s: *mx.Scope, h: A, ssp: A, down: Weight, up: Weight, norm: A, eps: A, streams: i32, low: i32) ![2]A {
    const dg = try down.geometry(2);
    const ug = try up.geometry(2);
    const rows = mx.dim(h, 0);
    const wide = mx.dim(h, 1);
    if (mx.shape(h).len != 2 or rows < 1 or rows > 128 or streams < 1 or low < 1 or @mod(low, 32) != 0 or @mod(wide, streams * 256) != 0 or @mod(wide, 1024) != 0 or dg.k != wide or ug.n != wide or ug.k != low or (dg.n != low and dg.n != low + streams)) return error.InvalidTensorShape;
    const dims = @divExact(wide, streams);
    const splits = @divExact(wide, 1024);
    const generic = down.format.bits != 4 or down.format.group_size != 32 or up.format.bits != 4 or up.format.group_size != 32;
    const scalar = rows <= 2;
    const tiles = if (scalar) rows else @divTrunc(rows + 7, 8);
    const down_spec = if (scalar) (if (generic) src.flash_qa_hc_down_row else src.flash_q4_hc_down_row) else (if (generic) src.flash_qa_hc_down_mma else src.flash_q4_hc_down_mma);
    const dp = [_]mx.Template{ mx.ti("S", streams), mx.ti("D", dims), mx.ti("ND", dg.n), mx.ti("BITS", down.format.bits), mx.ti("GS", down.format.group_size) };
    const count = try s.ints(&.{rows});
    const dt: i32 = if (rows <= 4) 8 else if (rows <= 16) 16 else 32;
    const threads: i32 = if (scalar) 1024 else 256;
    const columns: i32 = if (scalar) 32 else 8;
    const part = (try kernels.run(s, down_spec, &.{ h, ssp, norm, down.arrays[0], down.arrays[1], down.arrays[2], eps, count }, dp[0..if (generic) @as(usize, 5) else 3], .{ @divTrunc(dg.n + columns - 1, columns) * threads, splits, tiles }, .{ threads, 1, 1 }, &.{.{ .shape = &.{ splits, rows, dg.n }, .dtype = mx.f32t }}))[0];
    const up_spec = if (scalar) (if (generic) src.flash_qa_hc_up_row else src.flash_q4_hc_up_row) else (if (generic) src.flash_qa_hc_up_mma else src.flash_q4_hc_up_mma);
    const up_threads = if (scalar) 8 * streams * @divExact(low, 32) else 32 * streams;
    if (up_threads > 1024) return error.InvalidThreadgroup;
    var params = [_]mx.Template{ mx.ti("S", streams), mx.ti("D", dims), mx.ti("LOW", low), mx.ti("ND", dg.n), mx.ti("KS", splits), mx.ti("DT", dt), mx.ti("BITS", up.format.bits), mx.ti("GS", up.format.group_size) };
    if (scalar) params[5..7].* = params[6..8].*;
    const width = if (scalar) 8 else dt;
    const grid: [3]i32 = if (scalar) .{ @divExact(dims, width) * up_threads, 1, rows } else .{ @divExact(dims, width) * up_threads, tiles, 1 };
    const out = try kernels.run(s, up_spec, &.{ h, ssp, norm, part, up.arrays[0], up.arrays[1], up.arrays[2], eps, count }, params[0 .. (if (scalar) @as(usize, 5) else 6) + (if (generic) @as(usize, 2) else 0)], grid, .{ up_threads, 1, 1 }, &.{ .{ .shape = &.{ rows, dims } }, .{ .shape = &.{ @max(rows, 2), streams } } });
    return .{ out[0], try s.slice(out[1], 0, 0, rows) };
}
