//! LFM2's pipelines: the family's kernels and the core row projections (2-8 bit, plain and by expert).
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const row = @import("../../core/row_projection.zig");

const names = [_][:0]const u8{ "lfm2_embed_b4", "lfm2_embed_b6", "lfm2_embed_b8", "lfm2_norm", "lfm2_conv", "lfm2_qk", "lfm2_attn_partial", "lfm2_attn_merge", "lfm2_route_logits", "lfm2_route_select", "lfm2_swiglu", "lfm2_combine", "lfm2_sort", "lfm2_gather", "lfm2_combine_sorted" };

/// A 4-bit shape's simdgroup tiles, and whether its scalar twin's 1-2 rows equal them on this GPU.
pub const Twin = struct { n: usize, k: usize, scalar: bool };

pub const Kernels = struct {
    pipelines: [names.len]mtl.Pipeline,
    rows: row.Pipelines,
    simd: row.simd.Pipelines,
    twins: [8]Twin = undefined,
    shapes: usize = 0,

    /// Tiles for `w` (4-bit groups of 64) with the twin's verdict, checked once per shape; null for other widths.
    pub fn tiles(self: *Kernels, device: mtl.Device, w: row.Weights) !?bool {
        if (w.bits != 4 or w.group != 64 or w.sum != .f32) return null;
        for (self.twins[0..self.shapes]) |t| if (t.n == w.n and t.k == w.k) return t.scalar;
        if (self.shapes == self.twins.len) return error.TooManyLfmShapes;
        const m = row.simd.Matrix{ .w = w.w, .w_off = w.w_off, .scales = w.scales, .s_off = w.s_off, .biases = w.biases, .b_off = w.b_off, .n = w.n, .k = w.k };
        const scalar = try row.simd.twinExact(&self.simd, device, m);
        if (!scalar) std.log.info("lfm2: {d}x{d} rows take simdgroup tiles at every width (the scalar twin differs here)", .{ w.n, w.k });
        self.twins[self.shapes] = .{ .n = w.n, .k = w.k, .scalar = scalar };
        self.shapes += 1;
        return scalar;
    }

    /// The verdict `tiles` recorded for `w`'s shape; null when it takes the row kernels.
    pub fn twin(self: *const Kernels, w: row.Weights) ?bool {
        for (self.twins[0..self.shapes]) |t| if (t.n == w.n and t.k == w.k and w.bits == 4) return t.scalar;
        return null;
    }

    pub fn get(self: *const Kernels, comptime key: []const u8) mtl.Pipeline {
        inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, key)) return self.pipelines[i];
        @compileError("no LFM2 kernel " ++ key);
    }

    /// The embedding gather for a table of `bits`-wide codes.
    pub fn embed(self: *const Kernels, bits: u8) mtl.Pipeline {
        return switch (bits) {
            4 => self.get("lfm2_embed_b4"),
            6 => self.get("lfm2_embed_b6"),
            else => self.get("lfm2_embed_b8"),
        };
    }

    pub fn deinit(self: *Kernels) void {
        for (self.pipelines) |p| p.deinit();
        self.rows.deinit();
        self.simd.deinit();
    }
};

pub fn load(device: mtl.Device) !Kernels {
    var out: Kernels = undefined;
    const lib = try mtl.Library.fromSource(device, sources.lfm2, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var loaded: usize = 0;
    errdefer for (out.pipelines[0..loaded]) |p| p.deinit();
    for (names, 0..) |name, i| {
        out.pipelines[i] = try mtl.Pipeline.init(device, lib, name, false);
        loaded += 1;
    }
    out.rows = try row.Pipelines.load(device);
    errdefer out.rows.deinit();
    out.simd = try row.simd.Pipelines.load(device);
    out.shapes = 0;
    return out;
}
