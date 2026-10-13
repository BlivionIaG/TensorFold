//! LFM2's pipelines: the family's kernels and the core row projections (2-8 bit, plain and by expert).
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const row = @import("../../core/row_projection.zig");

const names = [_][:0]const u8{ "lfm2_embed_b4", "lfm2_embed_b6", "lfm2_embed_b8", "lfm2_norm", "lfm2_conv", "lfm2_qk", "lfm2_attn_partial", "lfm2_attn_merge", "lfm2_route", "lfm2_swiglu", "lfm2_combine" };

pub const Kernels = struct {
    pipelines: [names.len]mtl.Pipeline,
    rows: row.Pipelines,

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
    return out;
}
