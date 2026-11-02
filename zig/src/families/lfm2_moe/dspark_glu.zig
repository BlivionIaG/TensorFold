//! Native BF16 gate/up into SwiGLU; the same 16-row boundary as the hold's unfused projections.
const std = @import("std");
const mtl = @import("metal");
const kernels = @import("kernels.zig");
const dw = @import("dspark_weights.zig");

const Args = extern struct { rows: u32, n: u32, k: u32 };

/// `in` and `out` are buffer/byte-offset refs; out is the existing activation scratch, never gate/up scratch.
pub fn encode(e: mtl.ComputeEncoder, pipes: *const kernels.Kernels, gate: dw.Bf16, up: dw.Bf16, in: anytype, out: anytype, rows: usize) void {
    std.debug.assert(rows > 0 and rows <= 16 and gate.n == up.n and gate.k == up.k);
    e.setPipeline(pipes.get("lfm2_bf16_glu_rows"));
    e.setBuffer(in.buf, in.off, 0);
    e.setBuffer(gate.w, 0, 1);
    e.setValue(Args{ .rows = @intCast(rows), .n = @intCast(gate.n), .k = @intCast(gate.k) }, 2);
    e.setBuffer(out.buf, out.off, 3);
    e.setBuffer(up.w, 0, 4);
    e.dispatchGroups(mtl.Size.of((gate.n + 7) / 8, (rows + 15) / 16, 1), mtl.Size.of(256, 1, 1));
}
