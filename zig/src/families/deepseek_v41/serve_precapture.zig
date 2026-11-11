//! The served engine's graph pre-capture, before it listens (TF_DSV41_GRAPH_PRECAPTURE=0: none).

const model = @import("cuda_model.zig");
const pre = @import("graph_precapture.zig");

/// Rank 0: every planned decode graph up to the served window's last position captured on every rank; nothing runs.
pub fn run(m: *model.Model, window: u32) !void {
    if (!try pre.enabled()) return;
    const last: u64 = @as(u64, window) -| 1;
    if (m.rows) |b| return b.precapture(last);
    return m.f.precaptureGraphs(last);
}
