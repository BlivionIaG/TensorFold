//! A rate or time of the remote route, seeded and then blended with what is measured.
const std = @import("std");

pub const Rate = struct {
    v: f64,
    seen: bool = false,

    /// A sample within a third to three times the standing value: the first replaces the seed, then 70% / 30%.
    pub fn blend(r: *Rate, sample: f64) void {
        if (!std.math.isFinite(sample) or sample <= 0) return;
        const s = std.math.clamp(sample, r.v / 3, r.v * 3);
        r.v = if (r.seen) 0.7 * r.v + 0.3 * s else s;
        r.seen = true;
    }
};
