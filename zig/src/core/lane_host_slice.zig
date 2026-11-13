//! Time-sliced prompt pieces (TF_DSV41_PREFILL_SLICE_MS): while a stream decodes, a prompt piece takes at most a slice.
const std = @import("std");

pub const Slice = struct {
    ns: f64 = 0, // the time one piece may take; 0: off
    ns_per_row: f64 = 450_000, // the prompt time a row takes, followed from the pieces that ran

    // a forward reads the routed weights whatever its rows (~0.2 s on M5): a smaller piece wastes most of it
    pub const min_rows: u32 = 512;
    pub const max_rows: u32 = 16384;
    const min_sample: u64 = 64;

    /// TF_DSV41_PREFILL_SLICE_MS: milliseconds; none, zero or anything else is off.
    pub fn of(text: []const u8) Slice {
        const ms = std.fmt.parseFloat(f64, std.mem.trim(u8, text, " \t")) catch return .{};
        return if (std.math.isFinite(ms) and ms > 0) .{ .ns = ms * 1e6 } else .{};
    }

    pub fn on(s: Slice) bool {
        return s.ns > 0;
    }

    /// The rows one piece may take to fit the slice: a multiple of 64, at least `min_rows`, at most the largest chunk.
    pub fn rows(s: Slice) u32 {
        const fit = s.ns / s.ns_per_row;
        const whole: u32 = if (fit >= @as(f64, max_rows)) max_rows else @intFromFloat(fit);
        return @max(min_rows, whole / 64 * 64);
    }

    /// A piece of `n` rows took `ns`: the row time follows it (the final piece, which replays, is not a sample).
    pub fn sampled(s: *Slice, n: u64, ns: i96) void {
        if (n < min_sample or ns <= 0) return;
        const per = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(n));
        s.ns_per_row = 0.7 * s.ns_per_row + 0.3 * per;
    }
};
