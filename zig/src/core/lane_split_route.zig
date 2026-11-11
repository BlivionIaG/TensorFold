//! Layer-split prefill in the remote route: what a split of a prompt costs
const std = @import("std");
const Rate = @import("lane_remote_rate.zig").Rate;

/// A split is only worth its fill and landing past some length; the model prices both.
pub const Model = struct {
    /// TF_LS_SPLIT: the route may choose a split at all
    enabled: bool = false,
    /// the prompt tokens a split may take: shorter prompts do not hide its fill, longer ones its staging memory
    min_tokens: u64 = 16384,
    max_tokens: u64 = 131072,
    /// the pair's layers below the cut, prompt tokens a second (seed 3500: 0.2825 ms a row at cut 14, measured)
    stage_tps: Rate = .{ .v = 3500 },
    /// this machine's layers from the cut onward, tokens a second while rows arrive (seed 6000: ~0.16 ms a row)
    m5_tps: Rate = .{ .v = 6000 },
    /// what a split costs beyond its rows, seconds: ramp fill, the last chunk, the cone replay, landing (seed 1.4)
    fixed_s: Rate = .{ .v = 1.4 },

    /// Whether the prefix of `prefix` prompt tokens may be split.
    pub fn allows(m: *const Model, prefix: u64) bool {
        return m.enabled and prefix >= m.min_tokens and prefix <= m.max_tokens;
    }

    /// The pipeline's rate: the slower of its two stages.
    pub fn tps(m: *const Model) f64 {
        return @min(m.stage_tps.v, m.m5_tps.v);
    }

    /// Seconds from the trigger to the state landed, for `prefix` rows.
    pub fn seconds(m: *const Model, prefix: u64) f64 {
        return @as(f64, @floatFromInt(prefix)) / m.tps() + m.fixed_s.v;
    }

    /// A split's end to end time as the pair side measured it (trigger to state landed)
    pub fn observeTotal(m: *Model, rows: u64, ns: u64) void {
        const total = @as(f64, @floatFromInt(ns)) / 1e9;
        const streaming = total - m.fixed_s.v;
        if (rows == 0 or streaming < total / 2) return; // a split shorter than its own fixed cost says nothing of a rate
        m.stage_tps.blend(@as(f64, @floatFromInt(rows)) / streaming);
    }

    /// This machine's own passes over the rows (cut..19 and the cone's publish), as its engine timed them.
    pub fn observeM5(m: *Model, rows: u64, ns: u64) void {
        if (rows == 0 or ns == 0) return;
        m.m5_tps.blend(@as(f64, @floatFromInt(rows)) / (@as(f64, @floatFromInt(ns)) / 1e9));
    }
};

const t = std.testing;

test "a split costs its rows at the slower stage plus its fixed cost, and only a prompt in its window may split" {
    var m: Model = .{ .enabled = true };
    try t.expectApproxEqAbs(@as(f64, 32767) / 3500 + 1.4, m.seconds(32767), 1e-9);
    try t.expect(m.allows(16384) and m.allows(131072) and !m.allows(16383) and !m.allows(131073));
    m.m5_tps.v = 2000; // this machine the bottleneck
    try t.expectEqual(@as(f64, 2000), m.tps());
    m.enabled = false;
    try t.expect(!m.allows(32768));
}

test "measured passes move the stage rates, and a split no longer than its fixed cost says nothing" {
    var m: Model = .{ .enabled = true };
    m.observeTotal(32767, 10_800_000_000); // 9.4 s of rows: 3486 a second
    try t.expectApproxEqAbs(@as(f64, 32767) / 9.4, m.stage_tps.v, 1e-6);
    const before = m.stage_tps.v;
    m.observeTotal(100, 1_500_000_000); // 0.1 s over the fixed cost
    m.observeTotal(0, 5_000_000_000);
    try t.expectEqual(before, m.stage_tps.v);
    m.observeM5(32767, 5_000_000_000);
    try t.expectApproxEqAbs(@as(f64, 32767) / 5.0, m.m5_tps.v, 1e-6);
    m.observeM5(0, 1);
    try t.expectApproxEqAbs(@as(f64, 32767) / 5.0, m.m5_tps.v, 1e-6);
}
