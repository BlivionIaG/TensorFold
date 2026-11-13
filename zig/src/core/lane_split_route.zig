//! Layer-split prefill in the remote route: what a split of a prompt costs, and which cut it takes
const std = @import("std");
const Rate = @import("lane_remote_rate.zig").Rate;

/// The layers this machine may start at: index sources, so the selection a layer reuses is made on its own side.
pub const cuts = [_]u32{ 2, 8, 14 };

/// Per cut, ms a boundary row on the pair (layers below the cut) and on this machine (layers from it, then its cone), as the
/// first two windows measured them with the pipeline's overheads in: the pair 0.178 / 0.2825 GPU ms at cuts 8 / 14 (x1.3 in
/// flight), this machine 0.38 ms at cut 8 and ~0.0317 per layer elsewhere. The speeds below scale all cuts of one machine.
pub const pair_ms = [cuts.len]f64{ 0.070, 0.234, 0.370 };
pub const mac_ms = [cuts.len]f64{ 0.570, 0.380, 0.190 };

/// How much of the streaming time this machine may wait for rows before the pair is called the bottleneck.
const starved_share = 0.15;

/// A split is only worth its fill and landing past some length; the model prices both.
pub const Model = struct {
    /// TF_LS_SPLIT: the route may choose a split at all
    enabled: bool = false,
    /// the prompt tokens a split may take: shorter prompts do not hide its fill; the pair's pool bounds the longest
    min_tokens: u64 = 16384,
    max_tokens: u64 = 1048576,
    /// TF_LS_CUT: the cut every split takes (0: the cheapest of `cuts` at the speeds below)
    forced_cut: u32 = 0,
    /// the pair's speed against `pair_ms` (1: as measured in the first window), until a split of its own is timed
    pair_speed: Rate = .{ .v = 1 },
    /// this machine's speed against `mac_ms`
    mac_speed: Rate = .{ .v = 1 },
    /// what a split costs beyond its rows, seconds: ramp fill, the last chunk, the cone replay, landing (seed 1.4)
    fixed_s: Rate = .{ .v = 1.4 },

    /// The pair's seed at the default cut (TF_LS_SPARKS_TPS, tokens a second): the speed that gives it.
    pub fn seedPair(m: *Model, tps_at_deepest: f64) void {
        m.pair_speed.v = tps_at_deepest * pair_ms[cuts.len - 1] / 1000;
    }

    /// Whether the prefix of `prefix` prompt tokens may be split.
    pub fn allows(m: *const Model, prefix: u64) bool {
        return m.enabled and prefix >= m.min_tokens and prefix <= m.max_tokens;
    }

    fn slot(cut: u32) ?usize {
        return std.mem.indexOfScalar(u32, &cuts, cut);
    }

    /// Prompt tokens a second the pair's layers below `cut` run at (the stage's own, streaming).
    pub fn pairTps(m: *const Model, i: usize) f64 {
        return 1000 * m.pair_speed.v / pair_ms[i];
    }

    pub fn macTps(m: *const Model, i: usize) f64 {
        return 1000 * m.mac_speed.v / mac_ms[i];
    }

    /// The pipeline's rate at cut `i`: the slower of its two stages.
    pub fn tpsAt(m: *const Model, i: usize) f64 {
        return @min(m.pairTps(i), m.macTps(i));
    }

    /// The cut a split takes: the forced one, else the one whose slower stage is fastest.
    pub fn pick(m: *const Model) u32 {
        if (m.forced_cut != 0) return m.forced_cut;
        var best: usize = 0;
        for (1..cuts.len) |i| if (m.tpsAt(i) > m.tpsAt(best)) {
            best = i;
        };
        return cuts[best];
    }

    /// The pipeline's rate at the cut a split takes.
    pub fn tps(m: *const Model) f64 {
        return m.tpsAt(slot(m.pick()) orelse cuts.len - 1);
    }

    /// Seconds from the trigger to the state landed, for `prefix` rows.
    pub fn seconds(m: *const Model, prefix: u64) f64 {
        return @as(f64, @floatFromInt(prefix)) / m.tps() + m.fixed_s.v;
    }

    /// A split at `cut` of `rows` rows as timed (trigger to state landed), this machine having waited `wait_ns` for rows: the
    /// rate it streamed at is the pair's when this machine starved, else this machine's own (the pipeline runs at the slower).
    pub fn observeSplit(m: *Model, cut: u32, rows: u64, ns: u64, wait_ns: u64) void {
        const i = slot(cut) orelse return;
        const total = @as(f64, @floatFromInt(ns)) / 1e9;
        const streaming = total - m.fixed_s.v;
        if (rows == 0 or streaming < total / 2) return; // a split shorter than its own fixed cost says nothing of a rate
        const rate = @as(f64, @floatFromInt(rows)) / streaming;
        const waited = @as(f64, @floatFromInt(wait_ns)) / 1e9;
        if (waited > starved_share * streaming) {
            m.pair_speed.blend(rate * pair_ms[i] / 1000);
        } else m.mac_speed.blend(rate * mac_ms[i] / 1000);
    }
};

const t = std.testing;

test "a split costs its rows at the slower stage of the cut it takes plus its fixed cost, and only a prompt in its window may split" {
    var m: Model = .{ .enabled = true };
    // seeds: cut 14 runs at 1000/0.37 = 2703 a second (this machine 5263), cut 8 at 2632 (the pair 4274)
    try t.expectApproxEqAbs(@as(f64, 2703), m.tpsAt(2), 1);
    try t.expectApproxEqAbs(@as(f64, 2632), m.tpsAt(1), 1);
    try t.expectEqual(@as(u32, 14), m.pick());
    try t.expectApproxEqAbs(@as(f64, 32767) / m.tps() + 1.4, m.seconds(32767), 1e-9);
    try t.expect(m.allows(16384) and m.allows(1048576) and !m.allows(16383) and !m.allows(1048577));
    m.forced_cut = 8;
    try t.expectEqual(@as(u32, 8), m.pick());
    m.enabled = false;
    try t.expect(!m.allows(32768));
}

test "the cut follows the measured speeds: a slow pair moves it down, a slow machine up" {
    var m: Model = .{ .enabled = true };
    m.pair_speed.v = 0.5; // the pair half as fast: cut 14 now 1351 a second, cut 8 2137, cut 2 7.1k but this machine 1754
    try t.expectEqual(@as(u32, 8), m.pick());
    m.pair_speed.v = 1;
    m.mac_speed.v = 2; // this machine twice as fast: cut 8 is the pair's 4274, cut 14 still the pair's 2703, cut 2 3509
    try t.expectEqual(@as(u32, 8), m.pick());
    m.mac_speed.v = 0.3;
    try t.expectEqual(@as(u32, 14), m.pick());
    var s: Model = .{ .enabled = true };
    s.seedPair(3500); // TF_LS_SPARKS_TPS: the pair at cut 14
    try t.expectApproxEqAbs(@as(f64, 3500), s.pairTps(2), 1e-6);
    try t.expectEqual(@as(u32, 14), s.pick());
}

test "a timed split moves the speed of the stage that was the bottleneck, and a split no longer than its fixed cost says nothing" {
    var m: Model = .{ .enabled = true };
    // cut 14, 32767 rows in 9.4 s over the fixed cost, this machine waited 3 s for rows: the pair streamed 3486 a second
    m.observeSplit(14, 32767, 10_800_000_000, 3_000_000_000);
    try t.expectApproxEqAbs(@as(f64, 32767) / 9.4 * 0.370 / 1000, m.pair_speed.v, 1e-6);
    try t.expectEqual(@as(f64, 1), m.mac_speed.v);
    // cut 8, this machine never waited: its own 2000 a second
    m.observeSplit(8, 20000, 11_400_000_000, 100_000_000);
    try t.expectApproxEqAbs(@as(f64, 20000) / 10.0 * 0.380 / 1000, m.mac_speed.v, 1e-6);
    const before = m.pair_speed.v;
    m.observeSplit(14, 100, 1_500_000_000, 0); // 0.1 s over the fixed cost
    m.observeSplit(14, 0, 5_000_000_000, 0);
    m.observeSplit(11, 20000, 11_400_000_000, 0); // not a cut this machine takes
    try t.expectEqual(before, m.pair_speed.v);
}
