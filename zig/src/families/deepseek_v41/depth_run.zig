//! The acceptance depth controller (counts only, never a clock; after omlx's AcceptanceDepthController): a stream's
//! per-position acceptance is a decayed hit count, and the depth maximising tokens a millisecond under a fixed cost table wins.
const std = @import("std");

pub const max_depth = 3; // a window of five rows is slow and not bit-exact on this engine (m5-verify.md)

/// A round in ms by drafts held (plain, 1, 2, 3): the verify window, the drafter's forward and the host's 0.5 ms (M5 Ultra).
pub const round_ms = [max_depth + 1]f64{ 19.35, 29.9, 39.7, 44.3 };

/// A drafted depth must beat plain's tokens a ms by this much to stay on.
pub const margin = 1.05;
/// Rounds a request must run before the controller may switch drafting off.
pub const min_rounds = 4;
const decay = 0.9;
const prior_trials = 2.0;
const prior_rate = 0.7;

pub const State = struct {
    hits: [max_depth]f64 = @splat(0),
    trials: [max_depth]f64 = @splat(0),
    rounds: u32 = 0,
    /// The drafts the next window carries (0: drafting no longer pays).
    depth: u32 = max_depth,

    /// A round that held `used` drafts and kept `kept` rows (the pending row included).
    pub fn observe(s: *State, used: u32, kept: u32, cap: u32) void {
        const u = @min(used, max_depth);
        const accepted = @min(kept -| 1, u);
        for (0..max_depth) |j| {
            s.hits[j] *= decay;
            s.trials[j] *= decay;
            // Draft j was tried only when drafts before it were accepted; a rejected one ends the chain.
            if (j < u and j <= accepted) {
                s.trials[j] += 1;
                if (j < accepted) s.hits[j] += 1;
            }
        }
        s.rounds += 1;
        s.depth = s.best(cap);
    }

    fn rate(s: *const State, j: usize) f64 {
        return (s.hits[j] + prior_trials * prior_rate) / (s.trials[j] + prior_trials);
    }

    /// The depth with the most expected tokens a ms (0 when none beats plain by `margin`).
    pub fn best(s: *const State, cap: u32) u32 {
        var pick: u32 = 0;
        var top: f64 = margin / round_ms[0];
        var expected: f64 = 1;
        var chain: f64 = 1;
        for (1..@as(usize, @min(cap, max_depth)) + 1) |d| {
            chain *= s.rate(d - 1);
            expected += chain;
            const per_ms = expected / round_ms[d];
            if (per_ms > top) {
                top = per_ms;
                pick = @intCast(d);
            }
        }
        return pick;
    }

    /// Drafting stays off for good once a request has run `min_rounds` and no depth beats plain.
    pub fn off(s: *const State) bool {
        return s.rounds >= min_rounds and s.depth == 0;
    }
};

test "a request that keeps its drafts holds three, one that rarely does stops drafting" {
    var code: State = .{};
    for (0..8) |_| code.observe(3, 4, 3); // every draft accepted
    try std.testing.expectEqual(@as(u32, 3), code.depth);
    try std.testing.expect(!code.off());
    var chat: State = .{};
    for (0..6) |_| chat.observe(1, 1, 3); // the first draft never lands
    try std.testing.expectEqual(@as(u32, 0), chat.depth);
    try std.testing.expect(chat.off());
}

test "the depth follows the accepted run and the cost table" {
    var s: State = .{};
    for (0..12) |i| s.observe(3, if (i % 2 == 0) 2 else 1, 3); // the first draft lands half the time
    try std.testing.expect(s.depth <= 1);
    var young: State = .{};
    young.observe(1, 1, 3);
    try std.testing.expect(!young.off()); // too few rounds to cut
}
