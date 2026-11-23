//! The trained drafter's depth from the round's own measurements: expected tokens a round over the round's ms.
const std = @import("std");

pub const max_depth = 4; // the proposer's cap (proposeTokens returns four ids)
pub const max_rows = max_depth + 1;

/// Where the round's drafts came from: the trained head's chain, or the stream's own suffix match.
pub const Source = enum(u1) { trained, lookup };

pub const sources = std.enums.values(Source);

/// A mean over the first eight samples, then an EWMA: a new workload counts in full, a drifted one within a few rounds.
pub const Mean = struct {
    v: f64 = 0,
    n: u32 = 0,

    pub fn add(m: *Mean, x: f64) void {
        m.n += 1;
        const n: f64 = @floatFromInt(m.n);
        m.v = if (m.n <= 8) (m.v * (n - 1.0) + x) / n else m.v * 0.85 + x * 0.15;
    }
};

/// The round's own ms: each window's rows, and the drafter's pass (the lookup's costs nothing).
const Rows = struct {
    window: [max_rows + 1]Mean = @splat(.{}), // by the window's rows (1 .. max_rows)
    draft: Mean = .{},

    /// A guessed window of `rows` rows while the row count is unmeasured: one row's measured ms a row.
    fn windowMs(r: *const Rows, rows: usize) f64 {
        if (r.window[rows].n != 0) return r.window[rows].v;
        const one = r.window[1].v;
        const step = if (r.window[2].n != 0) @max(r.window[2].v - one, 0.2 * one) else 0.5 * one;
        return one + step * @as(f64, @floatFromInt(rows - 1));
    }

    /// The ms of a round verifying `k` trained drafts: its k + 1 rows and the drafter's pass.
    pub fn roundMs(r: *const Rows, k: usize, src: Source) f64 {
        const paid = if (k == 0 or src == .lookup) 0.0 else if (r.draft.n != 0) r.draft.v else 0.3 * r.window[1].v;
        return r.windowMs(k + 1) + paid;
    }
};

/// Tokens a round verifying `k` drafts commits on average, from the per-draft chances `q` (len >= k).
pub fn expectedTokens(q: []const f64, k: usize) f64 {
    var run: f64 = 1.0;
    var e: f64 = 1.0;
    for (q[0..@min(k, q.len)]) |c| {
        run *= c;
        e += run;
    }
    return e;
}

pub const hysteresis = 0.99; // a switch needs the new depth 1 % better: a round's ms is noisy
pub const probe_rounds = 64; // a plain stream re-checks the deeper window once this often
pub const explore_rounds = 12; // rounds that measure an unmeasured depth before the best one is taken

/// The depth of the next window, the measurements behind it and the choice it made last.
pub const Policy = struct {
    rows: Rows = .{},
    q: [sources.len][max_depth]Mean = @splat(@splat(Mean{})),
    depth: [sources.len]usize = @splat(0),
    quiet: [sources.len]u32 = @splat(0),
    cap: usize = 3,
    rounds: u32 = 0,

    /// A new request: the workload's chances and the choice start over, the machine's row costs stay.
    pub fn reset(p: *Policy) void {
        p.q = @splat(@splat(Mean{}));
        p.depth = @splat(0);
        p.quiet = @splat(0);
        p.rounds = 0;
    }

    /// A solo round's window of `rows` rows (a trained round's drafter pass is not part of it).
    pub fn noteRound(p: *Policy, rows: u32, ms: f64) void {
        p.rounds += 1;
        if (rows < 1 or rows > max_rows or !std.math.isFinite(ms) or ms <= 0) return;
        p.rows.window[rows].add(ms);
    }

    pub fn noteDraft(p: *Policy, ms: f64) void {
        if (!std.math.isFinite(ms) or ms < 0) return;
        p.rows.draft.add(ms);
    }

    /// The drafts a window offered and the ones that landed (a chain's kept rows but its pending token).
    pub fn noteLanded(p: *Policy, src: Source, landed: usize, offered: usize) void {
        for (0..@min(offered, max_depth)) |j| p.q[@backingInt(src)][j].add(@floatFromInt(@intFromBool(j < landed)));
    }

    /// The per-draft chances, shrunk toward the prior until the samples speak: one draft is not a trend.
    fn chances(p: *const Policy, src: Source, out: *[max_depth]f64) void {
        const prior = 0.75;
        const weight = 3.0;
        for (out, p.q[@backingInt(src)]) |*c, m| {
            const n: f64 = @floatFromInt(m.n);
            c.* = (m.v * n + prior * weight) / (n + weight);
        }
    }

    /// The tokens a round of depth `k` buys for its ms; 0 is the plain one-row round.
    pub fn rate(p: *const Policy, src: Source, k: usize) f64 {
        if (p.rows.window[1].n == 0) return 0;
        var q: [max_depth]f64 = undefined;
        p.chances(src, &q);
        return expectedTokens(&q, k) / p.rows.roundMs(k, src);
    }

    /// The next depth: 0 until a plain round is priced, then the shallow one and each deeper one while it pays.
    pub fn pick(p: *Policy, src: Source) usize {
        const s = @backingInt(src);
        const cap = @min(p.cap, max_depth);
        if (cap == 0 or p.rows.window[1].n == 0) return 0;
        if (p.rows.window[2].n == 0) return 1;
        if (p.rate(src, 1) <= p.rate(src, 0)) return 0;
        if (p.rounds < explore_rounds) for (2..cap + 1) |k| {
            if (p.rows.window[k + 1].n == 0) return k;
        };
        var best: usize = 0;
        for (1..cap + 1) |k| if (p.rate(src, k) > p.rate(src, best)) {
            best = k;
        };
        if (best == 0 and p.quiet[s] >= probe_rounds) return cap;
        if (p.depth[s] <= cap and p.rate(src, best) < p.rate(src, p.depth[s]) / hysteresis) return p.depth[s];
        return best;
    }

    /// The depth the round that is starting has: the choice prices the next one against it.
    pub fn noteDepth(p: *Policy, src: Source, depth: usize) void {
        const s = @backingInt(src);
        p.depth[s] = depth;
        if (depth == 0) p.quiet[s] += 1 else p.quiet[s] = 0;
    }
};

test "the plain round is measured before any depth is taken" {
    var p: Policy = .{};
    try std.testing.expectEqual(@as(usize, 0), p.pick(.trained));
    p.noteDepth(.trained, 0);
    p.noteRound(1, 19.0);
    try std.testing.expectEqual(@as(usize, 1), p.pick(.trained));
}

test "each unmeasured depth runs once before the best one is taken" {
    var p: Policy = .{};
    p.noteRound(1, 19.0);
    p.noteDraft(4.0);
    try std.testing.expectEqual(@as(usize, 1), p.pick(.trained));
    p.noteRound(2, 27.0);
    try std.testing.expectEqual(@as(usize, 2), p.pick(.trained));
    p.noteRound(3, 37.0);
    try std.testing.expectEqual(@as(usize, 3), p.pick(.trained));
    p.noteRound(4, 41.0);
    // 19 / 31 / 41 / 45 ms a round at 0.8 a draft: the four-row window commits the most tokens a ms.
    for (&p.q[@backingInt(Source.trained)]) |*m| m.* = .{ .v = 0.8, .n = 4 };
    try std.testing.expectEqual(@as(usize, 3), p.pick(.trained));
}

test "a weak first draft keeps the plain round: prose is not slowed" {
    var p: Policy = .{};
    p.noteRound(1, 19.0);
    p.noteRound(2, 27.0);
    p.noteRound(3, 37.0);
    p.noteRound(4, 41.0);
    p.noteDraft(4.0);
    for (&p.q[@backingInt(Source.trained)]) |*m| m.* = .{ .v = 0.25, .n = 6 };
    p.rounds = explore_rounds;
    try std.testing.expectEqual(@as(usize, 0), p.pick(.trained));
}

test "a row-count cliff leaves depth two behind the shallower window" {
    var p: Policy = .{};
    p.cap = 2;
    p.noteRound(1, 19.0);
    p.noteRound(2, 27.0);
    p.noteRound(3, 37.0); // a three-row round costs most of what a four-row one would
    p.noteRound(4, 41.0);
    p.noteDraft(4.0);
    for (&p.q[@backingInt(Source.trained)]) |*m| m.* = .{ .v = 0.7, .n = 8 };
    p.rounds = explore_rounds;
    // One more drafted token in the same row would have to pay ten ms: 1.7 tokens in 31 ms beat 2.19 in 41.
    try std.testing.expect(p.rate(.trained, 2) < p.rate(.trained, 1));
}

test "a lookup match is priced without the drafter's pass" {
    var p: Policy = .{};
    p.noteRound(1, 19.0);
    p.noteRound(2, 27.0);
    p.noteRound(3, 37.0);
    p.noteRound(4, 41.0);
    p.noteDraft(4.0);
    for (&p.q[@backingInt(Source.lookup)]) |*m| m.* = .{ .v = 0.9, .n = 8 };
    p.rounds = explore_rounds;
    try std.testing.expectEqual(@as(usize, 3), p.pick(.lookup));
    // the same chances through the trained head also pay the pass, so its deeper round is the dearer one
    for (&p.q[@backingInt(Source.trained)]) |*m| m.* = .{ .v = 0.9, .n = 8 };
    try std.testing.expect(p.rate(.trained, 3) < p.rate(.lookup, 3));
}

test "a shallow window that does not pay is not deepened" {
    var p: Policy = .{};
    p.cap = 3;
    p.noteRound(1, 19.0);
    p.noteRound(2, 27.0);
    for (&p.q[@backingInt(Source.trained)]) |*m| m.* = .{ .v = 0.0, .n = 12 };
    // A draft that never lands leaves 1.15 tokens in 31 ms against the plain round's 1 in 19: no third row is spent.
    try std.testing.expectEqual(@as(usize, 0), p.pick(.trained));
}

test "the choice holds until another depth is clearly better" {
    var p: Policy = .{};
    p.noteRound(1, 19.0);
    p.noteRound(2, 27.0);
    p.noteRound(3, 37.0);
    p.noteRound(4, 41.0);
    p.noteDraft(4.0);
    for (&p.q[@backingInt(Source.trained)]) |*m| m.* = .{ .v = 0.8, .n = 8 };
    p.rounds = explore_rounds;
    p.noteDepth(.trained, 1);
    const k = p.pick(.trained);
    try std.testing.expectEqual(@as(usize, 3), k);
    p.noteDepth(.trained, k);
    try std.testing.expectEqual(k, p.pick(.trained));
}

test "a depth a hair ahead follows the best past the exploration's depth" {
    var p: Policy = .{};
    p.noteRound(1, 19.0);
    p.noteRound(2, 27.0);
    p.noteRound(3, 37.0);
    p.noteRound(4, 43.0);
    p.noteDraft(4.0);
    for (&p.q[@backingInt(Source.trained)]) |*m| m.* = .{ .v = 0.8, .n = 8 };
    p.rounds = explore_rounds;
    // The exploration measured depth two last: three rows buy 4.9 % more tokens a ms, inside a 5 % band.
    p.noteDepth(.trained, 2);
    try std.testing.expectEqual(@as(usize, 3), p.pick(.trained));
}

test "a landed prefix updates each position it saw" {
    var p: Policy = .{};
    p.noteLanded(.trained, 2, 3);
    try std.testing.expectEqual(@as(f64, 1.0), p.q[0][0].v);
    try std.testing.expectEqual(@as(f64, 1.0), p.q[0][1].v);
    try std.testing.expectEqual(@as(f64, 0.0), p.q[0][2].v);
    p.noteLanded(.trained, 0, 3);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), p.q[0][0].v, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), p.q[0][1].v, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), p.q[0][2].v, 1e-9);
    try std.testing.expectEqual(@as(usize, 2), p.q[0][0].n);
}
