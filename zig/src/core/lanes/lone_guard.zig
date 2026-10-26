//! Whether a lone stream holds drafts: its drafted rounds' tokens a ms against plain rounds', the loser re-measured.
const std = @import("std");

pub const Constants = struct {
    drafted: u32 = 16, // measured drafted rounds a verdict takes, again and again while drafting wins
    plain: u32 = 16, // measured plain rounds that take the plain rate when there is none yet
    wait: u32 = 16, // plain rounds before drafting is measured again; doubles each time plain wins again
    wait_most: u32 = 512,
    margin: f64 = 1.05, // drafted rounds must beat plain ones' tokens a ms by this much
    probe_cost: f64 = 0.03, // after two losses a probe costs at most this share of plain time
    confirm: u32 = 2, // losing verdicts in a row (a remembered one counts) before the wait follows probe_cost
    blend: f64 = 0.5, // the newest stretch's weight in a rate
};

pub const Mode = enum { drafted, plain };

/// Plain rounds' tokens a ms by sampling mode and context band, kept from stream to stream (no draft changes them).
pub const Plain = struct {
    rates: [2][bands]?f64 = @splat(@splat(null)),
    verdicts: [2][bands]?Verdict = @splat(@splat(null)), // each band's last losing verdict, until drafting wins there

    pub const Verdict = struct { drafted: f64, wait: u32 };

    pub fn memo(p: *Plain, sampled: bool, band: usize) *?Verdict {
        return &p.verdicts[@intFromBool(sampled)][@min(band, bands - 1)];
    }

    pub const bands = 6;

    pub fn at(p: *Plain, sampled: bool, band: usize) *?f64 {
        return &p.rates[@intFromBool(sampled)][@min(band, bands - 1)];
    }
};

/// One lone stream's guard: the mode its holds follow, the stretch measured and its drafted rate.
pub const Lone = struct {
    mode: Mode = .drafted,
    left: u32 = 0,
    tokens: f64 = 0,
    ms: f64 = 0,
    skip: bool = true, // a mode's first round still verifies the other mode's hold (or none): not measured
    started: bool = false,
    pending: bool = false, // a drafted verdict waits for the plain rate this stretch takes
    fresh: bool = false, // a drafted stretch's first hold: one draft and its slack, whatever the last grant
    drafted: ?f64 = null,
    drafted_ms: f64 = 0, // the last drafted stretch's wall ms
    drafted_now: f64 = 0, // the last drafted stretch's own tokens a ms (unblended)
    losses: u32 = 0, // losing verdicts in a row
    wait: u32 = 0,

    /// Whether the next hold may draft.
    pub fn drafts(l: *const Lone) bool {
        return l.mode == .drafted;
    }

    /// A lone round of held drafts or none (copies left out): its tokens and wall ms; `plain` the band's rate.
    pub fn observe(l: *Lone, k: Constants, plain: *?f64, tokens: usize, ms: f64) void {
        l.observeBand(k, plain, null, tokens, ms);
    }

    /// observe, with the band's last verdict: a stream starting where drafting lost starts plain for its wait.
    pub fn observeBand(l: *Lone, k: Constants, plain: *?f64, memo: ?*?Plain.Verdict, tokens: usize, ms: f64) void {
        var switched = false;
        if (!l.started) {
            l.started = true;
            l.left = k.drafted;
            l.wait = k.wait;
            if (memo) |m| if (m.*) |v| if (plain.*) |p| if (v.drafted < p * k.margin) {
                l.drafted = v.drafted;
                l.losses = k.confirm - 1; // one more losing probe confirms the band's loss again
                l.switchTo(.plain, v.wait);
                l.wait = @min(v.wait *| 2, k.wait_most);
                switched = true;
            };
        }
        if (l.skip) {
            l.skip = switched; // the next round still verifies this round's drafted hold
            return;
        }
        l.tokens += @floatFromInt(tokens);
        l.ms += ms;
        l.left -|= 1;
        if (l.left > 0) return;
        const now = if (l.ms > 0) l.tokens / l.ms else 0;
        const stretch_ms = l.ms;
        l.tokens = 0;
        l.ms = 0;
        switch (l.mode) {
            .drafted => {
                l.drafted = blend(k, l.drafted, now);
                l.drafted_ms = stretch_ms;
                l.drafted_now = now;
                if (plain.* == null) {
                    l.pending = true;
                    l.switchTo(.plain, k.plain);
                } else l.verdict(k, plain.*.?, memo);
            },
            .plain => {
                plain.* = blend(k, plain.*, now);
                if (l.pending) {
                    l.pending = false;
                    l.verdict(k, plain.*.?, memo);
                } else l.switchTo(.drafted, k.drafted);
            },
        }
    }

    /// Drafting goes on while it beats plain by the margin, else plain rounds for a wait that doubles.
    fn verdict(l: *Lone, k: Constants, plain: f64, memo: ?*?Plain.Verdict) void {
        if (l.drafted.? >= plain * k.margin) {
            l.wait = k.wait;
            l.losses = 0;
            l.switchTo(.drafted, k.drafted);
            if (memo) |m| m.* = null;
            return;
        }
        var wait = l.wait;
        if (l.drafted_now >= plain * k.margin) {
            // the blend lost but this stretch won: no loss counted, the band forgets its loss
            l.losses = 0;
            if (memo) |m| m.* = null;
        } else {
            l.losses += 1;
            if (l.losses >= k.confirm) {
                const lost_ms = l.drafted_ms * @max(0, 1 - l.drafted_now / plain);
                const need: f64 = @ceil(lost_ms * plain / k.probe_cost); // plain rounds of about one token each
                wait = @min(k.wait_most, @max(wait, @as(u32, @intFromFloat(@min(need, 1e6)))));
            }
            if (memo) |m| m.* = .{ .drafted = l.drafted.?, .wait = wait };
        }
        l.switchTo(.plain, wait);
        l.wait = @min(wait *| 2, k.wait_most);
    }

    fn switchTo(l: *Lone, mode: Mode, rounds: u32) void {
        if (mode != l.mode) {
            l.skip = true;
            l.fresh = mode == .drafted;
        }
        l.mode = mode;
        l.left = @max(1, rounds);
    }
};

fn blend(k: Constants, old: ?f64, now: f64) f64 {
    return if (old) |r| r + k.blend * (now - r) else now;
}

/// Rounds of one stream whose drafted rounds give `drafted` and plain ones `plain` tokens a ms: how many drafted.
fn run(l: *Lone, k: Constants, rate: *?f64, rounds: usize, plain: f64, drafted: f64) usize {
    var n: usize = 0;
    for (0..rounds) |_| {
        const d = l.drafts();
        n += @intFromBool(d);
        l.observe(k, rate, if (d) 2 else 1, if (d) 2 / drafted else 1 / plain);
    }
    return n;
}

test "a lone stream whose drafted rounds lose runs plain but for probes that grow rarer" {
    var l: Lone = .{};
    var rate: ?f64 = null;
    const k: Constants = .{};
    const drafted = run(&l, k, &rate, 1000, 0.128, 0.122);
    // the first drafted stretch, the plain rate's own, then probes after 16, 32, 64, 128, 256 and 512 plain rounds
    try std.testing.expect(drafted <= 7 * (k.drafted + 1));
    try std.testing.expect(drafted > k.drafted);
    try std.testing.expect(!l.drafts() or l.left <= k.drafted);
    try std.testing.expectApproxEqAbs(@as(f64, 0.128), rate.?, 1e-9);
}

test "a lone stream whose drafted rounds win keeps drafting; a later stream reuses the plain rate" {
    var rate: ?f64 = null;
    const k: Constants = .{};
    var first: Lone = .{};
    const n = run(&first, k, &rate, 1000, 0.128, 0.15);
    try std.testing.expect(n > 1000 - k.plain - 2); // one plain stretch takes the plain rate
    var second: Lone = .{};
    try std.testing.expectEqual(@as(usize, 300), run(&second, k, &rate, 300, 0.128, 0.15)); // no plain stretch again
    var third: Lone = .{};
    const lost = run(&third, k, &rate, 300, 0.128, 0.11);
    try std.testing.expect(lost < 300 / 3); // plain from its first verdict, without measuring plain again
}

test "a drafted stretch's first round is not measured and its first hold is fresh" {
    var l: Lone = .{};
    var rate: ?f64 = 0.2;
    const k: Constants = .{ .drafted = 2, .wait = 2 };
    l.observe(k, &rate, 9, 1); // skipped: the stream's first round
    l.observe(k, &rate, 1, 10);
    l.observe(k, &rate, 1, 10);
    try std.testing.expect(!l.drafts()); // 0.1 against 0.2
    try std.testing.expect(!l.fresh);
    l.observe(k, &rate, 9, 1); // skipped: the last drafted round's hold
    l.observe(k, &rate, 1, 5);
    l.observe(k, &rate, 1, 5);
    try std.testing.expect(l.drafts() and l.fresh); // the probe after the wait
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), rate.?, 1e-12);
}

test "a probe that loses a third waits long enough that probes cost at most probe_cost; a band's verdict carries over" {
    const k: Constants = .{};
    var rate: ?f64 = null;
    var memo: ?Plain.Verdict = null;
    var l: Lone = .{};
    // 130k on the M5: plain 8.67 ms a token, drafted 15.7 ms for 1.2 tokens
    var drafted: usize = 0;
    for (0..200) |_| {
        const d = l.drafts();
        drafted += @intFromBool(d);
        l.observeBand(k, &rate, &memo, if (d) 12 else 10, if (d) 157 else 86.7);
    }
    try std.testing.expect(drafted <= 2 * (1 + k.drafted)); // two losing stretches, then the probe_cost wait
    try std.testing.expect(memo != null and memo.?.wait > 200);
    var next: Lone = .{};
    var later: usize = 0;
    for (0..200) |_| {
        const d = next.drafts();
        later += @intFromBool(d);
        next.observeBand(k, &rate, &memo, if (d) 12 else 10, if (d) 157 else 86.7);
    }
    try std.testing.expect(later <= 1);
}

test "a winning stretch the blend still calls a loss counts no loss: no probe_cost wait" {
    const k: Constants = .{ .drafted = 2, .wait = 2 };
    var rate: ?f64 = 0.2;
    var memo: ?Plain.Verdict = null;
    var l: Lone = .{};
    l.observeBand(k, &rate, &memo, 9, 1); // skipped
    l.observeBand(k, &rate, &memo, 1, 10);
    l.observeBand(k, &rate, &memo, 1, 10); // 0.1: a loss, wait 2
    try std.testing.expect(!l.drafts() and l.losses == 1 and memo != null);
    l.observeBand(k, &rate, &memo, 9, 1); // skipped
    l.observeBand(k, &rate, &memo, 1, 5);
    l.observeBand(k, &rate, &memo, 1, 5);
    try std.testing.expect(l.drafts());
    l.observeBand(k, &rate, &memo, 9, 1); // skipped
    l.observeBand(k, &rate, &memo, 2, 8);
    l.observeBand(k, &rate, &memo, 2, 8); // 0.25 alone wins; blended 0.175 loses
    try std.testing.expect(!l.drafts() and l.losses == 0 and memo == null and l.left == 4);
}
