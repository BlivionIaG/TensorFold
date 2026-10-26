//! Whether a shared round drafts at all: drafted and plain rounds' measured tokens a ms, by stream count.
const std = @import("std");
const Allocator = std.mem.Allocator;
const depth = @import("depth.zig");

pub const Constants = struct {
    explore: [2]u32 = .{ 16, 4 }, // measured rounds plain and drafted rounds run when their rate is (re)taken
    exploit: u32 = 512, // measured rounds the faster mode runs before the slower one is measured again
    longest: u5 = 6, // each time the same mode wins again its run doubles, up to exploit << longest
    margin: f64 = 1.05, // drafted rounds must beat plain ones' tokens a ms by this much
    blend: f64 = 0.5, // the newest stretch's weight in a mode's rate
};

/// One stream count's state: each mode's rate (0 plain, 1 drafted), the mode running and its stretch.
const Entry = struct {
    streams: u32,
    rate: [2]?f64 = .{ null, null },
    mode: u1 = 0, // plain first: drafted rounds that lose cost more than plain ones, so they are measured short
    exploring: bool = true,
    left: u32,
    tokens: f64 = 0,
    ms: f64 = 0,
    skip: bool = true, // a mode's first round still carries the other mode's drafts or holds: not measured
    best: ?u1 = null, // the mode the last measurement chose
    streak: u5 = 0, // how many times in a row it chose that mode
};

pub const Guard = struct {
    gpa: Allocator,
    k: Constants = .{},
    list: std.ArrayList(Entry) = .empty,

    pub fn deinit(g: *Guard) void {
        g.list.deinit(g.gpa);
    }

    fn entry(g: *Guard, streams: usize) !*Entry {
        const n: u32 = @intCast(streams);
        for (g.list.items) |*e| if (e.streams == n) return e;
        try g.list.append(g.gpa, .{ .streams = n, .left = g.k.explore[0] });
        return &g.list.items[g.list.items.len - 1];
    }

    /// Whether a shared round of `streams` streams may draft (false: every stream runs its row alone).
    pub fn drafts(g: *Guard, streams: usize) !bool {
        return (try g.entry(streams)).mode == 1;
    }

    /// drafts(); a drafted stretch's first round grants every stream one draft, so it measures drafted rounds.
    pub fn draftsFresh(g: *Guard, streams: usize, whos: []depth.Who) !bool {
        const e = try g.entry(streams);
        if (e.mode == 1 and e.skip) for (whos) |*w| {
            w.granted = 1;
            w.grant_short = false;
            w.gain = null; // priced by its first draft's rate, not a grant of none
        };
        return e.mode == 1;
    }

    /// A finished shared round: its committed tokens and wall ms, in the mode drafts() gave it.
    pub fn observe(g: *Guard, streams: usize, tokens: u64, ms: f64) !void {
        const e = try g.entry(streams);
        if (e.skip) {
            e.skip = false;
            return;
        }
        e.tokens += @floatFromInt(tokens);
        e.ms += ms;
        e.left -|= 1;
        if (e.left > 0) return;
        if (e.ms > 0) {
            const now = e.tokens / e.ms;
            e.rate[e.mode] = if (e.rate[e.mode]) |r| r + g.k.blend * (now - r) else now;
        }
        e.tokens = 0;
        e.ms = 0;
        const before = e.mode;
        if (e.exploring and e.rate[e.mode ^ 1] == null) {
            e.mode ^= 1; // the other mode has no rate yet: measure it next
        } else if (e.exploring) {
            e.mode = if (e.rate[1].? >= e.rate[0].? * g.k.margin) 1 else 0;
            e.streak = if (e.best == e.mode) @min(e.streak + 1, g.k.longest) else 0;
            e.best = e.mode;
            e.exploring = false;
        } else {
            e.mode ^= 1; // the slower mode is measured again, in case the streams' text changed
            e.exploring = true;
        }
        e.left = if (e.exploring) g.k.explore[e.mode] else g.k.exploit << e.streak;
        e.skip = e.mode != before;
    }
};

/// The tokens a round's outcomes committed (each outcome's `got`).
pub fn committed(outcomes: anytype) u64 {
    var n: u64 = 0;
    for (outcomes) |o| n += o.got.len;
    return n;
}

/// Rounds of a guard at `streams` with rates `plain` and `drafted` tokens a ms: how many it drafted.
fn run(g: *Guard, streams: usize, rounds: usize, plain: f64, drafted: f64) !usize {
    var n: usize = 0;
    for (0..rounds) |_| {
        const d = try g.drafts(streams);
        n += @intFromBool(d);
        try g.observe(streams, if (d) 3 else 1, if (d) 3 / drafted else 1 / plain);
    }
    return n;
}

test "a shared round drafts only while drafted rounds beat plain ones, measuring the slower mode again now and then" {
    var g: Guard = .{ .gpa = std.testing.allocator };
    defer g.deinit();
    // drafted rounds slower at four streams: after one stretch of each, plain but for a re-measure a period
    const slow = try run(&g, 4, 1000, 0.25, 0.20);
    try std.testing.expect(slow <= 4 * (g.k.explore[1] + 1)); // the first measure and three re-measures, short ones
    try std.testing.expect(slow > 0);
    // the same verdict again and again: the slower mode is measured ever more rarely, under 1% of rounds
    const later = try run(&g, 4, 20000, 0.25, 0.20);
    try std.testing.expect(later * 500 < 20000);
    // faster at one stream: drafted but for the plain re-measures
    const fast = try run(&g, 1, 1000, 0.13, 0.15);
    try std.testing.expect(fast > 800);
    // each stream count keeps its own rates
    try std.testing.expect(!try g.drafts(4) or (try g.entry(4)).exploring);
    try std.testing.expect(try g.drafts(1) or (try g.entry(1)).exploring);
}

test "a mode's first round is not measured: it carries the other mode's drafts" {
    var g: Guard = .{ .gpa = std.testing.allocator, .k = .{ .explore = .{ 2, 2 }, .exploit = 4 } };
    defer g.deinit();
    try std.testing.expect(!try g.drafts(2)); // plain first
    try g.observe(2, 0, 1); // skipped: the stream count's first round
    try g.observe(2, 1, 4);
    try g.observe(2, 1, 4);
    try std.testing.expect(try g.drafts(2)); // plain measured at 0.25 a ms; drafted next
    try g.observe(2, 100, 1); // skipped: plain's last held rows, a fantasy rate that would keep drafting
    try g.observe(2, 1, 10);
    try g.observe(2, 1, 10);
    try std.testing.expect(!try g.drafts(2)); // plain at 0.25 beats drafted at 0.1
    try std.testing.expectEqual(@as(?f64, 0.1), (try g.entry(2)).rate[1]);
}

test "a drafted stretch starts from a grant of one: a plain stretch's grants of none would hold nothing to measure" {
    var g: Guard = .{ .gpa = std.testing.allocator, .k = .{ .explore = .{ 2, 2 }, .exploit = 4 } };
    defer g.deinit();
    var st: ?depth.State = null;
    var whos = [_]depth.Who{.{ .state = &st, .draft_room = 100, .finished = false, .forced = false, .granted = 0 }};
    try std.testing.expect(!try g.draftsFresh(2, &whos)); // plain first: grants as they are
    try std.testing.expectEqual(@as(?u32, 0), whos[0].granted);
    for (0..3) |_| try g.observe(2, 1, 4);
    try std.testing.expect(try g.draftsFresh(2, &whos)); // the drafted stretch's first round
    try std.testing.expectEqual(@as(?u32, 1), whos[0].granted);
    try g.observe(2, 1, 4); // its first round, skipped
    whos[0].granted = 0;
    try std.testing.expect(try g.draftsFresh(2, &whos)); // later rounds keep the allocation's grants
    try std.testing.expectEqual(@as(?u32, 0), whos[0].granted);
}

test "a guard allocates once a stream count: its rounds after that take no allocation" {
    var counted = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var g: Guard = .{ .gpa = counted.allocator(), .k = .{ .explore = .{ 2, 2 }, .exploit = 4 } };
    defer g.deinit();
    for ([_]usize{ 1, 4, 8, 16 }) |streams| _ = try run(&g, streams, 1, 0.25, 0.2);
    const first = .{ counted.allocations, counted.resize_index };
    var st: [16]?depth.State = @splat(null);
    var whos: [16]depth.Who = undefined;
    for (&whos, &st) |*w, *s| w.* = .{ .state = s, .draft_room = 100, .finished = false, .forced = false, .granted = 1 };
    for (0..200) |i| for ([_]usize{ 1, 4, 8, 16 }) |streams| {
        _ = try g.draftsFresh(streams, whos[0..streams]);
        try g.observe(streams, i % 3 + 1, 4);
    };
    try std.testing.expectEqual(first[0], counted.allocations);
    try std.testing.expectEqual(first[1], counted.resize_index);
}
