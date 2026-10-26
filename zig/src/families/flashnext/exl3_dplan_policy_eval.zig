//! CPU counterfactuals for one ordered pool-demand timeline.

const std = @import("std");
const trace = @import("exl3_dplan_trace.zig");

pub const expert_count: usize = 512;
pub const max_unique: usize = 256;

pub const Policy = enum { lifetime, lru, decayed_frequency, decayed_admission };

pub const Seed = struct {
    line_ids: []const i32,
    stamps: []const u32,
    counts: [expert_count]u32,
    now: u32,
};

pub const Options = struct {
    capacity: usize,
    seed: ?Seed = null,
    decay_interval_rounds: u64 = 64,
    decay_shift: u5 = 1,
};

pub const Metrics = struct {
    rounds: u64 = 0,
    hit_picks: u64 = 0,
    miss_picks: u64 = 0,
    hit_experts: u64 = 0,
    miss_experts: u64 = 0,
    admissions: u64 = 0,
    copies: u64 = 0,
    evictions: u64 = 0,
    mapped_experts: u64 = 0,
    mapped_picks: u64 = 0,

    fn plus(self: Metrics, other: Metrics) !Metrics {
        var result: Metrics = undefined;
        inline for (@typeInfo(Metrics).@"struct".field_names) |field| {
            @field(result, field) = try std.math.add(u64, @field(self, field), @field(other, field));
        }
        return result;
    }
};

pub const Comparison = struct {
    lifetime: Metrics,
    lru: Metrics,
    decayed_frequency: Metrics,
    decayed_admission: Metrics,
    belady_unique_misses_lower_bound: u64,
};

const Demand = struct {
    counts: [expert_count]u32 = @splat(0),
    ordered: [expert_count]usize = undefined,
    unique: usize = 0,
    valid_picks: u64 = 0,
    picks: []const u32,
};

pub const State = struct {
    allocator: std.mem.Allocator,
    policy: Policy,
    options: Options,
    line_ids: []i32,
    stamps: []u32,
    scores: [expert_count]u32,
    last_access: [expert_count]u64,
    access_clock: u64,
    now: u32,
    total: Metrics = .{},

    pub fn init(allocator: std.mem.Allocator, policy: Policy, options: Options) !State {
        try validateOptions(options);
        const line_ids = try allocator.alloc(i32, options.capacity);
        errdefer allocator.free(line_ids);
        const stamps = try allocator.alloc(u32, options.capacity);
        @memset(line_ids, -1);
        @memset(stamps, 0);
        var self: State = .{
            .allocator = allocator,
            .policy = policy,
            .options = options,
            .line_ids = line_ids,
            .stamps = stamps,
            .scores = @splat(0),
            .last_access = @splat(0),
            .access_clock = 0,
            .now = 0,
        };
        if (options.seed) |seed| {
            @memcpy(line_ids, seed.line_ids);
            @memcpy(stamps, seed.stamps);
            self.scores = seed.counts;
            self.now = seed.now;
            self.access_clock = seed.now;
            for (line_ids, stamps) |expert, stamp| {
                self.access_clock = @max(self.access_clock, stamp);
                if (expert >= 0) self.last_access[@intCast(expert)] = stamp;
            }
        }
        self.options.seed = null;
        return self;
    }

    pub fn deinit(self: *State) void {
        self.allocator.free(self.line_ids);
        self.allocator.free(self.stamps);
        self.* = undefined;
    }

    pub fn observe(self: *State, record: trace.Round) !Metrics {
        const demand = try readDemand(record);
        const round_number = try std.math.add(u64, self.total.rounds, 1);
        const next_clock = try std.math.add(u64, self.access_clock, demand.valid_picks);
        var delta: Metrics = .{ .rounds = 1 };
        for (demand.ordered[0..demand.unique]) |expert| {
            if (self.lineOf(expert) != null) {
                delta.hit_picks += demand.counts[expert];
                delta.hit_experts += 1;
            } else {
                delta.miss_picks += demand.counts[expert];
                delta.miss_experts += 1;
            }
        }
        _ = try self.total.plus(delta);
        if ((self.policy == .decayed_frequency or self.policy == .decayed_admission) and round_number % self.options.decay_interval_rounds == 0) {
            for (&self.scores) |*score| score.* >>= self.options.decay_shift;
        }
        self.now +%= 1;
        for (demand.ordered[0..demand.unique]) |expert| {
            self.scores[expert] +|= demand.counts[expert];
            if (self.lineOf(expert)) |line| self.stamps[line] = self.now;
        }
        for (demand.picks) |bits| {
            const expert = routed(bits) orelse continue;
            self.access_clock += 1;
            self.last_access[expert] = self.access_clock;
        }
        std.debug.assert(self.access_clock == next_clock);

        var misses: [max_unique]usize = undefined;
        var miss_count: usize = 0;
        for (demand.ordered[0..demand.unique]) |expert| {
            if (self.lineOf(expert) == null) {
                misses[miss_count] = expert;
                miss_count += 1;
            }
        }
        var index: usize = 1;
        while (index < miss_count) : (index += 1) {
            const expert = misses[index];
            var position = index;
            while (position > 0 and self.preferAdmission(expert, misses[position - 1], &demand)) : (position -= 1) {
                misses[position] = misses[position - 1];
            }
            misses[position] = expert;
        }
        for (misses[0..miss_count]) |expert| {
            if (self.victim(&demand)) |line| {
                if (self.policy == .decayed_admission and self.line_ids[line] >= 0 and self.scores[expert] < self.scores[@intCast(self.line_ids[line])]) {
                    delta.mapped_experts += 1;
                    delta.mapped_picks += demand.counts[expert];
                    continue;
                }
                if (self.line_ids[line] >= 0) delta.evictions += 1;
                self.line_ids[line] = @intCast(expert);
                self.stamps[line] = self.now;
                delta.admissions += 1;
                delta.copies += 1;
            } else {
                delta.mapped_experts += 1;
                delta.mapped_picks += demand.counts[expert];
            }
        }
        self.total = try self.total.plus(delta);
        return delta;
    }

    fn lineOf(self: *const State, expert: usize) ?usize {
        for (self.line_ids, 0..) |resident, line| {
            if (resident == @as(i32, @intCast(expert))) return line;
        }
        return null;
    }

    fn preferAdmission(self: *const State, left: usize, right: usize, demand: *const Demand) bool {
        if (self.policy == .lru) {
            if (self.last_access[left] != self.last_access[right]) return self.last_access[left] > self.last_access[right];
        } else {
            if (self.scores[left] != self.scores[right]) return self.scores[left] > self.scores[right];
            if (demand.counts[left] != demand.counts[right]) return demand.counts[left] > demand.counts[right];
        }
        return left < right;
    }

    fn victim(self: *const State, demand: *const Demand) ?usize {
        var coldest: ?usize = null;
        for (self.line_ids, 0..) |expert, line| {
            if (expert < 0) return line;
            if (demand.counts[@intCast(expert)] != 0) continue;
            if (coldest == null or self.colder(line, coldest.?)) coldest = line;
        }
        return coldest;
    }

    fn colder(self: *const State, left: usize, right: usize) bool {
        const left_expert: usize = @intCast(self.line_ids[left]);
        const right_expert: usize = @intCast(self.line_ids[right]);
        if (self.policy != .lru and self.scores[left_expert] != self.scores[right_expert]) return self.scores[left_expert] < self.scores[right_expert];
        if (self.policy == .lifetime) {
            if (self.stamps[left] != self.stamps[right]) return self.stamps[left] < self.stamps[right];
        } else {
            if (self.last_access[left_expert] != self.last_access[right_expert]) return self.last_access[left_expert] < self.last_access[right_expert];
        }
        return left > right;
    }
};

pub fn evaluate(allocator: std.mem.Allocator, rounds: []const trace.Round, options: Options) !Comparison {
    var lifetime = try State.init(allocator, .lifetime, options);
    defer lifetime.deinit();
    var lru = try State.init(allocator, .lru, options);
    defer lru.deinit();
    var decayed = try State.init(allocator, .decayed_frequency, options);
    defer decayed.deinit();
    var decayed_admission = try State.init(allocator, .decayed_admission, options);
    defer decayed_admission.deinit();
    for (rounds) |record| {
        _ = try lifetime.observe(record);
        _ = try lru.observe(record);
        _ = try decayed.observe(record);
        _ = try decayed_admission.observe(record);
    }
    return .{
        .lifetime = lifetime.total,
        .lru = lru.total,
        .decayed_frequency = decayed.total,
        .decayed_admission = decayed_admission.total,
        .belady_unique_misses_lower_bound = try beladyLowerBound(allocator, rounds, options),
    };
}

pub fn evaluateTrace(allocator: std.mem.Allocator, view: trace.View) !Comparison {
    const verification = try @import("exl3_dplan_replay.zig").verifyTrace(allocator, view);
    const rounds = try allocator.alloc(trace.Round, view.recorded());
    defer allocator.free(rounds);
    for (rounds, 0..) |*round, index| round.* = try view.round(index);
    var options: Options = .{ .capacity = view.words[4], .seed = view.seedState(Seed) };
    var total: Comparison = .{ .lifetime = .{}, .lru = .{}, .decayed_frequency = .{}, .decayed_admission = .{}, .belady_unique_misses_lower_bound = 0 };
    var begin: usize = 0;
    for (0..view.checkpoints() + 1) |index| {
        const checkpoint = if (index < view.checkpoints()) try view.checkpoint(index) else null;
        const end = if (checkpoint) |entry| entry.roundIndex() else view.recorded();
        if (end > begin) {
            const segment = try evaluate(allocator, rounds[begin..end], options);
            inline for (.{ "lifetime", "lru", "decayed_frequency", "decayed_admission" }) |field| {
                @field(total, field) = try @field(total, field).plus(@field(segment, field));
            }
            total.belady_unique_misses_lower_bound = try std.math.add(u64, total.belady_unique_misses_lower_bound, segment.belady_unique_misses_lower_bound);
        }
        if (checkpoint) |entry| options.seed = trace.seedFromState(Seed, view.words[4], entry.after());
        begin = end;
    }
    const baseline = total.lifetime;
    const exact = verification.delta;
    if (baseline.rounds != @as(u64, @intCast(verification.rounds)) or
        baseline.hit_picks != exact.hit_picks or
        baseline.miss_picks != exact.miss_picks or
        baseline.copies != exact.copies or
        baseline.evictions != exact.evictions or
        baseline.mapped_experts != exact.inplace or
        baseline.mapped_picks != exact.mapped_picks)
        return error.PolicyBaselineMismatch;
    return total;
}

/// Belady with optional admission is optimal for uniform-cost sequential requests: on a full-cache miss, discard the
pub fn beladyLowerBound(allocator: std.mem.Allocator, rounds: []const trace.Round, options: Options) !u64 {
    try validateOptions(options);
    var count: usize = 0;
    for (rounds) |record| count = try std.math.add(usize, count, (try readDemand(record)).unique);
    const events = try allocator.alloc(Event, count);
    defer allocator.free(events);
    var offset: usize = 0;
    for (rounds) |record| {
        const demand = try readDemand(record);
        for (demand.ordered[0..demand.unique]) |expert| {
            events[offset] = .{ .expert = expert };
            offset += 1;
        }
    }
    return belady(events, options.capacity, if (options.seed) |seed| seed.line_ids else &.{});
}

const Event = struct { expert: usize, next: usize = std.math.maxInt(usize) };

fn belady(events: []Event, capacity: usize, initial: []const i32) u64 {
    var next_use: [expert_count]usize = @splat(std.math.maxInt(usize));
    var position = events.len;
    while (position > 0) {
        position -= 1;
        const event = &events[position];
        event.next = next_use[event.expert];
        next_use[event.expert] = position;
    }
    var resident: [expert_count]i32 = @splat(-1);
    @memcpy(resident[0..initial.len], initial);
    var misses: u64 = 0;
    for (events) |event| {
        var hit = false;
        var empty: ?usize = null;
        var farthest: ?usize = null;
        for (resident[0..capacity], 0..) |expert, line| {
            if (expert == @as(i32, @intCast(event.expert))) hit = true;
            if (expert < 0) {
                if (empty == null) empty = line;
            } else if (farthest == null or next_use[@intCast(expert)] > next_use[@intCast(resident[farthest.?])]) {
                farthest = line;
            }
        }
        next_use[event.expert] = event.next;
        if (hit) continue;
        misses += 1;
        if (empty) |line| {
            resident[line] = @intCast(event.expert);
        } else if (farthest) |line| {
            if (event.next < next_use[@intCast(resident[line])]) resident[line] = @intCast(event.expert);
        }
    }
    return misses;
}

fn validateOptions(options: Options) !void {
    if (options.capacity > expert_count or options.decay_interval_rounds == 0 or options.decay_shift == 0) return error.InvalidOptions;
    if (options.seed) |seed| {
        if (seed.line_ids.len != options.capacity or seed.stamps.len != options.capacity) return error.InvalidSeed;
        var seen: [expert_count]bool = @splat(false);
        for (seed.line_ids) |expert| {
            if (expert < -1 or expert >= expert_count) return error.InvalidSeed;
            if (expert >= 0) {
                const index: usize = @intCast(expert);
                if (seen[index]) return error.InvalidSeed;
                seen[index] = true;
            }
        }
    }
}

fn readDemand(record: trace.Round) !Demand {
    const shape = trace.header(0, 0, record.lines, record.max_rows) catch return error.InvalidRound;
    if (record.words.len != shape[7]) return error.InvalidRound;
    if (record.recentEnabled()) return error.UnsupportedRecentPolicy;
    if (record.rows() == 0 or record.rows() > record.max_rows or record.words[3] > 1) return error.InvalidRound;
    var demand: Demand = .{ .picks = record.picks() };
    for (demand.picks) |bits| {
        const expert = routed(bits) orelse continue;
        if (demand.counts[expert] == 0) {
            demand.ordered[demand.unique] = expert;
            demand.unique += 1;
        }
        demand.counts[expert] += 1;
        demand.valid_picks += 1;
    }
    if (demand.valid_picks != record.validPicks() or demand.unique != record.uniqueCount()) return error.InvalidRound;
    if (demand.unique > max_unique) return error.PlannerOverflow;
    return demand;
}

fn routed(bits: u32) ?usize {
    const expert: i32 = @bitCast(bits);
    return if (expert >= 0 and expert < expert_count) @intCast(expert) else null;
}

const Synthetic = struct {
    words: []u32,
    max_rows: usize,

    fn init(picks: []const i32) !Synthetic {
        const rows = @max(1, (picks.len + 10) / 11);
        const shape = try trace.header(0, 0, 2, rows);
        const words = try std.testing.allocator.alloc(u32, shape[7]);
        @memset(words, 0);
        words[0] = @intCast(rows);
        const self: Synthetic = .{ .words = words, .max_rows = rows };
        const raw_offset = trace.record_header_words;
        @memset(words[raw_offset..][0 .. rows * 11], std.math.maxInt(u32));
        var counts: [expert_count]u32 = @splat(0);
        for (picks, 0..) |expert, index| {
            words[raw_offset + index] = @bitCast(expert);
            if (routed(@bitCast(expert))) |valid| {
                if (counts[valid] == 0) words[2] += 1;
                counts[valid] += 1;
                words[1] += 1;
            }
        }
        return self;
    }

    fn round(self: Synthetic) trace.Round {
        return .{ .words = self.words, .lines = 2, .max_rows = self.max_rows };
    }

    fn deinit(self: Synthetic) void {
        std.testing.allocator.free(self.words);
    }
};

test "exl3: policy evaluation pins round hits and admissions and separates mapped misses" {
    var fixture = try Synthetic.init(&.{ 0, 2, 2, 2, 3, 512, -1, 999, -9 });
    defer fixture.deinit();
    var state = try State.init(std.testing.allocator, .lifetime, .{
        .capacity = 2,
        .seed = .{ .line_ids = &.{ 0, 1 }, .stamps = &.{ 1, 2 }, .counts = @splat(0), .now = 2 },
    });
    defer state.deinit();
    const result = try state.observe(fixture.round());
    try std.testing.expectEqual(Metrics{ .rounds = 1, .hit_picks = 1, .miss_picks = 4, .hit_experts = 1, .miss_experts = 2, .admissions = 1, .copies = 1, .evictions = 1, .mapped_experts = 1, .mapped_picks = 1 }, result);
    try std.testing.expectEqualSlices(i32, &.{ 0, 2 }, state.line_ids);
}

test "exl3: lifetime admission saturation and victim ties match current planner" {
    var fixture = try Synthetic.init(&.{ 2, 2, 2, 2, 2, 3 });
    defer fixture.deinit();
    var counts: [expert_count]u32 = @splat(0);
    counts[0] = 5;
    counts[1] = 3;
    counts[2] = std.math.maxInt(u32) - 2;
    var state = try State.init(std.testing.allocator, .lifetime, .{
        .capacity = 2,
        .seed = .{ .line_ids = &.{ 0, 1 }, .stamps = &.{ 1, 2 }, .counts = counts, .now = std.math.maxInt(u32) },
    });
    defer state.deinit();
    const result = try state.observe(fixture.round());
    try std.testing.expectEqual(@as(u64, 2), result.copies);
    try std.testing.expectEqualSlices(i32, &.{ 3, 2 }, state.line_ids);
    try std.testing.expectEqual(std.math.maxInt(u32), state.scores[2]);
    try std.testing.expectEqual(@as(u32, 0), state.now);

    var tie = try State.init(std.testing.allocator, .lifetime, .{
        .capacity = 2,
        .seed = .{ .line_ids = &.{ 0, 1 }, .stamps = &.{ 7, 7 }, .counts = @splat(0), .now = 7 },
    });
    defer tie.deinit();
    var single = try Synthetic.init(&.{9});
    defer single.deinit();
    _ = try tie.observe(single.round());
    try std.testing.expectEqualSlices(i32, &.{ 0, 9 }, tie.line_ids);
}

test "exl3: LRU preserves raw last occurrence order when the round exceeds capacity" {
    var first = try Synthetic.init(&.{ 1, 2, 1 });
    defer first.deinit();
    var second = try Synthetic.init(&.{ 1, 1, 2 });
    defer second.deinit();
    var left = try State.init(std.testing.allocator, .lru, .{ .capacity = 1 });
    defer left.deinit();
    var right = try State.init(std.testing.allocator, .lru, .{ .capacity = 1 });
    defer right.deinit();
    _ = try left.observe(first.round());
    _ = try right.observe(second.round());
    try std.testing.expectEqualSlices(i32, &.{1}, left.line_ids);
    try std.testing.expectEqualSlices(i32, &.{2}, right.line_ids);
    _ = try left.observe(second.round());
    try std.testing.expectEqualSlices(i32, &.{1}, left.line_ids);
}

test "exl3: frequency decay halves scores at configured round boundaries before new demand" {
    var fixture = try Synthetic.init(&.{ 0, 1, 1, 1 });
    defer fixture.deinit();
    var empty = try Synthetic.init(&.{});
    defer empty.deinit();
    var counts: [expert_count]u32 = @splat(0);
    counts[0] = 64;
    const options: Options = .{
        .capacity = 1,
        .seed = .{ .line_ids = &.{-1}, .stamps = &.{0}, .counts = counts, .now = 0 },
        .decay_interval_rounds = 1,
        .decay_shift = 6,
    };
    var lifetime = try State.init(std.testing.allocator, .lifetime, options);
    defer lifetime.deinit();
    var decayed = try State.init(std.testing.allocator, .decayed_frequency, options);
    defer decayed.deinit();
    _ = try lifetime.observe(fixture.round());
    _ = try decayed.observe(fixture.round());
    try std.testing.expectEqualSlices(i32, &.{0}, lifetime.line_ids);
    try std.testing.expectEqualSlices(i32, &.{1}, decayed.line_ids);

    var interval = try State.init(std.testing.allocator, .decayed_frequency, .{ .capacity = 0, .decay_interval_rounds = 2 });
    defer interval.deinit();
    _ = try interval.observe(fixture.round());
    try std.testing.expectEqual(@as(u32, 3), interval.scores[1]);
    _ = try interval.observe(empty.round());
    try std.testing.expectEqual(@as(u32, 1), interval.scores[1]);
}

test "exl3: LRU evicts by within-round recency and decayed eviction forgets old frequency" {
    var touching = try Synthetic.init(&.{ 0, 1, 0 });
    defer touching.deinit();
    var incoming = try Synthetic.init(&.{2});
    defer incoming.deinit();
    var empty = try Synthetic.init(&.{});
    defer empty.deinit();
    var counts: [expert_count]u32 = @splat(0);
    counts[0] = 128;
    counts[1] = 4;
    const options: Options = .{
        .capacity = 2,
        .seed = .{ .line_ids = &.{ 0, 1 }, .stamps = &.{ 1, 2 }, .counts = counts, .now = 2 },
        .decay_interval_rounds = 1,
        .decay_shift = 6,
    };
    var lru = try State.init(std.testing.allocator, .lru, options);
    defer lru.deinit();
    _ = try lru.observe(touching.round());
    _ = try lru.observe(incoming.round());
    try std.testing.expectEqualSlices(i32, &.{ 0, 2 }, lru.line_ids);

    var lifetime = try State.init(std.testing.allocator, .lifetime, options);
    defer lifetime.deinit();
    var decayed = try State.init(std.testing.allocator, .decayed_frequency, options);
    defer decayed.deinit();
    for (0..2) |_| {
        _ = try lifetime.observe(empty.round());
        _ = try decayed.observe(empty.round());
    }
    _ = try lifetime.observe(incoming.round());
    _ = try decayed.observe(incoming.round());
    try std.testing.expectEqualSlices(i32, &.{ 0, 2 }, lifetime.line_ids);
    try std.testing.expectEqualSlices(i32, &.{ 2, 1 }, decayed.line_ids);
}

test "exl3: frequency admission bypasses a colder newcomer and later admits it when hot" {
    var warm = try Synthetic.init(&.{ 1, 1 });
    defer warm.deinit();
    var hot = try Synthetic.init(&.{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
    defer hot.deinit();
    var counts: [expert_count]u32 = @splat(0);
    counts[0] = 10;
    var state = try State.init(std.testing.allocator, .decayed_admission, .{
        .capacity = 1,
        .seed = .{ .line_ids = &.{0}, .stamps = &.{1}, .counts = counts, .now = 1 },
    });
    defer state.deinit();
    const cold = try state.observe(warm.round());
    try std.testing.expectEqual(@as(u64, 0), cold.copies);
    try std.testing.expectEqual(@as(u64, 1), cold.mapped_experts);
    try std.testing.expectEqualSlices(i32, &.{0}, state.line_ids);
    const now_hot = try state.observe(hot.round());
    try std.testing.expectEqual(@as(u64, 1), now_hot.copies);
    try std.testing.expectEqual(@as(u64, 1), now_hot.evictions);
    try std.testing.expectEqualSlices(i32, &.{1}, state.line_ids);
}

test "exl3: comparison evolves independently from the initial seed and supports zero capacity" {
    var fixture = try Synthetic.init(&.{ 5, 5 });
    defer fixture.deinit();
    const rounds = [_]trace.Round{ fixture.round(), fixture.round() };
    const result = try evaluate(std.testing.allocator, &rounds, .{ .capacity = 1 });
    try std.testing.expectEqual(@as(u64, 2), result.lifetime.miss_picks);
    try std.testing.expectEqual(@as(u64, 2), result.lifetime.hit_picks);
    try std.testing.expectEqual(@as(u64, 1), result.lifetime.copies);
    try std.testing.expectEqual(result.lifetime, result.lru);
    try std.testing.expectEqual(result.lifetime, result.decayed_frequency);
    try std.testing.expectEqual(result.lifetime, result.decayed_admission);
    try std.testing.expectEqual(@as(u64, 1), result.belady_unique_misses_lower_bound);
    const zero = try evaluate(std.testing.allocator, &rounds, .{ .capacity = 0 });
    try std.testing.expectEqual(@as(u64, 4), zero.lifetime.mapped_picks);
    try std.testing.expectEqual(@as(u64, 0), zero.lifetime.copies);
    try std.testing.expectEqual(@as(u64, 4), zero.decayed_admission.mapped_picks);
    try std.testing.expectEqual(@as(u64, 0), zero.decayed_admission.copies);
    try std.testing.expectEqual(@as(u64, 2), zero.belady_unique_misses_lower_bound);
    const none = try evaluate(std.testing.allocator, &.{}, .{ .capacity = 2 });
    try std.testing.expectEqual(Metrics{}, none.lifetime);
    try std.testing.expectEqual(@as(u64, 0), none.belady_unique_misses_lower_bound);
}

test "exl3: Belady bypass retains a warm expert and bounds pinned round misses only" {
    var first = try Synthetic.init(&.{ 1, 1, 2 });
    defer first.deinit();
    var second = try Synthetic.init(&.{ 0, 0 });
    defer second.deinit();
    const options: Options = .{
        .capacity = 1,
        .seed = .{ .line_ids = &.{0}, .stamps = &.{0}, .counts = @splat(0), .now = 0 },
    };
    const result = try evaluate(std.testing.allocator, &.{ first.round(), second.round() }, options);
    try std.testing.expectEqual(@as(u64, 2), result.belady_unique_misses_lower_bound);
    try std.testing.expectEqual(@as(u64, 3), result.lifetime.miss_experts);
    try std.testing.expect(result.belady_unique_misses_lower_bound <= result.lru.miss_experts);
    try std.testing.expect(result.belady_unique_misses_lower_bound <= result.decayed_frequency.miss_experts);
}

fn exhaustiveMisses(events: []const Event, position: usize, mask: u8, capacity: usize) u64 {
    if (position == events.len) return 0;
    const bit = @as(u8, 1) << @as(u3, @intCast(events[position].expert));
    if (mask & bit != 0) return exhaustiveMisses(events, position + 1, mask, capacity);
    var best = exhaustiveMisses(events, position + 1, mask, capacity);
    if (@popCount(mask) < capacity) {
        best = @min(best, exhaustiveMisses(events, position + 1, mask | bit, capacity));
    } else {
        for (0..3) |expert| {
            const victim_bit = @as(u8, 1) << @as(u3, @intCast(expert));
            if (mask & victim_bit != 0) best = @min(best, exhaustiveMisses(events, position + 1, (mask & ~victim_bit) | bit, capacity));
        }
    }
    return 1 + best;
}

test "exl3: Belady equals exhaustive optional-admission optimum for short cold and warm streams" {
    for (0..243) |sequence| {
        var encoded = sequence;
        var events: [5]Event = undefined;
        for (&events) |*event| {
            event.* = .{ .expert = encoded % 3 };
            encoded /= 3;
        }
        for (0..3) |capacity| {
            try std.testing.expectEqual(exhaustiveMisses(&events, 0, 0, capacity), belady(&events, capacity, &.{}));
            if (capacity > 0) try std.testing.expectEqual(exhaustiveMisses(&events, 0, 1, capacity), belady(&events, capacity, &.{0}));
        }
    }
}

test "exl3: invalid input is rejected before state changes including planner unique overflow" {
    var state = try State.init(std.testing.allocator, .lifetime, .{ .capacity = 1 });
    defer state.deinit();
    var fixture = try Synthetic.init(&.{1});
    defer fixture.deinit();
    var truncated = fixture.round();
    truncated.words = truncated.words[0..3];
    try std.testing.expectError(error.InvalidRound, state.observe(truncated));
    fixture.words[1] = 2;
    try std.testing.expectError(error.InvalidRound, state.observe(fixture.round()));
    var picks: [257]i32 = undefined;
    for (&picks, 0..) |*expert, index| expert.* = @intCast(index);
    var overflow = try Synthetic.init(&picks);
    defer overflow.deinit();
    try std.testing.expectError(error.PlannerOverflow, state.observe(overflow.round()));
    try std.testing.expectEqual(Metrics{}, state.total);
    try std.testing.expectEqualSlices(i32, &.{-1}, state.line_ids);
    try std.testing.expectError(error.InvalidOptions, State.init(std.testing.allocator, .lru, .{ .capacity = 513 }));
    try std.testing.expectError(error.InvalidOptions, State.init(std.testing.allocator, .lru, .{ .capacity = 1, .decay_interval_rounds = 0 }));
    try std.testing.expectError(error.InvalidSeed, State.init(std.testing.allocator, .lru, .{
        .capacity = 2,
        .seed = .{ .line_ids = &.{ 1, 1 }, .stamps = &.{ 0, 0 }, .counts = @splat(0), .now = 0 },
    }));
}
