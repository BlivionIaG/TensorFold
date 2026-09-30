const std = @import("std");
const Proposal = @import("drafter.zig").Proposal;

pub const Stage = enum { idle, bound, prepared, forwarded, settled, failed };

pub const Owner = struct {
    epoch: u64 = 0,
    stage: Stage = .idle,

    pub fn begin(o: *Owner) !Ticket {
        if (o.stage != .idle) return error.ModelRoundActive;
        if (o.epoch == std.math.maxInt(u64)) return error.RoundEpochExhausted;
        o.epoch += 1;
        o.stage = .bound;
        return .{ .owner = o, .epoch = o.epoch };
    }
};

pub const Ticket = struct {
    owner: *Owner,
    epoch: u64,

    pub fn active(t: Ticket) bool {
        return t.owner.epoch == t.epoch and t.owner.stage != .idle;
    }

    pub fn expect(t: Ticket, stage: Stage) !void {
        if (!t.active()) return error.StaleRound;
        if (t.owner.stage != stage) return error.InvalidRoundStage;
    }

    pub fn advance(t: Ticket, before: Stage, after: Stage) !void {
        try t.expect(before);
        t.owner.stage = after;
    }

    pub fn release(t: Ticket) void {
        if (t.active()) t.owner.stage = .idle;
    }
};

pub const Window = struct {
    tokens: [16]i32 = undefined,
    parents: [16]i32 = undefined,
    positions: [16]i32 = undefined,
    draft: Proposal,
    from_neural: bool,
    count: usize,

    pub fn init(pending: i32, start: i32, draft: Proposal, from_neural: bool) !Window {
        if (draft.len > 15 or start < 0) return error.InvalidDecodeWindow;
        var w = Window{ .draft = draft, .from_neural = from_neural, .count = draft.len + 1 };
        w.tokens[0] = pending;
        w.parents[0] = -1;
        w.positions[0] = std.math.add(i32, start, 1) catch return error.InvalidDecodeWindow;
        @memcpy(w.tokens[1..w.count], draft.tokens[0..draft.len]);
        for (0..draft.len) |i| {
            const parent = draft.parents[i];
            if (parent < -1 or parent >= i) return error.InvalidDecodeWindow;
            w.parents[i + 1] = parent + 1;
            w.positions[i + 1] = std.math.add(i32, w.positions[@intCast(parent + 1)], 1) catch return error.InvalidDecodeWindow;
        }
        return w;
    }
};

test "round ownership rejects overlap, duplicate settlement and stale tickets" {
    var owner = Owner{};
    const first = try owner.begin();
    try std.testing.expectError(error.ModelRoundActive, owner.begin());
    try std.testing.expectError(error.InvalidRoundStage, first.expect(.forwarded));
    try first.advance(.bound, .prepared);
    try first.advance(.prepared, .forwarded);
    try first.advance(.forwarded, .settled);
    try std.testing.expectError(error.InvalidRoundStage, first.advance(.forwarded, .settled));
    first.release();
    const second = try owner.begin();
    try std.testing.expectError(error.StaleRound, first.expect(.bound));
    first.release();
    try second.expect(.bound);
    second.release();
    try std.testing.expectError(error.StaleRound, second.expect(.bound));
}

test "ragged tree sampling uses request positions and parent depths" {
    var draft = Proposal{ .len = 4 };
    @memcpy(draft.tokens[0..4], &[_]i32{ 12, 13, 14, 15 });
    @memcpy(draft.parents[0..4], &[_]i32{ -1, -1, 0, 2 });
    const w = try Window.init(11, 37, draft, true);
    try std.testing.expectEqualSlices(i32, &.{ 11, 12, 13, 14, 15 }, w.tokens[0..w.count]);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0, 1, 3 }, w.parents[0..w.count]);
    try std.testing.expectEqualSlices(i32, &.{ 38, 39, 39, 40, 41 }, w.positions[0..w.count]);
    const single = try Window.init(11, 4096, .{}, false);
    try std.testing.expectEqual(@as(usize, 1), single.count);
    try std.testing.expectEqual(@as(i32, 4097), single.positions[0]);
}

test "decode windows reject cycles, forward parents, excess rows and position overflow" {
    var draft = Proposal{ .len = 1 };
    draft.tokens[0] = 12;
    for ([_]i32{ -2, 0, 1 }) |parent| {
        draft.parents[0] = parent;
        try std.testing.expectError(error.InvalidDecodeWindow, Window.init(11, 0, draft, true));
    }
    draft.parents[0] = -1;
    try std.testing.expectError(error.InvalidDecodeWindow, Window.init(11, std.math.maxInt(i32) - 1, draft, true));
    try std.testing.expectError(error.InvalidDecodeWindow, Window.init(11, -1, .{}, false));
    draft.len = 16;
    try std.testing.expectError(error.InvalidDecodeWindow, Window.init(11, 0, draft, true));
}
