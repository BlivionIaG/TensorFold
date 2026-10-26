//! The lone guard through the round loop: a lone stream drafts while drafting measures faster, else runs plain.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const be = @import("backend.zig");
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const dr = @import("drafter.zig");
const Drafted = @import("drafted.zig").Drafted;
const dt = @import("drafted_test.zig");
const ct = @import("drafted_chances_test.zig");

const gpa = std.testing.allocator;

/// A round's ms from its verify's rows and whether a hold ran before it, as a GPU that runs both in turn.
const Costs = struct { row: f64, extra: f64, hold: f64 };

const Cost = struct {
    var costs: Costs = undefined;
    var inner_verify: *const fn (*anyopaque, []const be.Window, []be.Verified) anyerror!void = undefined;
    var inner_hold: *const fn (*anyopaque, []const dr.Hold) anyerror!void = undefined;
    var rows: usize = 1;
    var held: bool = false; // a hold since the last verify
    var before: bool = false; // the last verify ran after a hold
    var holds: u32 = 0;
    var verifies: u32 = 0;

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        rows = windows[0].rows();
        before = held;
        held = false;
        verifies += 1;
        return inner_verify(ptr, windows, out);
    }

    fn hold(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        for (items) |it| held = held or it.depth > 0;
        holds += @intFromBool(held);
        return inner_hold(ptr, items);
    }

    fn clock() be.Clock {
        return .{ .ptr = @ptrCast(&costs), .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(_: *anyopaque) void {}

    fn elapsed(_: *anyopaque, _: be.Mark) f64 {
        const extra: f64 = @floatFromInt(rows - 1);
        return costs.row + costs.extra * extra + (if (before) costs.hold else 0);
    }
};

const Run = struct { tokens: [400]u32 = undefined, len: usize = 0, holds: u32 = 0, rounds: u32 = 0, drafted: u64 = 0 };

/// One lone stream of 300 tokens; `guard` the drafter's guard switch, `costs` null: drafts off.
fn run(costs: ?Costs, guard: bool) !Run {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var p: ct.Priced = .{ .head = .{ .facts_ = .{ .depth = 9, .step_ms = 0.5, .batched = true, .chances = true, .shared_guard = guard } }, .sure = 0.95, .unsure = 0.05 };
    defer p.deinit();
    var strict: dt.Strict = .{ .inner = target.backend() };
    defer strict.deinit();
    var seen = strict.backend();
    var table = seen.vtable.*;
    Cost.inner_verify = table.verify;
    table.verify = Cost.verify;
    seen.vtable = &table;
    var drafter = p.drafter();
    var hold_table = drafter.vtable.*;
    Cost.inner_hold = hold_table.hold;
    hold_table.hold = Cost.hold;
    drafter.vtable = &hold_table;
    Cost.costs = costs orelse .{ .row = 8, .extra = 0, .hold = 0 };
    Cost.holds = 0;
    Cost.verifies = 0;
    Cost.held = false;
    var wrapped = try Drafted.init(gpa, seen, drafter);
    defer wrapped.deinit();
    const drafts = costs != null;
    var cfg = try Config.init(gpa, if (drafts) wrapped.facts(ct.base()) else ct.base(), 16, 15);
    defer cfg.deinit(gpa);
    var engine = Engine.init(gpa, &cfg, if (drafts) wrapped.backend() else target.backend(), Cost.clock());
    defer engine.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &dt.p1, .max_new = 300, .eos = &.{}, .drafts = drafts });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    var out: Run = .{ .len = s.emitted().len, .holds = Cost.holds, .rounds = Cost.verifies, .drafted = s.drafted };
    @memcpy(out.tokens[0..out.len], s.emitted());
    return out;
}

const slow: Costs = .{ .row = 8, .extra = 4, .hold = 9 }; // a hold and a verify row cost more than drafts save
const fast: Costs = .{ .row = 8, .extra = 1, .hold = 1 };

test "a lone stream whose drafted rounds lose runs plain rounds and probes now and then, replies unchanged" {
    const plain = try run(null, true);
    const unguarded = try run(slow, false);
    const guarded = try run(slow, true);
    try std.testing.expectEqualSlices(u32, plain.tokens[0..plain.len], unguarded.tokens[0..unguarded.len]);
    try std.testing.expectEqualSlices(u32, plain.tokens[0..plain.len], guarded.tokens[0..guarded.len]);
    try std.testing.expect(unguarded.holds * 10 > unguarded.rounds * 9); // without the guard nearly every round holds
    try std.testing.expect(guarded.holds * 3 < guarded.rounds); // with it most rounds are plain
    try std.testing.expect(guarded.holds > 16); // the first drafted stretch and probes
}

test "a lone stream whose drafted rounds win keeps drafting with the guard on" {
    const plain = try run(null, true);
    const unguarded = try run(fast, false);
    const guarded = try run(fast, true);
    try std.testing.expectEqualSlices(u32, plain.tokens[0..plain.len], guarded.tokens[0..guarded.len]);
    try std.testing.expect(guarded.holds * 10 > guarded.rounds * 8); // drafting but for the plain rate's one stretch
    try std.testing.expect(guarded.rounds * 10 < unguarded.rounds * 12);
}
