//! One decode row through a layer range, encoded into a relay's parked command buffer (the range's `Runner.advance`).
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const forward = @import("forward.zig");
const Runner = @import("decode_round.zig").Runner;
const Round = @import("round_plan.zig").Round;

const Pass = struct {
    stages: *forward.Stages,
    err: ?anyerror = null,
};

fn encode(ctx: *anyopaque, cb: mtl.CommandBuffer) void {
    const pass: *Pass = @ptrCast(@alignCast(ctx));
    var lane = core.segments.Lane{ .k = 0, .cb = cb, .enc = cb.compute(.concurrent), .fence = undefined };
    run(pass.stages, &lane) catch |err| {
        pass.err = err;
    };
    lane.enc.end();
}

fn run(s: *forward.Stages, lane: *core.segments.Lane) anyerror!void {
    try s.begin(lane);
    for (0..s.range.?.len()) |i| {
        try s.pre(lane, i);
        try s.mixer(lane, i);
        try s.post(lane, i);
    }
    try s.finish(lane);
}

/// Encode `slot`'s next position for `r`'s range into `relay`'s parked buffer; the caller opens the ticket.
pub fn launch(r: *Runner, relay: *mtl.Relay, slot: u32, id: u32, io: forward.Io) !mtl.Ticket {
    const m = r.model;
    if (r.failed or r.active != null or slot >= r.ready) return error.RoundNotReady;
    const ids = [1]u32{id};
    var round = try Round.init(r.allocator, &.{.{ .slot = slot, .start = r.offsets[slot], .capacity = r.capacity, .ids = &ids }}, @intCast(m.config.conv_kernel), r.slots, @intCast(m.shard.global.vocab));
    defer round.deinit();
    try r.cover(slot, round.windows[0].start + 1);
    const row = [1]u32{0};
    const paths = [1][]const u32{&row};
    var recurrent = try round.recurrent(try r.params(1), &paths);
    defer recurrent.deinit();
    var attention_plan = try r.attentionPlan(&round);
    defer attention_plan.deinit();
    try r.gdn.prepare(m.device, recurrent, true);
    try m.frame.setRows(round.ids, round.feed_positions);
    try r.scratch.upload(attention_plan);
    r.current_round = &round;
    r.current_gdn = &recurrent;
    r.current_attention = &attention_plan;
    r.accepted_chain = true;
    defer {
        r.current_round = null;
        r.current_gdn = null;
        r.current_attention = null;
        r.accepted_chain = false;
    }
    var stages = forward.Stages{
        .model = m,
        .rows = 1,
        .hooks = .{ .ptr = r, .gdn = Runner.gdnForward, .attention = Runner.attentionForward },
        .head = if (r.range.head(m.config.layers)) .last else .none,
        .range = r.range,
        .io = io,
    };
    var pass = Pass{ .stages = &stages };
    const ticket = relay.submit(&pass, encode) catch |err| {
        r.failed = true;
        return err;
    };
    if (pass.err) |err| {
        r.failed = true;
        relay.abort();
        return err;
    }
    r.offsets[slot] += 1;
    return ticket;
}
