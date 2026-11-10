//! One decode row through a layer range on the GPU, encoded into a relay's parked command buffer: the accepted-chain
//! decode of `Runner.advance` for a runner over a model that holds only that range (`Model.loadRange`).
//! The caller writes the stage's input row, opens the ticket and waits for it; the same program as the whole model's,
//! cut at the range's ends by `forward.Stages` (row load in front, hand-off row store behind, or the head).
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

/// The most positions one launch carries (the pipeline's `max_rows_limit`).
const max_chain = 256;

/// Encode the next `ids.len` consecutive positions of `slot` for `r`'s range into `relay`'s parked command buffer; the
/// caller opens the ticket once the input rows are in place. `ids` are the tokens a range that embeds reads (any ids for
/// one that does not). One position is a decode step; more are a prompt chunk, run with the prompt kernels as
/// `Session.promptChunk` runs it. An encoding error ends the runner and the relay's pass; a GPU failure shows at the ticket.
pub fn launch(r: *Runner, relay: *mtl.Relay, slot: u32, ids: []const u32, io: forward.Io) !mtl.Ticket {
    const m = r.model;
    if (r.failed or r.active != null or slot >= r.ready) return error.RoundNotReady;
    if (ids.len == 0 or ids.len > m.frame.capacity or ids.len > max_chain) return error.BadFrameRows;
    m.kernels.prompt = ids.len > 1;
    defer m.kernels.prompt = false;
    var round = try Round.init(r.allocator, &.{.{ .slot = slot, .start = r.offsets[slot], .capacity = r.capacity, .ids = ids }}, @intCast(m.config.conv_kernel), r.slots, @intCast(m.shard.global.vocab));
    defer round.deinit();
    try r.cover(slot, round.windows[0].start + @as(u32, @intCast(ids.len)));
    var chain: [max_chain]u32 = undefined;
    for (chain[0..ids.len], 0..) |*c, i| c.* = @intCast(i);
    const paths = [1][]const u32{chain[0..ids.len]};
    var recurrent = try round.recurrent(try r.params(ids.len), &paths);
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
        .rows = @intCast(ids.len),
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
    r.offsets[slot] += @intCast(ids.len);
    return ticket;
}
