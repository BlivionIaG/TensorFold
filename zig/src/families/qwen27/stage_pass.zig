//! Positions through a layer range on the GPU, encoded into a relay's parked command buffer: the accepted-chain decode
//! of `Runner.advance` for a runner over a model that holds only that range (`Model.loadRange`). A launch is a decode
//! step or a prompt chunk of one sequence slot, or a decode batch: the next position of each of several slots in one round.
//! The caller writes the stage's input rows, opens the ticket and waits for it; the same program as the whole model's,
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
/// The most sequences a decode batch names (the pipeline's `max_slots`).
pub const max_slots = 8;

/// Encode the next positions of the sequences `slots` name for `r`'s range into `relay`'s parked command buffer; the
/// caller opens the ticket once the input rows are in place. `ids` are the tokens a range that embeds reads (any ids for
/// one that does not). With one slot, `ids.len` consecutive positions of it: one is a decode step, more are a prompt
/// chunk run with the prompt kernels as `Session.promptChunk` runs it, and the head (if the range has it) is taken after
/// the last row. With `ids.len` slots (no slot twice), a decode batch: row i is the next position of `slots[i]`, and the
/// head is taken on every row. An encoding error ends the runner and the relay's pass; a GPU failure shows at the ticket.
pub fn launch(r: *Runner, relay: *mtl.Relay, slots: []const u8, ids: []const u32, io: forward.Io) !mtl.Ticket {
    const m = r.model;
    const batch = slots.len > 1;
    if (r.failed or r.active != null or slots.len == 0 or slots.len > max_slots) return error.RoundNotReady;
    for (slots) |slot| if (slot >= r.ready) return error.RoundNotReady;
    if (ids.len == 0 or ids.len > m.frame.capacity or ids.len > max_chain or (batch and ids.len != slots.len)) return error.BadFrameRows;
    m.kernels.prompt = !batch and ids.len > 1;
    defer m.kernels.prompt = false;
    var inputs: [max_slots]@import("round_plan.zig").Input = undefined;
    for (slots, inputs[0..slots.len], 0..) |slot, *input, i| {
        input.* = .{ .slot = slot, .start = r.offsets[slot], .capacity = r.capacity, .ids = if (batch) ids[i..][0..1] else ids };
    }
    var round = try Round.init(r.allocator, inputs[0..slots.len], @intCast(m.config.conv_kernel), r.slots, @intCast(m.shard.global.vocab));
    defer round.deinit();
    var chain: [max_chain]u32 = undefined;
    for (&chain, 0..) |*c, i| c.* = @intCast(i);
    var paths: [max_slots][]const u32 = undefined;
    for (round.slots, round.windows, paths[0..slots.len]) |slot, window, *path| {
        try r.cover(slot, window.start + @as(u32, @intCast(window.parents.len)));
        path.* = chain[0..window.parents.len];
    }
    var recurrent = try round.recurrent(try r.params(ids.len), paths[0..slots.len]);
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
        .head = if (!r.range.head(m.config.layers)) .none else if (batch) .all else .last,
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
    for (round.slots, round.windows) |slot, window| r.offsets[slot] = window.start + @as(u32, @intCast(window.parents.len));
    return ticket;
}
