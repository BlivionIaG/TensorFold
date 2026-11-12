//! A tied head on generated weights: the embedding table read as the head decodes what a stored copy of it decodes.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const graph = @import("synthetic_graph.zig");

const decoded = 16;

/// Greedy tokens after `prompt`, then the fingerprint of the final state and logits.
fn run(a: std.mem.Allocator, head: graph.Head, prompt: []const u32, tokens: *[decoded]u32) ![32]u8 {
    const m = try graph.createHead(a, 4, head);
    defer m.deinit();
    var runner = try q.decode_round.Runner.init(a, m, 1, 256);
    defer runner.deinit();
    const session = q.session.Session{ .runner = &runner };
    try session.prefill(prompt, 4);
    for (tokens, 0..) |*token, i| {
        token.* = try session.greedy();
        if (i + 1 < tokens.len) try session.step(token.*);
    }
    return session.fingerprint();
}

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var prompt: [12]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast((i * 17 + 7) % 96);
    var copied: [decoded]u32 = undefined;
    var tied: [decoded]u32 = undefined;
    var other: [decoded]u32 = undefined;
    const copy_state = try run(a, .copy, &prompt, &copied);
    const tied_state = try run(a, .tied, &prompt, &tied);
    _ = try run(a, .untied, &prompt, &other);
    var distinct: usize = 0;
    for (tied, 0..) |t, i| distinct += @intFromBool(std.mem.indexOfScalar(u32, tied[0..i], t) == null);
    if (distinct < 2) return error.TiedDecodeIsConstant;
    if (!std.mem.eql(u32, &copied, &tied)) return error.TiedTokensDiffer;
    if (!std.mem.eql(u8, &copy_state, &tied_state)) return error.TiedStateDiffers;
    const own_head = std.mem.eql(u32, &other, &tied);
    std.debug.print("qwen27 tied head: {d} greedy tokens ({d} distinct) and the final state/logits are byte-equal to a stored copy of the table; a separate random head {s}\n", .{ decoded, distinct, if (own_head) "decodes the same (the control is blind)" else "decodes differently (the control sees a head)" });
    if (own_head) return error.ControlIsBlind;
}
