//! Several sequences through one split at once. A `Lane` is a `Pipeline` over its own two slots of one window with its own
//! stage on each side, and the sequences it serves, a slot each in both stages (a stage owns the state of its layers for each
//! of its sequences). The lanes share the device: its stream runs the steps in the order they were queued, so after the host
//! has handed the device a lane's hidden rows it works on the next lane's while the device works on that lane's: both sides
//! stay busy and a step costs about the slower side, not the sum. Per round the schedule is: collect lane A, submit A's next
//! step, collect lane B, submit B's next step, so a lane's next step is queued as soon as its tokens are known.
//!
//! A lane's step is a prompt chunk of one of its sequences (prompts first, one sequence at a time) or a decode batch: the next
//! position of every sequence that still has tokens to make, in one step, so the weights are read once for all of them.
const std = @import("std");
const pipeline = @import("pipeline.zig");

const Pipeline = pipeline.Pipeline;

/// The most lanes one `generate` runs.
pub const max_streams = 8;

/// One sequence: a prompt, and the greedy tokens to take after it.
pub const Work = struct { prompt: []const u32, out: []u32 };

/// A pipeline and the sequences it serves, at most `pipeline.max_slots`; sequence i is slot i.
pub const Lane = struct { pipe: *Pipeline, works: []const Work };

const Seq = struct { at: usize = 0, produced: usize = 0, token: u32 = 0 };

const State = struct {
    seq: [pipeline.max_slots]Seq = @splat(.{}),
    flight: bool = false,
    /// The step in flight: a prompt chunk (`chunk` positions of slot `slots[0]`) or a batch of `n` decode rows.
    chunk: usize = 0,
    n: usize = 0,
    slots: [pipeline.max_slots]u8 = undefined,
    tokens: [pipeline.max_slots]u32 = undefined,

    /// Queues the lane's next step, if it has one; false when every sequence is done.
    fn submit(s: *State, lane: Lane, chunk: usize) pipeline.Error!bool {
        for (lane.works, 0..) |w, i| {
            const q = &s.seq[i];
            if (q.at >= w.prompt.len) continue;
            s.chunk = @min(chunk, w.prompt.len - q.at);
            s.n = 0;
            s.slots[0] = @intCast(i);
            try lane.pipe.submitSlots(w.prompt[q.at..][0..s.chunk], s.slots[0..1]);
            s.flight = true;
            return true;
        }
        s.chunk = 0;
        s.n = 0;
        for (lane.works, 0..) |w, i| {
            const q = &s.seq[i];
            if (q.produced >= w.out.len) continue;
            w.out[q.produced] = q.token;
            q.produced += 1;
            s.slots[s.n] = @intCast(i);
            s.tokens[s.n] = q.token;
            s.n += 1;
        }
        if (s.n == 0) return false;
        try lane.pipe.submitSlots(s.tokens[0..s.n], s.slots[0..s.n]);
        s.flight = true;
        return true;
    }

    fn landed(s: *State, lane: Lane) pipeline.Error!void {
        if (s.chunk > 0) {
            const q = &s.seq[s.slots[0]];
            q.token = try lane.pipe.collect();
            q.at += s.chunk;
        } else {
            var out: [pipeline.max_slots]u32 = undefined;
            try lane.pipe.collectAll(out[0..s.n]);
            for (s.slots[0..s.n], out[0..s.n]) |slot, t| s.seq[slot].token = t;
        }
        s.flight = false;
    }
};

/// Runs every lane's sequences: each prompt in steps of up to `chunk` positions, then each sequence's `out.len` greedy tokens,
/// the sequence `toy.generateChunked` makes for each alone. The lanes share a device; their windows must not overlap
/// (`Config.window_offset`). On an error every pipeline is released so the device's stream drains; the caller resets them.
pub fn generateLanes(lanes: []const Lane, chunk: usize) pipeline.Error!void {
    std.debug.assert(lanes.len <= max_streams and chunk >= 1);
    for (lanes) |l| std.debug.assert(l.works.len >= 1 and l.works.len <= pipeline.max_slots and l.pipe.cfg.max_rows >= l.works.len);
    run(lanes, chunk) catch |err| {
        for (lanes) |l| l.pipe.release();
        return err;
    };
}

/// One sequence a pipeline: `generateLanes` with a lane each.
pub fn generate(pipes: []const *Pipeline, works: []const Work, chunk: usize) pipeline.Error!void {
    std.debug.assert(pipes.len == works.len and pipes.len <= max_streams);
    var lanes: [max_streams]Lane = undefined;
    for (pipes, 0..) |p, i| lanes[i] = .{ .pipe = p, .works = works[i..][0..1] };
    return generateLanes(lanes[0..pipes.len], chunk);
}

fn run(lanes: []const Lane, chunk: usize) pipeline.Error!void {
    var st: [max_streams]State = @splat(.{});
    var live: usize = 0;
    for (lanes, st[0..lanes.len]) |l, *s| live += @intFromBool(try s.submit(l, chunk));
    while (live > 0) {
        for (lanes, st[0..lanes.len]) |l, *s| {
            if (!s.flight) continue;
            try s.landed(l);
            if (!try s.submit(l, chunk)) live -= 1;
        }
    }
}
