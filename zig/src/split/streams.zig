//! Several sequences through one split at once. Each sequence has its own `Pipeline`, over its own two slots of one
//! window, its own stage on each side (a stage owns the state of its layers for one sequence) and the one device. The
//! device's stream runs the steps in the order they were queued, so after the host has handed the device sequence A's
//! hidden row it works on sequence B's while the device works on A's: the two sides stay busy and a token costs the
//! slower side, not the sum of both. Per round the schedule is: collect A, submit A's next step, collect B, submit B's
//! next step, so a sequence's next step is queued as soon as its token is known.
const std = @import("std");
const pipeline = @import("pipeline.zig");

const Pipeline = pipeline.Pipeline;

pub const max_streams = 8;

/// One sequence: a prompt, and the greedy tokens to take after it.
pub const Work = struct { prompt: []const u32, out: []u32 };

const State = struct {
    at: usize = 0,
    produced: usize = 0,
    token: u32 = 0,
    flight: bool = false,
    chunk: usize = 0,
    one: [1]u32 = undefined,

    /// Queues the sequence's next step, if it has one; false when it is done.
    fn submit(s: *State, p: *Pipeline, w: Work, chunk: usize) pipeline.Error!bool {
        if (s.at < w.prompt.len) {
            s.chunk = @min(chunk, w.prompt.len - s.at);
            try p.submit(w.prompt[s.at..][0..s.chunk]);
        } else if (s.produced < w.out.len) {
            w.out[s.produced] = s.token;
            s.produced += 1;
            s.chunk = 0;
            s.one[0] = s.token;
            try p.submit(&s.one);
        } else return false;
        s.flight = true;
        return true;
    }

    fn landed(s: *State, token: u32) void {
        s.at += s.chunk;
        s.token = token;
        s.flight = false;
    }
};

/// Runs every work item through its pipeline: the prompt in steps of up to `chunk` positions, then `out.len` greedy
/// tokens, the sequence `toy.generateChunked` makes for each alone. The pipelines share a device; their windows must not
/// overlap (`Config.window_offset`). On an error every pipeline is released so the device's stream drains; the caller resets them.
pub fn generate(pipes: []const *Pipeline, works: []const Work, chunk: usize) pipeline.Error!void {
    std.debug.assert(pipes.len == works.len and pipes.len <= max_streams and chunk >= 1);
    run(pipes, works, chunk) catch |err| {
        for (pipes) |p| p.release();
        return err;
    };
}

fn run(pipes: []const *Pipeline, works: []const Work, chunk: usize) pipeline.Error!void {
    var st: [max_streams]State = @splat(.{});
    var live: usize = 0;
    for (pipes, works, st[0..pipes.len]) |p, w, *s| live += @intFromBool(try s.submit(p, w, chunk));
    while (live > 0) {
        for (pipes, works, st[0..pipes.len]) |p, w, *s| {
            if (!s.flight) continue;
            s.landed(try p.collect());
            if (!try s.submit(p, w, chunk)) live -= 1;
        }
    }
}
