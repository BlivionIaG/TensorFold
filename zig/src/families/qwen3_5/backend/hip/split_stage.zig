//! A layer range of the HIP Qwen3.5 / Qwen3.6 engine as the external stage of a split model (`split.pipeline`).

const std = @import("std");
const hip = @import("hip");
const split = hip.split;
const Engine = @import("engine/engine.zig").Engine;
const state = @import("forward/state.zig");
const fwd = @import("forward/forward.zig");

const pipeline = split.pipeline;

pub const Error = error{ BadStep, NotSized };

/// The pipeline's device stream is [wait on the input slot] [`run`'s launches] [release write of the output slot].
pub const HipLayerStage = struct {
    e: *Engine,
    caches: state.Caches,
    /// The next position this stage's caches take: one more after every recorded step.
    pos: usize = 0,
    /// The u32 the stamped consume kernel stores in (0: every group arrived, 1: a stamp stayed stale).
    status: hip.DeviceBuffer,

    /// A stage over the sized engine `e` for sequences of up to `positions`; open the engine with `batch_rows = 1`.
    pub fn init(e: *Engine, positions: usize) !HipLayerStage {
        if (!e.sized) return error.NotSized;
        var caches = try e.newCaches(positions);
        errdefer caches.deinit(e.gpa);
        var status = try hip.DeviceBuffer.alloc(&e.driver, 4);
        errdefer status.free();
        try status.fill8(0);
        return .{ .e = e, .caches = caches, .status = status };
    }

    /// Waits for the stream, then frees the caches.
    pub fn deinit(s: *HipLayerStage) void {
        s.e.drain();
        s.status.free();
        s.caches.deinit(s.e.gpa);
    }

    /// A new sequence: zeroed linear state, the first position again. The stream is waited for first.
    pub fn reset(s: *HipLayerStage) !void {
        s.e.drain();
        const total = s.caches.total;
        s.caches.deinit(s.e.gpa);
        s.caches = try s.e.newCaches(total);
        s.pos = 0;
        try s.status.fill8(0);
    }

    pub fn external(s: *HipLayerStage) pipeline.ExternalStage {
        return .{ .ptr = s, .run = run };
    }

    /// Pays first-launch costs before the pipeline's first window wait (a held queue cannot drain), then resets.
    pub fn warm(s: *HipLayerStage) !void {
        const e = s.e;
        const m = e.model();
        const embeds = m.range.embeds();
        const heads = m.headed();
        const hidden_bytes = 4 * m.spec.hidden;
        var rows = try hip.DeviceBuffer.alloc(&e.driver, split.wire.stampedLen(hidden_bytes) + hidden_bytes + 64);
        defer rows.free();
        try rows.fill8(0);
        const at = rows.base();
        const out_at = at + split.wire.stampedLen(hidden_bytes);
        for ([2]bool{ false, true }) |stamped| {
            if (embeds and stamped) continue;
            const hand: fwd.Handoff = .{
                .src = if (embeds) 0 else at,
                .stamp = if (stamped) .{ .seq = 0, .status = s.status.base(), .retries = 1 } else null,
                .dst = if (heads) 0 else out_at,
            };
            try e.stage(&s.caches, 0, if (embeds) at else 0, hand, if (heads) out_at else 0);
        }
        try s.reset();
    }

    /// The consume kernel's verdict on the last stamped input: 0 when every group's stamp was current (stream idle).
    pub fn stampStatus(s: *const HipLayerStage) !u32 {
        var word: [4]u8 = undefined;
        try s.status.download(0, &word);
        return std.mem.readInt(u32, &word, .little);
    }

    fn run(ptr: *anyopaque, io: pipeline.StepIo) anyerror!void {
        const s: *HipLayerStage = @ptrCast(@alignCast(ptr));
        return s.record(io);
    }

    /// Records one step on the engine's stream; the window's slots are read and written in place by GPU address.
    pub fn record(s: *HipLayerStage, io: pipeline.StepIo) !void {
        const e = s.e;
        const m = e.model();
        const embeds = m.range.embeds();
        const heads = m.headed();
        const hidden_bytes = 4 * m.spec.hidden;
        if (io.hidden_bytes != hidden_bytes or io.rows != 1 or io.slots.len != 1) return error.BadStep;
        // a token is not a stamped row, and every other stage has a hidden row to read
        if (embeds and io.stamped) return error.BadStep;
        const in = io.in.data();
        const out = io.out.data();
        if (embeds and in.len < 4) return error.BadStep;
        if (!embeds and in.len < if (io.stamped) split.wire.stampedLen(hidden_bytes) else hidden_bytes) return error.BadStep;
        if (out.len < if (heads) 4 else hidden_bytes) return error.BadStep;
        const in_at = @intFromPtr(in.ptr);
        const out_at = @intFromPtr(out.ptr);
        const hand: fwd.Handoff = .{
            .src = if (embeds) 0 else in_at,
            .stamp = if (io.stamped) .{ .seq = io.seq, .status = s.status.base(), .retries = io.retries } else null,
            .dst = if (heads) 0 else out_at,
        };
        try e.stage(&s.caches, s.pos, if (embeds) in_at else 0, hand, if (heads) out_at else 0);
        s.pos += 1;
    }
};
