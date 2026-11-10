//! A layer range of the HIP Qwen3.5 / Qwen3.6 engine as the external stage of a split model (`split.pipeline`).
//!
//! The pipeline's device stream is [wait on the input slot] [this stage's launches] [release write of the output slot];
//! `run` only records the middle: it never synchronizes. A window's host pointer is its GPU address (hip/offload.zig),
//! so the kernels read the input slot's payload and write the output slot's payload in place:
//!
//!   layer 0 in the range:  embed the u32 token at the input payload, run the layers, then
//!   otherwise:             take the hidden row at the input payload (f32; 16-byte stamped groups under level B),
//!   range ends the model:  final norm, head, greedy argmax: the u32 token lands in the output payload;
//!   otherwise:             the residual sum after the range's last layer, before the next input norm, lands in the
//!                          output payload as f32.
//!
//! Prompts run the same one-token step once per position (batch prefill across the window is not in this stage): a
//! stage's KV pages and linear state live only on the stage that owns the layer, and advance one position a `run`.
//! The step is recorded eagerly, not replayed from a captured graph: a graph would bake in the stamp's sequence
//! number and the window addresses. [INFERENCE] a graph of the step with patchable edges is the follow-up once the
//! card's launch cost is measured.

const std = @import("std");
const hip = @import("hip");
const split = @import("split");
const Engine = @import("engine/engine.zig").Engine;
const state = @import("forward/state.zig");
const fwd = @import("forward/forward.zig");

const pipeline = split.pipeline;

pub const Error = error{ BadStep, NotSized };

pub const HipLayerStage = struct {
    e: *Engine,
    caches: state.Caches,
    /// The next position this stage's caches take: one more after every recorded step.
    pos: usize = 0,
    /// The u32 the stamped consume kernel stores in (0: every group arrived, 1: a stamp stayed stale).
    status: hip.DeviceBuffer,
    /// The head's logits row of the last recorded step (a range with the head), valid until the next step is recorded.
    logits: ?hip.ops.Tensor = null,

    /// A stage over the sized engine `e` for sequences of up to `positions` positions (at most the engine's capacity).
    /// The engine should be opened with `batch_rows = 1`: a stage's step is one row.
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

    /// Pays the first launches' costs before the pipeline queues its first wait on the window: a kernel that needs more
    /// scratch than the queue holds makes the runtime wait for the queue to drain, and a queue that is waiting on the
    /// window never does. One step runs on zero rows, plain and then stamped, and the stage starts over.
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
            _ = try e.stage(&s.caches, 0, if (embeds) at else 0, hand, if (heads) out_at else 0);
        }
        try s.reset();
    }

    /// The last step's logits as f32, whatever kind the head produced: for fidelity checks, with the stream idle (after
    /// the pipeline's step returned). `out.len` is the vocabulary.
    pub fn readLogits(s: *const HipLayerStage, out: []f32) !void {
        const y = s.logits orelse return error.BadStep;
        const width: usize = if (y.kind == .f32) 4 else 2;
        const raw = try s.e.gpa.alloc(u8, out.len * width);
        defer s.e.gpa.free(raw);
        try hip.runtime.check(s.e.driver.api.hipMemcpyDtoH(raw.ptr, @ptrFromInt(y.ptr), raw.len));
        for (out, 0..) |*o, i| o.* = switch (y.kind) {
            .f32 => @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little)),
            .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[2 * i ..][0..2], .little)))),
            .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, raw[2 * i ..][0..2], .little)) << 16),
        };
    }

    /// The consume kernel's verdict on the last stamped input: 0 when every group's stamp was current. Reads device
    /// memory with a blocking copy: call it with the stream idle (after the pipeline's step returned).
    pub fn stampStatus(s: *const HipLayerStage) !u32 {
        var word: [4]u8 = undefined;
        try s.status.download(0, &word);
        return std.mem.readInt(u32, &word, .little);
    }

    fn run(ptr: *anyopaque, io: pipeline.StepIo) anyerror!void {
        const s: *HipLayerStage = @ptrCast(@alignCast(ptr));
        return s.record(io);
    }

    /// Records one step on the engine's stream.
    pub fn record(s: *HipLayerStage, io: pipeline.StepIo) !void {
        const e = s.e;
        const m = e.model();
        const embeds = m.range.embeds();
        const heads = m.headed();
        const hidden_bytes = 4 * m.spec.hidden;
        if (io.hidden_bytes != hidden_bytes) return error.BadStep;
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
        s.logits = try e.stage(&s.caches, s.pos, if (embeds) in_at else 0, hand, if (heads) out_at else 0);
        s.pos += 1;
    }
};
