//! A layer range of the HIP Qwen3.5 / Qwen3.6 engine as the external stage of a split model (`split.pipeline`).
//!
//! The pipeline's device stream is [wait on the input slot] [this stage's launches] [release write of the output slot];
//! `run` only records the middle: it never synchronizes. A window's host pointer is its GPU address (hip/offload.zig),
//! so the kernels read the input slot's payload and write the output slot's payload in place. A step carries `rows`
//! consecutive positions of one sequence (1: a decode step; more: a prompt chunk):
//!
//!   layer 0 in the range:  embed the `rows` u32 tokens at the input payload, run the layers, then
//!   otherwise:             take `rows` hidden rows at the input payload (f32; per row 16-byte stamped groups under
//!                          `metal_first`: one consume launch a row, each with its own status word),
//!   range ends the model:  final norm, head, greedy argmax of the LAST row: one u32 token lands in the output payload;
//!   otherwise:             the residual sums after the range's last layer, before the next input norm, land in the
//!                          output payload as `rows` f32 rows.
//!
//! One row runs the engine's decode round (`Engine.stage`), more rows of one sequence its prompt span
//! (`Engine.spanStage`, the prompt kernels), so a prompt chunk costs one forward and one hand-off pair for all its rows.
//! A stage's KV pages and linear state live only on the stage that owns the layer. All are recorded eagerly, not
//! replayed from a captured graph: a graph would bake in the stamp's sequence number and the window addresses
//! [INFERENCE: patchable graph edges are the follow-up once the card's launch cost is measured].
//!
//! A stage serves `Options.slots` sequences, slot i = sequence i of the pipeline's lane: its own caches and position.
//! A step names its slots (`StepIo.slots`): one slot, and its `rows` are consecutive positions of that sequence (a prompt
//! chunk or one decode position); `rows` slots, and row i is the next position of the sequence `slots[i]` at its own
//! position (a decode batch). A batch is ONE decode round over `rows` windows of different caches
//! (`Engine.stageRows`): the weights are read once for all of them, each row's hand-off is its own consume launch and
//! its head writes a u32 token a row (rows * 4 bytes in the output payload). Only the named slots advance, by one.
//!
//! Several lanes over one engine: every lane has its own `HipLayerStage` (own caches, status words and plan buffers),
//! their steps are recorded on the engine's one stream in the order the pipelines queue them. Invariant the stage
//! relies on: a lane has at most `Options.plans - 1` earlier steps recorded and not collected when it records another
//! (a pipeline collects a step's tokens before it queues the lane's next step, so the default of two is slack), which is
//! what keeps a plan buffer's pinned words from being refilled before the stream's copy of them ran. Open the engine
//! with `batch_rows >= slots` and `prompt_rows >= max_rows`; the pool holds the caches of every slot of every stage.

const std = @import("std");
const hip = @import("hip");
const split = @import("split");
const Engine = @import("engine/engine.zig").Engine;
const state = @import("forward/state.zig");
const plan = @import("forward/plan.zig");
const fwd = @import("forward/forward.zig");
const stage_steps = @import("engine/stage.zig");

const pipeline = split.pipeline;
const wire = split.wire;

pub const Error = error{ BadStep, NotSized, TooManySlots };

pub const Options = struct {
    /// The most rows one step carries: the pipeline's `Config.max_rows`; at most the engine's `prompt_rows`.
    max_rows: usize = 1,
    /// Decode plan buffers taken in turn (see the invariant above).
    plans: usize = 2,
    /// Sequences the stage serves (the lane's); at most `pipeline.max_slots` and the engine's `batch_rows`.
    slots: usize = 1,
};

/// One sequence's state: its caches and the next position they take.
pub const Seq = struct { caches: state.Caches, pos: usize = 0 };

pub const HipLayerStage = struct {
    e: *Engine,
    /// One per slot; a slot's position advances by the rows a step of its own gives it (a batch: one).
    seqs: []Seq,
    max_rows: usize,
    /// One u32 a row: what the stamped consume kernel stores for that row of the last step (0: every group arrived,
    /// 1: a stamp stayed stale).
    status: hip.DeviceBuffer,
    /// The rows of the last recorded step that were stamped (0: it was not).
    stamped_rows: usize = 0,
    /// The decode steps' device plans, used in turn.
    plans: []plan.Buffer,
    turn: usize = 0,
    /// The head's logits row of the last recorded step (a range with the head), valid until the next step of any stage
    /// of the engine is recorded.
    logits: ?hip.ops.Tensor = null,

    /// A stage over the sized engine `e` for `opts.slots` sequences of up to `positions` positions each (at most the
    /// engine's capacity).
    pub fn init(e: *Engine, positions: usize, opts: Options) !HipLayerStage {
        if (!e.sized) return error.NotSized;
        if (opts.max_rows == 0 or opts.max_rows > @min(e.o.prompt_rows, pipeline.max_rows_limit) or opts.plans == 0) return error.BadStep;
        if (opts.slots == 0 or opts.slots > @min(pipeline.max_slots, e.o.batch_rows)) return error.TooManySlots;
        const seqs = try e.gpa.alloc(Seq, opts.slots);
        errdefer e.gpa.free(seqs);
        var opened: usize = 0;
        errdefer for (seqs[0..opened]) |*q| q.caches.deinit(e.gpa);
        while (opened < seqs.len) : (opened += 1) seqs[opened] = .{ .caches = try e.newCaches(positions) };
        var status = try hip.DeviceBuffer.alloc(&e.driver, 4 * opts.max_rows);
        errdefer status.free();
        try status.fill8(0);
        const plans = try e.gpa.alloc(plan.Buffer, opts.plans);
        errdefer e.gpa.free(plans);
        var made: usize = 0;
        errdefer for (plans[0..made]) |*b| b.deinit();
        while (made < plans.len) : (made += 1) plans[made] = try e.newPlan();
        return .{ .e = e, .seqs = seqs, .max_rows = opts.max_rows, .status = status, .plans = plans };
    }

    /// Waits for the stream, then frees the caches.
    pub fn deinit(s: *HipLayerStage) void {
        s.e.drain();
        for (s.plans) |*b| b.deinit();
        s.e.gpa.free(s.plans);
        s.status.free();
        for (s.seqs) |*q| q.caches.deinit(s.e.gpa);
        s.e.gpa.free(s.seqs);
    }

    /// The next position slot `slot`'s caches take: the positions its sequence has run.
    pub fn position(s: *const HipLayerStage, slot: usize) usize {
        return s.seqs[slot].pos;
    }

    /// A new sequence in `slot`: zeroed linear state, the first position again. The stream is waited for first.
    pub fn resetSlot(s: *HipLayerStage, slot: usize) !void {
        s.e.drain();
        const q = &s.seqs[slot];
        const total = q.caches.total;
        q.caches.deinit(s.e.gpa);
        q.caches = try s.e.newCaches(total);
        q.pos = 0;
    }

    /// Every slot a new sequence, and the plan buffers and status words as new.
    pub fn reset(s: *HipLayerStage) !void {
        for (0..s.seqs.len) |slot| try s.resetSlot(slot);
        s.stamped_rows = 0;
        s.turn = 0;
        try s.status.fill8(0);
    }

    pub fn external(s: *HipLayerStage) pipeline.ExternalStage {
        return .{ .ptr = s, .run = run };
    }

    /// Row counts whose kernels differ: every count the small-row kernels are picked by, then each power of two with its
    /// neighbours (the tile and chunk boundaries), then `max_rows`. At least 2 (one row is the decode step).
    fn ladder(max_rows: usize, out: *[64]usize) []const usize {
        var n: usize = 0;
        var cand: usize = 2;
        while (cand <= @min(max_rows, 8)) : (cand += 1) {
            out[n] = cand;
            n += 1;
        }
        var p: usize = 16;
        while (p / 2 < max_rows) : (p *= 2) {
            for ([3]usize{ p - 1, p, p + 1 }) |r| {
                if (r > 8 and r <= max_rows and out[n - 1] < r) {
                    out[n] = r;
                    n += 1;
                }
            }
        }
        if (max_rows >= 2 and (n == 0 or out[n - 1] != max_rows)) {
            out[n] = max_rows;
            n += 1;
        }
        return out[0..n];
    }

    /// Pays the first launches' costs before the pipeline queues its first wait on the window: a kernel that needs more
    /// scratch than the queue holds makes the runtime wait for the queue to drain, and a queue that is waiting on the
    /// window never does. So every kernel and shape a step can take runs once here: the decode step, the decode
    /// batches of 2 to `slots` rows and the prompt span at each row count of `ladder` up to `max_rows`, plain and stamped (not stamped for the first stage), at
    /// the start of the caches and (plain) at their far end, with the head where the range has it; then the stage
    /// starts over.
    pub fn warm(s: *HipLayerStage) !void {
        var warm_rows: [pipeline.max_slots]stage_steps.Row = undefined;
        const e = s.e;
        const m = e.model();
        const embeds = m.range.embeds();
        const heads = m.headed();
        const hidden_bytes = 4 * m.spec.hidden;
        const in_bytes = wire.stampedRowsLen(s.max_rows, hidden_bytes);
        var rows = try hip.DeviceBuffer.alloc(&e.driver, in_bytes + @max(s.max_rows * hidden_bytes, 64));
        defer rows.free();
        try rows.fill8(0);
        const at = rows.base();
        const out_at = at + in_bytes;
        var lad: [64]usize = undefined;
        const counts = ladder(s.max_rows, &lad);
        const caches = &s.seqs[0].caches;
        const far = caches.total;
        const batches = s.seqs.len;
        for ([2]bool{ false, true }) |stamped| {
            if (embeds and stamped) continue;
            const hand: fwd.Handoff = .{
                .src = if (embeds) 0 else at,
                .stamp = if (stamped) .{ .seq = 0, .status = s.status.base(), .retries = 1 } else null,
                .dst = if (heads) 0 else out_at,
            };
            const argmax = if (heads) out_at else 0;
            const tokens = if (embeds) at else 0;
            _ = try e.stage(&s.plans[0], caches, 0, tokens, hand, argmax);
            // a decode batch of every row count from 2 to the slots (rows from distinct sequences at their own positions)
            for (2..batches + 1) |n| _ = try e.stageRows(&s.plans[0], s.batchRows(n, 0, &warm_rows), tokens, hand, argmax);
            for (counts) |n| _ = try e.spanStage(caches, tokens, n, 0, hand, argmax);
            if (stamped) continue;
            // and the far end of the caches: the longest attention walk, one row and the widest batch
            _ = try e.stage(&s.plans[0], caches, far - 1, tokens, hand, argmax);
            if (batches > 1) _ = try e.stageRows(&s.plans[0], s.batchRows(batches, far - 1, &warm_rows), tokens, hand, argmax);
            for (counts) |n| if (n == counts[0] or n == s.max_rows) {
                if (far >= n) _ = try e.spanStage(caches, tokens, n, far - n, hand, argmax);
            };
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

    /// The consume kernel's verdict on the last stamped input: 0 when every group of every row was current. Reads device
    /// memory with a blocking copy: call it with the stream idle (after the pipeline's step returned).
    pub fn stampStatus(s: *const HipLayerStage) !u32 {
        if (s.stamped_rows == 0) return 0;
        var words: [4 * pipeline.max_rows_limit]u8 = undefined;
        const bytes = words[0 .. 4 * s.stamped_rows];
        try s.status.download(0, bytes);
        var worst: u32 = 0;
        for (0..s.stamped_rows) |r| worst = @max(worst, std.mem.readInt(u32, bytes[4 * r ..][0..4], .little));
        return worst;
    }

    fn run(ptr: *anyopaque, io: pipeline.StepIo) anyerror!void {
        const s: *HipLayerStage = @ptrCast(@alignCast(ptr));
        return s.record(io);
    }

    /// The first `n` slots as the rows of a decode round, each at position `pos`.
    fn batchRows(s: *HipLayerStage, n: usize, pos: usize, buf: *[pipeline.max_slots]stage_steps.Row) []const stage_steps.Row {
        for (buf[0..n], s.seqs[0..n]) |*r, *q| r.* = .{ .caches = &q.caches, .pos = pos };
        return buf[0..n];
    }

    /// Records one step on the engine's stream.
    pub fn record(s: *HipLayerStage, io: pipeline.StepIo) !void {
        const e = s.e;
        const m = e.model();
        const embeds = m.range.embeds();
        const heads = m.headed();
        const hidden_bytes = 4 * m.spec.hidden;
        const rows_n: usize = io.rows;
        if (io.hidden_bytes != hidden_bytes or rows_n == 0 or rows_n > s.max_rows) return error.BadStep;
        // one slot: `rows` positions of that sequence; else a decode batch, a row each of `rows` different sequences
        const batch = io.slots.len > 1;
        if (io.slots.len == 0 or (batch and (io.slots.len != rows_n or rows_n > e.o.batch_rows))) return error.BadStep;
        for (io.slots, 0..) |slot, i| {
            if (slot >= s.seqs.len) return error.BadStep;
            for (io.slots[0..i]) |other| if (other == slot) return error.BadStep;
        }
        // a token is not a stamped row, and every other stage has hidden rows to read
        if (embeds and io.stamped) return error.BadStep;
        const in = io.in.data();
        const out = io.out.data();
        if (embeds and in.len < 4 * rows_n) return error.BadStep;
        if (!embeds and in.len < if (io.stamped) wire.stampedRowsLen(rows_n, hidden_bytes) else rows_n * hidden_bytes) return error.BadStep;
        // a head writes one token for a single slot, one a row for a batch
        if (out.len < if (heads) (if (batch) 4 * rows_n else 4) else rows_n * hidden_bytes) return error.BadStep;
        const advance: usize = if (batch) 1 else rows_n;
        for (io.slots) |slot| if (s.seqs[slot].pos + advance > s.seqs[slot].caches.total) return error.ContextFull;
        const in_at = @intFromPtr(in.ptr);
        const out_at = @intFromPtr(out.ptr);
        const hand: fwd.Handoff = .{
            .src = if (embeds) 0 else in_at,
            .stamp = if (io.stamped) .{ .seq = io.seq, .status = s.status.base(), .retries = io.retries } else null,
            .dst = if (heads) 0 else out_at,
        };
        const tokens = if (embeds) in_at else 0;
        const argmax = if (heads) out_at else 0;
        const first = &s.seqs[io.slots[0]];
        if (batch) {
            var seqs: [pipeline.max_slots]stage_steps.Row = undefined;
            for (io.slots, seqs[0..io.slots.len]) |slot, *r| r.* = .{ .caches = &s.seqs[slot].caches, .pos = s.seqs[slot].pos };
            s.logits = try e.stageRows(&s.plans[s.turn], seqs[0..io.slots.len], tokens, hand, argmax);
            s.turn = (s.turn + 1) % s.plans.len;
        } else if (rows_n == 1) {
            s.logits = try e.stage(&s.plans[s.turn], &first.caches, first.pos, tokens, hand, argmax);
            s.turn = (s.turn + 1) % s.plans.len;
        } else {
            s.logits = try e.spanStage(&first.caches, tokens, rows_n, first.pos, hand, argmax);
        }
        s.stamped_rows = if (io.stamped) rows_n else 0;
        for (io.slots) |slot| s.seqs[slot].pos += advance;
    }
};

test "split stage: warm-up row counts cover the small counts, every tile boundary and the maximum" {
    var buf: [64]usize = undefined;
    try std.testing.expectEqual(@as(usize, 0), HipLayerStage.ladder(1, &buf).len);
    try std.testing.expectEqualSlices(usize, &.{2}, HipLayerStage.ladder(2, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4, 5, 6, 7, 8 }, HipLayerStage.ladder(8, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4, 5, 6, 7, 8, 9 }, HipLayerStage.ladder(9, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4, 5, 6, 7, 8, 15, 16 }, HipLayerStage.ladder(16, &buf));
    const full = HipLayerStage.ladder(256, &buf);
    try std.testing.expectEqual(@as(usize, 256), full[full.len - 1]);
    for (full[1..], 0..) |r, i| try std.testing.expect(r > full[i]);
    for ([_]usize{ 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255 }) |r| try std.testing.expect(std.mem.indexOfScalar(usize, full, r) != null);
}
