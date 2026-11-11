//! One layer range of a dense Qwen model as a stage of the split pipeline, on this Mac's GPU: `LayerPeer` is the host
//! stage (`split.pipeline.Peer`, driven through `metal.Relay`), `External` the same range run synchronously from the
//! mock device's thread. A stage serves up to `max_slots` sequences, a slot each (their caches and positions). A step
//! carries one position of one slot (a decode step), up to `max_rows` consecutive positions of one slot (a prompt chunk,
//! run with the prompt kernels as `Session.promptChunk` runs it), or one position of each of several slots (a decode
//! batch, one shared round). The state of a layer lives with the stage that owns it; stages of one model (one a lane)
//! share its frame, see `Serial`.
const std = @import("std");
const mtl = @import("metal");
const split = @import("split");
const Model = @import("model.zig").Model;
const Runner = @import("decode_round.zig").Runner;
const Rows = @import("stage_rows.zig").Rows;
const stage_pass = @import("stage_pass.zig");
const session = @import("session.zig");
const range_mod = @import("range.zig");
const forward = @import("forward.zig");

const pipeline = split.pipeline;
const wire = split.wire;

/// How long a stage waits for its command buffer; the pipeline's own deadline is shorter or equal.
const gpu_wait_ms = 10_000;
/// The most positions a step carries (the pipeline's `max_rows_limit`).
pub const max_rows = pipeline.max_rows_limit;
pub const max_slots = stage_pass.max_slots;

/// What a step reads: the tokens a range that embeds takes, or the f32 rows (little-endian words, `rows * hidden * 4`
/// bytes) of one that does not.
pub const Input = union(enum) { tokens: []const u32, rows: []const u8 };
/// Where a range that does not end the model leaves its rows: this stage's own buffer, or the window at a byte offset,
/// each row on its own stamped groups of `stamp`.
pub const Output = union(enum) { own, window: struct { offset: usize, stamp: u32 } };

/// The stages that share one model share its frame (the rows, positions and scratch of the pass that encodes next), so
/// at most one of their passes may be in flight: a stage waits for the last stage's pass before it encodes its own.
/// One `Serial` a model; every stage of the model holds it. Stages of different models need no ordering.
pub const Serial = struct { last: ?*Stage = null };

/// A runner over a model that holds one layer range, with its relay and boundary rows. The model is the caller's.
pub const Stage = struct {
    allocator: std.mem.Allocator,
    model: *Model,
    serial: *Serial,
    runner: Runner,
    rows: Rows,
    relay: mtl.Relay,
    window: ?mtl.HostWindow,
    row_in: mtl.Buffer,
    row_out: mtl.Buffer,
    range: range_mod.Range,
    pending: ?mtl.Ticket = null,

    /// `model` was loaded for a range (`Model.loadRange`); the stage holds the state of `slots` sequences of it. `warm`
    /// keeps the first steps' shader compiles out of a pipeline deadline.
    pub fn init(a: std.mem.Allocator, model: *Model, capacity: u32, slots: u32, serial: *Serial) !*Stage {
        const range = model.range orelse return error.BadLayerRange;
        if (slots == 0 or slots > max_slots) return error.BadSlots;
        const s = try a.create(Stage);
        errdefer a.destroy(s);
        const bytes = model.config.hidden * 4 * @min(model.frame.capacity, max_rows);
        const opts = mtl.ResourceOptions.shared;
        s.* = .{
            .allocator = a,
            .model = model,
            .serial = serial,
            .runner = try Runner.init(a, model, slots, capacity),
            .rows = undefined,
            .relay = undefined,
            .window = null,
            .row_in = undefined,
            .row_out = undefined,
            .range = range,
        };
        errdefer s.runner.deinit();
        s.rows = try Rows.init(model.device);
        errdefer s.rows.deinit();
        s.row_in = try model.device.buffer(bytes, opts);
        errdefer s.row_in.deinit();
        s.row_out = try model.device.buffer(bytes, opts);
        errdefer s.row_out.deinit();
        s.relay = try mtl.Relay.init(model.device, model.queue);
        return s;
    }

    /// The memory the pipeline's slots live in (16 KiB aligned): a step that leaves its rows there writes them through
    /// this wrap. Replaces an earlier window.
    pub fn bindWindow(s: *Stage, memory: []u8) !void {
        const win = try mtl.HostWindow.wrap(s.model.device, memory.ptr, memory.len);
        if (s.window) |old| old.deinit();
        s.window = win;
    }

    /// The window's memory is about to go: its wrap ends first.
    pub fn unbindWindow(s: *Stage) void {
        if (s.window) |w| w.deinit();
        s.window = null;
    }

    pub fn deinit(s: *Stage) void {
        s.relay.deinit();
        if (s.serial.last == s) s.serial.last = null;
        if (s.window) |w| w.deinit();
        s.row_out.deinit();
        s.row_in.deinit();
        s.rows.deinit();
        s.runner.deinit();
        s.allocator.destroy(s);
    }

    fn rowBytes(s: *const Stage) usize {
        return s.model.config.hidden * 4;
    }

    /// Encode this stage's pass over `input` for the next positions of the sequences `slots` name (one slot: its
    /// consecutive positions; as many slots as rows: a decode batch) and open it; `finish` waits for it.
    pub fn launch(s: *Stage, input: Input, out: Output, slots: []const u8) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var zeros: [max_rows]u32 = @splat(0);
        var n: usize = undefined;
        var ids: []const u32 = undefined;
        switch (input) {
            .tokens => |t| {
                if (!s.range.embed() or t.len > max_rows) return error.BadStageInput;
                n = t.len;
                ids = t;
            },
            .rows => |rows| {
                n = rows.len / s.rowBytes();
                if (s.range.embed() or n == 0 or n > max_rows or rows.len != n * s.rowBytes() or n > s.model.frame.capacity) return error.BadHidden;
                @memcpy(s.row_in.contents()[0..rows.len], rows);
                ids = zeros[0..n];
            },
        }
        var io = forward.Io{ .rows = &s.rows, .input = .{ .buffer = s.row_in } };
        if (!s.range.head(s.model.config.layers)) switch (out) {
            .own => io.output = .{ .buffer = s.row_out },
            .window => |w| {
                const win = s.window orelse return error.OutsideWindow;
                if (w.offset + Rows.bytes(s.model.config.hidden, n, true) > win.len or w.stamp == 0) return error.OutsideWindow;
                io.output = .{ .buffer = win.buffer, .offset = w.offset };
                io.stamp_out = w.stamp;
            },
        };
        if (s.serial.last) |other| if (other != s) try other.finish();
        const ticket = try stage_pass.launch(&s.runner, &s.relay, slots, ids, io);
        s.relay.open(ticket);
        s.pending = ticket;
        s.serial.last = s;
    }

    /// Wait for the pass `launch` opened: `error.Timeout`, or `error.BadStage` when the GPU reported a failure.
    pub fn finish(s: *Stage) !void {
        const t = s.pending orelse return;
        s.pending = null;
        if (!s.relay.waitDone(t, gpu_wait_ms)) return error.Timeout;
        if (s.relay.failed(t) != null) {
            s.runner.failed = true;
            return error.BadStage;
        }
    }

    /// The tokens a range that ends the model drew, once `finish` returned: greedy over the logits rows, as
    /// `Session.greedy`. One for a single slot's step (after its last row), else one a row.
    pub fn draw(s: *Stage, out: []u32) !void {
        if (!s.range.head(s.model.config.layers) or out.len == 0) return error.BadStageInput;
        const vocab = s.model.config.vocab;
        const logits = s.model.frame.get(.logits).buffer.slice(u16, out.len * vocab);
        for (out, 0..) |*o, i| o.* = try session.argmax(logits[i * vocab ..][0..vocab]);
    }

    /// The `rows` f32 rows `Output.own` left, after `finish`.
    pub fn ownRows(s: *Stage, rows: usize) []const u8 {
        return s.row_out.contents()[0 .. rows * s.rowBytes()];
    }

    /// Back to position 0 with no state, also after an error or a stuck link: a fresh relay once the old one has
    /// drained, a recovered runner and a cleared recurrent state.
    pub fn reset(s: *Stage) !void {
        s.pending = null;
        if (s.serial.last == s) s.serial.last = null;
        s.relay.deinit();
        s.relay = try mtl.Relay.init(s.model.device, s.model.queue);
        s.runner.recover();
        for (0..s.runner.ready) |slot| try s.runner.reset(@intCast(slot));
    }

    /// A decode step, a two-row chunk and (with two slots) a two-row batch, then `reset`: pipelines compile and weights
    /// page in outside a deadline.
    pub fn warm(s: *Stage) !void {
        const row = try s.allocator.alloc(u8, 2 * s.rowBytes());
        defer s.allocator.free(row);
        @memset(row, 0);
        const tokens = [2]u32{ 0, 0 };
        const slots = [2]u8{ 0, 1 };
        for ([3]usize{ 1, 2, 3 }) |kind| {
            if (kind > 1 and s.model.frame.capacity < 2) break;
            if (kind == 3 and s.runner.ready < 2) break;
            const n: usize = if (kind == 1) 1 else 2;
            try s.launch(if (s.range.embed()) .{ .tokens = tokens[0..n] } else .{ .rows = row[0 .. n * s.rowBytes()] }, .own, if (kind == 3) slots[0..2] else slots[0..1]);
            try s.finish();
            try s.reset();
        }
    }
};

/// The host stage of the pipeline: `step` encodes the range and, at level A or when it draws the token, waits.
pub const LayerPeer = struct {
    stage: *Stage,

    pub fn peer(p: *LayerPeer) pipeline.Peer {
        return .{ .ptr = p, .step = stepFn, .finish = finishFn };
    }

    fn offsetOf(s: *const Stage, bytes: []const u8) !usize {
        const win = s.window orelse return error.OutsideWindow;
        const base = @intFromPtr(win.ptr);
        const at = @intFromPtr(bytes.ptr);
        if (at < base or at + bytes.len > base + win.len) return error.OutsideWindow;
        return at - base;
    }

    fn stepFn(ptr: *anyopaque, seq: u32, slots: []const u8, rows: u32, input: []const u8, out: []u8, level: pipeline.Level) anyerror!usize {
        const st = @as(*LayerPeer, @ptrCast(@alignCast(ptr))).stage;
        if (rows == 0 or rows > max_rows or slots.len == 0 or (slots.len > 1 and slots.len != rows)) return error.BadStep;
        const head = st.range.head(st.model.config.layers);
        var tokens: [max_rows]u32 = undefined;
        var in: Input = .{ .rows = input[0 .. rows * st.rowBytes()] };
        if (st.range.embed()) {
            for (tokens[0..rows], 0..) |*t, i| t.* = std.mem.readInt(u32, input[4 * i ..][0..4], .little);
            in = .{ .tokens = tokens[0..rows] };
        }
        try st.launch(in, if (head) .own else .{ .window = .{ .offset = try offsetOf(st, out), .stamp = seq } }, slots);
        if (head or level == .a) try st.finish();
        if (head) return try writeTokens(st, slots, out);
        return rows * st.rowBytes();
    }

    fn finishFn(ptr: *anyopaque, _: u32) anyerror!void {
        return @as(*LayerPeer, @ptrCast(@alignCast(ptr))).stage.finish();
    }
};

/// The head's tokens into `out` as u32 words: one for a single slot, else one a row. Returns the bytes written.
fn writeTokens(st: *Stage, slots: []const u8, out: []u8) !usize {
    var tokens: [max_slots]u32 = undefined;
    const n = slots.len;
    try st.draw(tokens[0..n]);
    for (tokens[0..n], 0..) |t, i| std.mem.writeInt(u32, out[4 * i ..][0..4], t, .little);
    return 4 * n;
}

/// The device stage of the pipeline: the same range run on this GPU from the mock device's thread, a step at a time.
pub const External = struct {
    stage: *Stage,

    pub fn external(e: *External) pipeline.ExternalStage {
        return .{ .ptr = e, .run = runFn };
    }

    fn runFn(ptr: *anyopaque, io: pipeline.StepIo) anyerror!void {
        const st = @as(*External, @ptrCast(@alignCast(ptr))).stage;
        const h = io.in.header().*;
        if (h.status != 0 or h.kind == @backingInt(wire.Kind.abort)) return error.PeerLost;
        const rows: usize = io.rows;
        if (rows == 0 or rows > max_rows or io.hidden_bytes != st.rowBytes() or io.slots.len == 0 or (io.slots.len > 1 and io.slots.len != rows)) return error.BadStep;
        const head = st.range.head(st.model.config.layers);
        var tokens: [max_rows]u32 = undefined;
        var in: Input = undefined;
        if (st.range.embed()) {
            for (tokens[0..rows], 0..) |*t, i| t.* = std.mem.readInt(u32, io.in.data()[4 * i ..][0..4], .little);
            in = .{ .tokens = tokens[0..rows] };
        } else if (io.stamped) {
            const plain = io.scratch[0 .. rows * io.hidden_bytes];
            _ = try wire.readStampedRows(plain, io.in.data(), rows, io.hidden_bytes, io.seq, io.retries);
            in = .{ .rows = plain };
        } else in = .{ .rows = io.in.data()[0 .. rows * io.hidden_bytes] };
        try st.launch(in, .own, io.slots);
        try st.finish();
        if (head) {
            _ = try writeTokens(st, io.slots, io.out.data());
        } else @memcpy(io.out.data()[0 .. rows * io.hidden_bytes], st.ownRows(rows));
    }
};
