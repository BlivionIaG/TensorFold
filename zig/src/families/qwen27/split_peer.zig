//! One layer range of a dense Qwen model as a stage of the split pipeline, on this Mac's GPU: `LayerPeer` is the host
//! stage (`split.pipeline.Peer`, driven through `metal.Relay`), `External` the same range run synchronously from the
//! mock device's thread. A step carries one position (a decode step) or up to `max_rows` consecutive positions of one
//! sequence (a prompt chunk, run with the prompt kernels as `Session.promptChunk` runs it). The state of a layer lives
//! with the stage that owns it, one stage a sequence; stages of one model share its frame, see `Serial`.
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

    /// `model` was loaded for a range (`Model.loadRange`); the stage is one sequence's state of it. `warm` keeps the
    /// first steps' shader compiles out of a pipeline deadline.
    pub fn init(a: std.mem.Allocator, model: *Model, capacity: u32, serial: *Serial) !*Stage {
        const range = model.range orelse return error.BadLayerRange;
        const s = try a.create(Stage);
        errdefer a.destroy(s);
        const bytes = model.config.hidden * 4 * @min(model.frame.capacity, max_rows);
        const opts = mtl.ResourceOptions.shared;
        s.* = .{
            .allocator = a,
            .model = model,
            .serial = serial,
            .runner = try Runner.init(a, model, 1, capacity),
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

    /// Encode this stage's pass over `input` for the next positions and open it; `finish` waits for it.
    pub fn launch(s: *Stage, input: Input, out: Output) !void {
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
        const ticket = try stage_pass.launch(&s.runner, &s.relay, 0, ids, io);
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

    /// The token a range that ends the model drew after the last row of its step, once `finish` returned: greedy over
    /// the logits, as `Session.greedy`.
    pub fn draw(s: *Stage) !u32 {
        if (!s.range.head(s.model.config.layers)) return error.BadStageInput;
        const logits = s.model.frame.get(.logits).buffer.slice(u16, s.model.config.vocab);
        return session.argmax(logits);
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
        try s.runner.reset(0);
    }

    /// One decode step and one two-row chunk, then `reset`: pipelines compile and weights page in outside a deadline.
    pub fn warm(s: *Stage) !void {
        const row = try s.allocator.alloc(u8, 2 * s.rowBytes());
        defer s.allocator.free(row);
        @memset(row, 0);
        const tokens = [2]u32{ 0, 0 };
        for ([2]usize{ 1, 2 }) |n| {
            if (n > s.model.frame.capacity) break;
            try s.launch(if (s.range.embed()) .{ .tokens = tokens[0..n] } else .{ .rows = row[0 .. n * s.rowBytes()] }, .own);
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
        if (rows == 0 or rows > max_rows or slots.len != 1 or slots[0] != 0) return error.BadStep;
        const head = st.range.head(st.model.config.layers);
        var tokens: [max_rows]u32 = undefined;
        var in: Input = .{ .rows = input[0 .. rows * st.rowBytes()] };
        if (st.range.embed()) {
            for (tokens[0..rows], 0..) |*t, i| t.* = std.mem.readInt(u32, input[4 * i ..][0..4], .little);
            in = .{ .tokens = tokens[0..rows] };
        }
        try st.launch(in, if (head) .own else .{ .window = .{ .offset = try offsetOf(st, out), .stamp = seq } });
        if (head or level == .a) try st.finish();
        if (head) {
            std.mem.writeInt(u32, out[0..4], try st.draw(), .little);
            return 4;
        }
        return rows * st.rowBytes();
    }

    fn finishFn(ptr: *anyopaque, _: u32) anyerror!void {
        return @as(*LayerPeer, @ptrCast(@alignCast(ptr))).stage.finish();
    }
};

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
        if (io.slots.len != 1 or io.slots[0] != 0) return error.BadStep;
        const rows: usize = io.rows;
        if (rows == 0 or rows > max_rows or io.hidden_bytes != st.rowBytes()) return error.BadStep;
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
        try st.launch(in, .own);
        try st.finish();
        if (head) {
            std.mem.writeInt(u32, io.out.data()[0..4], try st.draw(), .little);
        } else @memcpy(io.out.data()[0 .. rows * io.hidden_bytes], st.ownRows(rows));
    }
};
