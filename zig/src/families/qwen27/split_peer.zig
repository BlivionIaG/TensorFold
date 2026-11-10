//! One layer range of a dense Qwen model as a stage of the split pipeline, on this Mac's GPU: `LayerPeer` is the host
//! stage (`split.pipeline.Peer`, driven through `metal.Relay`), `externalStage` the same range run synchronously from
//! the mock device's thread. Decode only: every position, prompt included, goes through the pipeline one token at a
//! time (batch prefill across the window is not part of this). The state of a layer lives with the stage that owns it.
const std = @import("std");
const mtl = @import("metal");
const split = @import("split");
const Model = @import("model.zig").Model;
const Runner = @import("decode_round.zig").Runner;
const Rows = @import("stage_rows.zig").Rows;
const stage_pass = @import("stage_pass.zig");
const session = @import("session.zig");
const range_mod = @import("range.zig");

const pipeline = split.pipeline;
const wire = split.wire;

/// How long a stage waits for its command buffer; the pipeline's own deadline is shorter or equal.
const gpu_wait_ms = 10_000;

/// What a step reads: the token a range that embeds takes, or the f32 row (little-endian words) of one that does not.
pub const Input = union(enum) { token: u32, row: []const u8 };
/// Where a range that does not end the model leaves its row: this stage's own buffer, or the window at a byte offset,
/// in stamped groups of `seq`.
pub const Output = union(enum) { own, window: struct { offset: usize, stamp: u32 } };

/// A runner over a model that holds one layer range, with its relay and boundary rows. The model is the caller's.
pub const Stage = struct {
    allocator: std.mem.Allocator,
    model: *Model,
    runner: Runner,
    rows: Rows,
    relay: mtl.Relay,
    window: ?mtl.HostWindow,
    row_in: mtl.Buffer,
    row_out: mtl.Buffer,
    range: range_mod.Range,
    pending: ?mtl.Ticket = null,

    /// `model` was loaded for a range (`Model.loadRange`). The first step compiles and warms pipelines: `warm` keeps
    /// that out of a pipeline deadline.
    pub fn init(a: std.mem.Allocator, model: *Model, capacity: u32) !*Stage {
        const range = model.range orelse return error.BadLayerRange;
        const s = try a.create(Stage);
        errdefer a.destroy(s);
        const row_bytes = model.config.hidden * 4;
        const opts = mtl.ResourceOptions.shared;
        s.* = .{
            .allocator = a,
            .model = model,
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
        s.row_in = try model.device.buffer(row_bytes, opts);
        errdefer s.row_in.deinit();
        s.row_out = try model.device.buffer(row_bytes, opts);
        errdefer s.row_out.deinit();
        s.relay = try mtl.Relay.init(model.device, model.queue);
        return s;
    }

    /// The memory the pipeline's slots live in (16 KiB aligned): a step that leaves its row there writes it through
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

    /// Encode this stage's pass over `input` for the next position and open it; `finish` waits for it.
    pub fn launch(s: *Stage, input: Input, out: Output) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var token: u32 = 0;
        switch (input) {
            .token => |t| {
                if (!s.range.embed()) return error.BadStageInput;
                token = t;
            },
            .row => |row| {
                if (s.range.embed() or row.len != s.rowBytes()) return error.BadHidden;
                @memcpy(s.row_in.contents()[0..row.len], row);
            },
        }
        var io = @import("forward.zig").Io{ .rows = &s.rows, .input = .{ .buffer = s.row_in } };
        if (!s.range.head(s.model.config.layers)) switch (out) {
            .own => io.output = .{ .buffer = s.row_out },
            .window => |w| {
                const win = s.window orelse return error.OutsideWindow;
                if (w.offset + Rows.bytes(s.model.config.hidden, 1, true) > win.len or w.stamp == 0) return error.OutsideWindow;
                io.output = .{ .buffer = win.buffer, .offset = w.offset };
                io.stamp_out = w.stamp;
            },
        };
        const ticket = try stage_pass.launch(&s.runner, &s.relay, 0, token, io);
        s.relay.open(ticket);
        s.pending = ticket;
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

    /// The token a range that ends the model drew, after `finish`: greedy over the logits, as `Session.greedy`.
    pub fn draw(s: *Stage) !u32 {
        if (!s.range.head(s.model.config.layers)) return error.BadStageInput;
        const logits = s.model.frame.get(.logits).buffer.slice(u16, s.model.config.vocab);
        return session.argmax(logits);
    }

    /// The f32 row `Output.own` left, after `finish`.
    pub fn ownRow(s: *Stage) []const u8 {
        return s.row_out.contents()[0..s.rowBytes()];
    }

    /// Back to position 0 with no state, also after an error or a stuck link: a fresh relay once the old one has
    /// drained, a recovered runner and a cleared recurrent state.
    pub fn reset(s: *Stage) !void {
        s.pending = null;
        s.relay.deinit();
        s.relay = try mtl.Relay.init(s.model.device, s.model.queue);
        s.runner.recover();
        try s.runner.reset(0);
    }

    /// One throwaway step, then `reset`: pipelines compile and weights page in outside a pipeline deadline.
    pub fn warm(s: *Stage) !void {
        const row = try s.allocator.alloc(u8, s.rowBytes());
        defer s.allocator.free(row);
        @memset(row, 0);
        try s.launch(if (s.range.embed()) .{ .token = 0 } else .{ .row = row }, .own);
        try s.finish();
        try s.reset();
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

    fn stepFn(ptr: *anyopaque, seq: u32, input: []const u8, out: []u8, level: pipeline.Level) anyerror!usize {
        const st = @as(*LayerPeer, @ptrCast(@alignCast(ptr))).stage;
        const head = st.range.head(st.model.config.layers);
        const in: Input = if (st.range.embed()) .{ .token = std.mem.readInt(u32, input[0..4], .little) } else .{ .row = input };
        try st.launch(in, if (head) .own else .{ .window = .{ .offset = try offsetOf(st, out), .stamp = seq } });
        if (head or level == .a) try st.finish();
        if (head) {
            std.mem.writeInt(u32, out[0..4], try st.draw(), .little);
            return 4;
        }
        return st.rowBytes();
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
        const head = st.range.head(st.model.config.layers);
        var in: Input = undefined;
        if (st.range.embed()) {
            in = .{ .token = std.mem.readInt(u32, io.in.data()[0..4], .little) };
        } else if (io.stamped) {
            const row = io.scratch[0..io.hidden_bytes];
            _ = try wire.readStamped(row, io.in.data(), io.seq, io.retries);
            in = .{ .row = row };
        } else in = .{ .row = io.in.data()[0..h.bytes] };
        try st.launch(in, .own);
        try st.finish();
        if (head) {
            std.mem.writeInt(u32, io.out.data()[0..4], try st.draw(), .little);
        } else @memcpy(io.out.data()[0..st.rowBytes()], st.ownRow());
    }
};
