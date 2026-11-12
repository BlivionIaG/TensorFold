//! A two-stage layer pipeline across an external device and the host's own GPU; a step is a position or a prompt chunk.
const std = @import("std");
const device = @import("device.zig");
const wire = @import("wire.zig");

pub const Order = enum {
    /// The device runs the embedding and layers `[0, k)`, the host the rest and the head; hidden rows go up.
    external_first,
    /// The host runs the embedding and layers `[0, k)`, the device the rest and the head; hidden rows go down.
    metal_first,
    /// The device runs everything; the host only supplies and reads tokens.
    external_only,
};

/// How the host-to-device hidden row is protected in `metal_first`.
pub const Level = enum {
    /// The sequence word is stored after the producer finished.
    a,
    /// The sequence word may be stored while the data is still landing; every 16-byte group carries the sequence stamp.
    b,
};

pub const Error = device.Error || wire.Error || error{ PeerLost, BadStep };

/// What a device stage is given for one step: its input and output slots in the window and how the input is protected.
pub const StepIo = struct {
    in: wire.Slot,
    out: wire.Slot,
    seq: u32,
    /// Positions in the step: a stage that embeds reads that many u32 tokens, any other that many hidden rows.
    rows: u32 = 1,
    /// The step's sequences: one slot takes `rows` consecutive positions of it, else row i is slot i's next position.
    slots: []const u8 = &.{0},
    /// The input rows are 16-byte stamped groups (`metal_first`): the stage must see every group's stamp equal `seq`.
    stamped: bool,
    retries: u32,
    /// Host memory a CPU stage may stage the stamped rows in, `rows * hidden_bytes` long.
    scratch: []u8,
    hidden_bytes: usize,
};

/// The stage the device runs in its stream: records launches that read `io.in` and write `io.out`.
pub const ExternalStage = struct {
    ptr: *anyopaque,
    run: *const fn (ptr: *anyopaque, io: StepIo) anyerror!void,
};

/// The host's stage on its own GPU: `step` takes `rows` positions of the sequences `slots` names, as in `StepIo`.
pub const Peer = struct {
    ptr: *anyopaque,
    step: *const fn (ptr: *anyopaque, seq: u32, slots: []const u8, rows: u32, input: []const u8, out: []u8, level: Level) anyerror!usize,
    finish: *const fn (ptr: *anyopaque, seq: u32) anyerror!void,
};

pub const Config = struct {
    order: Order,
    level: Level = .a,
    /// Bytes of one hidden row as the stages exchange it.
    hidden_bytes: usize,
    /// The most positions one step carries; 1 is decode only.
    max_rows: u32 = 1,
    /// Where this pipeline's two slots start in the window: one `Pipeline` a sequence, `slotBytes * 2` apart.
    window_offset: usize = 0,
    /// Bound on every wait, host and device side; 0 is unbounded.
    timeout_ns: u64 = 5_000_000_000,
    /// Reads of an unarrived group before level B gives up on a reply.
    stamp_retries: u32 = 20_000_000,
};

/// Where a pipeline's steps spent their time, summed: recording the device's work, the host stage, waiting for a reply.
pub const Phases = struct { queue_ns: u64 = 0, peer_ns: u64 = 0, finish_ns: u64 = 0, wait_ns: u64 = 0 };
pub const max_rows_limit = 256;
/// The most sequences one decode batch holds.
pub const max_slots = 8;

pub const Pipeline = struct {
    dev: device.ExternalDevice,
    window: device.Window,
    cfg: Config,
    external: ExternalStage,
    peer: ?Peer,
    down: wire.Slot,
    up: wire.Slot,
    seq: u32 = 0,
    phases: Phases = .{},
    /// The rows of the step in flight; 0 when none is.
    rows: u32 = 0,
    /// The sequences of the step in flight, stable until `collect`: the device's work reads them.
    slots: [max_slots]u8 = undefined,
    nslots: u8 = 0,
    work: Step = undefined,
    /// Host memory for the stamped hidden rows the device stage reads (`metal_first`); `max_rows * hidden_bytes` long.
    scratch: []u8,
    tokens: [max_rows_limit * 4]u8 = undefined,

    /// Room at the start of the up slot's payload for the device's tokens, one a row of a batch; staged tokens follow.
    fn replyBytes(cfg: Config) usize {
        return (4 * @as(usize, @max(cfg.max_rows, max_slots)) + 63) / 64 * 64;
    }

    /// Slot size for `cfg`: header plus the larger of the stamped hidden rows and a token reply with the staged tokens.
    pub fn slotBytes(cfg: Config) usize {
        const rows = @max(cfg.max_rows, 1);
        const payload = @max(wire.stampedRowsLen(rows, cfg.hidden_bytes), replyBytes(cfg) + 4 * rows);
        return (wire.header_bytes + payload + 63) / 64 * 64;
    }

    /// Bytes of window for `streams` sequences.
    pub fn windowBytes(cfg: Config, streams: usize) usize {
        return 2 * slotBytes(cfg) * streams;
    }

    pub fn init(dev: device.ExternalDevice, window: device.Window, cfg: Config, external: ExternalStage, peer: ?Peer, scratch: []u8) Pipeline {
        const n = slotBytes(cfg);
        std.debug.assert(cfg.max_rows >= 1 and cfg.max_rows <= max_rows_limit);
        std.debug.assert(window.len >= cfg.window_offset + 2 * n and scratch.len >= cfg.max_rows * cfg.hidden_bytes);
        std.debug.assert(cfg.order == .external_only or peer != null);
        const base = window.ptr + cfg.window_offset;
        var p: Pipeline = .{ .dev = dev, .window = window, .cfg = cfg, .external = external, .peer = peer, .scratch = scratch, .down = .init(base, n), .up = .init(base + n, n) };
        p.prepare();
        return p;
    }

    /// Empty slots; the up slot's payload kind is fixed by the order, its size by each step.
    fn prepare(p: *Pipeline) void {
        p.down.clear();
        p.up.clear();
        const hidden = p.cfg.order == .external_first;
        p.up.header().kind = @backingInt(if (hidden) wire.Kind.hidden else wire.Kind.token);
        p.up.header().bytes = @intCast(if (hidden) p.cfg.hidden_bytes else 4);
        p.up.header().rows = 1;
    }

    /// One token step of sequence 0; returns the token the head chose. The step's number is the pipeline's count.
    pub fn step(p: *Pipeline, token: u32) Error!u32 {
        return p.stepRows(&.{token});
    }

    /// `tokens.len` consecutive positions of sequence 0 in one step (a prompt chunk); returns the token after the last.
    pub fn stepRows(p: *Pipeline, tokens: []const u32) Error!u32 {
        try p.submit(tokens);
        return p.collect();
    }

    /// Starts a step of sequence 0 (`submitSlots` with one slot).
    pub fn submit(p: *Pipeline, tokens: []const u32) Error!void {
        return p.submitSlots(tokens, &.{0});
    }

    /// Starts a step; `collect` finishes it. One slot: consecutive positions; several: `tokens[i]` is slot i's next.
    pub fn submitSlots(p: *Pipeline, tokens: []const u32, slots: []const u8) Error!void {
        if (tokens.len == 0 or tokens.len > p.cfg.max_rows or p.rows != 0) return error.BadStep;
        if (slots.len == 0 or slots.len > max_slots or (slots.len > 1 and slots.len != tokens.len)) return error.BadStep;
        for (slots, 0..) |a, i| for (slots[0..i]) |b| if (a == b) return error.BadStep;
        const rows: u32 = @intCast(tokens.len);
        const nout: u32 = if (slots.len == 1) 1 else rows;
        p.seq += 1;
        const seq = p.seq;
        p.work = .{ .p = p, .seq = seq, .rows = rows };
        p.rows = rows;
        p.nslots = @intCast(slots.len);
        @memcpy(p.slots[0..slots.len], slots);
        const tok_bytes = p.tokens[0 .. 4 * tokens.len];
        for (tokens, 0..) |t, i| std.mem.writeInt(u32, tok_bytes[4 * i ..][0..4], t, .little);
        const hidden = p.cfg.order == .external_first;
        p.up.header().bytes = @intCast(if (hidden) rows * p.cfg.hidden_bytes else 4 * nout);
        p.up.header().rows = rows;
        switch (p.cfg.order) {
            .external_first, .external_only => {
                const t = device.nowNs();
                try p.queue(seq);
                p.down.publish(seq, .token, 0, rows, tok_bytes, .ok) catch return error.Oversize;
                _ = p.lap(&p.phases.queue_ns, t);
            },
            .metal_first => {
                var t = device.nowNs();
                try p.queue(seq);
                t = p.lap(&p.phases.queue_ns, t);
                const peer = p.peer.?;
                const region = p.down.data();
                const level = p.cfg.level;
                // the tokens the host stage reads sit in the window, behind the room the device's token reply has
                const staged = p.up.data()[replyBytes(p.cfg)..][0..tok_bytes.len];
                @memcpy(staged, tok_bytes);
                const n = peer.step(peer.ptr, seq, p.slots[0..p.nslots], rows, staged, region, level) catch return p.fail(error.PeerLost);
                std.debug.assert(n == rows * p.cfg.hidden_bytes);
                t = p.lap(&p.phases.peer_ns, t);
                // level A: the stage is complete, so the word follows the data; level B: the word may go first
                const bytes: u32 = @intCast(wire.stampedRowsLen(rows, p.cfg.hidden_bytes));
                if (level == .a) peer.finish(peer.ptr, seq) catch return p.fail(error.PeerLost);
                p.down.publishHeader(seq, .hidden, 0, rows, bytes, .ok);
                if (level == .b) peer.finish(peer.ptr, seq) catch return p.fail(error.PeerLost);
                _ = p.lap(&p.phases.finish_ns, t);
            },
        }
    }

    /// Adds the time since `since` to `slot` and returns now.
    fn lap(_: *Pipeline, slot: *u64, since: u64) u64 {
        const now = device.nowNs();
        slot.* += now - since;
        return now;
    }

    /// Finishes the step `submit` started and returns its token (a decode batch's first: `collectAll` takes them all).
    pub fn collect(p: *Pipeline) Error!u32 {
        var one: [1]u32 = undefined;
        if (p.nslots > 1) return error.BadStep;
        try p.collectAll(&one);
        return one[0];
    }

    /// Finishes the step and writes its tokens: one for a single slot, one a row for a decode batch.
    pub fn collectAll(p: *Pipeline, out: []u32) Error!void {
        if (p.rows == 0) return error.BadStep;
        const seq = p.seq;
        const rows = p.rows;
        const nout: usize = if (p.nslots == 1) 1 else rows;
        if (out.len != nout) return error.BadStep;
        p.rows = 0;
        var t = device.nowNs();
        const h = p.up.wait(seq, p.cfg.timeout_ns) catch |err| return p.lose(err);
        t = p.lap(&p.phases.wait_ns, t);
        var from = p.up.data()[0 .. 4 * nout];
        if (p.cfg.order == .external_first) {
            const peer = p.peer.?;
            // the tokens land where the device's input tokens were: the device has read them
            const reply = p.down.data()[0 .. 4 * nout];
            const n = peer.step(peer.ptr, seq, p.slots[0..p.nslots], rows, p.up.data()[0..h.bytes], reply, .a) catch return p.fail(error.PeerLost);
            _ = p.lap(&p.phases.peer_ns, t);
            std.debug.assert(n == 4 * nout);
            from = reply;
        }
        for (out, 0..) |*o, i| o.* = std.mem.readInt(u32, from[4 * i ..][0..4], .little);
    }

    fn queue(p: *Pipeline, seq: u32) Error!void {
        try p.dev.enqueueWait(p.window, p.downOffset(), seq, p.cfg.timeout_ns);
        try p.dev.enqueue(.{ .ctx = &p.work, .run = Step.run });
        try p.dev.enqueueWrite(p.window, p.upOffset(), seq);
    }

    fn downOffset(p: *const Pipeline) usize {
        return @intFromPtr(p.down.base) - @intFromPtr(p.window.ptr);
    }

    fn upOffset(p: *const Pipeline) usize {
        return @intFromPtr(p.up.base) - @intFromPtr(p.window.ptr);
    }

    /// A wait on the device failed: say so on both slots so nothing keeps waiting, and name the error.
    fn lose(p: *Pipeline, err: wire.Error) Error {
        p.release();
        return if (err == error.Timeout) error.Timeout else error.PeerLost;
    }

    /// Release the device's wait for this step with an abort so its stream drains.
    pub fn release(p: *Pipeline) void {
        p.down.publishHeader(p.seq, .abort, 0, 0, 0, .peer_lost);
    }

    fn fail(p: *Pipeline, err: Error) Error {
        p.release();
        return err;
    }

    /// After a failed step: drain the device stream and start over; on a shared device release every pipeline first.
    pub fn reset(p: *Pipeline) void {
        p.dev.sync(p.cfg.timeout_ns) catch {};
        p.prepare();
        p.seq = 0;
        p.rows = 0;
        p.nslots = 0;
    }
};

/// The device-side step: the stage's work between the wait on the down slot's word and the write of the up slot's.
const Step = struct {
    p: *Pipeline,
    seq: u32,
    rows: u32,

    fn run(ctx: *anyopaque) anyerror!void {
        const s: *Step = @ptrCast(@alignCast(ctx));
        const p = s.p;
        try p.external.run(p.external.ptr, .{
            .in = p.down,
            .out = p.up,
            .seq = s.seq,
            .rows = s.rows,
            .slots = p.slots[0..p.nslots],
            .stamped = p.cfg.order == .metal_first,
            .retries = p.cfg.stamp_retries,
            .scratch = p.scratch,
            .hidden_bytes = p.cfg.hidden_bytes,
        });
    }
};
