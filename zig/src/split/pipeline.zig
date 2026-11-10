//! A two-stage layer pipeline across an external device and the host's own GPU.
//! One window holds two slots a stream. The device stream waits for its input slot, runs its stage, and releases its
//! output slot; the host waits for that slot, runs its stage through a `Peer` and publishes the next input. Which side
//! runs the first layers (and the embedding) is `Order`; the other runs the rest and the head. Planning chooses the split.
//! A step carries one position (decode) or up to `max_rows` consecutive positions of one sequence (a prompt chunk). Several
//! sequences share one device through one `Pipeline` each over its own slots of the window: the device's stream runs their
//! steps in the order they were queued, so while the host works on one sequence the device works on another (see `streams.zig`).
const std = @import("std");
const device = @import("device.zig");
const wire = @import("wire.zig");

pub const Order = enum {
    /// The device runs the embedding and layers `[0, k)`, the host layers `[k, L)` and the head; hidden rows go up, tokens down.
    external_first,
    /// The host runs the embedding and layers `[0, k)`, the device the rest and the head; hidden rows go down, tokens back.
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

/// What a device stage is given for one step: its input and output slots in the window (kernels read and write them
/// by their GPU addresses, which are the host's pointers), and how the input is protected. The output slot's kind and
/// byte count are fixed by the pipeline before the step; the stage's work fills the payload.
///
/// `rows` positions travel in one step. A stage that embeds reads `rows` u32 tokens at the input payload; any other
/// reads `rows` hidden rows of `hidden_bytes` f32 words, plain, or each row on its own stamped groups when `stamped`.
/// A stage that ends the model writes one u32 (the greedy token after the last row); any other writes `rows` plain rows.
pub const StepIo = struct {
    in: wire.Slot,
    out: wire.Slot,
    seq: u32,
    rows: u32 = 1,
    /// The input rows are 16-byte stamped groups (`metal_first`): the stage must see every group's stamp equal `seq`.
    stamped: bool,
    retries: u32,
    /// Host memory a CPU stage may stage stamped rows in, `rows * hidden_bytes` long.
    scratch: []u8,
    hidden_bytes: usize,
};

/// The stage the device runs in its stream. A stream device records launches that read `io.in` and write `io.out` when
/// `run` is called at enqueue time; a mock device calls it later, in stream order, on its own thread and computes on the CPU.
pub const ExternalStage = struct {
    ptr: *anyopaque,
    run: *const fn (ptr: *anyopaque, io: StepIo) anyerror!void,
};

/// The stage the host runs on its own GPU, for one sequence. `step` takes `rows` positions (`input` is `rows` u32 tokens
/// or `rows` plain f32 rows), returns when the stage's result is complete in `out` (level A) or has been launched and will
/// land there (level B, hidden rows only), and returns the bytes of plain result (`rows` rows, or 4 for a token);
/// `finish` waits for a launched step to complete. A hidden result goes to `out` as `rows` stamped rows.
pub const Peer = struct {
    ptr: *anyopaque,
    step: *const fn (ptr: *anyopaque, seq: u32, rows: u32, input: []const u8, out: []u8, level: Level) anyerror!usize,
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

/// Where a pipeline's steps spent their time, summed over its steps: recording the device's work, the host stage's step
/// and finish, and waiting for the device's reply.
pub const Phases = struct { queue_ns: u64 = 0, peer_ns: u64 = 0, finish_ns: u64 = 0, wait_ns: u64 = 0 };

/// Bytes the pipeline keeps clear at the start of the up slot's payload for the token the device writes; the tokens a
/// host stage reads are staged behind them.
const token_reply_bytes = 64;
pub const max_rows_limit = 256;

pub const Pipeline = struct {
    dev: device.ExternalDevice,
    window: device.Window,
    cfg: Config,
    external: ExternalStage,
    peer: ?Peer,
    down: wire.Slot,
    up: wire.Slot,
    seq: u32 = 0,
    /// The rows of the step in flight; 0 when none is.
    rows: u32 = 0,
    phases: Phases = .{},
    work: Step = undefined,
    /// Host memory for the stamped hidden rows the device stage reads (`metal_first`); `max_rows * hidden_bytes` long.
    scratch: []u8,
    tokens: [max_rows_limit * 4]u8 = undefined,

    /// Slot size for `cfg`: header plus the larger of the stamped hidden rows and a token reply with the staged tokens.
    pub fn slotBytes(cfg: Config) usize {
        const rows = @max(cfg.max_rows, 1);
        const payload = @max(wire.stampedRowsLen(rows, cfg.hidden_bytes), token_reply_bytes + 4 * rows);
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

    /// One token step; returns the token the last layer's head chose. The step's sequence number is the pipeline's count.
    pub fn step(p: *Pipeline, token: u32) Error!u32 {
        return p.stepRows(&.{token});
    }

    /// `tokens.len` consecutive positions in one step (a prompt chunk); returns the token after the last of them.
    pub fn stepRows(p: *Pipeline, tokens: []const u32) Error!u32 {
        try p.submit(tokens);
        return p.collect();
    }

    /// Starts a step: the device's work is queued and the host's part that comes first has run. `collect` finishes it.
    /// Another pipeline over the same device may submit in between; the device runs the queued steps in order.
    pub fn submit(p: *Pipeline, tokens: []const u32) Error!void {
        if (tokens.len == 0 or tokens.len > p.cfg.max_rows or p.rows != 0) return error.BadStep;
        const rows: u32 = @intCast(tokens.len);
        p.seq += 1;
        const seq = p.seq;
        p.work = .{ .p = p, .seq = seq, .rows = rows };
        p.rows = rows;
        const tok_bytes = p.tokens[0 .. 4 * tokens.len];
        for (tokens, 0..) |t, i| std.mem.writeInt(u32, tok_bytes[4 * i ..][0..4], t, .little);
        const hidden = p.cfg.order == .external_first;
        p.up.header().bytes = @intCast(if (hidden) rows * p.cfg.hidden_bytes else 4);
        p.up.header().rows = rows;
        var t = device.nowNs();
        switch (p.cfg.order) {
            .external_first, .external_only => {
                try p.queue(seq);
                p.down.publish(seq, .token, 0, rows, tok_bytes, .ok) catch return error.Oversize;
                _ = p.lap(&p.phases.queue_ns, t);
            },
            .metal_first => {
                try p.queue(seq);
                t = p.lap(&p.phases.queue_ns, t);
                const peer = p.peer.?;
                const region = p.down.data();
                const level = p.cfg.level;
                // the tokens the host stage reads sit in the window, behind the room the device's token reply has
                const staged = p.up.data()[token_reply_bytes..][0..tok_bytes.len];
                @memcpy(staged, tok_bytes);
                const n = peer.step(peer.ptr, seq, rows, staged, region, level) catch return p.fail(error.PeerLost);
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

    /// Finishes the step `submit` started: waits for the device's reply, runs the host's part that comes last, and returns the token.
    pub fn collect(p: *Pipeline) Error!u32 {
        if (p.rows == 0) return error.BadStep;
        const seq = p.seq;
        const rows = p.rows;
        p.rows = 0;
        const t = device.nowNs();
        const h = p.up.wait(seq, p.cfg.timeout_ns) catch |err| return p.lose(err);
        var t2 = p.lap(&p.phases.wait_ns, t);
        if (p.cfg.order != .external_first) return std.mem.readInt(u32, p.up.data()[0..4], .little);
        const peer = p.peer.?;
        // the token lands where the device's input tokens were: the device has read them
        const out = p.down.data()[0..4];
        const n = peer.step(peer.ptr, seq, rows, p.up.data()[0..h.bytes], out, .a) catch return p.fail(error.PeerLost);
        std.debug.assert(n == 4);
        t2 = p.lap(&p.phases.peer_ns, t2);
        return std.mem.readInt(u32, out, .little);
    }

    /// Adds the time since `since` to `slot` and returns now.
    fn lap(_: *Pipeline, slot: *u64, since: u64) u64 {
        const now = device.nowNs();
        slot.* += now - since;
        return now;
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

    /// After a failed step: drain the device stream (its error is the failure already reported) and start over.
    /// With several pipelines on one device, release every one first, then reset them.
    pub fn reset(p: *Pipeline) void {
        p.dev.sync(p.cfg.timeout_ns) catch {};
        p.prepare();
        p.seq = 0;
        p.rows = 0;
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
            .stamped = p.cfg.order == .metal_first,
            .retries = p.cfg.stamp_retries,
            .scratch = p.scratch,
            .hidden_bytes = p.cfg.hidden_bytes,
        });
    }
};
