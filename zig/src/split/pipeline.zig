//! A two-stage layer pipeline across an external device and the host's own GPU, one token at a time.
//! One window holds two slots. The device stream waits for its input slot, runs its stage, and releases its output slot;
//! the host waits for that slot, runs its stage through a `Peer` and publishes the next input. Which side runs the
//! first layers (and the embedding) is `Order`; the other runs the rest and the head. Planning chooses the split.
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
/// byte count are fixed by the pipeline before the first step; the stage's work fills the payload.
pub const StepIo = struct {
    in: wire.Slot,
    out: wire.Slot,
    seq: u32,
    /// The input row is 16-byte stamped groups (`metal_first`): the stage must see every group's stamp equal `seq`.
    stamped: bool,
    retries: u32,
    /// Host memory a CPU stage may stage a stamped row in, `hidden_bytes` long.
    scratch: []u8,
    hidden_bytes: usize,
};

/// The stage the device runs in its stream. A stream device records launches that read `io.in` and write `io.out` when
/// `run` is called at enqueue time; a mock device calls it later, in stream order, on its own thread and computes on the CPU.
pub const ExternalStage = struct {
    ptr: *anyopaque,
    run: *const fn (ptr: *anyopaque, io: StepIo) anyerror!void,
};

/// The stage the host runs on its own GPU. `step` returns when the stage's result is complete in `out` (level A) or has
/// been launched and will land there (level B, hidden rows only); `finish` waits for a launched step to complete.
pub const Peer = struct {
    ptr: *anyopaque,
    step: *const fn (ptr: *anyopaque, seq: u32, input: []const u8, out: []u8, level: Level) anyerror!usize,
    finish: *const fn (ptr: *anyopaque, seq: u32) anyerror!void,
};

pub const Config = struct {
    order: Order,
    level: Level = .a,
    /// Bytes of one hidden row as the stages exchange it.
    hidden_bytes: usize,
    /// Bound on every wait, host and device side; 0 is unbounded.
    timeout_ns: u64 = 5_000_000_000,
    /// Reads of an unarrived group before level B gives up on a reply.
    stamp_retries: u32 = 20_000_000,
};

/// Where a pipeline's steps spent their time, summed over its steps: recording the device's work, the host stage's step
/// and finish, and waiting for the device's reply.
pub const Phases = struct { queue_ns: u64 = 0, peer_ns: u64 = 0, finish_ns: u64 = 0, wait_ns: u64 = 0 };

pub const Pipeline = struct {
    dev: device.ExternalDevice,
    window: device.Window,
    cfg: Config,
    external: ExternalStage,
    peer: ?Peer,
    /// Host to device slot and device to host slot.
    down: wire.Slot,
    up: wire.Slot,
    seq: u32 = 0,
    phases: Phases = .{},
    work: Step = undefined,
    /// Host memory for the stamped hidden row the device stage reads (`metal_first`); `hidden_bytes` long.
    scratch: []u8,

    /// Slot size for `cfg`: header plus the larger of a stamped hidden row and a token.
    pub fn slotBytes(cfg: Config) usize {
        const payload = @max(wire.stampedLen(cfg.hidden_bytes), 64);
        return (wire.header_bytes + payload + 63) / 64 * 64;
    }

    pub fn init(dev: device.ExternalDevice, window: device.Window, cfg: Config, external: ExternalStage, peer: ?Peer, scratch: []u8) Pipeline {
        const n = slotBytes(cfg);
        std.debug.assert(window.len >= 2 * n and scratch.len >= cfg.hidden_bytes);
        std.debug.assert(cfg.order == .external_only or peer != null);
        var p: Pipeline = .{ .dev = dev, .window = window, .cfg = cfg, .external = external, .peer = peer, .scratch = scratch, .down = .init(window.ptr, n), .up = .init(window.ptr + n, n) };
        p.prepare();
        return p;
    }

    /// Empty slots; the up slot's payload kind and size are fixed by the order.
    fn prepare(p: *Pipeline) void {
        p.down.clear();
        p.up.clear();
        const hidden = p.cfg.order == .external_first;
        p.up.header().kind = @backingInt(if (hidden) wire.Kind.hidden else wire.Kind.token);
        p.up.header().bytes = @intCast(if (hidden) p.cfg.hidden_bytes else 4);
    }

    /// One token step; returns the token the last layer's head chose. The step's sequence number is the pipeline's count.
    pub fn step(p: *Pipeline, token: u32) Error!u32 {
        p.seq += 1;
        const seq = p.seq;
        p.work = .{ .p = p, .seq = seq };
        const timeout = p.cfg.timeout_ns;
        var tok_bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &tok_bytes, token, .little);
        switch (p.cfg.order) {
            .external_first, .external_only => {
                var t = device.nowNs();
                try p.queue(seq);
                p.down.publish(seq, .token, 0, 1, &tok_bytes, .ok) catch return error.Oversize;
                t = p.lap(&p.phases.queue_ns, t);
                const h = p.up.wait(seq, timeout) catch |err| return p.lose(err);
                t = p.lap(&p.phases.wait_ns, t);
                if (p.cfg.order == .external_only) return std.mem.readInt(u32, p.up.data()[0..4], .little);
                var out: [4]u8 = undefined;
                const peer = p.peer.?;
                const n = peer.step(peer.ptr, seq, p.up.data()[0..h.bytes], &out, .a) catch return p.fail(error.PeerLost);
                _ = p.lap(&p.phases.peer_ns, t);
                std.debug.assert(n == 4);
                return std.mem.readInt(u32, &out, .little);
            },
            .metal_first => {
                var t = device.nowNs();
                try p.queue(seq);
                t = p.lap(&p.phases.queue_ns, t);
                const peer = p.peer.?;
                const region = p.down.data();
                const level = p.cfg.level;
                const n = peer.step(peer.ptr, seq, &tok_bytes, region, level) catch return p.fail(error.PeerLost);
                std.debug.assert(n == p.cfg.hidden_bytes);
                t = p.lap(&p.phases.peer_ns, t);
                // level A: the stage is complete, so the word follows the data; level B: the word may go first
                if (level == .a) peer.finish(peer.ptr, seq) catch return p.fail(error.PeerLost);
                p.down.publishHeader(seq, .hidden, 0, 1, @intCast(wire.stampedLen(n)), .ok);
                if (level == .b) peer.finish(peer.ptr, seq) catch return p.fail(error.PeerLost);
                t = p.lap(&p.phases.finish_ns, t);
                _ = p.up.wait(seq, timeout) catch |err| return p.lose(err);
                _ = p.lap(&p.phases.wait_ns, t);
                return std.mem.readInt(u32, p.up.data()[0..4], .little);
            },
        }
    }

    /// Adds the time since `since` to `slot` and returns now.
    fn lap(_: *Pipeline, slot: *u64, since: u64) u64 {
        const now = device.nowNs();
        slot.* += now - since;
        return now;
    }

    fn queue(p: *Pipeline, seq: u32) Error!void {
        try p.dev.enqueueWait(p.window, 0, seq, p.cfg.timeout_ns);
        try p.dev.enqueue(.{ .ctx = &p.work, .run = Step.run });
        try p.dev.enqueueWrite(p.window, @intCast(p.upOffset()), seq);
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
    fn release(p: *Pipeline) void {
        p.down.publishHeader(p.seq, .abort, 0, 0, 0, .peer_lost);
    }

    fn fail(p: *Pipeline, err: Error) Error {
        p.release();
        return err;
    }

    /// After a failed step: drain the device stream (its error is the failure already reported) and start over.
    pub fn reset(p: *Pipeline) void {
        p.dev.sync(p.cfg.timeout_ns) catch {};
        p.prepare();
        p.seq = 0;
    }
};

/// The device-side step: the stage's work between the wait on the down slot's word and the write of the up slot's.
const Step = struct {
    p: *Pipeline,
    seq: u32,

    fn run(ctx: *anyopaque) anyerror!void {
        const s: *Step = @ptrCast(@alignCast(ctx));
        const p = s.p;
        try p.external.run(p.external.ptr, .{
            .in = p.down,
            .out = p.up,
            .seq = s.seq,
            .stamped = p.cfg.order == .metal_first,
            .retries = p.cfg.stamp_retries,
            .scratch = p.scratch,
            .hidden_bytes = p.cfg.hidden_bytes,
        });
    }
};
