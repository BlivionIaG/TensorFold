//! A two-stage layer pipeline across an external device and the host's own GPU, one token at a time.
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
    /// The input row is 16-byte stamped groups (`metal_first`): the stage must see every group's stamp equal `seq`.
    stamped: bool,
    retries: u32,
    /// Host memory a CPU stage may stage a stamped row in, `hidden_bytes` long.
    scratch: []u8,
    hidden_bytes: usize,
};

/// The stage the device runs in its stream: records launches that read `io.in` and write `io.out`.
pub const ExternalStage = struct {
    ptr: *anyopaque,
    run: *const fn (ptr: *anyopaque, io: StepIo) anyerror!void,
};

/// The stage the host runs on its own GPU: `step` returns once its result is in `out` or launched; `finish` waits.
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

    /// One token step; returns the token the head chose. The step's sequence number is the pipeline's count.
    pub fn step(p: *Pipeline, token: u32) Error!u32 {
        p.seq += 1;
        const seq = p.seq;
        p.work = .{ .p = p, .seq = seq };
        const timeout = p.cfg.timeout_ns;
        var tok_bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &tok_bytes, token, .little);
        switch (p.cfg.order) {
            .external_first, .external_only => {
                try p.queue(seq);
                p.down.publish(seq, .token, 0, 1, &tok_bytes, .ok) catch return error.Oversize;
                const h = p.up.wait(seq, timeout) catch |err| return p.lose(err);
                if (p.cfg.order == .external_only) return std.mem.readInt(u32, p.up.data()[0..4], .little);
                var out: [4]u8 = undefined;
                const peer = p.peer.?;
                const n = peer.step(peer.ptr, seq, p.up.data()[0..h.bytes], &out, .a) catch return p.fail(error.PeerLost);
                std.debug.assert(n == 4);
                return std.mem.readInt(u32, &out, .little);
            },
            .metal_first => {
                try p.queue(seq);
                const peer = p.peer.?;
                const region = p.down.data();
                const level = p.cfg.level;
                const n = peer.step(peer.ptr, seq, &tok_bytes, region, level) catch return p.fail(error.PeerLost);
                std.debug.assert(n == p.cfg.hidden_bytes);
                // level A: the stage is complete, so the word follows the data; level B: the word may go first
                if (level == .a) peer.finish(peer.ptr, seq) catch return p.fail(error.PeerLost);
                p.down.publishHeader(seq, .hidden, 0, 1, @intCast(wire.stampedLen(n)), .ok);
                if (level == .b) peer.finish(peer.ptr, seq) catch return p.fail(error.PeerLost);
                _ = p.up.wait(seq, timeout) catch |err| return p.lose(err);
                return std.mem.readInt(u32, p.up.data()[0..4], .little);
            },
        }
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
