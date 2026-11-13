//! The decoder's side of a handoff: OPEN until the export is recorded, PULL every frame into a sink, then CLOSE.
const std = @import("std");
const wire = @import("wire.zig");
const manifest = @import("manifest.zig");
const mailbox = @import("mailbox.zig");
const words = @import("words.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ Refused, Protocol, Mismatch, Corrupt, Expired, Aborted } || mailbox.Error || manifest.Error;

/// What the decoder knows the export must be: the same checkpoint, prompt and tensor-parallel rank.
pub const Expect = struct {
    model: []const u8,
    tokens: []const u32,
    tp_rank: u64 = 0,
    tp_size: u64 = 1,
    checksum: bool = true,
    open_timeout_ns: u64 = 30 * std.time.ns_per_s,
    /// The newest protocol this decoder reads: 2 takes recurrent layers; 1 refuses a protocol 2 export as foreign.
    protocol: u16 = wire.version,
    /// Set by the caller when the request is gone: the pull stops between calls, after a CLOSE if it had opened.
    abort: ?*const std.atomic.Value(bool) = null,
    /// True while the producer is still prefilling (the pull began with its trigger): OPEN and PULL wait for it.
    streaming: ?*const std.atomic.Value(bool) = null,
    /// Set when the producer's prefill failed: the pull stops as if its request were gone.
    broken: ?*const std.atomic.Value(bool) = null,
    /// The longest a request waits for its link to come back when the link dropped under it (the same mailbox, a new
    /// generation) before the pull fails; the request is then asked again. 0 fails at once, as a pull always did.
    relink_ns: u64 = 0,

    pub fn aborted(e: Expect) bool {
        if (e.broken) |b| if (b.load(.acquire)) return true;
        return if (e.abort) |a| a.load(.acquire) else false;
    }

    fn live(e: Expect) bool {
        return if (e.streaming) |s| s.load(.acquire) else false;
    }
};

/// Where pulled rows go: rows [row_start, row_start + rows) of `layer`, C-contiguous in the manifest's shape.
pub const Sink = struct {
    ptr: *anyopaque,
    put_fn: *const fn (ptr: *anyopaque, layer: *const manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void,
};

/// The producer's ERROR text, kept for the caller's message.
pub const Reason = struct {
    buf: [256]u8 = undefined,
    len: usize = 0,

    pub fn text(r: *const Reason) []const u8 {
        return r.buf[0..r.len];
    }

    fn set(r: *Reason, s: []const u8) void {
        r.len = @min(s.len, r.buf.len);
        @memcpy(r.buf[0..r.len], s[0..r.len]);
    }
};

pub const Pulled = struct { manifest: manifest.Manifest, frames: u64, bytes: u64 };

/// Pull one handoff over `client`; any failure after OPEN still asks the producer to CLOSE it.
pub fn pull(arena: Allocator, client: *mailbox.Client, handoff: [16]u8, expect: Expect, sink: Sink, reason: *Reason) Error!Pulled {
    const m = try open(arena, client, handoff, expect, reason);
    const got = frames(arena, client, handoff, m, expect, sink, reason) catch |err| {
        if (!client.poisoned) close(client, handoff) catch {};
        return err;
    };
    // every frame landed: a CLOSE the link dropped under only leaves the export to age out on the producer
    close(client, handoff) catch |err| if (!lost(err)) return err;
    return got;
}

const log = std.log.scoped(.decoder);
const relink_poll_ns = 100 * std.time.ns_per_ms;

/// The errors of a link that dropped: its client works again once the mailbox's next generation is up.
fn lost(err: anyerror) bool {
    return err == error.LinkDown or err == error.LinkLost or err == error.Poisoned;
}

/// `client.call`; a call the link dropped under waits up to `expect.relink_ns` for the link to come back and asks again.
/// OPEN and PULL are answered the same each time and a frame lands only from a reply, so none lands twice or is skipped.
fn call(client: *mailbox.Client, expect: ?*const Expect, msg: []const u8) mailbox.Error![]const u8 {
    const e = expect orelse return client.call(msg);
    var deadline: u64 = 0;
    while (true) {
        const reply = client.call(msg) catch |err| {
            if (e.relink_ns == 0 or !lost(err)) return err;
            if (deadline == 0) deadline = words.nowNs() +| e.relink_ns;
            try relinked(client, e, deadline, err);
            continue;
        };
        return reply;
    }
}

/// The client made afresh for the link's new generation, or `err` (what the drop was) when the wait ran out or the pull is over.
fn relinked(client: *mailbox.Client, expect: *const Expect, deadline: u64, err: mailbox.Error) mailbox.Error!void {
    while (!expect.aborted() and words.nowNs() < deadline) {
        client.renew() catch |again| {
            if (again != error.LinkDown) return err; // a request still owed a reply, or a mailbox gone: not this link's flap
            words.pause(relink_poll_ns);
            continue;
        };
        log.info("a link re-linked during the pull ({s}); the request is asked again", .{@errorName(err)});
        return;
    }
    return err;
}

fn request(client: *mailbox.Client, expect: ?*const Expect, kind: wire.Kind, handoff: [16]u8, frame: u32, payload: []const u8, most: u16) Error!wire.Header {
    var msg: [wire.header_bytes + 32]u8 = undefined;
    msg[0..wire.header_bytes].* = wire.pack(.{ .kind = kind, .handoff = handoff, .frame = frame, .nbytes = payload.len });
    @memcpy(msg[wire.header_bytes..][0..payload.len], payload);
    const reply = try call(client, expect, msg[0 .. wire.header_bytes + payload.len]);
    const h = switch (wire.unpackUpTo(reply, most)) {
        .ok => |h| h,
        .bad => return error.Protocol,
    };
    if (!std.mem.eql(u8, &h.handoff, &handoff) and h.kind != .@"error") return error.Protocol;
    return h;
}

fn open(arena: Allocator, client: *mailbox.Client, handoff: [16]u8, expect: Expect, reason: *Reason) Error!manifest.Manifest {
    const options = if (expect.checksum) "{\"checksum\": true}" else "{\"checksum\": false}";
    var deadline = words.nowNs() + expect.open_timeout_ns;
    var nap: u64 = std.time.ns_per_ms;
    while (true) {
        const h = try request(client, &expect, .open, handoff, 0, options, expect.protocol);
        const reply = client.box.replyPayload();
        switch (h.kind) {
            .wait => {
                if (expect.aborted()) return error.Aborted;
                if (expect.live()) deadline = words.nowNs() + expect.open_timeout_ns;
                if (words.nowNs() >= deadline) return error.Expired;
                words.pause(nap);
                nap = @min(nap * 2, 20 * std.time.ns_per_ms);
            },
            .@"error" => {
                reason.set(wire.body(reply, h));
                return error.Refused;
            },
            .manifest => {
                const m = try manifest.parseUpTo(arena, try arena.dupe(u8, wire.body(reply, h)), expect.protocol);
                if (m.protocol != h.version) return error.Protocol;
                try check(m, h, handoff, expect);
                return m;
            },
            else => return error.Protocol,
        }
    }
}

/// The manifest must be for this prompt, checkpoint and rank, and its layers must be ones the sink can place.
fn check(m: manifest.Manifest, h: wire.Header, handoff: [16]u8, expect: Expect) Error!void {
    if (!std.mem.eql(u8, &m.handoff, &handoff) or m.frames != h.frames) return error.Protocol;
    if (!std.mem.eql(u8, m.model, expect.model) or m.prompt_tokens != expect.tokens.len) return error.Mismatch;
    if (!std.mem.eql(u8, &m.token_sha256, &wire.tokenSha256(expect.tokens))) return error.Mismatch;
    if (m.tp_rank != expect.tp_rank or m.tp_size != expect.tp_size or m.first_token % m.block_size != 0) return error.Mismatch;
    for (m.layers, 0..) |l, i| {
        for (m.layers[0..i]) |o| if (o.index == l.index) return error.Protocol;
    }
}

/// The most frames a manifest may announce: a peer's count sizes the pull's bookkeeping.
const max_frames: u64 = 1 << 22;

/// The most PULLs a sweep answered `wait` before it rests: later frames of a streamed export are ready later.
const max_waits: u32 = 16;

/// Every frame into `sink`; recurrent parts are independent, so a frame still being built (`wait`) is skipped for now.
fn frames(arena: Allocator, client: *mailbox.Client, handoff: [16]u8, m: manifest.Manifest, expect: Expect, sink: Sink, reason: *Reason) Error!Pulled {
    if (m.layers.len > 512 or m.frames > max_frames) return error.Protocol;
    var got: [512]u64 = @splat(0);
    var base: [512]u64 = undefined; // the first frame of each layer when every layer is one frame a part
    var any_order = m.layers.len > 0;
    var at_frame: u64 = 0;
    for (m.layers, 0..) |l, j| {
        base[j] = at_frame;
        at_frame += l.parts.len;
        any_order = any_order and l.kind == .recurrent and l.parts.len > 0;
    }
    any_order = any_order and at_frame == m.frames;
    const n: usize = @intCast(m.frames);
    const done = try arena.alloc(bool, n);
    @memset(done, false);
    var left = n;
    var first: usize = 0;
    var bytes: u64 = 0;
    var nap: u64 = std.time.ns_per_ms;
    var moved = words.nowNs(); // the last time a frame landed
    while (left > 0) {
        var progressed = false;
        var waits: u32 = 0;
        var i = first;
        while (i < n) : (i += 1) {
            if (done[i]) continue;
            if (expect.aborted()) return error.Aborted;
            const frame: u32 = @intCast(i);
            const h = try request(client, &expect, .pull, handoff, frame, "", m.protocol);
            const reply = client.box.replyPayload();
            if (h.kind == .wait) {
                waits += 1;
                if (!any_order or waits >= max_waits) break;
                continue;
            }
            if (h.kind == .@"error") {
                reason.set(wire.body(reply, h));
                return error.Refused;
            }
            if (h.kind != .data or h.frame != frame or h.frames != m.frames or h.version != m.protocol) return error.Protocol;
            const at = for (m.layers, 0..) |l, j| {
                if (l.index == h.layer) break j;
            } else return error.Protocol;
            const layer = &m.layers[at];
            if (!rowsFit(h.row_start, h.rows, layer.rows())) return error.Protocol;
            if (any_order) {
                if (base[at] + h.row_start != i) return error.Protocol;
            } else if (h.row_start != got[at]) return error.Protocol;
            if (h.nbytes != layer.frameBytes(h.row_start, h.rows) orelse return error.Protocol) return error.Protocol;
            const payload = wire.body(reply, h);
            if (expect.checksum and (h.flags & wire.checked == 0 or wire.crc32(payload) != h.crc)) return error.Corrupt;
            sink.put_fn(sink.ptr, layer, h.row_start, h.rows, payload) catch return error.Protocol;
            got[at] += h.rows;
            bytes += h.nbytes;
            done[i] = true;
            left -= 1;
            progressed = true;
            waits = 0;
        }
        while (first < n and done[first]) first += 1;
        if (left == 0) break;
        if (progressed) {
            nap = std.time.ns_per_ms;
            moved = words.nowNs();
            continue;
        }
        if (expect.aborted()) return error.Aborted;
        if (expect.live()) moved = words.nowNs() else if (words.nowNs() - moved >= expect.open_timeout_ns) return error.Expired;
        words.pause(nap);
        nap = @min(nap * 2, 20 * std.time.ns_per_ms);
    }
    for (m.layers, 0..) |l, j| if (got[j] != l.rows()) return error.Protocol;
    return .{ .manifest = m, .frames = m.frames, .bytes = bytes };
}

/// Rows [start, start + count) inside a layer of `total`, without the sum wrapping for a peer's huge values.
fn rowsFit(start: u64, count: u64, total: u64) bool {
    return count > 0 and count <= total and start <= total - count;
}

test "a frame's rows fit its layer only when their end does, however large a peer's values" {
    try std.testing.expect(rowsFit(0, 3, 3) and rowsFit(2, 1, 3));
    try std.testing.expect(!rowsFit(3, 1, 3) and !rowsFit(0, 0, 3) and !rowsFit(0, 4, 3));
    try std.testing.expect(!rowsFit(std.math.maxInt(u64), 1, 3) and !rowsFit(1, std.math.maxInt(u64), 3));
}

fn close(client: *mailbox.Client, handoff: [16]u8) Error!void {
    const h = try request(client, null, .close, handoff, 0, "", wire.recurrent_version);
    if (h.kind != .ack) return error.Protocol;
}
