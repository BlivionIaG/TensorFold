//! A hand-off as a file, for moving one between machines with no MCDMA link (opt-in: TF_DSV41_HANDOFF_DUMP on the producer).
//!
//! The file is the producer's replies, byte for byte, to a decoder's OPEN and then each PULL: one MKVH protocol 2 MANIFEST
//! message, then every DATA frame in order, each message a 128-byte header followed by `nbytes` of payload (so the stream
//! delimits itself). `write` runs the real `producer.Responder` over the export; `pull` is `fabric.decoder.pull` reading the
//! same messages from memory: the same checks in the same order, the rows landed in the same `Sink`.
const std = @import("std");
const fabric = @import("fabric");
const wire = fabric.wire;
const manifest = fabric.manifest;
const producer = fabric.producer;
const decoder = fabric.decoder;

pub const extension = ".mkvh2";
/// The reply area a link of 4 MiB segments gives the responder: frames are planned and cut for it.
pub const reply_area = 4 << 20;

/// `<id as 32 hex digits>.mkvh2`.
pub fn fileName(buf: *[32 + extension.len]u8, id: [16]u8) []const u8 {
    return std.fmt.bufPrint(buf, "{x}{s}", .{ &id, extension }) catch unreachable;
}

const Capture = struct {
    len: usize = 0,
    fn publish(ptr: *anyopaque, _: u32, len: usize) anyerror!void {
        const c: *Capture = @ptrCast(@alignCast(ptr));
        c.len = len;
    }
};

/// `exp` as the replies a responder gives its decoder; `exp` itself is left as it was (its frames are planned on a copy).
pub fn write(gpa: std.mem.Allocator, w: *std.Io.Writer, exp: *const producer.Export, model: []const u8, tp_rank: u64, tp_size: u64) !void {
    var copy = exp.*;
    copy.frames = &.{};
    defer if (copy.frames.len > 0) gpa.free(copy.frames);
    var table: producer.Table = .{ .gpa = gpa, .ttl_ns = std.math.maxInt(u64), .clock = fabric.words.nowNs };
    defer table.deinit();
    try table.add(exp.handoff, "dump", .{ .ready = &copy });
    const area = try gpa.alloc(u8, reply_area);
    defer gpa.free(area);
    var cap: Capture = .{};
    var responder: producer.Responder = .{ .gpa = gpa, .table = &table, .out = .{ .area = area, .ptr = &cap, .publish_fn = Capture.publish }, .model = model, .tp_rank = tp_rank, .tp_size = tp_size };
    defer responder.deinit();

    const options = "{\"checksum\": true}";
    var open: [wire.header_bytes + options.len]u8 = undefined;
    open[0..wire.header_bytes].* = wire.pack(.{ .kind = .open, .handoff = exp.handoff, .nbytes = options.len });
    @memcpy(open[wire.header_bytes..], options);
    try responder.handle(1, &open);
    const head = switch (wire.unpackUpTo(area[0..cap.len], wire.recurrent_version)) {
        .ok => |h| h,
        .bad => return error.Protocol,
    };
    if (head.kind != .manifest) return error.Protocol;
    try w.writeAll(area[0..cap.len]);
    for (0..head.frames) |i| {
        const request = wire.pack(.{ .kind = .pull, .handoff = exp.handoff, .frame = @intCast(i) });
        try responder.handle(@intCast(2 + i), &request);
        try w.writeAll(area[0..cap.len]);
    }
}

const Message = struct { h: wire.Header, payload: []const u8 };

fn next(bytes: []const u8, at: *usize, most: u16) decoder.Error!Message {
    const msg = bytes[at.*..];
    const h = switch (wire.unpackUpTo(msg, most)) {
        .ok => |h| h,
        .bad => return error.Protocol,
    };
    // every message of a file carries a payload: manifest, data or error
    if (h.kind != .manifest and h.kind != .data and h.kind != .@"error") return error.Protocol;
    at.* += wire.header_bytes + @as(usize, @intCast(h.nbytes));
    return .{ .h = h, .payload = wire.body(msg, h) };
}

/// The file's hand-off pulled into `sink` as `fabric.decoder.pull` would from a producer: the manifest must be for this
/// checkpoint, prompt and shard; every frame arrives in order, whole, checksummed; nothing may follow the last.
pub fn pull(arena: std.mem.Allocator, bytes: []const u8, expect: decoder.Expect, sink: decoder.Sink, reason: *decoder.Reason) decoder.Error!decoder.Pulled {
    var at: usize = 0;
    const first = try next(bytes, &at, expect.protocol);
    switch (first.h.kind) {
        .@"error" => {
            reason.len = @min(first.payload.len, reason.buf.len);
            @memcpy(reason.buf[0..reason.len], first.payload[0..reason.len]);
            return error.Refused;
        },
        .manifest => {},
        else => return error.Protocol,
    }
    const m = try manifest.parseUpTo(arena, try arena.dupe(u8, first.payload), expect.protocol);
    if (m.protocol != first.h.version) return error.Protocol;
    const handoff = first.h.handoff;
    if (!std.mem.eql(u8, &m.handoff, &handoff) or m.frames != first.h.frames) return error.Protocol;
    if (!std.mem.eql(u8, m.model, expect.model) or m.prompt_tokens != expect.tokens.len) return error.Mismatch;
    if (!std.mem.eql(u8, &m.token_sha256, &wire.tokenSha256(expect.tokens))) return error.Mismatch;
    if (m.tp_rank != expect.tp_rank or m.tp_size != expect.tp_size or m.first_token % m.block_size != 0) return error.Mismatch;
    for (m.layers, 0..) |l, i| for (m.layers[0..i]) |o| if (o.index == l.index) return error.Protocol;

    var rows_next: [512]u64 = @splat(0);
    if (m.layers.len > rows_next.len) return error.Protocol;
    var total: u64 = 0;
    for (0..m.frames) |i| {
        const frame: u32 = @intCast(i);
        const msg = try next(bytes, &at, m.protocol);
        const h = msg.h;
        if (h.kind == .@"error") {
            reason.len = @min(msg.payload.len, reason.buf.len);
            @memcpy(reason.buf[0..reason.len], msg.payload[0..reason.len]);
            return error.Refused;
        }
        if (h.kind != .data or h.frame != frame or h.frames != m.frames or h.version != m.protocol or !std.mem.eql(u8, &h.handoff, &handoff)) return error.Protocol;
        const idx = for (m.layers, 0..) |l, j| {
            if (l.index == h.layer) break j;
        } else return error.Protocol;
        const layer = &m.layers[idx];
        if (!rowsFit(h.row_start, h.rows, layer.rows()) or h.row_start != rows_next[idx]) return error.Protocol;
        if (h.nbytes != layer.frameBytes(h.row_start, h.rows) orelse return error.Protocol) return error.Protocol;
        if (expect.checksum and (h.flags & wire.checked == 0 or wire.crc32(msg.payload) != h.crc)) return error.Corrupt;
        sink.put_fn(sink.ptr, layer, h.row_start, h.rows, msg.payload) catch return error.Protocol;
        rows_next[idx] += h.rows;
        total += h.nbytes;
    }
    if (at != bytes.len) return error.Protocol;
    for (m.layers, 0..) |l, j| if (rows_next[j] != l.rows()) return error.Protocol;
    return .{ .manifest = m, .frames = m.frames, .bytes = total };
}

fn rowsFit(start: u64, count: u64, total: u64) bool {
    return count > 0 and count <= total and start <= total - count;
}
