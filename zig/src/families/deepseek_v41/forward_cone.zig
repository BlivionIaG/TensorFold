//! Cone prefill (TF_DSV41_CONE=1): encoder layers over every prompt row, decoder layers over the last rows only.
//!
//! A prompt's non-final chunks run layers 0..d-1 and layer d's compressor (the kv source's rows, every position), and
//! stash the stream state entering layer d for the chunk's last `rows` rows. The final chunk does the same, then the
//! decoder layers d..39 run over the stashed rows with the full path's chunk boundaries (`ced.zig`'s exact rule: the
//! rows that influence the kept outputs and the kept window ring).
const std = @import("std");
const mtl = @import("metal");
const ced = @import("ced.zig");
const ft = @import("forward_types.zig");
const model = @import("model.zig");
const ops = @import("ops.zig");
const view = @import("view.zig");
const impl = @import("forward_impl.zig");
const Model = model.Model;
const Seg = model.Seg;
const Logits = model.Logits;
const View = view.View;

/// Stash ring rows (positions modulo this); at least the cone's rows.
pub const cap: u32 = 4096;
/// A chunk shorter than this is not worth a cone (decode rows, tails).
const min_rows: u32 = 17;
/// The kept rows: the window ring's last rows.
const kept: u32 = ft.ring;
const max_chunks = 64;

pub const Phase = enum { off, encoder, decoder };

/// Bytes a row occupies in each stash section: stream, branch, pre, post, comb.
const sect = [5]usize{ 20480 * 2, 5120 * 2, 4 * 4, 4 * 4, 16 * 4 };

const Chunk = struct { pos: u32, rows: u32 };

pub const Cone = struct {
    on: ?bool = null,
    phase: Phase = .off,
    /// the decoder's first layer
    d: u32 = 20,
    stash: ?mtl.Buffer = null,
    ids: [cap]u32 = undefined,
    chunks: [max_chunks]Chunk = undefined,
    count: usize = 0,
    pending: bool = false,
    slot: u32 = 0,
    first: u32 = 0,
    next: u32 = 0,
    /// the decoder chunk's first position (loadStash)
    at: u32 = 0,

    fn sectionOffset(i: usize) usize {
        var off: usize = 0;
        for (sect[0..i]) |b| off += b * cap;
        return off;
    }

    fn totalBytes() usize {
        return sectionOffset(5);
    }
};

fn enabled(m: *Model) bool {
    const c = &m.cone;
    if (c.on) |v| return v;
    var v = false;
    if (std.c.getenv("TF_DSV41_CONE")) |e| v = std.mem.eql(u8, std.mem.span(e), "1");
    if (v) {
        c.d = ced.decoderStart(&m.cfg) catch {
            c.on = false;
            return false;
        };
        const rows = ced.suffixRows(&m.cfg, kept) catch cap + 1;
        if (rows > cap) v = false;
    }
    c.on = v;
    return v;
}

fn eligible(m: *Model, segs: []const Seg, logits: Logits) bool {
    return segs.len == 1 and logits != .all and m.profile == null and m.observer == null and
        m.attention_observer == null and m.pipelines.hc_boundary != null and m.publish_limit == 0 and
        std.c.getenv("TF_DSV41_LAYER_CB") == null and std.c.getenv("TF_DSV41_HID_HASH") == null and
        std.c.getenv("TF_DSV41_STAGE_CB") == null;
}

/// The cone path of one forward, or null when the call takes the full path.
pub fn route(m: *Model, segs: []const Seg, logits: Logits) !?[]const f32 {
    if (!enabled(m)) return null;
    const c = &m.cone;
    if (c.pending and (segs.len != 1 or segs[0].slot != c.slot or segs[0].p0 != c.next or logits == .all)) _ = try replay(m, .none);
    if (!eligible(m, segs, logits)) return null;
    const sg = segs[0];
    const rows: u32 = @intCast(sg.ids.len);
    const final = logits != .none;
    if (!c.pending) {
        if (final or rows < min_rows) return null;
        c.pending = true;
        c.slot = sg.slot;
        c.first = sg.p0;
        c.next = sg.p0;
        c.count = 0;
    }
    try ensureStash(m);
    c.phase = .encoder;
    defer c.phase = .off;
    _ = try impl.forwardCore(m, segs, .none);
    for (sg.ids, 0..) |id, i| c.ids[(sg.p0 + i) % cap] = id;
    if (c.count == max_chunks) {
        std.mem.copyForwards(Chunk, c.chunks[0 .. max_chunks - 1], c.chunks[1..max_chunks]);
        c.count -= 1;
    }
    c.chunks[c.count] = .{ .pos = sg.p0, .rows = rows };
    c.count += 1;
    c.next = sg.p0 + rows;
    if (!final) return m.workspace.logits(0);
    return try replay(m, logits);
}

/// The decoder layers over the stashed rows, chunk by chunk, ending the pending prompt.
pub fn replay(m: *Model, logits: Logits) ![]const f32 {
    const c = &m.cone;
    const end = c.next;
    const r = ced.suffixRows(&m.cfg, kept) catch return error.InvalidPrefillChunk;
    const start = @max(end -| r, c.first);
    const state = &m.caches[c.slot];
    c.pending = false;
    defer c.phase = .off;
    var out: []const f32 = m.workspace.logits(0);
    for (c.chunks[0..c.count], 0..) |ch, i| {
        const a = @max(ch.pos, start);
        const b = ch.pos + ch.rows;
        if (a >= b) continue;
        state.position = a;
        c.at = a;
        c.phase = .decoder;
        var ids: [cap]u32 = undefined;
        for (0..b - a) |k| ids[k] = c.ids[(a + k) % cap];
        const last = i + 1 == c.count;
        out = try impl.forwardCore(m, &.{.{ .slot = c.slot, .ids = ids[0 .. b - a], .p0 = a }}, if (last) logits else .none);
    }
    c.count = 0;
    return out;
}

fn ensureStash(m: *Model) !void {
    const c = &m.cone;
    if (c.stash != null) return;
    const buf = try m.device.buffer(Cone.totalBytes(), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    m.residency.add(buf);
    c.stash = buf;
}

fn sources(m: *const Model) [5]View {
    const ws = &m.workspace;
    return .{ m.streamView(), ws.view(.branch), ws.view(.previous_pre), ws.view(.post), ws.view(.comb) };
}

/// Copy the state entering the decoder (the chunk's last rows, at most the cone's) into the stash ring.
pub fn stashRows(m: *Model, enc: mtl.ComputeEncoder, rows: u32, p0: u32) void {
    const c = &m.cone;
    const r = ced.suffixRows(&m.cfg, kept) catch cap;
    const take = @min(rows, r);
    const q0 = p0 + rows - take;
    const first = @min(take, cap - q0 % cap);
    const src = sources(m);
    for (src, 0..) |s, i| {
        const row = sect[i];
        const base = Cone.sectionOffset(i);
        const dst0: View = .{ .buffer = c.stash.?, .offset = base + @as(usize, q0 % cap) * row };
        ops.encodeDump(enc, &m.pipelines.ops, s.shifted(@as(usize, rows - take) * row), dst0, @intCast(first * row / 4), false);
        if (first < take) {
            const dst1: View = .{ .buffer = c.stash.?, .offset = base };
            ops.encodeDump(enc, &m.pipelines.ops, s.shifted(@as(usize, rows - take + first) * row), dst1, @intCast((take - first) * row / 4), false);
        }
    }
    enc.barrier();
}

/// Load the stashed rows [at, at + rows) as the pending state entering the decoder layers.
pub fn loadStash(m: *Model, enc: mtl.ComputeEncoder, rows: u32) void {
    const c = &m.cone;
    const ws = &m.workspace;
    const dst = [5]View{ ws.view(.stream), ws.view(.branch), ws.view(.previous_pre), ws.view(.post), ws.view(.comb) };
    const first = @min(rows, cap - c.at % cap);
    for (dst, 0..) |d, i| {
        const row = sect[i];
        const base = Cone.sectionOffset(i);
        const src0: View = .{ .buffer = c.stash.?, .offset = base + @as(usize, c.at % cap) * row };
        ops.encodeDump(enc, &m.pipelines.ops, src0, d, @intCast(first * row / 4), false);
        if (first < rows) {
            const src1: View = .{ .buffer = c.stash.?, .offset = base };
            ops.encodeDump(enc, &m.pipelines.ops, src1, d.shifted(@as(usize, first) * row), @intCast((rows - first) * row / 4), false);
        }
    }
    enc.barrier();
    m.stream_alt = false;
    m.pending_expand = true;
}
