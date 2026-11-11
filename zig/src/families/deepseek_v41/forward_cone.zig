//! Cone prefill (TF_DSV41_CONE=1): encoder layers over every prompt row, the decoder layers over the cone's last rows.
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

/// Stash ring rows (position p at row p % cap); at least the cone's rows.
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
    /// the rows the decoder replays at most
    rows: u32 = 0,
    stash: ?mtl.Buffer = null,
    ids: [cap]u32 = undefined,
    chunks: [max_chunks]Chunk = undefined,
    count: usize = 0,
    pending: bool = false,
    slot: u32 = 0,
    /// the pending run's first position and the next one it continues at
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
    if (m.cone.on) |v| return v;
    return configure(m, @import("sys").environment.read(bool, "TF_DSV41_CONE", false) catch false);
}

/// Set the cone on or off; on only when the config has a decoder the cone rule covers.
fn configure(m: *Model, knob: bool) bool {
    const c = &m.cone;
    c.on = false;
    if (!knob) return false;
    const d = ced.decoderStart(&m.cfg) catch return false;
    const rows = ced.suffixRows(&m.cfg, kept) catch return false;
    if (rows > cap) return false;
    // the decoder layers read the first one's pool: no other kv source among them
    for (d + 1..m.cfg.layers) |l| if (m.cfg.isKvSource(@intCast(l))) return false;
    c.d = d;
    c.rows = rows;
    c.on = true;
    return true;
}

/// The check harness's switch between the paths (no run pending).
pub fn force(m: *Model, on: bool) void {
    _ = configure(m, on);
}

fn eligible(m: *Model, segs: []const Seg, logits: Logits) bool {
    return segs.len == 1 and logits != .all and m.profile == null and m.observer == null and
        m.attention_observer == null and m.pipelines.hc_boundary != null and m.publish_limit == 0 and
        std.c.getenv("TF_DSV41_LAYER_CB") == null and std.c.getenv("TF_DSV41_HID_HASH") == null and
        std.c.getenv("TF_DSV41_STAGE_CB") == null and std.c.getenv("TF_DSV41_ATTN_PROF") == null;
}

/// The cone path of one forward, or null when the call takes the full path.
pub fn route(m: *Model, segs: []const Seg, logits: Logits) !?[]const f32 {
    if (!enabled(m)) return null;
    const c = &m.cone;
    if (segs.len == 0 or segs[0].slot >= m.lanes) return null;
    const sg = segs[0];
    var mine = false;
    if (c.pending) for (segs) |s| if (s.slot == c.slot) {
        mine = true;
    };
    // another prompt is half done in the stash: this call takes the full path
    if (c.pending and !mine) return null;
    const ok = eligible(m, segs, logits);
    if (mine) {
        // the slot's state was replaced under the pending run: nothing to replay
        if (segs.len != 1 or sg.p0 != c.next) {
            c.pending = false;
            return null;
        }
        if (!ok or c.count == max_chunks) _ = try replay(m, .none);
    }
    if (!ok) return null;
    const rows: u32 = @intCast(sg.ids.len);
    const final = logits != .none;
    if (!c.pending) {
        if (rows < min_rows or (final and rows <= c.rows)) return null;
        c.pending = true;
        c.slot = sg.slot;
        c.first = sg.p0;
        c.next = sg.p0;
        c.count = 0;
    }
    try ensureStash(m);
    c.phase = .encoder;
    defer c.phase = .off;
    const out = try impl.forwardCore(m, segs, .none);
    for (sg.ids, 0..) |id, i| c.ids[(sg.p0 + i) % cap] = id;
    c.chunks[c.count] = .{ .pos = sg.p0, .rows = rows };
    c.count += 1;
    c.next = sg.p0 + rows;
    if (!final) return out;
    return try replay(m, logits);
}

/// The decoder layers over the stashed rows, chunk by chunk, ending the pending run.
pub fn replay(m: *Model, logits: Logits) ![]const f32 {
    const c = &m.cone;
    const end = c.next;
    const start = replayStart(end, c.first, c.rows);
    const state = &m.caches[c.slot];
    c.pending = false;
    defer c.phase = .off;
    var out: []const f32 = m.workspace.logits(0);
    for (c.chunks[0..c.count], 0..) |ch, i| {
        const a = @max(ch.pos, start);
        const b = ch.pos + ch.rows;
        // the drafter's tap ring advances as the full pass advances it, over rows the replay skips too
        if (model.tapsOn()) {
            const extra = @min(ch.rows, 128) - (if (a < b) @min(b - a, 128) else 0);
            m.taps_row = (m.taps_row + extra) % 128;
            m.taps_n = @min(m.taps_n + extra, 128);
        }
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

/// Replay a pending run on `slot` so its state is exact at the position it stands at.
pub fn flush(m: *Model, slot: u32) !void {
    if (m.cone.pending and m.cone.slot == slot) _ = try replay(m, .none);
}

/// Forget a pending run on `slot` whose state is being replaced.
pub fn drop(m: *Model, slot: u32) void {
    if (m.cone.pending and m.cone.slot == slot) m.cone.pending = false;
}

pub fn release(m: *Model) void {
    if (m.cone.stash) |b| b.deinit();
    m.cone.stash = null;
}

fn ensureStash(m: *Model) !void {
    const c = &m.cone;
    if (c.stash != null) return;
    const buf = try m.device.buffer(Cone.totalBytes(), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer buf.deinit();
    m.residency.add(buf);
    m.residency.commit();
    m.residency.requestResidency();
    c.stash = buf;
}

fn sources(m: *const Model) [5]View {
    const ws = &m.workspace;
    return .{ m.streamView(), ws.view(.branch), ws.view(.previous_pre), ws.view(.post), ws.view(.comb) };
}

/// Copy the state entering the decoder (the chunk's last rows, at most the cone's) into the stash ring.
pub fn stashRows(m: *Model, enc: mtl.ComputeEncoder, rows: u32, p0: u32) void {
    const c = &m.cone;
    const take = @min(rows, c.rows);
    const q0 = p0 + rows - take;
    const split = ringRuns(q0, take);
    const src = sources(m);
    for (src, 0..) |s, i| {
        const row = sect[i];
        const base = Cone.sectionOffset(i);
        const dst0: View = .{ .buffer = c.stash.?, .offset = base + @as(usize, q0 % cap) * row };
        ops.encodeDump(enc, &m.pipelines.ops, s.shifted(@as(usize, rows - take) * row), dst0, @intCast(split[0] * row / 4), false);
        if (split[1] != 0) {
            const dst1: View = .{ .buffer = c.stash.?, .offset = base };
            ops.encodeDump(enc, &m.pipelines.ops, s.shifted(@as(usize, rows - take + split[0]) * row), dst1, @intCast(split[1] * row / 4), false);
        }
    }
    enc.barrier();
}

/// Load the stashed rows [at, at + rows) as the pending state entering the decoder layers.
pub fn loadStash(m: *Model, enc: mtl.ComputeEncoder, rows: u32) void {
    const c = &m.cone;
    const ws = &m.workspace;
    const dst = [5]View{ ws.view(.stream), ws.view(.branch), ws.view(.previous_pre), ws.view(.post), ws.view(.comb) };
    const split = ringRuns(c.at, rows);
    for (dst, 0..) |d, i| {
        const row = sect[i];
        const base = Cone.sectionOffset(i);
        const src0: View = .{ .buffer = c.stash.?, .offset = base + @as(usize, c.at % cap) * row };
        ops.encodeDump(enc, &m.pipelines.ops, src0, d, @intCast(split[0] * row / 4), false);
        if (split[1] != 0) {
            const src1: View = .{ .buffer = c.stash.?, .offset = base };
            ops.encodeDump(enc, &m.pipelines.ops, src1, d.shifted(@as(usize, split[0]) * row), @intCast(split[1] * row / 4), false);
        }
    }
    enc.barrier();
    m.stream_alt = false;
    m.pending_expand = true;
}

/// `n` stash rows from position `at` as up to two ring runs: to the ring's end, then from its start.
fn ringRuns(at: u32, n: u32) [2]u32 {
    const first = @min(n, cap - at % cap);
    return .{ first, n - first };
}

/// Where the decoder replay of a run that ends at `end` begins: the cone's rows back, but not before the run's start.
fn replayStart(end: u32, first: u32, rows: u32) u32 {
    return @max(end -| rows, first);
}

test "stash rows wrap at the ring end in two runs" {
    try std.testing.expectEqual([2]u32{ 100, 0 }, ringRuns(0, 100));
    try std.testing.expectEqual([2]u32{ 100, 0 }, ringRuns(cap - 100, 100));
    try std.testing.expectEqual([2]u32{ 100, 2568 }, ringRuns(cap - 100, 2668));
    try std.testing.expectEqual([2]u32{ 100, 2568 }, ringRuns(3 * cap - 100, 2668));
}

test "the replay starts a cone back, or at the run's first position" {
    try std.testing.expectEqual(@as(u32, 5332), replayStart(8000, 0, 2668));
    try std.testing.expectEqual(@as(u32, 4096), replayStart(8000, 4096, 4000));
    try std.testing.expectEqual(@as(u32, 0), replayStart(2000, 0, 2668));
    try std.testing.expectEqual(@as(u32, 1000), replayStart(2000, 1000, 2668));
}

test "the stash holds every section's rows for the whole ring" {
    var row: usize = 0;
    for (sect) |b| row += b;
    try std.testing.expectEqual(@as(usize, 51296), row);
    try std.testing.expectEqual(row * cap, Cone.totalBytes());
}
