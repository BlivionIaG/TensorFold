//! A blank host Snapshot an importer writes into, and the seal that makes it restorable by `snapshotRestore`.
const std = @import("std");
const mtl = @import("metal");
const kv = @import("kv.zig");
const prefix = @import("kv_prefix.zig");
const engram = @import("engram.zig");
const model = @import("model.zig");
const split_feed = @import("split_feed.zig");
const entry_async = @import("entry_async.zig");
const restore = @import("handoff/restore.zig");
const Packed = @import("handoff/metal.zig").Packed;
const Model = model.Model;
const Snapshot = Model.Snapshot;

/// A zeroed Snapshot for a prefix of `rows` tokens: `storage` laid out for those rows only, `live`, taps; position 0.
pub fn blank(m: *Model, rows: u32) !*Snapshot {
    const plan = m.caches[0].plan;
    if (rows > plan.capacity) return error.SnapshotOutOfRange;
    const capacity = prefix.capacityFor(rows, plan.capacity);
    const snap = try m.gpa.create(Snapshot);
    errdefer m.gpa.destroy(snap);
    const storage = try m.gpa.alloc(u8, kv.cache.pool_layout.storageBytes(capacity));
    errdefer m.gpa.free(storage);
    const live = try m.gpa.alloc(u8, plan.mutableBytes());
    errdefer m.gpa.free(live);
    const taps = try m.gpa.alloc(u8, m.taps.length());
    errdefer m.gpa.free(taps);
    @memset(storage, 0);
    @memset(live, 0);
    @memset(taps, 0);
    snap.* = .{ .storage = storage, .capacity = capacity, .live = live, .taps = taps, .position = 0, .history = @splat(0), .taps_row = 0, .taps_n = 0, .taps_rows_last = 0, .taps_rows_valid = 0, .taps_base = 0 };
    return snap;
}

/// Stands the imported state after `upto` tokens: position, Engram record, and taps without rows.
pub fn seal(m: *const Model, snap: *Snapshot, tokens: []const u32, upto: u32) !void {
    if (upto == 0 or upto > tokens.len) return error.InvalidPrefix;
    if (upto > snap.capacity) return error.SnapshotOutOfRange;
    const tm = m.workspace.view(.token_map);
    const map = engram.hash.TokenMap{ .bytes = (tm.buffer.contents() + tm.offset)[0 .. engram.hash.raw_vocab * 4] };
    snap.history = try engram.hash.resumeRecord(map, tokens, upto);
    snap.position = upto;
    snap.taps_row = upto % kv.cache.window;
    snap.taps_base = snap.taps_row;
    snap.taps_n = 0;
    snap.taps_rows_last = 0;
    snap.taps_rows_valid = 0;
}

/// A chunk of a layer-split prompt: layers `cut`.. over the rows of ids, entered from the producer's boundary rows.
/// The rows are copied before this returns; the pass may still run (entry_async.zig).
pub fn forwardEntered(m: *Model, slot: u32, ids: []const u32, rows: []const u8, cut: u32) !void {
    if (!m.valid) return error.InvalidModelState;
    if (slot >= m.lanes) return error.InvalidLaneSlot;
    if (cut == 0 or cut >= m.cfg.layers) return error.BadCut;
    if (ids.len == 0 or rows.len != ids.len * split_feed.row_bytes) return error.GateRows;
    if (m.gate.entered() or m.gate.stop_dir != null) return error.GateBusy;
    const n: u32 = @intCast(ids.len);
    // the host's copy runs while the pass before is on the GPU
    const b = try entry_async.stage(m, rows, n);
    m.pipe.reading = b;
    defer entry_async.done(m);
    try entry_async.prepare(m);
    m.gate.entry_buf = m.pipe.bufs[b].?;
    m.gate.k = cut;
    defer m.gate.off();
    _ = try m.forwardSegs(&.{.{ .slot = slot, .ids = ids, .p0 = m.caches[slot].position }}, .none);
}

/// Layers [0, cut) of a landed layer-split state (`snap`, sealed after its `upto` rows) into `slot`
pub fn restoreLayers(m: *Model, slot: u32, snap: *const Snapshot, cut: u32) !void {
    try entry_async.settle(m);
    if (slot >= m.lanes) return error.InvalidLaneSlot;
    if (snap.taps.len != m.taps.length()) return error.SnapshotLayout;
    try restoreCache(&m.caches[slot], snap, cut);
    m.attention_workspace.selection_layer = null;
    m.attention_workspace.selection_rows = 0;
    m.attention_workspace.candidate_rows = 0;
}

/// `restoreLayers` on the slot's cache alone.
fn restoreCache(state: *kv.cache.State, snap: *const Snapshot, cut: u32) !void {
    if (snap.live.len != state.plan.mutableBytes()) return error.SnapshotLayout;
    if (snap.position != state.position) return error.PositionMismatch;
    const dst: Packed = .{ .live = state.live.contents()[0..state.plan.mutableBytes()], .storage = state.storage.contents()[0..state.plan.packedBytes()], .layers = state.plan.layers, .capacity = state.plan.capacity };
    const src: Packed = .{ .live = snap.live, .storage = snap.storage, .layers = state.plan.layers, .capacity = snap.capacity };
    try restore.layers(dst, src, cut, state.position);
}

test "a layer-split restore writes where the engine's cache reads layers below the cut, and only there" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const gpa = std.testing.allocator;
    var no_taps: [0]u8 = .{};
    const cache = kv.cache;
    for ([_]struct { cut: u32, layers: u32, position: u32 }{ .{ .cut = 14, .layers = 40, .position = 301 }, .{ .cut = 8, .layers = 43, .position = 130 }, .{ .cut = 2, .layers = 40, .position = 7 } }) |cs| {
        var state = try cache.State.init(device, .{ .capacity = 512, .layers = cs.layers });
        defer state.deinit();
        @memset(state.live.contents()[0..state.plan.mutableBytes()], 0xd0);
        @memset(state.storage.contents()[0..state.plan.packedBytes()], 0xd0);
        state.position = cs.position;
        const cap = cs.position + cs.position % 2;
        const snap = try gpa.create(Snapshot);
        defer gpa.destroy(snap);
        const live = try gpa.alloc(u8, state.plan.mutableBytes());
        defer gpa.free(live);
        const storage = try gpa.alloc(u8, cache.pool_layout.storageBytes(cap));
        defer gpa.free(storage);
        @memset(live, 0x5c);
        @memset(storage, 0x5c);
        snap.* = .{ .storage = storage, .capacity = cap, .live = live, .taps = &no_taps, .position = cs.position, .history = @splat(0), .taps_row = 0, .taps_n = 0, .taps_rows_last = 0, .taps_rows_valid = 0, .taps_base = 0 };
        try restoreCache(&state, snap, cs.cut);
        const bytes = state.live.contents();
        for (0..cs.layers) |l| {
            const ring = try state.ring(@intCast(l));
            const want: u8 = if (l < cs.cut) 0x5c else 0xd0;
            for (bytes[ring.offset..][0 .. cache.window * cache.ring_stride]) |b| try std.testing.expectEqual(want, b);
        }
        // each owner's carry, capture and pool rows, found as the engine finds them
        for (cache.owners, cache.ratios) |owner, ratio| {
            const src = try state.source(owner, ratio);
            const below = owner < cs.cut;
            const want: u8 = if (below) 0x5c else 0xd0;
            for (bytes[src.partial.offset..][0 .. 2 * ratio * 512 * 4]) |b| try std.testing.expectEqual(want, b);
            for (bytes[src.capture.offset..][0..cache.captureBytes()]) |b| try std.testing.expectEqual(want, b);
            const pool_bytes = state.storage.contents();
            const used = (cs.position + ratio - 1) / ratio;
            for (0..state.plan.capacity / ratio) |row| {
                const w: u8 = if (below and row < used) 0x5c else 0xd0;
                try std.testing.expectEqual(w, pool_bytes[src.main.offset + row * cache.main_stride]);
                try std.testing.expectEqual(w, pool_bytes[src.index.offset + row * cache.index_stride + cache.index_stride - 1]);
            }
        }
    }
    // the slot must stand where the landed state does
    var state = try cache.State.init(device, .{ .capacity = 64, .layers = 40 });
    defer state.deinit();
    const live = try gpa.alloc(u8, state.plan.mutableBytes());
    defer gpa.free(live);
    const storage = try gpa.alloc(u8, cache.pool_layout.storageBytes(64));
    defer gpa.free(storage);
    var snap: Snapshot = .{ .storage = storage, .capacity = 64, .live = live, .taps = &no_taps, .position = 10, .history = @splat(0), .taps_row = 0, .taps_n = 0, .taps_rows_last = 0, .taps_rows_valid = 0, .taps_base = 0 };
    state.position = 9;
    try std.testing.expectError(error.PositionMismatch, restoreCache(&state, &snap, 8));
}
