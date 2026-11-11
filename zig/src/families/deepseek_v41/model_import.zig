//! A blank host Snapshot an importer writes into, and the seal that makes it restorable by `snapshotRestore`.
const std = @import("std");
const kv = @import("kv.zig");
const prefix = @import("kv_prefix.zig");
const engram = @import("engram.zig");
const model = @import("model.zig");
const split_feed = @import("split_feed.zig");
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

/// A chunk of a layer-split prompt: layers `cut`.. over the rows of ids, entered from the producer's boundary rows (`rows`:
/// [row][4 x 5120 bf16 stream, 4 f32 pre], as `split_feed` carries them) instead of the embedding and layers below the cut.
/// The slot's caches, positions and the cone's stash take the rows as a normal chunk's layers from the cut on do.
pub fn forwardEntered(m: *Model, slot: u32, ids: []const u32, rows: []const u8, cut: u32) !void {
    if (!m.valid) return error.InvalidModelState;
    if (slot >= m.lanes) return error.InvalidLaneSlot;
    if (cut == 0 or cut >= m.cfg.layers) return error.BadCut;
    if (ids.len == 0 or rows.len != ids.len * split_feed.row_bytes) return error.GateRows;
    if (m.gate.entered() or m.gate.stop_dir != null) return error.GateBusy;
    m.gate.entry_rows = rows;
    m.gate.k = cut;
    defer m.gate.off();
    _ = try m.forwardSegs(&.{.{ .slot = slot, .ids = ids, .p0 = m.caches[slot].position }}, .none);
}

/// Layers [0, cut) of a landed layer-split state (`snap`, sealed after its `upto` rows) into `slot`, which stands at `upto`
/// after its entered chunks; the layers from the cut on, the cone's pending run and the Engram record stay as they are.
pub fn restoreLayers(m: *Model, slot: u32, snap: *const Snapshot, cut: u32) !void {
    if (slot >= m.lanes) return error.InvalidLaneSlot;
    const state = &m.caches[slot];
    if (snap.live.len != state.plan.mutableBytes() or snap.taps.len != m.taps.length()) return error.SnapshotLayout;
    if (snap.position != state.position) return error.PositionMismatch;
    const dst: Packed = .{ .live = state.live.contents()[0..state.plan.mutableBytes()], .storage = state.storage.contents()[0..state.plan.packedBytes()], .layers = state.plan.layers, .capacity = state.plan.capacity };
    const src: Packed = .{ .live = snap.live, .storage = snap.storage, .layers = state.plan.layers, .capacity = snap.capacity };
    try restore.layers(dst, src, cut, state.position);
    m.attention_workspace.selection_layer = null;
    m.attention_workspace.selection_rows = 0;
    m.attention_workspace.candidate_rows = 0;
}
