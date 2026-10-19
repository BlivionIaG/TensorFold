//! Flash Next prompt states for the shared prompt cache: one stream's caches after `at` tokens, copied on the device.
//!
//! BF16 paged snapshot design (not implemented; KvPageSnapshot remains mandatory)
//!
//! A paged sequence is not a dense kc/vc allocation. Its kc/vc views alias compact device windows. Dividing their
//! lengths by seq.cap, or copying at * row_bytes from them, cannot describe the logical history. The host store
//! alone is also insufficient: rows still in each device window have not necessarily been written back.
//!
//! Format: keep the dense/int8 format separate, with an explicit version and paging discriminator in Snap. The
//! paged header records at, history, head presence, BF16 row width, attention-layer count, index ratio, cap,
//! window/rows/slots, wr and nblocks, plus checked offsets/lengths for every section. No stored GPU pointers are
//! executable after restore. The initial exact-state format requires equal pager geometry and capacity on the
//! destination; reject a mismatch before touching it. A future capacity-changing format must explicitly re-encode
//! slot2blk's layer * nblocks keys and rebuild both inverse tables, rather than copying those keys unchanged.
//!
//! Payload, at one completed sequence boundary:
//! - Existing convolution, recurrent state, PLE tail, IKC [0,at), complete pooled blocks and optional MTP head
//!   state. Include partially pooled raw IKC rows; they are needed when the next block completes.
//! - Canonical BF16 host K/V [layer][K|V][0,at), after flushing every pending window interval [win0[layer],at).
//! - The entire compact device K/V allocation, including all windows and shared slot payloads, and the entire
//!   tabs allocation: blk2slot, mark, ctl (including global clock, sticky status and per-layer epoch), slot2blk,
//!   stamp. Copying only host history and resetting tables would lose the requested mappings and LRU state.
//! - Host win0, rebases and written arrays. graph_prepared is transient execution state and restores as false.
//!   work lists and status_host are scratch: rebuild/overwrite them before use; never restore stale completion
//!   status as proof that an operation completed. Logical rows >= at are invalid irrespective of their bytes.
//!
//! Save ordering:
//! 1. Exclude scheduling/replay for this sequence. Establish an event dependency from every producer (including
//!    prefill reduction and copy streams) onto the snapshot stream; synchronizing this stream alone is insufficient
//!    unless the caller already joined those producers. Reject an outstanding draft or inconsistent head mark.
//! 2. Validate ALL layer ranges, geometry and checked sizes before any copy; require win0 aligned to index_ratio,
//!    win0 <= at <= min(cap,win0+wr), and reject a sticky paging failure after a completed status transfer.
//! 3. Allocate the budgeted snapshot, enqueue window writebacks, then canonical host-prefix copies, device K/V,
//!    tables and ordinary state on the ordered stream. Snapshot storage owns its copies; it never aliases a live
//!    pager. Preserve counters as well as data; flush itself does not advance epochs or rebase windows.
//! 4. Wait for the copies before publishing the immutable snapshot or releasing source/storage ownership. On a
//!    submission failure, drain already submitted copies before freeing their targets. If draining fails, retain
//!    ownership until the device/session is quiesced. A partial snapshot must never reach the prompt cache.
//!
//! Restore ordering:
//! 1. Validate format, bounds, byte counts, pager geometry, sequence capacity and optional MTP head availability
//!    before changing the destination. Drain prior uses and pending graph instantiation before overwriting state.
//! 2. Invalidate MTP verification generation before the first mutation, so previously published or delayed results
//!    cannot be consumed. Restore into the destination's own allocations; never transplant snapshot addresses.
//! 3. Copy canonical host rows, compact device payload, tables, IKC/pooled/recurrent/tail/head state, then wait.
//!    Restore host origins/counters only after device completion; clear graph_prepared and refresh paging status.
//!    Matching tables, slot payloads and clock must become visible together before another attention call.
//! 4. Commit seq.fed/history and reset Draft's transient drafted/rows state and len only after successful completion.
//!    On a failure after mutation, discard/poison the destination rather than publishing its previous position over
//!    partially restored memory. The snapshot remains immutable and reusable. Destroy snapshots only after their
//!    last asynchronous reader completes; dropping a source sequence must not invalidate saved storage.
//!
//! Integration work outside this file's permitted scope blocks safely enabling this format:
//! - cuda_lanes.zig snapBytes currently calls bytesKv(at,head,rows), which has no pager capacity/configuration. The
//!   prompt-cache bytes callback is an upper bound checked BEFORE allocation. It must include fixed device/window
//!   and table payloads and the paged header as well as logical rows. Adding hidden buffers to Snap would undercharge
//!   both host/device cache admission. Use an explicit pager-aware layout derived from the loaded engine, not a
//!   mutable environment lookup in bytesKv. Future sparse packing can shrink storage with a proven upper bound.
//! - cuda_rank_engine.zig must validate pager compatibility on both ranks, join their producers and coordinate
//!   success/failure before resumed execution. Its existing poison-on-copy-error behavior must remain in force;
//!   no rank may expose results from a partially restored pair. Validate on the leader before dispatching restore,
//!   and ensure a peer validation failure cannot leave the leader's mutated sequence available for reuse.
//! - The graph owner must join pending instantiations and invalidate incompatible executable keys on a restore;
//!   same-allocation graphs are reusable only when captured origin/shape arguments match the restored metadata.
//!   Both ranks must agree on graph/eager dispatch. Clearing graph_prepared alone does not establish that agreement.
//!
//! Acceptance gates are specified in zig/tests/cuda/exl3_kvpage_tests.zig. The CPU tests there establish the current
//! rejection and flush ordering only; they do not establish snapshot support or GPU/TP2 correctness.

const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const Draft = @import("cuda_mtp_draft.zig").Draft;
const kv = @import("cuda_exl3_kv.zig");

const conv_bytes = (gdn.taps - 1) * gdn.conv_dim * 2;
const state_bytes = wt.state_n * 4;
const index_row = wt.index_dim * 2;
const tail_bytes = wt.ple_tail_n * wt.wide * 2;

/// One attention cache's first `at` key, value and indexer-key rows, and the pooled blocks they complete.
fn attnBytes(at: usize) usize {
    return at * (2 * prompt.kv_row_bytes + index_row) + at / wt.index_ratio * index_row;
}

/// Device bytes a state after `at` tokens takes: the DeltaNet's are fixed, each attention cache grows by the token.
pub fn bytes(at: usize, head: bool) usize {
    return wt.linear_n * (conv_bytes + state_bytes) + (wt.attn_n + @intFromBool(head)) * attnBytes(at) + tail_bytes;
}

/// `bytes` for a sequence whose K/V rows are `r` (int8 codes and scales, or one KV head a rank: cuda_exl3_kv.zig);
/// the head's cache stays upstream's.
pub fn bytesKv(at: usize, head: bool, r: kv.Rows) usize {
    const layer = at * (2 * (r.kv + r.scales) + index_row) + at / wt.index_ratio * index_row;
    return wt.linear_n * (conv_bytes + state_bytes) + wt.attn_n * layer + @as(usize, @intFromBool(head)) * attnBytes(at) + tail_bytes;
}

/// A sequence's K/V rows a token, read from its buffers (cap rows each).
fn seqRows(seq: *const prompt.Seq) kv.Rows {
    return .{ .kv = seq.kc[0].len / seq.cap, .scales = if (seq.kv8) |c| c.ks[0].len / seq.cap else 0 };
}

/// One kept state; `head`: it holds the MTP head's cache over the same rows, which a drafting stream needs.
pub const Snap = struct {
    rank_id: u64 = 0,
    buf: cuda.DeviceBuffer,
    /// The pinned host memory `buf` borrows (states parked on the host: `on_host`), or null (device memory).
    host: ?cuda.HostBuffer = null,
    at: usize,
    history: [2]i64,
    head: bool,
};

const Way = enum { save, restore };

const Piece = struct { bufs: []cuda.DeviceBuffer, n: usize };

/// An attention cache's pieces for `at` rows: complete pooled blocks only (the rest pool again as later rows land).
fn attnPieces(kc: []cuda.DeviceBuffer, vc: []cuda.DeviceBuffer, ikc: []cuda.DeviceBuffer, pooled: []cuda.DeviceBuffer, at: usize) [4]Piece {
    return attnPiecesRow(kc, vc, ikc, pooled, at, prompt.kv_row_bytes);
}

fn attnPiecesRow(kc: []cuda.DeviceBuffer, vc: []cuda.DeviceBuffer, ikc: []cuda.DeviceBuffer, pooled: []cuda.DeviceBuffer, at: usize, kv_row: usize) [4]Piece {
    return .{
        .{ .bufs = kc, .n = at * kv_row },
        .{ .bufs = vc, .n = at * kv_row },
        .{ .bufs = ikc, .n = at * index_row },
        .{ .bufs = pooled, .n = at / wt.index_ratio * index_row },
    };
}

/// Each piece of the sequence (and the head's cache, with `d`) to or from its place in the buffer at `base`.
fn move(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, base: u64, at: usize, way: Way) !void {
    if (seq.paged != null) return error.KvPageSnapshot; // paged K/V (cuda_exl3_kvpage.zig) has no snapshot yet
    const dr = stream.d;
    var off: u64 = 0;
    var pieces: [13]Piece = undefined;
    pieces[0] = .{ .bufs = seq.conv, .n = conv_bytes };
    pieces[1] = .{ .bufs = seq.state, .n = state_bytes };
    const rows = seqRows(seq);
    pieces[2..6].* = attnPiecesRow(seq.kc, seq.vc, seq.ikc, seq.pooled, at, rows.kv);
    pieces[6] = .{ .bufs = (&seq.tail)[0..1], .n = tail_bytes };
    var count: usize = 7;
    if (seq.kv8) |c| {
        // the int8 cache's scales (its codes went with kc/vc)
        pieces[7] = .{ .bufs = c.ks, .n = at * rows.scales };
        pieces[8] = .{ .bufs = c.vs, .n = at * rows.scales };
        count = 9;
    }
    if (d) |h| {
        pieces[count..][0..4].* = attnPieces((&h.kc)[0..1], (&h.vc)[0..1], (&h.ikc)[0..1], (&h.pooled)[0..1], at);
        count += 4;
    }
    for (pieces[0..count]) |p| for (p.bufs) |b| {
        if (p.n == 0) continue;
        const dst, const src = if (way == .save) .{ base + off, b.ptr } else .{ b.ptr, base + off };
        try dr.check(dr.api.cuMemcpyDtoDAsync_v2(dst, src, p.n, stream.handle), "cuMemcpyDtoDAsync");
        off += p.n;
    };
    std.debug.assert(off == bytesKv(at, d != null, rows));
}

/// A copy of `seq` (and of the head's cache, with `d`) as it stands after `seq.fed` tokens.
/// Kept states go to mapped pinned host memory (copied at ~14 GB/s by the same DtoD copies; the device keeps none of
/// them): set once at load (cuda_lanes_load.zig, TF_FLASHNEXT_SNAP_HOST), and the native host then budgets the prompt
/// cache in host memory (Loaded.host_snaps).
pub var on_host: bool = false;

pub fn save(gpa: std.mem.Allocator, driver: *const cuda.Driver, stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft) !*Snap {
    const at = seq.fed;
    if (seq.paged != null) return error.KvPageSnapshot;
    if (d) |h| if (h.len != at or h.drafted != 0) return error.NotAtMark;
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    const n = bytesKv(at, d != null, seqRows(seq));
    snap.* = .{ .buf = undefined, .at = at, .history = seq.history, .head = d != null };
    if (on_host) {
        var h = try cuda.HostBuffer.allocMapped(driver, @max(n, 256));
        errdefer h.free();
        snap.buf = .{ .d = driver, .ptr = try h.device(), .len = n, .borrowed = true };
        snap.host = h;
    } else snap.buf = try cuda.DeviceBuffer.alloc(driver, n);
    errdefer if (snap.host) |*h| h.free() else snap.buf.free();
    try move(stream, seq, d, snap.buf.ptr, at, .save);
    try stream.synchronize();
    return snap;
}

/// `seq` (and the head's cache, with `d`) becomes `snap`'s state; refused before any copy when it cannot hold it.
pub fn restore(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, snap: *const Snap) !void {
    if (snap.at > seq.cap) return error.SequenceTooShort;
    if (d) |h| if (!snap.head or snap.at > h.cap) return error.NoHeadState;
    if (d) |h| h.verify_cache.invalidate();
    // a state with the head's rows serves a stream without drafts too: those rows stay in the buffer
    try move(stream, seq, d, snap.buf.ptr, snap.at, .restore);
    try stream.synchronize();
    seq.history = snap.history;
    seq.fed = snap.at;
    if (d) |h| {
        h.reset();
        h.len = snap.at;
    }
}

pub fn drop(gpa: std.mem.Allocator, stream: cuda.Stream, snap: *Snap) void {
    stream.synchronize() catch {};
    snap.buf.free();
    if (snap.host) |*h| h.free();
    gpa.destroy(snap);
}

test "a placed state's bytes: int8 codes and scales or one KV head a token, the head's rows upstream's" {
    const r = kv.snapRows(true, 2);
    try std.testing.expectEqual(@as(usize, 12 * (2 * (512 + 32) + 128 * 2)), bytesKv(1, false, r) - bytesKv(0, false, r));
    try std.testing.expectEqual(bytesKv(1, false, r) - bytesKv(0, false, r) + (2 * 2 * 256 * 2 + 128 * 2), bytesKv(1, true, r) - bytesKv(0, true, r));
    try std.testing.expectEqual(bytes(7, true), bytesKv(7, true, kv.snapRows(false, 2)));
    const half = kv.snapRows(false, 1);
    try std.testing.expectEqual(@as(usize, 12 * (2 * 256 * 2 + 128 * 2)), bytesKv(1, false, half) - bytesKv(0, false, half));
}

test "a state's bytes: about 110 MiB of DeltaNet states and the n-gram tail, then 27 KiB a token" {
    try std.testing.expectEqual(@as(usize, 12 * (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, false) - bytes(0, false));
    // a complete block of 4 adds its pooled row
    try std.testing.expectEqual(@as(usize, 12 * (4 * (2 * 2 * 256 * 2 + 128 * 2) + 128 * 2)), bytes(4, false) - bytes(0, false));
    // the head's cache is one more attention layer's rows
    try std.testing.expectEqual(@as(usize, 13 * (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, true) - bytes(0, true));
    try std.testing.expect(bytes(0, false) > 100 << 20 and bytes(0, false) < 120 << 20);
}
