//! CUDA graphs for an EXL3 Flash Next decode step with the expert offload pool (TF_FLASHNEXT_EXL3_GRAPH=1).
//!
//! With the pool planned on the device (cuda_exl3_dplan.zig) a decode step has no host wait between its first launch
//! and its logits, so the step is captured once and replayed: a decode round (Engine.round, one row a stream) and an
//! MTP verify window (Engine.forward with every row's logits, up to the planner's row bound) each take ONE graph
//! launch; at tp=2 the graph also holds the step's NCCL sums and the head's gathers (both ranks capture the same
//! calls in the same order, the rank engine keeps them in lockstep).
//!
//! What a replay may change and how (the graph-safety rules of exl3_graph.zig):
//!  - token ids, positions and the n-gram rows: pinned host bytes the graph's copies read, rewritten by `prepare`
//!    before each launch (upstream's cuda_round_graph.zig does the same for its tp round);
//!  - the experts each layer reads: the pool's pointer tables, rewritten in place by the device planner;
//!  - everything else (weights, scratch, caches, the sequence's buffers, the logits buffer) sits at fixed addresses,
//!    which the key holds: a different address, row count, attention chunk count or score-tile count captures anew.
//! A step the graph cannot take (more rows than the planner's bound, media) runs eagerly with the host cut.
//!
//! TF_FLASHNEXT_EXL3_DPLAN=1 alone plans on the device without capturing (the same kernels launched one by one).

const std = @import("std");
const cuda = @import("cuda");
const dplan = @import("cuda_exl3_dplan.zig");
const wt = @import("cuda_prompt_wt.zig");

const dims = wt.dims;
const vocab = wt.vocab;
const att_rows = @import("cuda_batch_state.zig").att_rows;

/// TF_FLASHNEXT_EXL3_GRAPH=1: decode rounds and verify windows replay captured graphs.
pub fn enabled() bool {
    if (force) |f| return f == .graph;
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_FLASHNEXT_EXL3_GRAPH_STRICT=1: a failed capture is an error, not a fallback (tests).
fn strict() bool {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_STRICT") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// A test's override of the environment: the host cut, the device plan launched eagerly, or graphs.
pub const Mode = enum { off, dplan, graph };
pub var force: ?Mode = null;

const max_lanes = 32;

pub const Key = struct {
    /// 1: a decode round; 2: a window.
    kind: u8,
    rows: usize,
    /// Per lane (a window: lane 0): the sequence's buffers, its attention chunks and score tiles at the step's end.
    seq: [max_lanes][4]u64 = @splat(@splat(0)),
    chunks: [max_lanes]u32 = @splat(0),
    score_tiles: [max_lanes]u32 = @splat(0),
    /// The pass's scratch (its first buffer and its row capacity, which sets every other buffer's offset).
    scratch: u64,
    scratch_cap: usize = 0,
    scores: u64,
    output: u64,
    streams: u64 = 0,
    keep: u64 = 0,
    /// The device planner's state (a new planner is new memory).
    plan: u64 = 0,

    fn seams(end: usize, chunks: *u32, tiles: *u32) void {
        chunks.* = @intCast((@min(end, wt.dense_keys) + 511) / 512);
        tiles.* = if (end / wt.index_ratio > wt.index_top) @intCast(((end + 3) / 4 + 63) / 64) else 0;
    }

    fn seqOf(q: anytype) [4]u64 {
        return .{ if (q.kc.len > 0) q.kc[0].ptr else 0, if (q.conv.len > 0) q.conv[0].ptr else 0, q.tail.ptr, q.cap };
    }
};

/// True while this module captures a step: the pass's pinned uploads become copy nodes (Uploads.copy).
pub var capturing: bool = false;

/// A copy node from pinned host bytes (the pass's staging): the graph reads them at every replay, as `prepare` left
/// them. A pageable source fails the capture (the driver refuses it), so stale custody copies cannot hide here.
pub fn pinnedCopy(s: cuda.Stream, dst: u64, bytes: []const u8) !void {
    if (bytes.len == 0) return;
    try s.d.check(s.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, s.handle), "cuMemcpyHtoDAsync");
}

/// `template`: the graph an update took this exec's parameters from, kept while the exec uses them (a tp graph's
/// NCCL resources are tied to the graph they were captured in).
const Entry = struct { key: Key, exec: cuda.graph.Exec, used: u64, template: ?cuda.graph.Graph = null };

fn drop(x: *Entry) void {
    x.exec.deinit();
    if (x.template) |*g| g.deinit();
}

pub const Stats = struct {
    captures: u64 = 0,
    replays: u64 = 0,
    /// Steps on the device plan launched one by one (TF_FLASHNEXT_EXL3_DPLAN=1 without graphs).
    eager_dplan: u64 = 0,
    /// Steps that left for the host cut (too many rows, media, no pools).
    declined: u64 = 0,
    capture_ns: u64 = 0,
    /// Captures that updated a cached graph of the same shape in place instead of instantiating one.
    updates: u64 = 0,
};

pub var stats: Stats = .{};

/// Graphs kept (LRU): a round and up to `max_rows` window sizes per stream, at two seams.
const cache_n = 32;
var entries: [cache_n]?Entry = @splat(null);
var clock: u64 = 0;
/// A capture that failed once turns graphs off for the process (the steps keep the device plan).
var broken: bool = false;

pub fn clear() void {
    clearPrefill();
    for (&entries) |*e| if (e.*) |*x| {
        drop(x);
        e.* = null;
    };
}

/// The engine's deinit: graphs and the device planner go.
pub fn releaseFor(owner: *const anyopaque) void {
    clear();
    dplan.releaseFor(owner);
}

fn find(key: Key) ?*Entry {
    for (&entries) |*e| if (e.*) |*x| if (std.meta.eql(x.key, key)) return x;
    return null;
}

fn slot() *?Entry {
    var best: usize = 0;
    var best_used: u64 = std.math.maxInt(u64);
    for (&entries, 0..) |*e, i| {
        if (e.* == null) return e;
        if (e.*.?.used < best_used) {
            best_used = e.*.?.used;
            best = i;
        }
    }
    drop(&entries[best].?);
    entries[best] = null;
    return &entries[best];
}

/// The key without the addresses a cudaGraphExecUpdate may change: equal shapes capture the same topology.
fn sameShape(a: Key, b: Key) bool {
    return a.kind == b.kind and a.rows == b.rows and std.mem.eql(u32, &a.chunks, &b.chunks) and
        std.mem.eql(u32, &a.score_tiles, &b.score_tiles) and a.scratch_cap == b.scratch_cap and
        (a.keep != 0) == (b.keep != 0) and (a.streams != 0) == (b.streams != 0);
}

/// Keys seen without a graph, and how often (a key is captured at its `after`-th sighting; one-off shapes, e.g. a
/// request's rare window sizes, stay eager).
const Seen = struct { key: Key, n: u32, used: u64 };
var seen: [64]?Seen = @splat(null);

fn sighting(key: Key) u32 {
    var oldest: usize = 0;
    var oldest_used: u64 = std.math.maxInt(u64);
    for (&seen, 0..) |*e, i| {
        if (e.*) |*x| {
            if (std.meta.eql(x.key, key)) {
                x.n += 1;
                x.used = clock;
                return x.n;
            }
            if (x.used < oldest_used) {
                oldest_used = x.used;
                oldest = i;
            }
        } else {
            oldest = i;
            oldest_used = 0;
        }
    }
    seen[oldest] = .{ .key = key, .n = 1, .used = clock };
    return 1;
}

/// TF_FLASHNEXT_EXL3_GRAPH_TP_UPDATE=1: tp graphs too are updated in place (their template kept alive); default off.
fn tpUpdate() bool {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_TP_UPDATE") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_FLASHNEXT_EXL3_GRAPH_AFTER: the sighting of a key that captures it (default 2; 1 captures at once).
fn after() u32 {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_AFTER") orelse return 2;
    return @max(1, std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2);
}

/// Captures `pass.chunk(ids, pos0, null)` (prepared) under `key`, or replays the graph captured under it. A key not
/// yet seen `after()` times runs eagerly (prepared, uncaptured). A new key whose shape matches a cached graph's
/// updates that graph in place (cudaGraphExecUpdate) instead of instantiating one, except with a collective (NCCL's
/// captured resources belong to the graph they were captured in, so a tp graph is always instantiated).
fn run(key: Key, pass: anytype, ids: []const u32, pos0: usize) !void {
    clock += 1;
    if (find(key)) |e| {
        e.used = clock;
        stats.replays += 1;
        return e.exec.launchOn(pass.stream);
    }
    if (sighting(key) < after()) {
        stats.eager_dplan += 1;
        pass.prepared = true;
        return pass.chunk(ids, pos0, null);
    }
    const t0 = std.Io.Clock.awake.now(pass.io).toNanoseconds();
    try cuda.graph.beginCapture(pass.stream, .thread_local);
    pass.prepared = true;
    capturing = true;
    defer capturing = false;
    pass.chunk(ids, pos0, null) catch |err| {
        if (cuda.graph.endCapture(pass.stream)) |g| {
            var discarded = g;
            discarded.deinit();
        } else |_| {}
        return err;
    };
    var graph = try cuda.graph.endCapture(pass.stream);
    var kept = false;
    defer if (!kept) graph.deinit();
    const t1 = std.Io.Clock.awake.now(pass.io).toNanoseconds();
    var how: []const u8 = "instantiated";
    const tp = pass.m.collective != null;
    const at: *Entry = blk: {
        if (!tp or tpUpdate()) {
            for (&entries) |*e| if (e.*) |*x| if (sameShape(x.key, key)) {
                // every earlier launch finished (each step ends on the host), so the old parameters are free to go
                const r = try x.exec.update(graph);
                if (r == .success) {
                    x.key = key;
                    x.used = clock;
                    if (x.template) |*old| old.deinit();
                    x.template = null;
                    if (tp) {
                        x.template = graph;
                        kept = true;
                    }
                    stats.updates += 1;
                    how = "updated";
                    break :blk x;
                }
            };
        }
        var exec = try graph.instantiate();
        errdefer exec.deinit();
        try exec.upload(pass.stream);
        const sl = slot();
        sl.* = .{ .key = key, .exec = exec, .used = clock };
        break :blk &sl.*.?;
    };
    stats.captures += 1;
    const t2 = std.Io.Clock.awake.now(pass.io).toNanoseconds();
    stats.capture_ns += @intCast(t2 - t0);
    std.log.info("flashnext exl3 graph: captured a {s} of {d} rows in {d:.1} ms + {s} in {d:.1} ms ({d} captures, {d} updates, {d} replays, {d} eager, {d} declined steps, {d} residency pulls so far)", .{ if (key.kind == 1) "decode round" else "window", key.rows, @as(f64, @floatFromInt(t1 - t0)) / 1e6, how, @as(f64, @floatFromInt(t2 - t1)) / 1e6, stats.captures, stats.updates, stats.replays, stats.eager_dplan, stats.declined, if (dplan.current()) |dp| dp.pulls else 0 });
    try at.exec.launchOn(pass.stream);
}

/// A round's host inputs (what chunk's stage and gather would write), outside capture and replay.
fn prepareRound(pass: anytype, ids: []const u32) !void {
    const s = pass.s;
    try s.ready();
    for (s.hostIds(ids.len), ids) |*out, token| out.* = @intCast(token);
    const positions = s.hostPos(ids.len * 4);
    @memset(positions, 0);
    for (pass.lanes, 0..) |lane, i| {
        positions[i * 4] = @intCast(lane.pos);
        try pass.m.ngram.rows(lane.seq.history[0..], ids[i..][0..1], s.hostEmb(ids.len)[i * dims ..][0..dims], s.hostXs(ids.len)[i * (dims / 32) ..][0 .. dims / 32]);
    }
}

/// A window's host inputs: ids, one position a block of attention rows, the n-gram rows after the sequence's history.
fn prepareWindow(pass: anytype, ids: []const u32, pos0: usize) !void {
    const s = pass.s;
    const n = ids.len;
    try s.ready();
    for (s.hostIds(n), ids) |*out, token| out.* = @intCast(token);
    const blocks = (n + att_rows - 1) / att_rows;
    const positions = s.hostPos(blocks * 4);
    @memset(positions, 0);
    for (0..blocks) |b| positions[4 * b] = @intCast(pos0 + b * att_rows);
    try pass.m.ngram.rows(pass.seq.history[0..], ids, s.hostEmb(n), s.hostXs(n));
}

/// The device planner for `e`, or null when its decode steps keep the host cut.
fn planner(e: anytype, rows: usize) !?*dplan.DevPlan {
    if (!dplan.enabled()) return null;
    const dp = (try dplan.forLayers(e.gpa, e.driver, e, e.layers[0..e.n_layers])) orelse return null;
    if (rows > dp.max_rows) {
        stats.declined += 1;
        return null;
    }
    return dp;
}

/// Engine.round's hook: true when this module ran the round (device plan, captured or not); false leaves it to the
/// engine. `p` is the round's pass (lanes set), `out` its logits [rows, vocab].
pub fn round(e: anytype, p: anytype, seqs: anytype, ids: []const u32, scores: u64, out: u64) !bool {
    for (seqs) |q| if (q.media != null) return false;
    if (seqs.len > max_lanes) return false;
    const dp = (try planner(e, ids.len)) orelse return false;
    try dp.toDevice(p.stream);
    dplan.active = dp;
    defer dplan.active = null;
    if (!enabled() or broken) {
        stats.eager_dplan += 1;
        try p.chunk(ids, 0, null);
        return true;
    }
    var key: Key = .{ .kind = 1, .rows = ids.len, .scratch = p.s.ids, .scratch_cap = p.s.cap, .scores = scores, .output = out, .plan = dp.state.ptr };
    for (seqs, 0..) |q, i| {
        key.seq[i] = Key.seqOf(q);
        Key.seams(q.fed + 1, &key.chunks[i], &key.score_tiles[i]);
    }
    var hist: [max_lanes][2]i64 = undefined;
    for (seqs, 0..) |q, i| hist[i] = q.history;
    try prepareRound(p, ids);
    run(key, p, ids, 0) catch |err| {
        if (strict()) return err;
        std.log.warn("flashnext exl3 graph: a decode round could not be captured ({t}); rounds run eagerly on the device plan", .{err});
        broken = true;
        for (seqs, 0..) |q, i| q.history = hist[i];
        p.prepared = false;
        try p.chunk(ids, 0, null);
    };
    return true;
}

/// Engine.forward's hook for a window: every row's logits into hooks.logits_out (an MTP verify window, or a one-row
/// step), at most the planner's rows. True when this module ran it (seq.fed and seq.logits advanced as forward does).
pub fn forward(e: anytype, p: anytype, seq: anytype, ids: []const u32) !bool {
    const hooks = p.hooks;
    if (hooks.logits != .all or hooks.logits_out == 0 or seq.media != null or ids.len == 0 or ids.len > att_rows) return false;
    if (seq.fed + ids.len > seq.cap or ids.len > p.s.cap) return false;
    const dp = (try planner(e, ids.len)) orelse return false;
    try dp.toDevice(p.stream);
    dplan.active = dp;
    defer dplan.active = null;
    const n = ids.len;
    if (!enabled() or broken) {
        stats.eager_dplan += 1;
        try p.chunk(ids, seq.fed, null);
    } else {
        var key: Key = .{ .kind = 2, .rows = n, .scratch = p.s.ids, .scratch_cap = p.s.cap, .scores = p.seq.scores, .output = hooks.logits_out, .streams = hooks.streams, .plan = dp.state.ptr };
        key.seq[0] = Key.seqOf(seq);
        Key.seams(seq.fed + n, &key.chunks[0], &key.score_tiles[0]);
        if (hooks.keep) |kp| key.keep = std.hash.Wyhash.hash(@intFromPtr(kp), std.mem.asBytes(kp));
        const hist = seq.history;
        try prepareWindow(p, ids, seq.fed);
        run(key, p, ids, seq.fed) catch |err| {
            if (strict()) return err;
            std.log.warn("flashnext exl3 graph: a window could not be captured ({t}); windows run eagerly on the device plan", .{err});
            broken = true;
            seq.history = hist;
            p.prepared = false;
            try p.chunk(ids, seq.fed, null);
        };
    }
    // the pass's head(.all) copies the last row's scores to the host after the step (outside the graph)
    if (p.m.collective) |c| try c.finish(p.stream) else try p.stream.synchronize();
    const d = p.stream.d;
    try d.check(d.api.cuMemcpyDtoH_v2(seq.logits.ptr, hooks.logits_out + (n - 1) * vocab * 2, vocab * 2), "cuMemcpyDtoH");
    seq.fed += n;
    return true;
}

// ---- prefill: one layer's sub-blocks of a deep chunk -------------------------------------------------------------
//
// The expert-major prefill (cuda_exl3_prefill.zig) runs each layer as its dense sub-blocks (HC read-outs, DeltaNet or
// attention, write-backs, router and top-k over every sub-block of the deep chunk) and then the layer's experts, whose
// passes the host plans from the layer's picks (the ring, the members, the copies): that half stays eager. The dense
// half has no host dependency and is captured per layer: a DeltaNet layer's graph depends only on the chunk's shape
// (rows, sub-block size) and replays for every later chunk of that shape; an attention layer's launch geometry follows
// the chunk's position, so each chunk re-captures it and updates the cached graph in place when the topology holds.

pub const Prefill = enum { eager, replayed, capturing };

const PKey = struct {
    layer: usize,
    rows: usize,
    sub: usize,
    /// The chunk's first position (attention layers only; 0 for DeltaNet layers).
    pos0: usize,
    deep: u64,
    base: u64,
    base_cap: usize,
    seq: [4]u64,
    scores: u64,
    tiles: bool,
};

const PEntry = struct { key: PKey, exec: cuda.graph.Exec, used: u64 };
const pcache_n = 112;
var pentries: [pcache_n]?PEntry = @splat(null);
var pseen: [128]?struct { key: PKey, n: u32 } = @splat(null);
var pseen_at: usize = 0;
var pcur: ?PKey = null;

pub var prefill_stats: struct { captures: u64 = 0, updates: u64 = 0, replays: u64 = 0, eager: u64 = 0, capture_ns: u64 = 0 } = .{};

/// TF_FLASHNEXT_EXL3_GRAPH_PREFILL=0 keeps prefill eager under TF_FLASHNEXT_EXL3_GRAPH=1.
fn prefillEnabled() bool {
    if (!enabled() or broken) return false;
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_PREFILL") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

fn pshape(k: PKey) PKey {
    var s = k;
    s.pos0 = 0;
    return s;
}

fn psighting(k: PKey) u32 {
    const s = pshape(k);
    for (&pseen) |*e| if (e.*) |*x| if (std.meta.eql(x.key, s)) {
        x.n += 1;
        return x.n;
    };
    pseen[pseen_at % pseen.len] = .{ .key = s, .n = 1 };
    pseen_at += 1;
    return 1;
}

pub fn clearPrefill() void {
    for (&pentries) |*e| if (e.*) |*x| {
        x.exec.deinit();
        e.* = null;
    };
    pseen = @splat(null);
}

/// Before a layer's sub-blocks: `.replayed` (the graph ran; skip the sub-blocks), `.capturing` (run them: they are
/// recorded; then `prefillEnd`), or `.eager` (run them; `prefillEnd` is a no-op).
/// `waits`: events of other streams the sub-blocks wait on (expert parallel's per-sub-block sums): a graph cannot wait
/// on work outside it, so a replay or a capture waits for all of them first.
pub fn prefillBegin(p: anytype, linear: bool, layer: usize, rows: usize, sub: usize, pos0: usize, deep: u64, base: anytype, tiles: bool, waits: []const cuda.Event) !Prefill {
    if (!prefillEnabled()) return .eager;
    const q = p.seq;
    const key: PKey = .{
        .layer = layer,
        .rows = rows,
        .sub = sub,
        .pos0 = if (linear) 0 else pos0,
        .deep = deep,
        .base = base.ids,
        .base_cap = base.cap,
        .seq = .{ if (q.conv.len > 0) q.conv[0].ptr else 0, if (q.kc.len > 0) q.kc[0].ptr else 0, q.tail.ptr, q.cap },
        .scores = q.scores,
        .tiles = tiles,
    };
    clock += 1;
    for (&pentries) |*e| if (e.*) |*x| if (std.meta.eql(x.key, key)) {
        x.used = clock;
        prefill_stats.replays += 1;
        for (waits) |ev| try p.stream.wait(ev);
        try x.exec.launchOn(p.stream);
        return .replayed;
    };
    if (psighting(key) < after()) {
        prefill_stats.eager += 1;
        return .eager;
    }
    for (waits) |ev| try p.stream.wait(ev);
    try cuda.graph.beginCapture(p.stream, .thread_local);
    capturing = true;
    pcur = key;
    return .capturing;
}

/// After a layer's sub-blocks ran under `.capturing`: the graph is instantiated (or a cached one of the same shape
/// updated) and launched.
pub fn prefillEnd(p: anytype, how: Prefill) !void {
    if (how != .capturing) return;
    const t0 = std.Io.Clock.awake.now(p.io).toNanoseconds();
    capturing = false;
    const key = pcur.?;
    pcur = null;
    var graph = try cuda.graph.endCapture(p.stream);
    defer graph.deinit();
    const at: *PEntry = blk: {
        for (&pentries) |*e| if (e.*) |*x| if (std.meta.eql(pshape(x.key), pshape(key))) {
            if ((try x.exec.update(graph)) == .success) {
                x.key = key;
                x.used = clock;
                prefill_stats.updates += 1;
                break :blk x;
            }
        };
        var exec = try graph.instantiate();
        errdefer exec.deinit();
        try exec.upload(p.stream);
        var best: usize = 0;
        var best_used: u64 = std.math.maxInt(u64);
        for (&pentries, 0..) |*e, i| {
            if (e.* == null) {
                best = i;
                break;
            }
            if (e.*.?.used < best_used) {
                best_used = e.*.?.used;
                best = i;
            }
        }
        if (pentries[best]) |*old| old.exec.deinit();
        pentries[best] = .{ .key = key, .exec = exec, .used = clock };
        break :blk &pentries[best].?;
    };
    prefill_stats.captures += 1;
    prefill_stats.capture_ns += @intCast(std.Io.Clock.awake.now(p.io).toNanoseconds() - t0);
    try at.exec.launchOn(p.stream);
}

/// A failed capture: closed and dropped; prefill graphs stay off for the process (the chunk's error propagates).
pub fn prefillAbort(p: anytype, how: Prefill) void {
    if (how != .capturing) return;
    capturing = false;
    pcur = null;
    if (cuda.graph.endCapture(p.stream)) |g| {
        var d = g;
        d.deinit();
    } else |_| {}
    std.log.warn("flashnext exl3 graph: a prefill layer could not be captured; prefill and decode graphs are off", .{});
    broken = true;
}
