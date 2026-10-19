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

const Entry = struct { key: Key, exec: cuda.graph.Exec, used: u64 };

pub const Stats = struct {
    captures: u64 = 0,
    replays: u64 = 0,
    /// Steps on the device plan launched one by one (TF_FLASHNEXT_EXL3_DPLAN=1 without graphs).
    eager_dplan: u64 = 0,
    /// Steps that left for the host cut (too many rows, media, no pools).
    declined: u64 = 0,
    capture_ns: u64 = 0,
};

pub var stats: Stats = .{};

const cache_n = 8;
var entries: [cache_n]?Entry = @splat(null);
var clock: u64 = 0;
/// A capture that failed once turns graphs off for the process (the steps keep the device plan).
var broken: bool = false;

pub fn clear() void {
    for (&entries) |*e| if (e.*) |*x| {
        x.exec.deinit();
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
    entries[best].?.exec.deinit();
    entries[best] = null;
    return &entries[best];
}

/// Captures `pass.chunk(ids, pos0, null)` (prepared) under `key`, or replays the graph captured under it.
fn run(key: Key, pass: anytype, ids: []const u32, pos0: usize) !void {
    clock += 1;
    if (find(key)) |e| {
        e.used = clock;
        stats.replays += 1;
        return e.exec.launchOn(pass.stream);
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
    defer graph.deinit();
    var exec = try graph.instantiate();
    errdefer exec.deinit();
    try exec.upload(pass.stream);
    const at = slot();
    at.* = .{ .key = key, .exec = exec, .used = clock };
    stats.captures += 1;
    const took: u64 = @intCast(std.Io.Clock.awake.now(pass.io).toNanoseconds() - t0);
    stats.capture_ns += took;
    std.log.info("flashnext exl3 graph: captured a {s} of {d} rows in {d:.1} ms ({d} captures, {d} replays, {d} declined steps so far)", .{ if (key.kind == 1) "decode round" else "window", key.rows, @as(f64, @floatFromInt(took)) / 1e6, stats.captures, stats.replays, stats.declined });
    try exec.launchOn(pass.stream);
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
