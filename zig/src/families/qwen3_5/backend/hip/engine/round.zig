//! A lane round on the engine: its device plan, its forward (replayed, captured or eager), its draws and its keep.

const std = @import("std");
const platform = @import("sys");
const hip = @import("hip");
const state = @import("../forward/state.zig");
const win = @import("../forward/window.zig");
const fwd = @import("../forward/forward.zig");
const plan = @import("../forward/plan.zig");
const view = @import("../model/view.zig");
const draw = @import("hip_serve").draw;
const round_graphs = @import("round_graphs.zig");
const Engine = @import("engine.zig").Engine;

/// What a round does with its shape's graph: run eagerly, replay it, or capture it (every rank does the same).
pub const Pick = enum(u32) { eager, replay, capture };

/// What a verify hands back: the round's final rows, padding included.
pub const Verified = struct { hidden: hip.ops.Tensor };

/// One stream's rows of a round: tokens (the pending one, then drafts) from slot `pos` over its caches.
pub const Rows = struct { caches: *state.Caches, pos: usize, tokens: []const u32, shift: i32 = 0 };

const Chosen = struct { pick: Pick, entry: ?*round_graphs.Entry, shape: plan.Shape };

/// The slots a keep names and the plan they index.
const Last = struct { plan: hip.plan_ops.Plan, keep_at: usize, hidden: u64 };

/// Keep lists in flight: a flush's pinned words stay until its copy has run, which a later round's sync guarantees.
const keep_ring = 4;

pub const State = struct {
    buffer: plan.Buffer,
    /// The caches the padding rows run in.
    scratch: state.Caches,
    /// Each linear layer's snapshots of the round in flight.
    snaps: [][2]u64,
    wins: []plan.Window,
    /// The rows each slot keeps at the next flush (negative: not listed), and the pinned words they go up in.
    staging: []i32,
    keeps: hip.HostBuffer,
    ring: usize = 0,
    /// The candidate count of each row of the round being planned: its top-k launch's words in the plan.
    ks: []i32,
    /// The mirror slot the next queued round stores its pick to (1 up, round to 1): slot 0 serves the rounds waited for at once.
    mirror_next: usize = 1,
    /// For checks: a layer-by-layer trace of the forward (graphs off), and a least bucket of rows.
    trace: ?fwd.Trace = null,
    pad_to: usize = 0,
    chosen: ?Chosen = null,
    last: ?Last = null,
    /// The layout of the plan in flight, for the copy a round starts with.
    layout: plan.Layout = undefined,
    votes: ?hip.Votes = null,

    pub fn init(e: *Engine) !State {
        const m = e.model();
        if (!win.supported(e.ops(&e.rounds), m)) return error.RoundsUnsupported;
        const rows = e.o.batch_rows;
        var buffer = try plan.Buffer.init(&e.driver, rows, m.spec.n_layers, e.pool.count);
        errdefer buffer.deinit();
        var scratch = try state.Caches.initFull(e.gpa, &e.pool, m, rows);
        errdefer scratch.deinit(e.gpa);
        const snaps = try e.gpa.alloc([2]u64, m.spec.n_layers);
        errdefer e.gpa.free(snaps);
        const wins = try e.gpa.alloc(plan.Window, rows);
        errdefer e.gpa.free(wins);
        const staging = try e.gpa.alloc(i32, rows + 1);
        errdefer e.gpa.free(staging);
        @memset(staging, -1);
        const ks = try e.gpa.alloc(i32, rows);
        errdefer e.gpa.free(ks);
        var votes: ?hip.Votes = if (e.o.world > 1) try hip.Votes.init(&e.driver, e.o.world) else null;
        errdefer if (votes) |*v| v.deinit();
        const keeps = try hip.HostBuffer.alloc(&e.driver, keep_ring * 4 * (rows + 1));
        return .{
            .buffer = buffer,
            .scratch = scratch,
            .snaps = snaps,
            .wins = wins,
            .staging = staging,
            .keeps = keeps,
            .ks = ks,
            .votes = votes,
        };
    }

    pub fn deinit(s: *State, e: *Engine) void {
        if (s.votes) |*v| v.deinit();
        s.keeps.free();
        e.gpa.free(s.ks);
        e.gpa.free(s.staging);
        e.gpa.free(s.wins);
        e.gpa.free(s.snaps);
        s.scratch.deinit(e.gpa);
        s.buffer.deinit();
    }
};

/// The shape of a round of `rows`: its bucket of rows, its slots and the keys its walk covers.
pub fn shapeOf(e: *Engine, rows: []const Rows) !plan.Shape {
    var total: usize = 0;
    var visible: usize = 0;
    for (rows) |r| {
        total += r.tokens.len;
        visible = @max(visible, r.pos + r.tokens.len);
    }
    if (rows.len == 0 or total == 0 or total > e.o.batch_rows) return error.WindowTooWide;
    const cap = std.mem.alignForward(usize, e.o.capacity, plan.min_span);
    const bucket = plan.bucketOf(@max(total, e.round.pad_to), e.o.batch_rows);
    const slots = plan.bucketOf(rows.len + 1, e.o.batch_rows + 1);
    return .{ .rows = @intCast(bucket), .slots = @intCast(slots), .span = @intCast(@min(plan.spanOf(visible), cap)) };
}

/// `shape` with what the round's rows draw on the device: the stride of its top-k launch and whether its argmax runs (a
/// tensor-parallel group draws from the joined rows afterwards, so its shape has neither). `reqs` are the rows' draws.
fn withDraws(e: *Engine, shape: *plan.Shape, reqs: []const draw.Request) !void {
    if (e.model().tp != null) return;
    if (reqs.len > e.round.ks.len) return error.WindowTooWide;
    const a = e.drawer.asks(reqs, e.round.ks);
    shape.stride = plan.strideOf(a.widest);
    // the argmax serves greedy rows; with none and a top-k launch it is left out, with neither it stays as it was
    shape.argmax = a.greedy or a.widest == 0;
}

/// The round's graph choice: from the shape's history, or `forced` (rank 0's pick, which a follower obeys); `reqs` the rows' draws.
pub fn choose(e: *Engine, rows: []const Rows, reqs: []const draw.Request, forced: ?Pick) !Pick {
    var shape = try shapeOf(e, rows);
    try withDraws(e, &shape, reqs);
    return pick(e, shape, forced);
}

fn pick(e: *Engine, shape: plan.Shape, forced: ?Pick) !Pick {
    const st = &e.round;
    st.chosen = .{ .pick = .eager, .entry = null, .shape = shape };
    if (!e.o.graphs) {
        if ((forced orelse .eager) != .eager) return error.GraphsDisagree;
        return .eager;
    }
    const entry = try e.graphs.find(shape);
    const mine: Pick = switch (entry.state) {
        .failed => .eager,
        .ready => .replay,
        .seen => .capture,
    };
    const chosen = forced orelse mine;
    if (chosen == .replay and entry.state != .ready) return error.GraphsDisagree;
    st.chosen = .{ .pick = chosen, .entry = entry, .shape = shape };
    return chosen;
}

fn nowNs() u64 {
    var ts: platform.c.timespec = undefined;
    _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Every window in one forward: `out` the token of every row per `reqs`; snapshots live until the next round's `keep`.
pub fn verify(e: *Engine, rows: []const Rows, reqs: []const draw.Request, out: []u32) !Verified {
    const st = &e.round;
    const m = e.model();
    const began = nowNs();
    if (st.chosen == null) _ = try choose(e, rows, reqs, null);
    const c = st.chosen.?;
    defer st.chosen = null;
    var total: usize = 0;
    for (rows, 0..) |r, i| {
        if (r.pos + r.tokens.len > r.caches.covered()) return error.ContextFull;
        st.wins[i] = .{ .desc = r.caches.desc.base(), .pos = r.pos, .tokens = r.tokens, .shift = r.shift };
        total += r.tokens.len;
    }
    e.rounds.reset();
    const o = e.ops(&e.rounds);
    try win.kept(o, m, c.shape.rows, st.snaps);
    const l = plan.Layout.of(c.shape.rows, c.shape.slots, m.spec.n_layers);
    // the draws the shape names: its rows' candidate counts, and the host words the argmax also stores to (slot 0: waited for at once)
    var draws: plan.Draws = .{};
    var done_draws: draw.Done = .{};
    if (m.tp == null) {
        _ = e.drawer.asks(reqs[0..total], st.ks);
        draws = .{ .ks = st.ks[0..total], .mirror = if (c.shape.argmax) e.drawer.mirrorAddr(0) else 0 };
        done_draws = .{ .argmax = c.shape.argmax, .mirror = if (c.shape.argmax) 0 else null, .stride = c.shape.stride, .rows = c.shape.rows };
    }
    st.buffer.fill(l, c.shape, st.wins[0..rows.len], st.scratch.desc.base(), st.snaps, draws);
    st.layout = l;
    const p: hip.plan_ops.Plan = .{ .args = st.buffer.args(l), .rows = c.shape.rows, .slots = c.shape.slots };
    const r: win.Round = .{ .plan = p, .tokens = st.buffer.dev.base() + 4 * l.tokens, .span = c.shape.span, .inputs = st.snaps, .positions3 = st.buffer.dev.base() + 4 * l.rope, .rotary = st.buffer.dev.base() + 4 * l.rotary };
    // experts whose copies landed since the last round are resident from this round's table
    if (e.weights.experts) |s| try s.beginRound();
    const done = try run(e, c, r);
    e.graphs.rounds += 1;
    e.graphs.submit_ns += nowNs() - began;
    try e.drawRows(o, done.y, total, reqs, out, done_draws);
    // the round is over (drawRows waited): its picks decide which experts the next rounds' copies move
    if (e.weights.experts) |s| try s.endRound(total, nowNs() - began);
    st.last = .{ .plan = p, .keep_at = 4 * l.keep, .hidden = done.hidden.ptr };
    return .{ .hidden = done.hidden };
}

/// Whether a one-token round can be queued ahead: one rank holds the head, and a lone row is the round's whole bucket.
pub fn canQueue(e: *Engine) bool {
    return e.model().tp == null and e.round.pad_to <= 1;
}

/// Launches the one-token greedy round of `row` and returns before it has run: its pick is stored to mirror slot `slot` of the drawer
/// (to be read once an event recorded behind the round has completed) and stays on the device as the plan's first word, the token
/// of the round queued after it. `fed`: this round's token is that word, the pick of the round launched before it (the plan's copy
/// leaves it in place); otherwise it is `row.tokens[0]`. The keep of the round before goes first, from the caller's `flush`.
pub fn queued(e: *Engine, row: Rows, fed: bool) !usize {
    const st = &e.round;
    const m = e.model();
    if (!canQueue(e) or row.tokens.len != 1) return error.NotPipelined;
    const began = nowNs();
    const one = [1]Rows{row};
    var shape = try shapeOf(e, &one);
    shape.piped = true;
    _ = try pick(e, shape, null);
    const c = st.chosen.?;
    defer st.chosen = null;
    if (row.pos + 1 > row.caches.covered()) return error.ContextFull;
    st.wins[0] = .{ .desc = row.caches.desc.base(), .pos = row.pos, .tokens = row.tokens, .shift = row.shift };
    e.rounds.reset();
    const o = e.ops(&e.rounds);
    try win.kept(o, m, c.shape.rows, st.snaps);
    const l = plan.Layout.of(c.shape.rows, c.shape.slots, m.spec.n_layers);
    const slot = st.mirror_next;
    st.mirror_next = if (slot + 1 == draw.mirror_slots) 1 else slot + 1;
    const words = st.buffer.nextSlot();
    plan.fillInto(words, l, c.shape, st.wins[0..1], st.scratch.desc.base(), st.snaps, .{ .mirror = e.drawer.mirrorAddr(slot) });
    st.layout = l;
    try st.buffer.sendQueued(e.stream.handle, l, words, fed);
    const p: hip.plan_ops.Plan = .{ .args = st.buffer.args(l), .rows = c.shape.rows, .slots = c.shape.slots };
    const r: win.Round = .{ .plan = p, .tokens = st.buffer.dev.base() + 4 * l.tokens, .span = c.shape.span, .inputs = st.snaps, .positions3 = st.buffer.dev.base() + 4 * l.rope, .rotary = st.buffer.dev.base() + 4 * l.rotary };
    const done = try run(e, c, r);
    e.graphs.rounds += 1;
    e.graphs.submit_ns += nowNs() - began;
    st.last = .{ .plan = p, .keep_at = 4 * l.keep, .hidden = done.hidden.ptr };
    return slot;
}

/// The forward and its logits projection, and the draws its shape names (the argmax and top-k of every row when this rank holds the whole head).
fn body(e: *Engine, r: win.Round) !round_graphs.Out {
    const st = &e.round;
    const shape = st.chosen.?.shape;
    const l = st.layout;
    const at = st.buffer.dev.base();
    // the plan goes up first, inside the graph; a queued round's went up ahead of it, and its pick replaces the plan's first word, its token
    if (!shape.piped) try st.buffer.send(e.stream.handle, l);
    const out = if (shape.piped) at + 4 * l.tokens else e.drawer.argmaxAt();
    const dl = e.drawer.launches(shape.rows, shape.stride, shape.argmax, out, at + 4 * l.mirror, at + 4 * l.ks);
    return forwardOn(e.ops(&e.rounds), e.model(), r, st.trace, dl);
}

/// `body` after the plan's copy, on `o`: on the counting stream it measures what a round takes.
pub fn forwardOn(ops: hip.ops.Ops, m: *const view.Model, r: win.Round, trace: ?fwd.Trace, dl: draw.Launches) !round_graphs.Out {
    var o = ops;
    // a round's rows keep their decode kernels however many share it
    o.window = true;
    const hidden = try win.forward(o, m, r, trace);
    const y = try o.project(hidden, m.head, r.plan.rows, false);
    if (m.tp == null) {
        if (dl.argmax) try o.argmaxRowsTo(y, r.plan.rows, m.head.n, dl.argmax_at, dl.mirror_at);
        if (dl.stride > 0) try o.topkRows(y, dl.rows, m.head.n, dl.ks, dl.stride, dl.ids, dl.vals);
    }
    return .{ .hidden = hidden, .y = y, .used = o.arena.used };
}

/// One round's forward as `choose` picked: replayed from the shape's graph, captured, or eager.
fn run(e: *Engine, c: Chosen, r: win.Round) !round_graphs.Out {
    const entry = c.entry orelse return body(e, r);
    switch (c.pick) {
        .eager => return body(e, r),
        .replay => {
            e.rounds.used = entry.out.used;
            try entry.exec.?.launchOn(e.stream);
            e.graphs.replayed += 1;
            return entry.out;
        },
        .capture => return capture(e, entry, r),
    }
}

/// Records the round, then every rank votes: all kept replay it, one decline runs it eagerly, one failure stops all.
fn capture(e: *Engine, entry: *round_graphs.Entry, r: win.Round) !round_graphs.Out {
    const at = e.rounds.mark();
    const began = nowNs();
    const m = &e.bridge.model;
    const got = hip.vote.record(round_graphs.Out, e.gpa, e.stream, &m.tp, e.o.policy.failsCapture(e.o.rank), Body{
        .e = e,
        .r = r,
    }, Body.run);
    e.graphs.capture_ns += nowNs() - began;
    if (got.split) |split| e.graphs.hold(entry, split, got.out);
    const all = if (e.round.votes) |*v| try v.agree(e.comm, got.vote, e.stream) else got.vote;
    if (all == .kept) {
        try entry.exec.?.launchOn(e.stream);
        return entry.out;
    }
    e.graphs.revoke(entry);
    e.rounds.release(at);
    if (all == .declined) return body(e, r);
    if (e.o.world == 1) return error.OutOfDeviceMemory;
    // no rank runs another collective once one could not take a round
    e.stopped = true;
    return error.GroupStopped;
}

/// A round's forward as the recorder runs it.
const Body = struct {
    e: *Engine,
    r: win.Round,

    fn run(b: Body) anyerror!round_graphs.Out {
        return body(b.e, b.r);
    }
};

/// Marks slot `slot` of the last verify to keep its first `rows` rows at the next `flush`.
pub fn mark(e: *Engine, slot: usize, rows: usize) void {
    e.round.staging[slot] = @intCast(rows);
}

/// Keeps every marked slot of the last verify in one launch: linear states after the kept rows, last kept final rows.
pub fn flush(e: *Engine) !void {
    const st = &e.round;
    const last = st.last orelse return;
    const slots = last.plan.slots;
    var any = false;
    for (st.staging[0..slots]) |k| any = any or k >= 0;
    if (!any) return;
    const per = e.o.batch_rows + 1;
    const from = 4 * st.ring * per;
    const words = st.keeps.slice(i32)[st.ring * per ..][0..slots];
    st.ring = (st.ring + 1) % keep_ring;
    @memcpy(words, st.staging[0..slots]);
    @memset(st.staging[0..slots], -1);
    try st.buffer.dev.uploadAsync(last.keep_at, st.keeps.view(from, from + 4 * slots), e.stream);
    try win.keep(e.ops(&e.rounds), e.model(), last.plan, st.buffer.dev.base() + last.keep_at, last.hidden);
}
