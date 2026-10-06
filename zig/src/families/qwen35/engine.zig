//! One loaded Qwen3.5 / 3.6 model on one HIP device: the kernel library, a stream, the forward's scratch, and the
//! pinned buffers rounds read and write through; prompts and lane windows run here, draws on the device and host.

const std = @import("std");
const hip = @import("hip");
const view = @import("view.zig");
const state = @import("state.zig");
const fwd = @import("forward.zig");
const win = @import("window.zig");
const weights = @import("weights.zig");
const bridge = @import("bridge.zig");
const sample = @import("sample.zig");
const memory = @import("memory.zig");
const draw = @import("draw.zig");
const slicing = @import("slicing.zig");
const reduce = @import("reduce.zig");
const round_graphs = @import("round_graphs.zig");

pub const Options = struct {
    /// Most positions a stream's caches hold (its prompt, its reply and a window's rows).
    capacity: usize = 0,
    /// Rows a shared forward holds at most.
    batch_rows: usize = 32,
    /// The device ordinal among the visible ones.
    device: c_int = 0,
    /// Positions past a reply a verify writes (the window's rows and one more).
    slack: usize = 0,
    /// Tensor parallelism: this rank of `world` (the model split across them; every rank runs every forward). `id` is
    /// the communicator's unique id, the same on every rank.
    rank: usize = 0,
    world: usize = 1,
    id: ?hip.rccl.UniqueId = null,
    /// Replay rounds from captured graphs (TF_HIP_GRAPHS=0 turns it off).
    graphs: bool = true,
};

/// What a round does with its shape's graph: run eagerly, replay it, or capture it (every rank does the same).
pub const Pick = enum(u32) { eager, replay, capture };

const Chosen = struct { pick: Pick, entry: ?*round_graphs.Entry };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    driver: hip.Driver,
    ctx: hip.Context,
    lib: hip.rocm.Library,
    stream: hip.Stream,
    weights: weights.Model,
    bridge: *bridge.Bridge,
    o: Options,
    /// The scratch below exists (`size` ran).
    sized: bool,
    act: view.Kind,
    dtype: sample.Dtype,
    /// A window's scratch, kept from its verify to its keep (the commit reads the per-row states).
    rounds: hip.Arena,
    /// A prompt's scratch: its rows, the step's temporaries.
    prompts: hip.Arena,
    ids: hip.HostBuffer,
    ids_dev: hip.DeviceBuffer,
    drawer: draw.Drawer,
    /// Tensor parallelism: RCCL and this rank's communicator.
    rccl: hip.rccl.Rccl = undefined,
    comm: hip.rccl.Comm = undefined,
    graphs: round_graphs.Graphs,
    /// The round's graph choice, made before its forward (rank 0 sends it to the others).
    chosen: ?Chosen = null,
    /// The serial of the next caches made, for graphs keyed by stream.
    serial: u64 = 1,

    /// The model on the device, its scratch not yet sized: `size` takes the capacity the memory plan fits.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.o = o;
        e.sized = false;
        // rounds replay captured graphs unless TF_HIP_GRAPHS=0 (the Python engine's TENSORFOLD_GRAPH=0); under tensor
        // parallelism the collectives are captured too, rank 0's pick goes to every rank with the round, and the
        // replays are slower than the eager rounds on RCCL 2.30 (V620 x2-4), so TF_HIP_GRAPHS_TP=1 turns them on
        for ([_][*:0]const u8{ "TF_HIP_GRAPHS", "TENSORFOLD_GRAPH" }) |name| {
            if (std.c.getenv(name)) |v| if (std.mem.eql(u8, std.mem.span(v), "0")) {
                e.o.graphs = false;
            };
        }
        if (o.world > 1) {
            const on = if (std.c.getenv("TF_HIP_GRAPHS_TP")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
            if (!on) e.o.graphs = false;
        }
        e.graphs = .{ .gpa = gpa };
        e.serial = 1;
        e.driver = try hip.Driver.open();
        errdefer e.driver.close();
        e.ctx = try hip.Context.init(&e.driver, o.device);
        errdefer e.ctx.deinit();
        const family = hip.rocm.familyOf(try e.ctx.capability()) orelse return error.UnsupportedGpu;
        e.lib = try hip.rocm.Library.open(e.ctx.d, family);
        errdefer e.lib.close();
        e.act = if (family == .rdna2) .f16 else .bf16;
        e.dtype = if (family == .rdna2) .f16 else .bf16;
        e.stream = try hip.Stream.init(&e.driver, true);
        errdefer e.stream.deinit();
        const group: ?slicing.Rank = if (o.world > 1) .{ .rank = o.rank, .world = o.world } else null;
        if (group != null) {
            e.rccl = try hip.rccl.Rccl.open();
            errdefer e.rccl.close();
            e.comm = try hip.rccl.Comm.init(&e.rccl, o.id orelse return error.NoUniqueId, o.rank, o.world);
        }
        errdefer if (group != null) {
            e.comm.deinit();
            e.rccl.close();
        };
        e.weights = try weights.Model.loadRank(gpa, io, &e.driver, dir, group);
        errdefer e.weights.deinit();
        e.bridge = try bridge.Bridge.init(gpa, &e.driver, &e.weights, e.act);
        errdefer e.bridge.deinit();
        if (group != null) e.bridge.model.tp = e.comm;
        return e;
    }

    /// Allocate the scratch for streams of `capacity` positions (`o.batch_rows` rows a shared forward).
    pub fn size(e: *Engine, capacity: usize) !void {
        const s = e.weights.spec;
        const rows = e.o.batch_rows;
        const need = memory.Scratch.of(s, capacity, rows);
        e.o.capacity = capacity;
        e.rounds = try hip.Arena.init(&e.driver, need.rounds);
        errdefer e.rounds.deinit();
        e.prompts = try hip.Arena.init(&e.driver, need.prompts);
        errdefer e.prompts.deinit();
        e.ids = try hip.HostBuffer.alloc(&e.driver, need.ids);
        errdefer e.ids.free();
        e.ids_dev = try hip.DeviceBuffer.alloc(&e.driver, need.ids);
        errdefer e.ids_dev.free();
        e.drawer = try draw.Drawer.init(e.gpa, &e.driver, e.dtype, e.bridge.model.head.n * e.o.world, rows);
        e.sized = true;
    }

    /// The memory plan for `streams` at once and a window of `target` tokens, from the memory free now.
    pub fn plan(e: *Engine, streams: usize, target: usize) !memory.Plan {
        const info = try e.ctx.memInfo();
        return memory.plan(.{ .spec = e.weights.spec, .act_bytes = e.act.size(), .streams = streams, .rows = e.o.batch_rows, .slack = e.o.slack, .target = target, .free = info.free, .total = info.total });
    }

    /// The least of each value over the ranks, the same on every rank (a window and a prompt cache they all fit).
    pub fn least(e: *Engine, mine: [2]u64) ![2]u64 {
        if (e.o.world < 2) return mine;
        const world = e.o.world;
        var host = try hip.HostBuffer.alloc(&e.driver, 16 * (1 + world));
        defer host.free();
        var send = try hip.DeviceBuffer.alloc(&e.driver, 16);
        defer send.free();
        var recv = try hip.DeviceBuffer.alloc(&e.driver, 16 * world);
        defer recv.free();
        host.slice(u64)[0..2].* = mine;
        try send.uploadAsync(0, host.bytes[0..16], e.stream.handle);
        try e.comm.allGather(send.ptr, recv.ptr, 2, .i64, e.stream.handle);
        try recv.downloadAsync(0, host.bytes[16 .. 16 * (1 + world)], e.stream.handle);
        try e.stream.synchronize();
        var out = mine;
        for (host.slice(u64)[2 .. 2 * (1 + world)], 0..) |v, i| out[i % 2] = @min(out[i % 2], v);
        return out;
    }

    /// Load and size in one: `o.capacity` positions.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        const e = try load(gpa, io, dir, o);
        errdefer e.deinit();
        try e.size(o.capacity);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        if (e.graphs.captured > 0) std.log.info("graphs: {d} captured, {d} rounds replayed", .{ e.graphs.captured, e.graphs.replayed });
        e.graphs.deinit();
        if (e.sized) {
            e.drawer.deinit();
            e.ids_dev.free();
            e.ids.free();
            e.prompts.deinit();
            e.rounds.deinit();
        }
        e.bridge.deinit();
        e.weights.deinit();
        e.stream.deinit();
        e.lib.close();
        if (e.o.world > 1) {
            e.comm.deinit();
            e.rccl.close();
        }
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }

    pub fn model(e: *const Engine) *const view.Model {
        return &e.bridge.model;
    }

    fn ops(e: *Engine, arena: *hip.Arena) hip.ops.Ops {
        return .{ .lib = &e.lib, .stream = e.stream.handle, .arena = arena };
    }

    /// Zeroed caches for `total` positions (at most the capacity), with the serial graphs key them by.
    pub fn newCaches(e: *Engine, total: usize) !state.Caches {
        var c = try state.Caches.init(e.gpa, &e.driver, e.model(), @min(total, e.o.capacity));
        c.serial = e.serial;
        e.serial += 1;
        return c;
    }

    /// Drops the graphs over `caches`, before they are freed.
    pub fn forget(e: *Engine, caches: *const state.Caches) void {
        e.stream.synchronize() catch {};
        e.graphs.forget(caches.serial);
    }

    /// The tokens of `rows` final rows at `hidden`: one projection as the engine's _logits (the ranks' vocabulary
    /// slices joined in rank order under tensor parallelism), each row drawn per `reqs`.
    fn project(e: *Engine, o: hip.ops.Ops, hidden: hip.ops.Tensor, rows: usize, reqs: []const draw.Request, out: []u32) !void {
        const y = try o.affine(hidden, e.model().head, rows, false);
        try e.drawRows(o, y, rows, reqs, out);
    }

    /// Each of `rows` logits rows `y` drawn per `reqs` (whole rows joined first under tensor parallelism).
    fn drawRows(e: *Engine, o: hip.ops.Ops, y: hip.ops.Tensor, rows: usize, reqs: []const draw.Request, out: []u32) !void {
        var whole = y;
        if (e.model().tp) |c| whole = try e.joined(o, c, y, rows);
        try e.drawer.draw(o, e.stream, whole, reqs[0..rows], out);
    }

    /// vocab_gather: every rank's slice of `rows` logits rows, joined on the device into whole rows in rank order.
    fn joined(e: *Engine, o: hip.ops.Ops, c: hip.rccl.Comm, slice: hip.ops.Tensor, rows: usize) !hip.ops.Tensor {
        const width = e.model().head.n;
        const n = rows * width;
        const size_of = slice.kind.size();
        const parts = try o.arena.take(c.world * n * size_of);
        try reduce.gather(o, c, slice, parts, n);
        const whole = try o.arena.take(c.world * n * size_of);
        for (0..rows) |row| for (0..c.world) |r| {
            const to = whole + (row * c.world + r) * width * size_of;
            const from = parts + (r * rows + row) * width * size_of;
            try e.driver.check(e.driver.api.hipMemcpyDtoDAsync(to, from, width * size_of, e.stream.handle), "join logits");
        };
        return .{ .ptr = whole, .kind = slice.kind };
    }

    /// Asked after each layer of a long prompt (the stream is idle then): true ends the pass with `error.Cancelled`.
    pub const Cancel = struct {
        ctx: *anyopaque,
        check: *const fn (ctx: *anyopaque) bool,
    };

    const Poll = struct {
        stream: hip.Stream,
        cancel: Cancel,

        fn layer(ctx: *anyopaque, _: usize, _: hip.ops.Tensor, _: usize) anyerror!void {
            const p: *Poll = @ptrCast(@alignCast(ctx));
            try p.stream.synchronize();
            if (p.cancel.check(p.cancel.ctx)) return error.Cancelled;
        }
    };

    /// The pass over `len` rows with the cancel polled after each layer; null where it cannot stop (a prompt shorter
    /// than a span, or tensor parallelism, where the other ranks would wait in a collective).
    fn poll(e: *Engine, cancel: ?Cancel, len: usize, p: *Poll) ?fwd.Trace {
        const c = cancel orelse return null;
        if (e.o.world > 1 or len < fwd.SPAN) return null;
        p.* = .{ .stream = e.stream, .cancel = c };
        return .{ .ctx = p, .layer = Poll.layer };
    }

    /// Run `prompt[pos0..end]` into `caches`, its logits not read: a cut where the caches are kept.
    pub fn advance(e: *Engine, caches: *state.Caches, prompt: []const u32, pos0: usize, end: usize, cancel: ?Cancel) !void {
        if (end <= pos0 or end > caches.total) return error.PromptTooLong;
        e.prompts.reset();
        const ids = e.ids.slice(u32)[0 .. end - pos0];
        @memcpy(ids, prompt[pos0..end]);
        try e.ids_dev.uploadAsync(0, std.mem.sliceAsBytes(ids), e.stream.handle);
        var p: Poll = undefined;
        _ = try fwd.span(e.ops(&e.prompts), e.model(), caches, e.ids_dev.ptr, end - pos0, pos0, e.poll(cancel, end - pos0, &p));
        try e.stream.synchronize();
    }

    /// Prefill `prompt[pos0..]` into `caches`; the token drawn per `req` from the last row, whose final row is copied to `last` (MTP's input).
    pub fn prefill(e: *Engine, caches: *state.Caches, prompt: []const u32, pos0: usize, last: ?hip.DeviceBuffer, req: draw.Request, cancel: ?Cancel) !u32 {
        const m = e.model();
        const len = prompt.len - pos0;
        if (len == 0 or prompt.len > caches.total) return error.PromptTooLong;
        e.prompts.reset();
        const ids = e.ids.slice(u32)[0..len];
        @memcpy(ids, prompt[pos0..]);
        try e.ids_dev.uploadAsync(0, std.mem.sliceAsBytes(ids), e.stream.handle);
        const o = e.ops(&e.prompts);
        var p: Poll = undefined;
        const hidden = try fwd.span(o, m, caches, e.ids_dev.ptr, len, pos0, e.poll(cancel, len, &p));
        const row = fwd.at(hidden, (len - 1) * m.spec.hidden);
        if (last) |b| try b.copyFrom(0, row.ptr, m.spec.hidden * m.act.size(), e.stream.handle);
        var token: [1]u32 = undefined;
        try e.project(o, row, 1, &.{req}, &token);
        return token[0];
    }

    /// One stream's rows of a round: tokens (the pending one, then drafts) from slot `pos` over its caches.
    pub const Rows = struct { caches: *state.Caches, pos: usize, tokens: []const u32 };

    /// Every window in one forward; `out` the token of every row (in order) per `reqs`, and the final rows. The windows' snapshots live in
    /// the round's scratch until the next round, so `keep` commits from them.
    pub fn verify(e: *Engine, rows: []const Rows, wins: []win.Window, snaps: []win.Snapshot, reqs: []const draw.Request, out: []u32) !struct { hidden: hip.ops.Tensor } {
        const m = e.model();
        var total: usize = 0;
        for (rows) |r| total += r.tokens.len;
        if (total > e.o.batch_rows) return error.WindowTooWide;
        if (e.chosen == null) _ = try e.choose(rows, null);
        defer e.chosen = null;
        e.rounds.reset();
        // ids, then each row's position, in one pinned copy
        const host = e.ids.slice(u32);
        var at: usize = 0;
        for (rows) |r| {
            for (r.tokens, 0..) |t, i| {
                host[at + i] = t;
                host[total + at + i] = @intCast(r.pos + i);
            }
            at += r.tokens.len;
        }
        try e.ids_dev.uploadAsync(0, std.mem.sliceAsBytes(host[0 .. 2 * total]), e.stream.handle);
        at = 0;
        const layers = m.spec.n_layers;
        for (rows, wins, 0..) |r, *w, i| {
            if (r.pos + r.tokens.len > r.caches.total) return error.ContextFull;
            w.* = .{ .caches = r.caches, .pos = r.pos, .rows = r.tokens.len, .at32 = e.ids_dev.ptr + (total + at) * 4, .snaps = snaps[i * layers ..][0..layers] };
            at += r.tokens.len;
        }
        const out_round = try e.round(wins, snaps[0 .. rows.len * layers], total);
        try e.drawRows(e.ops(&e.rounds), out_round.y, total, reqs, out);
        return .{ .hidden = out_round.hidden };
    }

    /// The forward and its logits projection.
    fn body(e: *Engine, wins: []win.Window, total: usize) !round_graphs.Out {
        const o = e.ops(&e.rounds);
        const hidden = try win.forward(o, e.model(), wins, e.ids_dev.ptr, null);
        const y = try o.affine(hidden, e.model().head, total, false);
        return .{ .hidden = hidden, .y = y, .used = e.rounds.used };
    }

    /// The round's graph choice: from the shape's history, or `forced` (rank 0's pick, which a follower obeys). Looks
    /// the shape up once, so every rank's table moves the same way.
    pub fn choose(e: *Engine, rows: []const Rows, forced: ?Pick) !Pick {
        e.chosen = .{ .pick = .eager, .entry = null };
        if (!e.o.graphs or rows.len > 64) {
            if ((forced orelse .eager) != .eager) return error.GraphsDisagree;
            return .eager;
        }
        var parts: [64]round_graphs.Part = undefined;
        for (rows, 0..) |r, i| parts[i] = .{ .serial = r.caches.serial, .rows = @intCast(r.tokens.len) };
        const entry = try e.graphs.find(parts[0..rows.len]);
        const mine: Pick = switch (entry.state) {
            .failed => .eager,
            .ready => .replay,
            .seen => if (entry.again) .capture else .eager,
        };
        const pick = forced orelse mine;
        if (pick == .replay and entry.state != .ready) return error.GraphsDisagree;
        if (pick == .eager and entry.state == .seen) entry.again = true;
        e.chosen = .{ .pick = pick, .entry = entry };
        return pick;
    }

    /// One round's forward as `choose` picked: replayed from the shape's graph, captured, or eager.
    fn round(e: *Engine, wins: []win.Window, snaps: []win.Snapshot, total: usize) !round_graphs.Out {
        const c = e.chosen.?;
        const entry = c.entry orelse return e.body(wins, total);
        switch (c.pick) {
            .eager => return e.body(wins, total),
            .replay => {
                @memcpy(snaps, entry.snaps);
                e.rounds.used = entry.out.used;
                try entry.exec.?.launchOn(e.stream);
                e.graphs.replayed += 1;
                return entry.out;
            },
            .capture => return e.capture(entry, wins, snaps, total),
        }
    }

    /// Records the round into a graph and launches it. A capture that fails on any rank runs the round eagerly on every
    /// rank, and the shape stays eager.
    fn capture(e: *Engine, entry: *round_graphs.Entry, wins: []win.Window, snaps: []win.Snapshot, total: usize) !round_graphs.Out {
        var kept = false;
        var fatal: ?anyerror = null;
        var out: round_graphs.Out = undefined;
        if (hip.graph.beginCapture(e.stream, .thread_local)) {
            const recorded = e.body(wins, total);
            if (hip.graph.endCapture(e.stream)) |graph| {
                if (recorded) |o| {
                    out = o;
                    kept = if (e.graphs.keep(entry, graph, e.stream, snaps, o)) true else |_| false;
                } else |err| {
                    var g = graph;
                    g.deinit();
                    if (err == error.OutOfDeviceMemory) fatal = err;
                }
            } else |_| {}
        } else |_| {}
        if (fatal) |err| return err;
        // TF_HIP_GRAPH_FAIL=R makes rank R's capture fail, to check that every rank falls back
        if (std.c.getenv("TF_HIP_GRAPH_FAIL")) |v| if (std.fmt.parseInt(usize, std.mem.span(v), 10) catch null == e.o.rank) {
            kept = false;
        };
        const all = if (e.o.world > 1) try e.agreed(kept) else kept;
        if (!all) {
            e.graphs.revoke(entry);
            e.rounds.reset();
            return e.body(wins, total);
        }
        try entry.exec.?.launchOn(e.stream);
        return out;
    }

    /// Whether every rank says yes (an all-gather of one word each).
    fn agreed(e: *Engine, yes: bool) !bool {
        const world = e.o.world;
        var host = try hip.HostBuffer.alloc(&e.driver, 8 * (1 + world));
        defer host.free();
        var send = try hip.DeviceBuffer.alloc(&e.driver, 8);
        defer send.free();
        var recv = try hip.DeviceBuffer.alloc(&e.driver, 8 * world);
        defer recv.free();
        host.slice(u64)[0] = @intFromBool(yes);
        try send.uploadAsync(0, host.bytes[0..8], e.stream.handle);
        try e.comm.allGather(send.ptr, recv.ptr, 1, .i64, e.stream.handle);
        try recv.downloadAsync(0, host.bytes[8 .. 8 * (1 + world)], e.stream.handle);
        try e.stream.synchronize();
        for (host.slice(u64)[1 .. 1 + world]) |v| if (v == 0) return false;
        return true;
    }

    /// Keep a verified window's first `rows` rows.
    pub fn keep(e: *Engine, w: win.Window, rows: usize) !void {
        try win.commit(e.ops(&e.rounds), e.model(), w, rows);
    }
};
