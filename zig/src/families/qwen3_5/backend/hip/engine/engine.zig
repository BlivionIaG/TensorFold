//! One loaded model on one HIP device: the kernel library, a stream, scratch and the pinned buffers rounds go through.

const std = @import("std");
const hip = @import("hip");
const view = @import("../model/view.zig");
const state = @import("../forward/state.zig");
const pages = @import("../forward/pages.zig");
const fwd = @import("../forward/forward.zig");
const weights = @import("../model/weights.zig");
const bridge = @import("../model/bridge.zig");
const sample = @import("sample.zig");
const memory = @import("memory.zig");
const draw = @import("draw.zig");
const slicing = @import("../../../weights/slicing.zig");
const reduce = @import("../forward/reduce.zig");
const round_graphs = @import("round_graphs.zig");
const lane_round = @import("round.zig");

pub const Options = struct {
    /// Most positions a stream's caches hold (its prompt, its reply and a window's rows).
    capacity: usize = 0,
    /// Pages of the KV pool (0: enough for `default_streams` streams of the capacity and the scratch caches).
    pool_pages: usize = 0,
    /// Rows a shared forward holds at most.
    batch_rows: usize = 32,
    /// The device ordinal among the visible ones.
    device: c_int = 0,
    /// Positions past a reply a verify writes (the window's rows and one more).
    slack: usize = 0,
    /// Tensor parallelism: this rank of `world`, `id` being the communicator's unique id, the same on every rank.
    rank: usize = 0,
    world: usize = 1,
    id: ?hip.rccl.UniqueId = null,
    /// Replay rounds from captured graphs (the policy's `graphs` turns it off).
    graphs: bool = true,
    /// What the run may use, resolved once by whoever opens the engine.
    policy: hip.Policy = .{},
};

pub const Pick = lane_round.Pick;

/// Streams the pool of an engine opened without a memory plan holds whole.
pub const default_streams = 12;

pub const Engine = struct {
    pub const Rows = lane_round.Rows;

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
    /// The keys and values of every stream and prefix entry.
    pool: pages.Pool,
    drawer: draw.Drawer,
    /// Tensor parallelism: RCCL and this rank's communicator.
    rccl: hip.rccl.Rccl = undefined,
    comm: hip.rccl.Comm = undefined,
    graphs: round_graphs.Graphs,
    /// The draft head's batches replay graphs (the head holds no collective, so under tp too unless TF_HIP_GRAPHS=0).
    head_graphs: bool = true,
    /// The lane rounds' plan, graph choice and keep.
    round: lane_round.State = undefined,

    /// The model on the device, its scratch not yet sized: `size` takes the capacity the memory plan fits.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.o = o;
        e.sized = false;
        // rounds replay captured graphs as the policy says: under tensor parallelism they stay eager unless it is `graphs=on`
        e.o.graphs = o.graphs and o.policy.graphsOn(o.world);
        e.head_graphs = o.graphs and o.policy.graphsOn(1);
        e.graphs = round_graphs.Graphs.init(gpa);
        e.driver = try hip.Driver.open();
        errdefer e.driver.close();
        e.ctx = try hip.Context.init(&e.driver, o.device);
        errdefer e.ctx.deinit();
        const caps = try e.ctx.caps();
        e.lib = try hip.rocm.Library.open(e.ctx.d, caps, o.policy);
        errdefer e.lib.close();
        e.act = if (caps.act == .f16) .f16 else .bf16;
        e.dtype = if (caps.act == .f16) .f16 else .bf16;
        e.stream = try hip.Stream.init(&e.driver, true);
        errdefer e.stream.deinit();
        const group: ?slicing.Rank = if (o.world > 1) .{ .rank = o.rank, .world = o.world } else null;
        if (group != null) {
            e.rccl = try hip.rccl.Rccl.open(o.policy.rccl_lib.slice());
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

    /// Allocate the scratch for streams of `capacity` positions (`o.batch_rows` rows a shared forward) and a pool of
    /// `pool_pages` pages (`o.pool_pages`, or the default, when zero).
    pub fn size(e: *Engine, capacity: usize, pool_pages: usize) !void {
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
        errdefer e.drawer.deinit();
        e.pool = try pages.Pool.init(e.gpa, &e.driver, &e.bridge.model, try e.fitPages(if (pool_pages > 0) pool_pages else default_streams * pages.pagesFor(capacity) + pages.pagesFor(rows), capacity), e.o.rank == 0);
        errdefer e.pool.deinit();
        e.round = try lane_round.State.init(e);
        e.sized = true;
    }

    /// `want` pages, or the fewer the memory left after the scratch holds beside the reserve (the same on every rank); it refuses
    /// when not even the scratch rows and one window fit.
    fn fitPages(e: *Engine, want: usize, capacity: usize) !usize {
        const info = try e.ctx.memInfo();
        const per = @max(memory.pageBytes(e.weights.spec, e.act.size()), 1);
        var fit = @min(want, (info.free -| memory.reserve(info.total)) / per);
        if (e.o.world > 1) fit = (try e.least(.{ fit, 0 }))[0];
        const least_pages = pages.pagesFor(e.o.batch_rows) + pages.pagesFor(capacity);
        if (fit < least_pages) {
            std.log.err("the weights and scratch leave room for {d} pages of {d} bytes, and {d} are the least a window needs", .{ fit, per, least_pages });
            return error.OutOfDeviceMemory;
        }
        if (fit < want) std.log.warn("the page pool holds {d} pages, not the {d} asked for: that is what the memory leaves", .{ fit, want });
        return fit;
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
        try e.size(o.capacity, o.pool_pages);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        if (e.graphs.captured > 0) std.log.info("graphs: {d} captured ({d:.0} ms each), {d} of {d} rounds replayed", .{ e.graphs.captured, @as(f64, @floatFromInt(e.graphs.capture_ns)) / @as(f64, @floatFromInt(e.graphs.captured)) / 1e6, e.graphs.replayed, e.graphs.rounds });
        e.graphs.deinit();
        if (e.sized) {
            e.round.deinit(e);
            e.drawer.deinit();
            e.ids_dev.free();
            e.ids.free();
            e.prompts.deinit();
            e.rounds.deinit();
            e.pool.deinit();
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

    pub fn ops(e: *Engine, arena: *hip.Arena) hip.ops.Ops {
        return .{ .lib = &e.lib, .stream = e.stream.handle, .arena = arena };
    }

    /// Caches for `total` positions (at most the capacity) with every page taken.
    pub fn newCaches(e: *Engine, total: usize) !state.Caches {
        return state.Caches.initFull(e.gpa, &e.pool, e.model(), @min(total, e.o.capacity));
    }

    /// Caches for `total` positions (at most the capacity) with no page yet: a stream takes them as it grows.
    pub fn emptyCaches(e: *Engine, total: usize) !state.Caches {
        return state.Caches.init(e.gpa, &e.pool, e.model(), @min(total, e.o.capacity));
    }

    /// Whether the draft head's greedy batches replay graphs.
    pub fn headGraphs(e: *const Engine) bool {
        return if (e.o.world > 1) e.head_graphs else e.o.graphs;
    }

    /// Waits for the stream, before caches are freed (a round may still be running on them).
    pub fn drain(e: *Engine) void {
        e.stream.synchronize() catch {};
    }

    /// The tokens of `rows` final rows at `hidden`: one projection, vocabulary slices joined in rank order, each row per `reqs`.
    fn project(e: *Engine, o: hip.ops.Ops, hidden: hip.ops.Tensor, rows: usize, reqs: []const draw.Request, out: []u32) !void {
        const y = try o.project(hidden, e.model().head, rows, false);
        try e.drawRows(o, y, rows, reqs, out, false);
    }

    /// Each of `rows` logits rows `y` drawn per `reqs` (whole rows joined first under tensor parallelism); `argmaxed`: the
    /// greedy rows are drawn already, where the drawer keeps them.
    pub fn drawRows(e: *Engine, o: hip.ops.Ops, y: hip.ops.Tensor, rows: usize, reqs: []const draw.Request, out: []u32, argmaxed: bool) !void {
        var whole = y;
        if (e.model().tp) |c| whole = try e.joined(o, c, y, rows);
        try e.drawer.draw(o, e.stream, whole, reqs[0..rows], out, argmaxed);
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

    /// The pass over `len` rows polling the cancel after each layer; null where it cannot stop (a short prompt, or tp).
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

    /// Every window in one forward: `out` the token of every row per `reqs`; the round's snapshots live until its keeps are flushed.
    pub fn verify(e: *Engine, rows: []const Rows, reqs: []const draw.Request, out: []u32) !lane_round.Verified {
        return lane_round.verify(e, rows, reqs, out);
    }

    /// The round's graph choice: from the shape's history, or `forced` (rank 0's pick, which a follower obeys).
    pub fn choose(e: *Engine, rows: []const Rows, forced: ?Pick) !Pick {
        return lane_round.choose(e, rows, forced);
    }

    /// Marks a slot of the last verify to keep its first `rows` rows at the next `flush`.
    pub fn keep(e: *Engine, slot: usize, rows: usize) void {
        lane_round.mark(e, slot, rows);
    }

    /// Keeps every marked slot in one launch.
    pub fn flush(e: *Engine) !void {
        try lane_round.flush(e);
    }
};
