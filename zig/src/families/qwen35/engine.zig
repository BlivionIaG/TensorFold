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
};

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

    /// The model on the device, its scratch not yet sized: `size` takes the capacity the memory plan fits.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.o = o;
        e.sized = false;
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

    /// Load and size in one: `o.capacity` positions.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        const e = try load(gpa, io, dir, o);
        errdefer e.deinit();
        try e.size(o.capacity);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
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

    /// Zeroed caches for `total` positions (at most the capacity).
    pub fn newCaches(e: *Engine, total: usize) !state.Caches {
        return state.Caches.init(e.gpa, &e.driver, e.model(), @min(total, e.o.capacity));
    }

    /// The tokens of `rows` final rows at `hidden`: one projection as the engine's _logits (the ranks' vocabulary
    /// slices joined in rank order under tensor parallelism), each row drawn per `reqs`.
    fn project(e: *Engine, o: hip.ops.Ops, hidden: hip.ops.Tensor, rows: usize, reqs: []const draw.Request, out: []u32) !void {
        const m = e.model();
        var y = try o.affine(hidden, m.head, rows, false);
        if (m.tp) |c| y = try e.joined(o, c, y, rows);
        try e.drawer.draw(o, e.stream, y, reqs[0..rows], out);
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

    /// Run `prompt[pos0..end]` into `caches`, its logits not read: a cut where the caches are kept.
    pub fn advance(e: *Engine, caches: *state.Caches, prompt: []const u32, pos0: usize, end: usize) !void {
        if (end <= pos0 or end > caches.total) return error.PromptTooLong;
        e.prompts.reset();
        const ids = e.ids.slice(u32)[0 .. end - pos0];
        @memcpy(ids, prompt[pos0..end]);
        try e.ids_dev.uploadAsync(0, std.mem.sliceAsBytes(ids), e.stream.handle);
        _ = try fwd.span(e.ops(&e.prompts), e.model(), caches, e.ids_dev.ptr, end - pos0, pos0, null);
        try e.stream.synchronize();
    }

    /// Prefill `prompt[pos0..]` into `caches`; the token drawn per `req` from the last row, whose final row is copied to `last` (MTP's input).
    pub fn prefill(e: *Engine, caches: *state.Caches, prompt: []const u32, pos0: usize, last: ?hip.DeviceBuffer, req: draw.Request) !u32 {
        const m = e.model();
        const len = prompt.len - pos0;
        if (len == 0 or prompt.len > caches.total) return error.PromptTooLong;
        e.prompts.reset();
        const ids = e.ids.slice(u32)[0..len];
        @memcpy(ids, prompt[pos0..]);
        try e.ids_dev.uploadAsync(0, std.mem.sliceAsBytes(ids), e.stream.handle);
        const o = e.ops(&e.prompts);
        const hidden = try fwd.span(o, m, caches, e.ids_dev.ptr, len, pos0, null);
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
        const o = e.ops(&e.rounds);
        const hidden = try win.forward(o, m, wins, e.ids_dev.ptr, null);
        try e.project(o, hidden, total, reqs, out);
        return .{ .hidden = hidden };
    }

    /// Keep a verified window's first `rows` rows.
    pub fn keep(e: *Engine, w: win.Window, rows: usize) !void {
        try win.commit(e.ops(&e.rounds), e.model(), w, rows);
    }
};
