//! One loaded Qwen3.5 / 3.6 model on one HIP device: the kernel library, a stream, the forward's scratch, and the
//! pinned buffers rounds read and write through; prompts and lane windows run here, draws on the host.

const std = @import("std");
const hip = @import("hip");
const view = @import("view.zig");
const state = @import("state.zig");
const fwd = @import("forward.zig");
const win = @import("window.zig");
const weights = @import("weights.zig");
const bridge = @import("bridge.zig");
const sample = @import("sample.zig");

pub const Options = struct {
    /// Positions a stream's caches hold (its prompt, its reply and a window's rows).
    capacity: usize,
    /// Rows a shared forward holds at most.
    batch_rows: usize = 32,
    /// The device ordinal among the visible ones.
    device: c_int = 0,
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
    act: view.Kind,
    dtype: sample.Dtype,
    /// A window's scratch, kept from its verify to its keep (the commit reads the per-row states).
    rounds: hip.Arena,
    /// A prompt's scratch: its rows, the step's temporaries.
    prompts: hip.Arena,
    ids: hip.HostBuffer,
    ids_dev: hip.DeviceBuffer,
    logits: hip.HostBuffer,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.o = o;
        e.driver = try hip.Driver.open();
        errdefer e.driver.close();
        e.ctx = try hip.Context.init(&e.driver, o.device);
        errdefer e.ctx.deinit();
        const family = hip.rocm.familyOf(try e.ctx.capability()) orelse return error.UnsupportedGpu;
        e.lib = try hip.rocm.Library.open(family);
        errdefer e.lib.close();
        e.act = if (family == .rdna2) .f16 else .bf16;
        e.dtype = if (family == .rdna2) .f16 else .bf16;
        e.stream = try hip.Stream.init(&e.driver, true);
        errdefer e.stream.deinit();
        e.weights = try weights.Model.load(gpa, io, &e.driver, dir);
        errdefer e.weights.deinit();
        e.bridge = try bridge.Bridge.init(gpa, &e.driver, &e.weights, e.act);
        errdefer e.bridge.deinit();
        const s = e.weights.spec;
        const rows = o.batch_rows;
        // causal_at's scores over a whole cache a query, then a forward's activations, plans and expert products
        const per_row = s.heads * o.capacity * 4 + 64 * s.hidden * 4 + (s.top_k + 1) * (3 * @max(s.moe_width, 1) + s.hidden) * 4 * 2;
        // a window of several rows keeps every linear layer's conv and DeltaNet state after each row
        var linear: usize = 0;
        for (0..s.n_layers) |i| linear += @intFromBool(!s.full(i));
        const snapshot = linear * ((s.conv - 1) * view.convChannels(s) + s.value_heads * s.value_dim * s.key_dim) * 4;
        e.rounds = try hip.Arena.init(&e.driver, (256 << 20) + rows * (per_row * 2 + snapshot));
        errdefer e.rounds.deinit();
        // a prompt's residual and final rows, plus one SPAN step's temporaries
        e.prompts = try hip.Arena.init(&e.driver, (768 << 20) + 2 * o.capacity * s.hidden * 2 + fwd.SPAN * per_row / 4);
        errdefer e.prompts.deinit();
        e.ids = try hip.HostBuffer.alloc(&e.driver, @max(o.capacity, 2 * rows) * 4);
        errdefer e.ids.free();
        e.ids_dev = try hip.DeviceBuffer.alloc(&e.driver, @max(o.capacity, 2 * rows) * 4);
        errdefer e.ids_dev.free();
        e.logits = try hip.HostBuffer.alloc(&e.driver, rows * e.bridge.model.head.n * 2);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        e.logits.free();
        e.ids_dev.free();
        e.ids.free();
        e.prompts.deinit();
        e.rounds.deinit();
        e.bridge.deinit();
        e.weights.deinit();
        e.stream.deinit();
        e.lib.close();
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

    pub fn newCaches(e: *Engine) !state.Caches {
        return state.Caches.init(e.gpa, &e.driver, e.model(), e.o.capacity);
    }

    /// The logits of `rows` final rows at `hidden`, one projection as the engine's _logits, read into `logits`.
    fn project(e: *Engine, o: hip.ops.Ops, hidden: hip.ops.Tensor, rows: usize) ![]const u16 {
        const m = e.model();
        const y = try o.affine(hidden, m.head, rows, false);
        const n = rows * m.head.n;
        try e.driver.check(e.driver.api.hipMemcpyDtoHAsync(e.logits.bytes.ptr, y.ptr, n * 2, e.stream.handle), "logits");
        try e.stream.synchronize();
        return e.logits.slice(u16)[0..n];
    }

    /// Prefill `prompt[pos0..]` into `caches`; the last row's logits, and its final row copied to `last` (MTP's input).
    pub fn prefill(e: *Engine, caches: *state.Caches, prompt: []const u32, pos0: usize, last: ?hip.DeviceBuffer) ![]const u16 {
        const m = e.model();
        const len = prompt.len - pos0;
        if (len == 0 or prompt.len > e.o.capacity) return error.PromptTooLong;
        e.prompts.reset();
        const ids = e.ids.slice(u32)[0..len];
        @memcpy(ids, prompt[pos0..]);
        try e.ids_dev.uploadAsync(0, std.mem.sliceAsBytes(ids), e.stream.handle);
        const o = e.ops(&e.prompts);
        const hidden = try fwd.span(o, m, caches, e.ids_dev.ptr, len, pos0, null);
        const row = fwd.at(hidden, (len - 1) * m.spec.hidden);
        if (last) |b| try b.copyFrom(0, row.ptr, m.spec.hidden * m.act.size(), e.stream.handle);
        return e.project(o, row, 1);
    }

    /// One stream's rows of a round: tokens (the pending one, then drafts) from slot `pos` over its caches.
    pub const Rows = struct { caches: *state.Caches, pos: usize, tokens: []const u32 };

    /// Every window in one forward; their logits (all rows, in order) and final rows. The windows' snapshots live in
    /// the round's scratch until the next round, so `keep` commits from them.
    pub fn verify(e: *Engine, rows: []const Rows, wins: []win.Window, snaps: []win.Snapshot) !struct { logits: []const u16, hidden: hip.ops.Tensor } {
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
            if (r.pos + r.tokens.len > e.o.capacity) return error.ContextFull;
            w.* = .{ .caches = r.caches, .pos = r.pos, .rows = r.tokens.len, .at32 = e.ids_dev.ptr + (total + at) * 4, .snaps = snaps[i * layers ..][0..layers] };
            at += r.tokens.len;
        }
        const o = e.ops(&e.rounds);
        const hidden = try win.forward(o, m, wins, e.ids_dev.ptr, null);
        return .{ .logits = try e.project(o, hidden, total), .hidden = hidden };
    }

    /// Keep a verified window's first `rows` rows.
    pub fn keep(e: *Engine, w: win.Window, rows: usize) !void {
        try win.commit(e.ops(&e.rounds), e.model(), w, rows);
    }
};
