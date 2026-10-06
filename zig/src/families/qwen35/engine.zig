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
const round_graphs = @import("round_graphs.zig");

pub const Options = struct {
    /// Positions a stream's caches hold (its prompt, its reply and a window's rows).
    capacity: usize,
    /// Rows a shared forward holds at most.
    batch_rows: usize = 32,
    /// The device ordinal among the visible ones.
    device: c_int = 0,
    /// Replay rounds from captured graphs (TF_HIP_GRAPHS=0 turns it off).
    graphs: bool = true,
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
    graphs: round_graphs.Graphs,
    /// The serial of the next caches made, for graphs keyed by stream.
    serial: u64 = 1,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.o = o;
        if (std.c.getenv("TF_HIP_GRAPHS")) |v| if (std.mem.eql(u8, std.mem.span(v), "0")) {
            e.o.graphs = false;
        };
        e.graphs = .{ .gpa = gpa };
        e.serial = 1;
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
        e.rounds = try hip.Arena.init(&e.driver, (256 << 20) + rows * per_row * 2);
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
        if (e.graphs.captured > 0) std.log.info("graphs: {d} captured, {d} rounds replayed", .{ e.graphs.captured, e.graphs.replayed });
        e.graphs.deinit();
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
        var c = try state.Caches.init(e.gpa, &e.driver, e.model(), e.o.capacity);
        c.serial = e.serial;
        e.serial += 1;
        return c;
    }

    /// Drops the graphs over `caches`, before they are freed.
    pub fn forget(e: *Engine, caches: *const state.Caches) void {
        e.stream.synchronize() catch {};
        e.graphs.forget(caches.serial);
    }

    /// The logits of `rows` final rows at `hidden`, one projection as the engine's _logits, read into `logits`.
    fn project(e: *Engine, o: hip.ops.Ops, hidden: hip.ops.Tensor, rows: usize) ![]const u16 {
        return e.readLogits(try o.affine(hidden, e.model().head, rows, false), rows);
    }

    /// `rows` rows of the projection `y`, copied to the host.
    fn readLogits(e: *Engine, y: hip.ops.Tensor, rows: usize) ![]const u16 {
        const n = rows * e.model().head.n;
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
        const out = try e.round(rows, wins, snaps[0 .. rows.len * layers], total);
        return .{ .logits = try e.readLogits(out.y, total), .hidden = out.hidden };
    }

    /// The forward and its logits projection.
    fn body(e: *Engine, wins: []win.Window, total: usize) !round_graphs.Out {
        const o = e.ops(&e.rounds);
        const hidden = try win.forward(o, e.model(), wins, e.ids_dev.ptr, null);
        const y = try o.affine(hidden, e.model().head, total, false);
        return .{ .hidden = hidden, .y = y, .used = e.rounds.used };
    }

    /// One round's forward: replayed from the shape's graph when it has one, captured on its second sighting, else eager.
    fn round(e: *Engine, rows: []const Rows, wins: []win.Window, snaps: []win.Snapshot, total: usize) !round_graphs.Out {
        if (!e.o.graphs or rows.len > 64) return e.body(wins, total);
        var parts: [64]round_graphs.Part = undefined;
        for (rows, 0..) |r, i| parts[i] = .{ .serial = r.caches.serial, .rows = @intCast(r.tokens.len) };
        const entry = try e.graphs.find(parts[0..rows.len]);
        switch (entry.state) {
            .failed => return e.body(wins, total),
            .ready => {
                @memcpy(snaps, entry.snaps);
                e.rounds.used = entry.out.used;
                try entry.exec.?.launchOn(e.stream);
                e.graphs.replayed += 1;
                return entry.out;
            },
            .seen => {
                if (!entry.again) {
                    entry.again = true;
                    return e.body(wins, total);
                }
                return e.capture(entry, wins, snaps, total);
            },
        }
    }

    /// Records the round into a graph and launches it; a capture that fails runs the round eagerly from then on.
    fn capture(e: *Engine, entry: *round_graphs.Entry, wins: []win.Window, snaps: []win.Snapshot, total: usize) !round_graphs.Out {
        hip.graph.beginCapture(e.stream, .thread_local) catch {
            entry.state = .failed;
            return e.body(wins, total);
        };
        const recorded = e.body(wins, total);
        var graph = hip.graph.endCapture(e.stream) catch {
            entry.state = .failed;
            e.rounds.reset();
            return e.body(wins, total);
        };
        const out = recorded catch |err| {
            graph.deinit();
            entry.state = .failed;
            e.rounds.reset();
            if (err == error.OutOfDeviceMemory) return err;
            return e.body(wins, total);
        };
        e.graphs.keep(entry, graph, e.stream, snaps, out) catch {
            entry.state = .failed;
            e.rounds.reset();
            return e.body(wins, total);
        };
        try entry.exec.?.launchOn(e.stream);
        return out;
    }

    /// Keep a verified window's first `rows` rows.
    pub fn keep(e: *Engine, w: win.Window, rows: usize) !void {
        try win.commit(e.ops(&e.rounds), e.model(), w, rows);
    }
};
