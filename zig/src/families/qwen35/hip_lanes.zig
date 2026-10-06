//! The lane core's HIP backend for Qwen3.5 / 3.6: every stream its own caches, every round's windows verified in one
//! forward, each row drawn on the host at its keyed position, the kept rows committed from the window's states.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const Engine = @import("engine.zig").Engine;
const state = @import("state.zig");
const win = @import("window.zig");
const sample = @import("sample.zig");

const be = lanes.backend;

/// Drawn tokens a handle names, newest last.
const ring = 1024;

const Lane = struct {
    caches: state.Caches,
    /// Slots written and kept: the next window starts here.
    len: usize,
    /// The last verify's window and rows: kept whole unless keep drops some first.
    pending: ?struct { window: usize, rows: usize } = null,
};

pub const Hip = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, *Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    /// The last verify's windows and snapshots, for keep.
    wins: []win.Window,
    snaps: []win.Snapshot,
    order: []*const lanes.Stream,

    pub fn init(gpa: std.mem.Allocator, e: *Engine) !*Hip {
        const h = try gpa.create(Hip);
        errdefer gpa.destroy(h);
        const rows = e.o.batch_rows;
        h.* = .{ .gpa = gpa, .e = e, .wins = try gpa.alloc(win.Window, rows), .snaps = undefined, .order = undefined };
        errdefer gpa.free(h.wins);
        h.snaps = try gpa.alloc(win.Snapshot, rows * e.model().spec.n_layers);
        errdefer gpa.free(h.snaps);
        h.order = try gpa.alloc(*const lanes.Stream, rows);
        return h;
    }

    pub fn deinit(h: *Hip) void {
        var it = h.lanes.valueIterator();
        while (it.next()) |l| {
            h.e.forget(&l.*.caches);
            l.*.caches.deinit(h.gpa);
            h.gpa.destroy(l.*);
        }
        h.lanes.deinit(h.gpa);
        h.gpa.free(h.order);
        h.gpa.free(h.snaps);
        h.gpa.free(h.wins);
        h.gpa.destroy(h);
    }

    pub fn backend(h: *Hip) be.Backend {
        return .{ .ptr = h, .vtable = &.{
            .prefill = prefillFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .release = releaseFn,
        } };
    }

    /// Rows a window holds: every width keeps a row's bits (the window forward), drafts come from the core.
    pub const max_window = 16;

    /// The facts the round loop reads at setup: shared rounds of exact windows, no draft head yet.
    pub fn facts(h: *const Hip) lanes.Model {
        return .{
            .exact_width = max_window,
            .gpu_tokens = false,
            .mtp = false,
            .speculate = false,
            .hidden_rows = true,
            .batch_rows = @intCast(h.e.o.batch_rows),
            .max_streams = @intCast(h.e.o.batch_rows),
        };
    }

    fn of(ptr: *anyopaque) *Hip {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(h: *Hip, token: u32) u64 {
        const at = h.next;
        h.drawn[at % ring] = token;
        h.next += 1;
        return at;
    }

    /// A verify the round loop did not trim keeps every row (the core keeps only on drops).
    fn settle(h: *Hip, lane: *Lane) !void {
        const p = lane.pending orelse return;
        try h.e.keep(h.wins[p.window], p.rows);
        lane.len += p.rows;
        lane.pending = null;
    }

    fn sampling(s: *const lanes.Stream) ?lanes.Sampling {
        return s.sampling;
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const h = of(ptr);
        const prompt = s.prompt();
        if (prompt.len == 0 or prompt.len + s.max_new + 1 > h.e.o.capacity) return error.PromptTooLong;
        const gop = try h.lanes.getOrPut(h.gpa, s);
        if (gop.found_existing) {
            h.e.forget(&gop.value_ptr.*.caches);
            gop.value_ptr.*.caches.deinit(h.gpa);
            h.gpa.destroy(gop.value_ptr.*);
        }
        const lane = h.gpa.create(Lane) catch |err| {
            h.lanes.removeByPtr(gop.key_ptr);
            return err;
        };
        lane.* = .{ .caches = h.e.newCaches() catch |err| {
            h.gpa.destroy(lane);
            h.lanes.removeByPtr(gop.key_ptr);
            return err;
        }, .len = prompt.len };
        gop.value_ptr.* = lane;
        const row = try h.e.prefill(&lane.caches, prompt, 0, null);
        _ = h.take(try sample.draw(h.gpa, row, h.e.dtype, sampling(s), prompt.len));
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const h = of(ptr);
        if (position != s.prompt_len) {
            std.log.err("first draw at {d}, the prompt has {d} tokens", .{ position, s.prompt_len });
            return error.PositionMismatch;
        }
        return h.next - 1;
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = ptr;
        _ = s;
        _ = feed;
        _ = position;
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const h = of(ptr);
        if (handle >= h.next or h.next - handle > ring) return error.NoSuchToken;
        return h.drawn[handle % ring];
    }

    /// Each stream's window from its kept length: the pending token, then the host's drafts; one forward for all.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const h = of(ptr);
        if (windows.len > h.wins.len) return error.WindowTooWide;
        var rows: [64]Engine.Rows = undefined;
        var tokens: [64][16]u32 = undefined;
        if (windows.len > rows.len) return error.WindowTooWide;
        for (windows, 0..) |w, i| {
            if (w.parents != null) return error.TreesNotBuilt;
            if (w.held > 0) return error.NoHeldDrafts;
            const lane = h.lanes.get(w.stream) orelse return error.UnknownStream;
            const n = w.rows();
            if (n > tokens[i].len) return error.WindowTooWide;
            try h.settle(lane);
            for (w.positions, 0..) |p, r| if (p != lane.len + 1 + r) {
                std.log.err("row {d} keyed at {d}, the stream holds {d} slots", .{ r, p, lane.len });
                return error.PositionMismatch;
            };
            tokens[i][0] = w.pending;
            @memcpy(tokens[i][1..n], w.tokens);
            rows[i] = .{ .caches = &lane.caches, .pos = lane.len, .tokens = tokens[i][0..n] };
            h.order[i] = w.stream;
        }
        const r = try h.e.verify(rows[0..windows.len], h.wins[0..windows.len], h.snaps);
        const vocab = h.e.model().head.n;
        var at: usize = 0;
        for (windows, out, 0..) |w, *o, i| {
            for (o.sampled, 0..) |*t, row| t.* = try sample.draw(h.gpa, r.logits[(at + row) * vocab ..][0..vocab], h.e.dtype, sampling(w.stream), w.positions[row]);
            @memcpy(o.drafts, w.tokens);
            const lane = h.lanes.get(w.stream).?;
            lane.pending = .{ .window = i, .rows = w.rows() };
            at += w.rows();
        }
    }

    /// Keep each stream's accepted prefix: lengths, and the linear states of its last kept row.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const h = of(ptr);
        for (windows, paths) |w, path| {
            const lane = h.lanes.get(w.stream) orelse return error.UnknownStream;
            const p = lane.pending orelse return error.NothingToKeep;
            for (path, 0..) |r, j| if (r != j) return error.TreesNotBuilt;
            try h.e.keep(h.wins[p.window], path.len);
            lane.len += path.len;
            lane.pending = null;
        }
    }

    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        _ = ptr;
        if (requests.len > 0) return error.NoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const h = of(ptr);
        const kv = h.lanes.fetchRemove(s) orelse return;
        h.e.forget(&kv.value.caches);
        kv.value.caches.deinit(h.gpa);
        h.gpa.destroy(kv.value);
    }
};
