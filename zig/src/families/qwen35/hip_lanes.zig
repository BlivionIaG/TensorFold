//! The lane core's HIP backend for Qwen3.5 / 3.6: every stream its own caches, every round's windows verified in one
//! forward, each row drawn on the host at its keyed position, the kept rows committed from the window's states.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const Engine = @import("engine.zig").Engine;
const state = @import("state.zig");
const win = @import("window.zig");
const sample = @import("sample.zig");
const prefix = @import("prefix.zig");

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
    /// Caches kept at prompt cuts for later turns (no entries until `keepPrompts`).
    kept: prefix.Cache,

    pub fn init(gpa: std.mem.Allocator, e: *Engine) !*Hip {
        const h = try gpa.create(Hip);
        errdefer gpa.destroy(h);
        const rows = e.o.batch_rows;
        h.* = .{ .gpa = gpa, .e = e, .wins = try gpa.alloc(win.Window, rows), .snaps = undefined, .order = undefined, .kept = prefix.Cache.init(gpa, 0, 0) };
        errdefer gpa.free(h.wins);
        h.snaps = try gpa.alloc(win.Snapshot, rows * e.model().spec.n_layers);
        errdefer gpa.free(h.snaps);
        h.order = try gpa.alloc(*const lanes.Stream, rows);
        return h;
    }

    /// Keep up to `entries` prompt caches within `budget` bytes of device memory.
    pub fn keepPrompts(h: *Hip, entries: usize, budget: usize) void {
        h.kept.keep = entries;
        h.kept.budget = budget;
    }

    pub fn deinit(h: *Hip) void {
        h.e.stream.synchronize() catch {};
        h.kept.deinit();
        var it = h.lanes.valueIterator();
        while (it.next()) |l| {
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

    /// Copy of `caches` at its first `len` positions, kept under the prompt's first `len` ids; a failure keeps nothing.
    fn remember(h: *Hip, ids: []const u32, caches: *const state.Caches) void {
        if (h.kept.keep == 0 or h.kept.has(ids)) return;
        var snap = state.Caches.blank(h.gpa, &h.e.driver, h.e.model(), ids.len) catch return;
        snap.copyPrefix(caches, h.e.model(), ids.len, h.e.stream.handle) catch return snap.deinit(h.gpa);
        h.kept.add(ids, snap) catch {};
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const h = of(ptr);
        const prompt = s.prompt();
        if (prompt.len == 0 or prompt.len + s.max_new + 1 > h.e.o.capacity) return error.PromptTooLong;
        const gop = try h.lanes.getOrPut(h.gpa, s);
        if (gop.found_existing) {
            gop.value_ptr.*.caches.deinit(h.gpa);
            h.gpa.destroy(gop.value_ptr.*);
        }
        const lane = h.gpa.create(Lane) catch |err| {
            h.lanes.removeByPtr(gop.key_ptr);
            return err;
        };
        lane.* = .{ .caches = h.e.newCaches(prompt.len + s.max_new + max_window + 1) catch |err| {
            h.gpa.destroy(lane);
            h.lanes.removeByPtr(gop.key_ptr);
            return err;
        }, .len = prompt.len };
        gop.value_ptr.* = lane;
        // a drafted request resumes from the longest kept prompt it extends and keeps its own cuts; a serial one neither
        var at: usize = 0;
        if (s.drafts) if (h.kept.longest(prompt)) |hit| {
            at = hit.ids.len;
            try lane.caches.copyPrefix(&hit.caches, h.e.model(), at, h.e.stream.handle);
        };
        s.cached = @intCast(at);
        var cut_ids: [16]u32 = undefined;
        const stops: []const u32 = if (s.drafts) prefix.cuts(&cut_ids, prompt.len, at, s.history_len, s.shared_prefixes) else &.{};
        for (stops) |stop| {
            try h.e.advance(&lane.caches, prompt, at, stop);
            at = stop;
            h.remember(prompt[0..at], &lane.caches);
        }
        const row = try h.e.prefill(&lane.caches, prompt, at, null);
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
        h.e.stream.synchronize() catch {};
        kv.value.caches.deinit(h.gpa);
        h.gpa.destroy(kv.value);
    }
};
