//! The lane core's HIP backend for Qwen3.5 / 3.6: every stream its own caches, every round's windows verified in one
//! forward, each row drawn on the host at its keyed position, the kept rows committed from the window's states.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const Engine = @import("engine.zig").Engine;
const state = @import("state.zig");
const win = @import("window.zig");
const draw = @import("draw.zig");
const mtp = @import("mtp.zig");
const prefix = @import("prefix.zig");
const worker = @import("worker.zig");

const be = lanes.backend;

/// Drawn tokens a handle names, newest last.
const ring = 1024;

/// Where a prompt pass stands: the row it has reached, and the cuts it still keeps a state at.
const Fill = struct { at: usize, stops: [16]u32, count: usize, next: usize };

const Lane = struct {
    caches: state.Caches,
    /// The last kept row's final hidden row: the draft head's input.
    hidden: hip.DeviceBuffer,
    /// Drafts the head holds for the next window.
    held: [mtp.max_depth]u32 = undefined,
    held_n: usize = 0,
    /// Slots written and kept: the next window starts here.
    len: usize,
    /// The stream's id on every rank.
    id: u32 = 0,
    /// The prompt pass in progress, a chunk a call (null once it is in).
    fill: ?Fill = null,
    /// The last verify's window and rows: kept whole unless keep drops some first.
    pending: ?struct { window: usize, rows: usize, start: usize } = null,
};

pub const Hip = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, *Lane) = .empty,
    head: ?*mtp.Head = null,
    /// The last verify's final rows, for the kept rows' hidden row.
    round_hidden: u64 = 0,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    /// The last verify's windows and snapshots, for keep.
    wins: []win.Window,
    snaps: []win.Snapshot,
    order: []*const lanes.Stream,
    /// Caches kept at prompt cuts for later turns (no entries until `keepPrompts`).
    kept: prefix.Cache,
    /// Tensor parallelism: the other ranks, which get every step before this rank runs it (`worker.follow`).
    link: ?*const hip.link.Link = null,
    msg: std.ArrayList(u32) = .empty,
    /// Each stream's id on every rank.
    ids: std.AutoHashMapUnmanaged(*const lanes.Stream, u32) = .empty,
    next_id: u32 = 0,

    pub fn init(gpa: std.mem.Allocator, e: *Engine) !*Hip {
        const h = try gpa.create(Hip);
        errdefer gpa.destroy(h);
        const rows = e.o.batch_rows;
        h.* = .{ .gpa = gpa, .e = e, .wins = try gpa.alloc(win.Window, rows), .snaps = undefined, .order = undefined, .kept = prefix.Cache.init(gpa, 0, 0) };
        errdefer gpa.free(h.wins);
        h.snaps = try gpa.alloc(win.Snapshot, rows * e.model().spec.n_layers);
        errdefer gpa.free(h.snaps);
        h.order = try gpa.alloc(*const lanes.Stream, rows);
        errdefer gpa.free(h.order);
        // under tensor parallelism only rank 0 holds the head (whole): drafts only choose the rows every rank verifies
        h.head = try mtp.Head.init(gpa, &e.driver, &e.weights, e.model());
        return h;
    }

    /// Keep up to `entries` prompt caches within `budget` bytes of device memory.
    pub fn keepPrompts(h: *Hip, entries: usize, budget: usize) void {
        h.kept.keep = entries;
        h.kept.budget = budget;
    }

    /// Rank 0 of a tensor-parallel group: every step also goes to the ranks behind `link`.
    pub fn withLink(h: *Hip, link: *const hip.link.Link) void {
        h.link = link;
    }

    fn send(h: *Hip, words: []const u32) !void {
        if (h.link) |l| try l.send(words);
    }

    /// A stream's id on every rank.
    fn idOf(h: *Hip, s: *const lanes.Stream) !u32 {
        const gop = try h.ids.getOrPut(h.gpa, s);
        if (!gop.found_existing) {
            gop.value_ptr.* = h.next_id;
            h.next_id += 1;
        }
        return gop.value_ptr.*;
    }

    pub fn deinit(h: *Hip) void {
        if (h.link != null) h.send(&.{@backingInt(worker.Op.stop)}) catch {};
        h.ids.deinit(h.gpa);
        h.msg.deinit(h.gpa);
        h.e.stream.synchronize() catch {};
        h.kept.deinit();
        var it = h.lanes.valueIterator();
        while (it.next()) |l| h.free(l.*);
        if (h.head) |hd| hd.deinit();
        h.lanes.deinit(h.gpa);
        h.gpa.free(h.order);
        h.gpa.free(h.snaps);
        h.gpa.free(h.wins);
        h.gpa.destroy(h);
    }

    pub fn backend(h: *Hip) be.Backend {
        return .{ .ptr = h, .vtable = &.{
            .prefill = prefillFn,
            .prefill_step = prefillStepFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .tree = heldFn,
            .release = releaseFn,
        } };
    }

    /// Rows a window holds: every width keeps a row's bits (the window forward), drafts come from the core.
    pub const max_window = 16;

    /// The facts the round loop reads at setup: shared rounds of exact windows, the MTP head's chains when it has one.
    pub fn facts(h: *const Hip) lanes.Model {
        const drafting = h.head != null;
        return .{
            .exact_width = max_window,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = mtp.max_depth,
            .hidden_rows = true,
            .batch_rows = @intCast(h.e.o.batch_rows),
            .max_streams = @intCast(h.e.o.batch_rows),
        };
    }

    fn free(h: *Hip, l: *Lane) void {
        h.e.forget(&l.caches);
        l.caches.deinit(h.gpa);
        l.hidden.free();
        h.gpa.destroy(l);
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
        try h.commit(lane, p.rows);
    }

    /// Keep the pending window's first `rows` rows, and its last kept final row for the head.
    fn commit(h: *Hip, lane: *Lane, rows: usize) !void {
        const p = lane.pending.?;
        if (h.link != null) try h.send(&.{ @backingInt(worker.Op.keep), 1, lane.id, @intCast(rows) });
        try h.e.keep(h.wins[p.window], rows);
        const width = h.e.model().spec.hidden * h.e.model().act.size();
        try lane.hidden.copyFrom(0, h.round_hidden + (p.start + rows - 1) * width, width, h.e.stream.handle);
        lane.len += rows;
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

    /// The whole prompt pass, chunk by chunk (a driver of its own, with no rounds between).
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        while (!try prefillStepFn(ptr, s)) {}
    }

    /// The stream's lane and its prompt pass begun: caches, the resume from a kept prompt, the cuts, the other ranks told.
    fn begin(h: *Hip, s: *lanes.Stream) !*Lane {
        const prompt = s.prompt();
        if (prompt.len == 0 or prompt.len + s.max_new + 1 > h.e.o.capacity) return error.PromptTooLong;
        const gop = try h.lanes.getOrPut(h.gpa, s);
        if (gop.found_existing) h.free(gop.value_ptr.*);
        const lane = h.gpa.create(Lane) catch |err| {
            h.lanes.removeByPtr(gop.key_ptr);
            return err;
        };
        const width = h.e.model().spec.hidden * h.e.model().act.size();
        lane.* = .{ .caches = h.e.newCaches(prompt.len + s.max_new + max_window + 1) catch |err| {
            h.gpa.destroy(lane);
            h.lanes.removeByPtr(gop.key_ptr);
            return err;
        }, .hidden = hip.DeviceBuffer.alloc(&h.e.driver, width) catch |err| {
            h.gpa.destroy(lane);
            h.lanes.removeByPtr(gop.key_ptr);
            return err;
        }, .len = prompt.len };
        gop.value_ptr.* = lane;
        lane.id = try h.idOf(s);
        const total = prompt.len + s.max_new + max_window + 1;
        // a drafted request resumes from the longest kept prompt it extends and keeps its own cuts; a serial one neither
        var at: usize = 0;
        if (s.drafts) if (h.kept.longest(prompt)) |hit| {
            at = hit.ids.len;
            try lane.caches.copyPrefix(&hit.caches, h.e.model(), at, h.e.stream.handle);
        };
        s.cached = @intCast(at);
        var fill: Fill = .{ .at = at, .stops = undefined, .count = 0, .next = 0 };
        if (s.drafts) fill.count = prefix.cuts(&fill.stops, prompt.len, at, s.history_len, s.shared_prefixes).len;
        lane.fill = fill;
        if (h.link != null) {
            // the other ranks keep the same prompts: where this one resumed and cut, and the cache's limits
            h.msg.clearRetainingCapacity();
            const budget: u64 = h.kept.budget;
            try h.msg.appendSlice(h.gpa, &.{ @backingInt(worker.Op.prefill), lane.id, @intCast(total), @intCast(prompt.len), @intCast(at), @intCast(fill.count), @intCast(h.kept.keep), @truncate(budget), @truncate(budget >> 32) });
            try h.msg.appendSlice(h.gpa, fill.stops[0..fill.count]);
            try h.msg.appendSlice(h.gpa, prompt);
            try h.send(h.msg.items);
        }
        return lane;
    }

    /// One chunk of the prompt pass: up to `prefix.chunk` rows, ending on the next cut or the prompt's end. The last
    /// chunk draws the first token. A cancelled request stops between chunks, on every rank.
    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
        const h = of(ptr);
        if (s.isCancelled()) return error.Cancelled;
        const lane = if (h.lanes.get(s)) |l| (if (l.fill != null) l else try h.begin(s)) else try h.begin(s);
        const prompt = s.prompt();
        const f = &lane.fill.?;
        const limit: usize = if (f.next < f.count) f.stops[f.next] else prompt.len;
        const to = prefix.chunkEnd(f.at, limit);
        if (h.link != null) try h.send(&.{ @backingInt(worker.Op.fill), lane.id, @intCast(to) });
        if (to == prompt.len) {
            _ = h.take(try h.e.prefill(&lane.caches, prompt, f.at, lane.hidden, .{ .sampling = sampling(s), .position = prompt.len }, null));
            lane.fill = null;
            return true;
        }
        try h.e.advance(&lane.caches, prompt, f.at, to, null);
        f.at = to;
        if (f.next < f.count and to == f.stops[f.next]) {
            h.remember(prompt[0..to], &lane.caches);
            f.next += 1;
        }
        return false;
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
            const lane = h.lanes.get(w.stream) orelse return error.UnknownStream;
            if (w.held > lane.held_n or (w.held > 0 and w.tokens.len > 0)) return error.NoHeldDrafts;
            const n = w.rows();
            if (n > tokens[i].len) return error.WindowTooWide;
            try h.settle(lane);
            for (w.positions, 0..) |p, r| if (p != lane.len + 1 + r) {
                std.log.err("row {d} keyed at {d}, the stream holds {d} slots", .{ r, p, lane.len });
                return error.PositionMismatch;
            };
            tokens[i][0] = w.pending;
            if (w.held > 0) @memcpy(tokens[i][1..n], lane.held[0..w.held]) else @memcpy(tokens[i][1..n], w.tokens);
            rows[i] = .{ .caches = &lane.caches, .pos = lane.len, .tokens = tokens[i][0..n] };
            h.order[i] = w.stream;
        }
        var reqs: [256]draw.Request = undefined;
        var drawn: [256]u32 = undefined;
        var total: usize = 0;
        for (windows) |w| {
            if (total + w.positions.len > reqs.len) return error.WindowTooWide;
            for (w.positions) |p| {
                reqs[total] = .{ .sampling = sampling(w.stream), .position = p };
                total += 1;
            }
        }
        // the graph choice is rank 0's, and goes with the round
        const pick = try h.e.choose(rows[0..windows.len], null);
        if (h.link != null) {
            h.msg.clearRetainingCapacity();
            try h.msg.appendSlice(h.gpa, &.{ @backingInt(worker.Op.verify), @backingInt(pick), @intCast(windows.len) });
            for (windows, rows[0..windows.len]) |w, r| {
                try h.msg.appendSlice(h.gpa, &.{ h.lanes.get(w.stream).?.id, @intCast(r.tokens.len) });
                try h.msg.appendSlice(h.gpa, r.tokens);
            }
            try h.send(h.msg.items);
        }
        const r = try h.e.verify(rows[0..windows.len], h.wins[0..windows.len], h.snaps, reqs[0..total], &drawn);
        h.round_hidden = r.hidden.ptr;
        var at: usize = 0;
        for (windows, out, 0..) |w, *o, i| {
            @memcpy(o.sampled, drawn[at..][0..o.sampled.len]);
            const lane = h.lanes.get(w.stream).?;
            @memcpy(o.drafts, if (w.held > 0) lane.held[0..w.held] else w.tokens);
            lane.held_n = 0;
            lane.pending = .{ .window = i, .rows = w.rows(), .start = at };
            at += w.rows();
        }
    }

    /// Keep each stream's accepted prefix: lengths, and the linear states of its last kept row.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const h = of(ptr);
        for (windows, paths) |w, path| {
            const lane = h.lanes.get(w.stream) orelse return error.UnknownStream;
            for (path, 0..) |r, j| if (r != j) return error.TreesNotBuilt;
            // the draft request of the same round may have kept these rows already
            if (lane.pending == null) continue;
            try h.commit(lane, path.len);
        }
    }

    /// The head drafts `depth` from the stream's last kept row and the token after it (its pending one).
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const h = of(ptr);
        const head = h.head orelse return error.NoDraftHead;
        const m = h.e.model();
        for (requests) |r| {
            const lane = h.lanes.get(r.stream) orelse return error.UnknownStream;
            // a shared round asks for drafts before its keep: the request names the kept rows
            if (lane.pending != null) {
                if (r.rows) |kept| try h.commit(lane, kept.len) else try h.settle(lane);
            }
            if (r.position != lane.len + 1) {
                std.log.err("draft for {s} at {d}, the stream holds {d} slots (rows {any}, depth {d})", .{ r.stream.id, r.position, lane.len, r.rows, r.depth });
                return error.PositionMismatch;
            }
            const token: u32 = if (r.rows != null) r.follow[r.follow.len - 1] else switch (r.first orelse return error.NoFirstToken) {
                .handle => |at| h.drawn[at % ring],
                .value => |v| v,
            };
            lane.held_n = 0;
            if (r.depth == 0) continue;
            // a chain ends after a draft the head gives under the confidence (`heldFn` tells the core how many it holds);
            // the prompt's first drafts are asked for before any hook reads the count, so they run whole
            const stop_under: f64 = if (r.rows != null) mtp.confidence else 0.0;
            lane.held_n = try head.chain(&h.e.lib, h.e.stream, &h.e.drawer, m, .{ .ptr = lane.hidden.ptr, .kind = m.act }, token, lane.len, r.depth, sampling(r.stream), stop_under, &lane.held);
            if (lane.held_n > r.depth or (lane.held_n < r.depth and r.rows == null)) return error.ShortChain;
        }
    }

    /// The drafts the head holds for the stream after its last chain: fewer than asked when a draft cut it.
    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, _: std.mem.Allocator) anyerror!?lanes.stream.Held {
        const lane = of(ptr).lanes.get(s) orelse return error.UnknownStream;
        return .{ .count = @intCast(lane.held_n) };
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const h = of(ptr);
        const kv = h.lanes.fetchRemove(s) orelse return;
        if (h.link != null) h.send(&.{ @backingInt(worker.Op.release), kv.value.id }) catch {};
        _ = h.ids.remove(s);
        h.e.stream.synchronize() catch {};
        h.free(kv.value);
    }
};
