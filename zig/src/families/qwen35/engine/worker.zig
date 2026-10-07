//! A tensor-parallel rank above 0: runs the steps rank 0 sends in the same order, so the collectives pair up.

const std = @import("std");
const hip = @import("hip");
const Engine = @import("engine.zig").Engine;
const Pick = @import("engine.zig").Pick;
const state = @import("../forward/state.zig");
const draw = @import("draw.zig");
const prefix = @import("prefix.zig");

/// A step rank 0 sends the other ranks (the first word of a message).
/// prefill (begins a prompt pass): id, total, len, resumed at, cut count, kept entries, byte budget (low, high), cuts..., prompt...;
/// fill: id, end of the next chunk of the prompt pass; verify: graph pick, count, then id, rows, tokens... each (the round's plan
/// shape follows from them: every rank derives the same); keep: count, then id, rows each;
/// release: id; stop.
pub const Op = enum(u32) { stop, prefill, verify, keep, release, fill };

/// Windows a round holds at most.
const max_windows = 128;

const Fill = struct { prompt: []u32, at: usize, stops: []u32, next: usize };

const Lane = struct {
    caches: state.Caches,
    /// Slots written and kept: the next window starts here.
    len: usize,
    /// The prompt pass in progress: the prompt, the row it has reached, the cuts it keeps a state at.
    fill: ?Fill = null,
    /// The last verify's slot and rows, until rank 0's keep.
    pending: ?struct { window: usize, rows: usize } = null,
};

pub const Worker = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    lanes: std.AutoHashMapUnmanaged(u32, *Lane) = .empty,
    reqs: []draw.Request,
    out: []u32,
    /// rank 0's kept prompts, mirrored: the same cuts in the same order keep the same entries
    kept: prefix.Cache,

    pub fn init(gpa: std.mem.Allocator, e: *Engine) !Worker {
        const rows = e.o.batch_rows;
        const reqs = try gpa.alloc(draw.Request, rows);
        errdefer gpa.free(reqs);
        // a follower's draws are greedy and unread: the forward and its collectives are what it shares
        @memset(reqs, .{ .sampling = null, .position = 0 });
        return .{ .gpa = gpa, .e = e, .reqs = reqs, .out = try gpa.alloc(u32, rows), .kept = prefix.Cache.init(gpa, 0, 0) };
    }

    pub fn deinit(w: *Worker) void {
        w.e.stream.synchronize() catch {};
        w.kept.deinit();
        var it = w.lanes.valueIterator();
        while (it.next()) |l| w.destroy(l.*);
        w.lanes.deinit(w.gpa);
        w.gpa.free(w.out);
        w.gpa.free(w.reqs);
    }

    fn destroy(w: *Worker, l: *Lane) void {
        if (l.fill) |f| {
            w.gpa.free(f.prompt);
            w.gpa.free(f.stops);
        }
        w.e.drain();
        l.caches.deinit(w.gpa);
        w.gpa.destroy(l);
    }

    fn prefill(w: *Worker, id: u32, total: usize, prompt: []const u32, at: usize, stops: []const u32) !void {
        const gop = try w.lanes.getOrPut(w.gpa, id);
        if (gop.found_existing) w.destroy(gop.value_ptr.*);
        errdefer w.lanes.removeByPtr(gop.key_ptr);
        const lane = try w.gpa.create(Lane);
        errdefer w.gpa.destroy(lane);
        lane.* = .{ .caches = try w.e.newCaches(total), .len = prompt.len };
        gop.value_ptr.* = lane;
        if (at > 0) {
            const hit = w.kept.longest(prompt) orelse return error.PrefixMissing;
            if (hit.ids.len != at) return error.PrefixMismatch;
            try lane.caches.copyPrefix(&hit.caches, w.e.model(), at, w.e.stream.handle);
        }
        const own = try w.gpa.dupe(u32, prompt);
        errdefer w.gpa.free(own);
        lane.fill = .{ .prompt = own, .at = at, .stops = try w.gpa.dupe(u32, stops), .next = 0 };
    }

    /// The next chunk of a prompt pass, to row `to`: rank 0's cuts and chunk ends, run the same way.
    fn fill(w: *Worker, id: u32, to: usize) !void {
        const lane = w.lanes.get(id) orelse return error.UnknownStream;
        const f = &(lane.fill orelse return error.NoPromptPass);
        if (to == f.prompt.len) {
            _ = try w.e.prefill(&lane.caches, f.prompt, f.at, null, w.reqs[0], null);
            w.gpa.free(f.prompt);
            w.gpa.free(f.stops);
            lane.fill = null;
            return;
        }
        try w.e.advance(&lane.caches, f.prompt, f.at, to, null);
        f.at = to;
        if (f.next < f.stops.len and to == f.stops[f.next]) {
            try w.remember(f.prompt[0..to], &lane.caches);
            f.next += 1;
        }
    }

    /// A copy of the caches at a cut, as rank 0's `remember` keeps it.
    fn remember(w: *Worker, ids: []const u32, caches: *const state.Caches) !void {
        if (w.kept.keep == 0 or w.kept.has(ids)) return;
        var snap = try state.Caches.blank(w.gpa, &w.e.driver, w.e.model(), ids.len);
        errdefer snap.deinit(w.gpa);
        try snap.copyPrefix(caches, w.e.model(), ids.len, w.e.stream.handle);
        try w.kept.add(ids, snap);
    }

    fn verify(w: *Worker, ids: []const u32, rows: []const Engine.Rows, pick: Pick) !void {
        var total: usize = 0;
        for (rows) |r| total += r.tokens.len;
        _ = try w.e.choose(rows, pick);
        _ = try w.e.verify(rows, w.reqs[0..total], w.out[0..total]);
        for (ids, 0..) |id, i| w.lanes.get(id).?.pending = .{ .window = i, .rows = rows[i].tokens.len };
    }

    fn keep(w: *Worker, id: u32, rows: usize) !void {
        const lane = w.lanes.get(id) orelse return error.UnknownStream;
        const p = lane.pending orelse return error.NothingToKeep;
        w.e.keep(p.window, rows);
        lane.len += rows;
        lane.pending = null;
    }

    fn release(w: *Worker, id: u32) void {
        const kv = w.lanes.fetchRemove(id) orelse return;
        w.e.stream.synchronize() catch {};
        w.destroy(kv.value);
    }

    /// Runs rank 0's steps until it says stop.
    pub fn follow(w: *Worker, link: *const hip.link.Link) !void {
        var msg: std.ArrayList(u32) = .empty;
        defer msg.deinit(w.gpa);
        var ids: [max_windows]u32 = undefined;
        var rows: [max_windows]Engine.Rows = undefined;
        while (true) {
            try link.recv(w.gpa, &msg);
            const m = msg.items;
            switch (@as(Op, @fromBackingInt(@as(u32, @intCast(m[0]))))) {
                .stop => return,
                .prefill => {
                    w.kept.keep = m[6];
                    w.kept.budget = @intCast(@as(u64, m[7]) | @as(u64, m[8]) << 32);
                    const stops = m[9..][0..m[5]];
                    try w.prefill(m[1], m[2], m[9 + m[5] ..][0..m[3]], m[4], stops);
                },
                .verify => {
                    const pick: Pick = @fromBackingInt(m[1]);
                    const n = m[2];
                    if (n > max_windows) return error.WindowTooWide;
                    var at: usize = 3;
                    for (0..n) |i| {
                        const lane = w.lanes.get(m[at]) orelse return error.UnknownStream;
                        ids[i] = m[at];
                        rows[i] = .{ .caches = &lane.caches, .pos = lane.len, .tokens = m[at + 2 ..][0..m[at + 1]] };
                        at += 2 + m[at + 1];
                    }
                    try w.verify(ids[0..n], rows[0..n], pick);
                },
                .keep => {
                    for (0..m[1]) |i| try w.keep(m[2 + 2 * i], m[3 + 2 * i]);
                    try w.e.flush();
                },
                .release => w.release(m[1]),
                .fill => try w.fill(m[1], m[2]),
            }
        }
    }
};
