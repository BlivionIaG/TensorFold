//! A tensor-parallel rank above 0: rank 0's lane backend sends every step (`Op`) and this rank runs the same forwards
//! in the same order over its own shares, so the collectives pair up. It draws nothing it keeps: tokens are rank 0's.

const std = @import("std");
const hip = @import("hip");
const Engine = @import("engine.zig").Engine;
const state = @import("state.zig");
const win = @import("window.zig");
const draw = @import("draw.zig");
const prefix = @import("prefix.zig");

/// A step rank 0 sends the other ranks (the first word of a message).
/// prefill: id, total, len, resumed at, cut count, kept entries, byte budget (low, high), cuts..., prompt...; verify: count, then id, rows, tokens... each; keep: count, then id, rows each;
/// release: id; stop.
pub const Op = enum(u32) { stop, prefill, verify, keep, release };

/// Windows a round holds at most.
const max_windows = 128;

const Lane = struct {
    caches: state.Caches,
    /// Slots written and kept: the next window starts here.
    len: usize,
    /// The last verify's window and rows, until rank 0's keep.
    pending: ?struct { window: usize, rows: usize } = null,
};

pub const Worker = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    lanes: std.AutoHashMapUnmanaged(u32, *Lane) = .empty,
    wins: []win.Window,
    snaps: []win.Snapshot,
    reqs: []draw.Request,
    out: []u32,
    /// rank 0's kept prompts, mirrored: the same cuts in the same order keep the same entries
    kept: prefix.Cache,

    pub fn init(gpa: std.mem.Allocator, e: *Engine) !Worker {
        const rows = e.o.batch_rows;
        const wins = try gpa.alloc(win.Window, rows);
        errdefer gpa.free(wins);
        const snaps = try gpa.alloc(win.Snapshot, rows * e.model().spec.n_layers);
        errdefer gpa.free(snaps);
        const reqs = try gpa.alloc(draw.Request, rows);
        errdefer gpa.free(reqs);
        // a follower's draws are greedy and unread: the forward and its collectives are what it shares
        @memset(reqs, .{ .sampling = null, .position = 0 });
        return .{ .gpa = gpa, .e = e, .wins = wins, .snaps = snaps, .reqs = reqs, .out = try gpa.alloc(u32, rows), .kept = prefix.Cache.init(gpa, 0, 0) };
    }

    pub fn deinit(w: *Worker) void {
        w.e.stream.synchronize() catch {};
        w.kept.deinit();
        var it = w.lanes.valueIterator();
        while (it.next()) |l| w.destroy(l.*);
        w.lanes.deinit(w.gpa);
        w.gpa.free(w.out);
        w.gpa.free(w.reqs);
        w.gpa.free(w.snaps);
        w.gpa.free(w.wins);
    }

    fn destroy(w: *Worker, l: *Lane) void {
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
        var from = at;
        for (stops) |stop| {
            try w.e.advance(&lane.caches, prompt, from, stop, null);
            from = stop;
            try w.remember(prompt[0..from], &lane.caches);
        }
        _ = try w.e.prefill(&lane.caches, prompt, from, null, w.reqs[0], null);
    }

    /// A copy of the caches at a cut, as rank 0's `remember` keeps it.
    fn remember(w: *Worker, ids: []const u32, caches: *const state.Caches) !void {
        if (w.kept.keep == 0 or w.kept.has(ids)) return;
        var snap = try state.Caches.blank(w.gpa, &w.e.driver, w.e.model(), ids.len);
        errdefer snap.deinit(w.gpa);
        try snap.copyPrefix(caches, w.e.model(), ids.len, w.e.stream.handle);
        try w.kept.add(ids, snap);
    }

    fn verify(w: *Worker, ids: []const u32, rows: []const Engine.Rows) !void {
        var total: usize = 0;
        for (rows) |r| total += r.tokens.len;
        _ = try w.e.verify(rows, w.wins[0..rows.len], w.snaps, w.reqs[0..total], w.out[0..total]);
        for (ids, 0..) |id, i| w.lanes.get(id).?.pending = .{ .window = i, .rows = rows[i].tokens.len };
    }

    fn keep(w: *Worker, id: u32, rows: usize) !void {
        const lane = w.lanes.get(id) orelse return error.UnknownStream;
        const p = lane.pending orelse return error.NothingToKeep;
        try w.e.keep(w.wins[p.window], rows);
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
                    const n = m[1];
                    if (n > max_windows) return error.WindowTooWide;
                    var at: usize = 2;
                    for (0..n) |i| {
                        const lane = w.lanes.get(m[at]) orelse return error.UnknownStream;
                        ids[i] = m[at];
                        rows[i] = .{ .caches = &lane.caches, .pos = lane.len, .tokens = m[at + 2 ..][0..m[at + 1]] };
                        at += 2 + m[at + 1];
                    }
                    try w.verify(ids[0..n], rows[0..n]);
                },
                .keep => for (0..m[1]) |i| try w.keep(m[2 + 2 * i], m[3 + 2 * i]),
                .release => w.release(m[1]),
            }
        }
    }
};
