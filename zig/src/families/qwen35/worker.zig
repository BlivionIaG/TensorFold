//! One rank's streams: each has its caches and the slots it holds, the last verify's windows wait for their keep. Rank 0's
//! lane backend drives a worker and, under tensor parallelism, sends every step to the other ranks' workers (`follow`),
//! which run the same forwards in the same order so the collectives pair up.

const std = @import("std");
const hip = @import("hip");
const Engine = @import("engine.zig").Engine;
const state = @import("state.zig");
const win = @import("window.zig");

/// A step rank 0 sends the other ranks (the first word of a message).
pub const Op = enum(u32) { stop, prefill, verify, keep, release };

/// Windows a round holds at most.
const max_windows = 64;
/// Rows of one window.
const window_rows = 16;

const Lane = struct {
    caches: state.Caches,
    /// Slots written and kept: the next window starts here.
    len: usize,
    /// The last verify's window and rows: kept whole unless a keep drops some first.
    pending: ?struct { window: usize, rows: usize } = null,
};

/// One stream's rows of a verify: its pending token, then its drafts.
pub const Item = struct { id: u32, tokens: []const u32 };

pub const Worker = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    lanes: std.AutoHashMapUnmanaged(u32, *Lane) = .empty,
    wins: []win.Window,
    snaps: []win.Snapshot,

    pub fn init(gpa: std.mem.Allocator, e: *Engine) !Worker {
        const rows = e.o.batch_rows;
        const wins = try gpa.alloc(win.Window, rows);
        errdefer gpa.free(wins);
        return .{ .gpa = gpa, .e = e, .wins = wins, .snaps = try gpa.alloc(win.Snapshot, rows * e.model().spec.n_layers) };
    }

    pub fn deinit(w: *Worker) void {
        w.e.stream.synchronize() catch {};
        var it = w.lanes.valueIterator();
        while (it.next()) |l| w.destroy(l.*);
        w.lanes.deinit(w.gpa);
        w.gpa.free(w.snaps);
        w.gpa.free(w.wins);
    }

    fn destroy(w: *Worker, l: *Lane) void {
        l.caches.deinit(w.gpa);
        w.gpa.destroy(l);
    }

    /// A new stream `id` over `prompt`, prefilled; the last row's logits (the vocabulary whole).
    pub fn prefill(w: *Worker, id: u32, prompt: []const u32) ![]const u16 {
        if (prompt.len == 0 or prompt.len > w.e.o.capacity) return error.PromptTooLong;
        const gop = try w.lanes.getOrPut(w.gpa, id);
        if (gop.found_existing) {
            w.destroy(gop.value_ptr.*);
            gop.value_ptr.* = undefined;
        }
        errdefer w.lanes.removeByPtr(gop.key_ptr);
        const lane = try w.gpa.create(Lane);
        errdefer w.gpa.destroy(lane);
        lane.* = .{ .caches = try w.e.newCaches(), .len = prompt.len };
        errdefer lane.caches.deinit(w.gpa);
        gop.value_ptr.* = lane;
        return w.e.prefill(&lane.caches, prompt, 0, null);
    }

    /// Keeps the pending window whole (a verify the round loop did not trim), then the slots the stream holds.
    pub fn settled(w: *Worker, id: u32) !usize {
        const lane = w.lanes.get(id) orelse return error.UnknownStream;
        try w.settle(lane);
        return lane.len;
    }

    /// A verify the round loop did not trim keeps every row (the core keeps only on drops).
    fn settle(w: *Worker, l: *Lane) !void {
        const p = l.pending orelse return;
        try w.e.keep(w.wins[p.window], p.rows);
        l.len += p.rows;
        l.pending = null;
    }

    /// Every item's window from its stream's kept length, in one forward; the logits of all rows in order.
    pub fn verify(w: *Worker, items: []const Item) ![]const u16 {
        if (items.len > w.wins.len or items.len > max_windows) return error.WindowTooWide;
        var rows: [max_windows]Engine.Rows = undefined;
        for (items, 0..) |item, i| {
            if (item.tokens.len > window_rows) return error.WindowTooWide;
            const lane = w.lanes.get(item.id) orelse return error.UnknownStream;
            try w.settle(lane);
            rows[i] = .{ .caches = &lane.caches, .pos = lane.len, .tokens = item.tokens };
        }
        const r = try w.e.verify(rows[0..items.len], w.wins[0..items.len], w.snaps);
        for (items, 0..) |item, i| w.lanes.get(item.id).?.pending = .{ .window = i, .rows = item.tokens.len };
        return r.logits;
    }

    /// Keeps the first `rows` rows of stream `id`'s verified window.
    pub fn keep(w: *Worker, id: u32, rows: usize) !void {
        const lane = w.lanes.get(id) orelse return error.UnknownStream;
        const p = lane.pending orelse return error.NothingToKeep;
        try w.e.keep(w.wins[p.window], rows);
        lane.len += rows;
        lane.pending = null;
    }

    pub fn release(w: *Worker, id: u32) void {
        const kv = w.lanes.fetchRemove(id) orelse return;
        w.e.stream.synchronize() catch {};
        w.destroy(kv.value);
    }
};

/// A rank above 0: runs rank 0's steps until it says stop.
pub fn follow(gpa: std.mem.Allocator, w: *Worker, link: *const hip.link.Link) !void {
    var msg: std.ArrayList(u32) = .empty;
    defer msg.deinit(gpa);
    var items: std.ArrayList(Item) = .empty;
    defer items.deinit(gpa);
    while (true) {
        try link.recv(gpa, &msg);
        const m = msg.items;
        switch (@as(Op, @fromBackingInt(@intCast(m[0])))) {
            .stop => return,
            .prefill => _ = try w.prefill(m[1], m[3..][0..m[2]]),
            .verify => {
                items.clearRetainingCapacity();
                var at: usize = 2;
                for (0..m[1]) |_| {
                    try items.append(gpa, .{ .id = m[at], .tokens = m[at + 2 ..][0..m[at + 1]] });
                    at += 2 + m[at + 1];
                }
                _ = try w.verify(items.items);
            },
            .keep => for (0..m[1]) |i| try w.keep(m[2 + 2 * i], m[3 + 2 * i]),
            .release => w.release(m[1]),
        }
    }
}
