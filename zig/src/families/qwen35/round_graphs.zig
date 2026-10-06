//! Captured lane rounds: a round's forward recorded once as a HIP graph per window shape (each stream's caches and row
//! count) and replayed on later rounds, which cost one launch instead of the forward's hundreds.

const std = @import("std");
const hip = @import("hip");
const win = @import("window.zig");

/// Shapes kept at once; the least recently used one is dropped for a new one.
const max_entries = 24;

/// One window of a shape: the stream's caches (by serial) and its row count.
pub const Part = struct { serial: u64, rows: u32 };

/// What a replay hands back: the forward's outputs and where the arena stands after it.
pub const Out = struct { hidden: hip.ops.Tensor, y: hip.ops.Tensor, used: usize };

const State = enum { seen, ready, failed };

pub const Entry = struct {
    parts: []Part,
    state: State = .seen,
    exec: ?hip.graph.Exec = null,
    snaps: []win.Snapshot = &.{},
    out: Out = undefined,
    tick: u64 = 0,
    /// Seen before: the next sighting captures.
    again: bool = false,
};

pub const Graphs = struct {
    gpa: std.mem.Allocator,
    entries: std.ArrayList(Entry) = .empty,
    clock: u64 = 0,
    /// Rounds replayed from a graph, and graphs captured.
    replayed: u64 = 0,
    captured: u64 = 0,

    pub fn deinit(g: *Graphs) void {
        for (g.entries.items) |*e| drop(g.gpa, e);
        g.entries.deinit(g.gpa);
    }

    fn drop(gpa: std.mem.Allocator, e: *Entry) void {
        if (e.exec) |*x| x.deinit();
        gpa.free(e.parts);
        gpa.free(e.snaps);
    }

    /// The entry for `parts`, made (state `seen`) if new.
    pub fn find(g: *Graphs, parts: []const Part) !*Entry {
        g.clock += 1;
        for (g.entries.items) |*e| if (std.mem.eql(u8, std.mem.sliceAsBytes(e.parts), std.mem.sliceAsBytes(parts))) {
            e.tick = g.clock;
            return e;
        };
        if (g.entries.items.len >= max_entries) {
            var oldest: usize = 0;
            for (g.entries.items, 0..) |e, i| if (e.tick < g.entries.items[oldest].tick) {
                oldest = i;
            };
            drop(g.gpa, &g.entries.items[oldest]);
            _ = g.entries.swapRemove(oldest);
        }
        try g.entries.append(g.gpa, .{ .parts = try g.gpa.dupe(Part, parts), .tick = g.clock });
        return &g.entries.items[g.entries.items.len - 1];
    }

    /// Forget every graph over the stream's caches (they are about to be freed).
    pub fn forget(g: *Graphs, serial: u64) void {
        var i: usize = 0;
        while (i < g.entries.items.len) {
            const hit = for (g.entries.items[i].parts) |p| {
                if (p.serial == serial) break true;
            } else false;
            if (hit) {
                drop(g.gpa, &g.entries.items[i]);
                _ = g.entries.swapRemove(i);
            } else i += 1;
        }
    }

    /// Instantiates a capture that just ended, keeps it with the round's outputs, and uploads it.
    pub fn keep(g: *Graphs, e: *Entry, graph: hip.graph.Graph, stream: hip.Stream, snaps: []const win.Snapshot, out: Out) !void {
        var tmp = graph;
        defer tmp.deinit();
        var exec = try tmp.instantiate();
        errdefer exec.deinit();
        try exec.upload(stream);
        e.snaps = try g.gpa.dupe(win.Snapshot, snaps);
        e.exec = exec;
        e.out = out;
        e.state = .ready;
        g.captured += 1;
    }
};
