//! A tensor-parallel rank above 0: runs the forwards rank 0 sends in its order, so the collectives pair up.
const std = @import("std");
const hip = @import("hip");
const Engine = @import("engine.zig").Engine;
const cache = @import("cache.zig");
const forward = @import("forward.zig");

/// A message's first word; rank 0 decides everything, a rank only applies it.
pub const Op = enum(u32) { stop, begin, fill, verify, keep, release };

const Lane = struct {
    caches: cache.Caches,
    pending: [cache.max_window]u32 = undefined,
};

/// Rank 0's messages until it says stop: a stream begun, a prompt chunk, a window, a keep, a release.
pub fn follow(gpa: std.mem.Allocator, e: *Engine, link: hip.collective.Link) !void {
    var lanes: std.AutoHashMapUnmanaged(u32, *Lane) = .empty;
    defer {
        var it = lanes.valueIterator();
        while (it.next()) |l| {
            l.*.caches.deinit(gpa);
            gpa.destroy(l.*);
        }
        lanes.deinit(gpa);
    }
    var msg: std.ArrayList(u32) = .empty;
    defer msg.deinit(gpa);
    const d = e.weights.model.dims;
    while (true) {
        msg.clearRetainingCapacity();
        try link.recv(gpa, &msg);
        const w = msg.items;
        if (w.len == 0) return error.EmptyMessage;
        switch (@as(Op, @fromBackingInt(@intCast(w[0])))) {
            .stop => return,
            .begin => {
                const lane = try gpa.create(Lane);
                errdefer gpa.destroy(lane);
                lane.* = .{ .caches = try e.caches(w[2]) };
                const gop = try lanes.getOrPut(gpa, w[1]);
                if (gop.found_existing) {
                    gop.value_ptr.*.caches.deinit(gpa);
                    gpa.destroy(gop.value_ptr.*);
                }
                gop.value_ptr.* = lane;
            },
            .fill => {
                const lane = lanes.get(w[1]) orelse return error.UnknownStream;
                const n = w[3];
                const out = try e.prompt(&lane.caches, w[4..][0..n]);
                if (w[2] != 0) _ = try e.logits(e.ops(&e.arena), forward.at(out.mixed, (n - 1) * d.hidden), 1);
                try e.stream.synchronize();
            },
            .verify => {
                const lane = lanes.get(w[1]) orelse return error.UnknownStream;
                const n = w[2];
                @memcpy(lane.pending[0..n], w[3..][0..n]);
                const out = try e.run(&lane.caches, lane.pending[0..n], lane.caches.len, true);
                _ = try e.logits(e.ops(&e.arena), out.mixed, n);
                try e.stream.synchronize();
            },
            .keep => {
                const lane = lanes.get(w[1]) orelse return error.UnknownStream;
                try e.keepWindow(&lane.caches, lane.pending[0..w[3]], w[2]);
            },
            .release => if (lanes.fetchRemove(w[1])) |kv| {
                try e.stream.synchronize();
                kv.value.caches.deinit(gpa);
                gpa.destroy(kv.value);
            },
        }
    }
}
