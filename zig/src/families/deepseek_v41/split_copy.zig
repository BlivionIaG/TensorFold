//! The layer-split producer's boundary copies: pinned slots, a copy stream and one event per slot (rank 0).
const std = @import("std");
const cuda = @import("cuda");
const ring_mod = @import("split_ring.zig");
const split = @import("handoff/split.zig");

pub const Ring = ring_mod.Ring;
pub const Sink = ring_mod.Sink;
const slots = ring_mod.slots;

pub const Copy = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    x: [slots]cuda.HostBuffer,
    pre: [slots]cuda.HostBuffer,
    /// per slot: its copy has run (recorded on the copy stream)
    done: [slots]cuda.Event,
    /// the compute stream has produced the chunk's rows (the copy stream waits on it)
    ready: cuda.Event,
    stream: cuda.Stream,
    ring: Ring,
    /// the newest slot a copy was queued for: the next chunk's boundary buffer waits for it
    last: ?usize = null,
    rows_max: u32,

    /// Pinned slots for chunks of up to `rows_max` rows, rows going to `sink`.
    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, sink: Sink, rows_max: u32) !*Copy {
        const c = try gpa.create(Copy);
        errdefer gpa.destroy(c);
        c.* = .{ .gpa = gpa, .d = d, .x = undefined, .pre = undefined, .done = undefined, .ready = undefined, .stream = undefined, .ring = undefined, .rows_max = rows_max };
        var made: usize = 0;
        errdefer for (0..made) |i| {
            c.x[i].free();
            c.pre[i].free();
            c.done[i].deinit();
        };
        for (0..slots) |i| {
            c.x[i] = try cuda.HostBuffer.alloc(d, @as(usize, rows_max) * @as(usize, @intCast(split.stream_bytes)));
            errdefer c.x[i].free();
            c.pre[i] = try cuda.HostBuffer.alloc(d, @as(usize, rows_max) * @as(usize, @intCast(split.pre_bytes)));
            errdefer c.pre[i].free();
            c.done[i] = try cuda.Event.init(d, false);
            made += 1;
        }
        c.ready = try cuda.Event.init(d, false);
        errdefer c.ready.deinit();
        c.stream = try cuda.Stream.init(d, true);
        var xs: [slots][]const u8 = undefined;
        var ps: [slots][]const u8 = undefined;
        for (0..slots) |i| {
            xs[i] = c.x[i].bytes;
            ps[i] = c.pre[i].bytes;
        }
        c.ring = .{ .sink = sink, .be = .{ .ptr = c, .done = eventDone, .wait = eventWait }, .x = xs, .pre = ps };
        return c;
    }

    /// Frees the slots once the copy stream is past every queued copy (a failed split drops its rows unpublished).
    pub fn destroy(c: *Copy) void {
        c.stream.deinit();
        c.ready.deinit();
        for (0..slots) |i| {
            c.x[i].free();
            c.pre[i].free();
            c.done[i].deinit();
        }
        c.gpa.destroy(c);
    }

    fn eventDone(ptr: *anyopaque, slot: usize) anyerror!bool {
        const c: *Copy = @ptrCast(@alignCast(ptr));
        return c.done[slot].done();
    }

    fn eventWait(ptr: *anyopaque, slot: usize) anyerror!void {
        const c: *Copy = @ptrCast(@alignCast(ptr));
        return c.done[slot].synchronize();
    }
};
