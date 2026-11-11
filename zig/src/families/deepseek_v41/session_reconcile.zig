//! A restart's NVMe tier on several ranks: each rank indexes the files every rank found, in the same order.

const std = @import("std");
const cuda = @import("cuda");
const kv = @import("kv");
const fwd = @import("forward.zig");

const sessions = kv.sessions;
const Disk = sessions.Disk;

/// The ranks' all-gather of i32 words over the pair's collective, outside any capture (boot: nothing else runs).
const Wire = struct {
    f: *fwd.Forward,

    fn all(ctx: *anyopaque, mine: []const i32, out: []i32) anyerror!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        const ranks: usize = w.f.comm.world();
        const n = mine.len;
        const stream = w.f.runner.stream;
        var dev = try cuda.DeviceBuffer.alloc(w.f.runner.d, 4 * n * (ranks + 1));
        defer dev.free();
        var host = try cuda.HostBuffer.alloc(w.f.runner.d, 4 * n * (ranks + 1));
        defer host.free();
        try stream.synchronize();
        const h = host.slice(i32);
        @memcpy(h[0..n], mine);
        try dev.uploadAsync(0, std.mem.sliceAsBytes(h[0..n]), stream.handle);
        try w.f.comm.allGather(dev.ptr, dev.ptr + 4 * n, n, .i32, stream.handle);
        try dev.downloadAsync(4 * n, std.mem.sliceAsBytes(h[n..][0 .. ranks * n]), stream.handle);
        try stream.synchronize();
        @memcpy(out, h[n..][0 .. ranks * n]);
    }
};

const Narrower = struct {
    gpa: std.mem.Allocator,
    wire: Wire,

    fn run(ctx: *anyopaque, d: *Disk, found: *std.ArrayList(Disk.Found)) anyerror!void {
        const n: *Narrower = @ptrCast(@alignCast(ctx));
        try sessions.rank_narrow.narrow(n.gpa, d, found, n.wire.f.comm.world(), .{ .ctx = &n.wire, .all = Wire.all });
    }
};

/// The files found on this rank's tier become index entries; returns how many (every rank the same).
pub fn reconcile(gpa: std.mem.Allocator, f: *fwd.Forward, store: *sessions.Store) !u32 {
    if (f.comm.world() <= 1) return store.reconcile();
    var n: Narrower = .{ .gpa = gpa, .wire = .{ .f = f } };
    return store.reconcileWith(.{ .ctx = &n, .run = Narrower.run });
}
