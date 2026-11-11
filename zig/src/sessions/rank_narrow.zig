//! A restart on two or more ranks: each keeps only the files every rank found, so their tier indexes match.

const std = @import("std");
const Disk = @import("disk.zig").Disk;
const Digest = @import("prefix.zig").Digest;

const words = @sizeOf(Digest) / 4;

/// Every rank's contribution laid end to end: `mine` (same length on all ranks) in, world x mine.len out.
pub const Gather = struct {
    ctx: *anyopaque,
    all: *const fn (ctx: *anyopaque, mine: []const i32, out: []i32) anyerror!void,
};

/// Most files a rank may offer; a tier past it is not reconciled (the caller purges it).
pub const max_files = 1 << 18;

/// Deletes this rank's collected files that another rank did not find: a file one rank lacks is gone everywhere.
pub fn narrow(gpa: std.mem.Allocator, d: *Disk, found: *std.ArrayList(Disk.Found), world: u32, g: Gather) !void {
    const w: usize = world;
    const counts = try gpa.alloc(i32, w);
    defer gpa.free(counts);
    try g.all(g.ctx, &.{@intCast(found.items.len)}, counts);
    var most: usize = 0;
    for (counts) |c| most = @max(most, @as(usize, @intCast(c)));
    if (most > max_files) return error.TooManyFiles;
    if (most == 0) return;
    const mine = try gpa.alloc(i32, most * words);
    defer gpa.free(mine);
    @memset(mine, 0);
    for (found.items, 0..) |f, i| @memcpy(std.mem.sliceAsBytes(mine[i * words ..][0..words]), &f.head.key);
    const all = try gpa.alloc(i32, w * mine.len);
    defer gpa.free(all);
    try g.all(g.ctx, mine, all);
    const sets = try gpa.alloc(std.AutoHashMapUnmanaged(Digest, void), w);
    @memset(sets, .empty);
    defer {
        for (sets) |*s| s.deinit(gpa);
        gpa.free(sets);
    }
    for (sets, 0..) |*s, r| {
        for (0..@as(usize, @intCast(counts[r]))) |i| {
            var k: Digest = undefined;
            @memcpy(&k, std.mem.sliceAsBytes(all[r * mine.len + i * words ..][0..words]));
            try s.put(gpa, k, {});
        }
    }
    var i: usize = 0;
    while (i < found.items.len) {
        const key = found.items[i].head.key;
        var everywhere = true;
        for (sets) |*s| everywhere = everywhere and s.contains(key);
        if (everywhere) i += 1 else d.discard(found, i);
    }
}
