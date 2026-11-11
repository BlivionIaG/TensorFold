//! A restart's view of the NVMe tier: the valid files of a namespace, collected, optionally narrowed, then indexed.

const std = @import("std");
const linux = std.os.linux;
const disk = @import("disk.zig");
const Disk = disk.Disk;
const lanes_mod = @import("lanes.zig");
const Head = @import("format.zig").Head;
const Digest = @import("prefix.zig").Digest;
const charge = @import("disk_charge.zig");

/// A valid file found on restart: its header, size and verified ids (tokens [first_page x page, pos), owned).
pub const Found = struct {
    head: Head,
    size: u64,
    mtime: i64,
    ids: []i32,

    /// Bases first (a delta starts past its base's first page), a rank-independent order after that.
    fn before(_: void, a: Found, b: Found) bool {
        if (a.head.first_page != b.head.first_page) return a.head.first_page < b.head.first_page;
        return std.mem.order(u8, &a.head.key, &b.head.key) == .lt;
    }

    fn older(_: void, a: Found, b: Found) bool {
        return a.mtime < b.mtime;
    }
};

pub fn deinitFound(d: *Disk, out: *std.ArrayList(Found)) void {
    for (out.items) |f| d.gpa.free(f.ids);
    out.deinit(d.gpa);
}

/// Reads the directory on restart: drops .tmp and bad files, checks ids; the valid files, in no order.
pub fn collect(d: *Disk, out: *std.ArrayList(Found)) !void {
    if (@import("builtin").os.tag != .linux) return error.UnsupportedPlatform;
    var it = d.dir.iterate();
    var nb: [48]u8 = undefined;
    while (try it.next(d.io)) |ent| {
        if (ent.name.len >= nb.len) continue;
        @memcpy(nb[0..ent.name.len], ent.name);
        nb[ent.name.len] = 0;
        const name = nb[0..ent.name.len :0];
        if (std.mem.endsWith(u8, name, ".tmp")) {
            _ = linux.unlinkat(d.dir.handle, name, 0);
            continue;
        }
        if (!std.mem.endsWith(u8, name, ".tfs")) continue;
        const f = found(d, name) catch {
            d.stats.bad += 1;
            _ = linux.unlinkat(d.dir.handle, name, 0);
            continue;
        };
        try out.append(d.gpa, f);
    }
}

/// Deletes the file of out[i] and drops it from the list.
pub fn discard(d: *Disk, out: *std.ArrayList(Found), i: usize) void {
    const f = out.orderedRemove(i);
    var kb: [48]u8 = undefined;
    _ = linux.unlinkat(d.dir.handle, Disk.fileName(f.head.key, "tfs", &kb), 0);
    d.gpa.free(f.ids);
}

/// Indexes collected files, bases first, in a rank-independent order (orphan deltas deleted); the LRU order is age.
pub fn adopt(d: *Disk, out: *std.ArrayList(Found)) !void {
    std.mem.sort(Found, out.items, {}, Found.before);
    var i: usize = 0;
    while (i < out.items.len) {
        const f = out.items[i];
        const base: ?Digest = if (f.head.hasBase()) f.head.base else null;
        if (base != null and d.by_key.get(base.?) == null) {
            d.stats.bad += 1;
            discard(d, out, i);
            continue;
        }
        _ = try d.add(f.head.key, base, f.head.tag, f.head.kind, f.head.pos, f.head.first_page, f.head.npages, f.size, try charge.ofFile(&f.head, d.opts.world, f.size));
        i += 1;
    }
    const by_age = try d.gpa.dupe(Found, out.items);
    defer d.gpa.free(by_age);
    std.mem.sort(Found, by_age, {}, Found.older);
    for (by_age) |f| d.touch(f.head.key);
}

/// Indexes the directory on restart: drops .tmp and bad files, checks ids, registers bases before their deltas.
pub fn scan(d: *Disk, out: *std.ArrayList(Found)) !void {
    try collect(d, out);
    try adopt(d, out);
}

fn found(d: *Disk, name: [:0]const u8) !Found {
    const f = try d.openFile(name, false);
    defer _ = linux.close(f.fd);
    var lay = try d.readHead(f.fd);
    defer lay.deinit(d.gpa);
    const h = lay.head;
    var kb: [48]u8 = undefined;
    if (!std.mem.eql(u8, Disk.fileName(h.key, "tfs", &kb), name)) return error.BadHeader;
    if (!std.mem.eql(u8, &h.compat, &d.opts.compat) or h.rank != d.opts.rank or h.world != d.opts.world) return error.Foreign;
    _ = try charge.ofFile(&h, d.opts.world, 0); // a split world's file without its logical charge is not adopted
    var st: linux.Statx = undefined;
    if (linux.errno(linux.statx(f.fd, "", linux.AT.EMPTY_PATH, .{ .SIZE = true, .MTIME = true }, &st)) != .SUCCESS or st.size < h.total) return error.Truncated;
    const seg = lay.find(.ids, "ids") orelse return error.BadHeader;
    if (seg.nbytes % 4 != 0 or h.pos < @as(u64, h.first_page) * h.page or seg.nbytes / 4 != h.pos - @as(u64, h.first_page) * h.page) return error.BadHeader;
    const ids = try d.gpa.alloc(i32, seg.nbytes / 4);
    errdefer d.gpa.free(ids);
    const sb = try d.staging.get();
    defer d.staging.put(sb);
    for (0..seg.nchunks) |ci| {
        const c = seg.chunkAt(@intCast(ci));
        const buf = d.staging.buf(sb);
        try lanes_mod.preadAll(f.fd, buf[0..c.span], c.off);
        var dg: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(buf[0..c.data], &dg, .{});
        if (!std.mem.eql(u8, &dg, &lay.digests[seg.first_digest + ci])) return error.ChecksumMismatch;
        @memcpy(std.mem.sliceAsBytes(ids)[@as(usize, ci) * seg.chunk ..][0..c.data], buf[0..c.data]);
    }
    return .{ .head = h, .size = st.size, .mtime = st.mtime.sec * std.time.ns_per_s + st.mtime.nsec, .ids = ids };
}
