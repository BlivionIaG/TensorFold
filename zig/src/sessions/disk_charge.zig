//! Split-rank eviction uses logical bytes; physical file sizes remain available for disk headroom reporting.

const std = @import("std");
const disk = @import("disk.zig");
const Head = @import("format.zig").Head;

const unit = 4096;

/// An entry's budget charge: file bytes, or in a split world logical bytes in 4 KiB units (alike on every rank).
pub fn charge(src: *const disk.WriteSrc, world: u32, physical: u64) u64 {
    if (world <= 1) return physical;
    var bytes: u64 = @as(u64, src.ids.len) * 4 + src.measure;
    for (src.families) |f| {
        const pages: u64 = src.npages - src.first_page;
        bytes += pages * f.pageBytes(src.page) / (if (f.split) world else 1);
    }
    return std.mem.alignForward(u64, bytes, unit);
}

/// The header's logical charge in 4 KiB units (0 in a world of one).
pub fn kib(src: *const disk.WriteSrc, world: u32) u32 {
    return if (world <= 1) 0 else @intCast(charge(src, world, 0) / unit);
}

/// A restart's charge for a found file: the header's logical bytes in a split world, else its size.
pub fn ofFile(h: *const Head, world: u32, size: u64) !u64 {
    if (world <= 1) return size;
    if (h.charge_kib == 0) return error.BadHeader;
    return @as(u64, h.charge_kib) * unit;
}
