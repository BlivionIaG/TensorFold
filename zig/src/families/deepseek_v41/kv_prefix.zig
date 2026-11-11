//! Row-sized copies of a Metal slot's packed pool: a kept state holds the rows its position uses, not the capacity.
const std = @import("std");

pub const Error = error{SnapshotLayout};

/// One block per owner, back to back: an owner of ratio r holds `capacity / r` rows of each sub-array (`strides`).
pub const Layout = struct {
    ratios: []const u32,
    strides: []const usize,

    fn rowBytes(l: Layout) usize {
        var n: usize = 0;
        for (l.strides) |s| n += s;
        return n;
    }

    /// Bytes of a pool laid out for `capacity` rows of the ratio-1 owner.
    pub fn storageBytes(l: Layout, capacity: u32) usize {
        var n: usize = 0;
        for (l.ratios) |r| n += @as(usize, capacity / r) * l.rowBytes();
        return n;
    }

    /// Where owner `id`'s block begins in a pool laid out for `capacity`.
    pub fn blockOffset(l: Layout, id: usize, capacity: u32) usize {
        var n: usize = 0;
        for (l.ratios[0..id]) |r| n += @as(usize, capacity / r) * l.rowBytes();
        return n;
    }

    /// Where sub-array `sub` of owner `id` begins in a pool laid out for `capacity`; its rows run on from there.
    pub fn subOffset(l: Layout, id: usize, sub: usize, capacity: u32) usize {
        var n = l.blockOffset(id, capacity);
        for (l.strides[0..sub]) |s| n += @as(usize, capacity / l.ratios[id]) * s;
        return n;
    }
};

/// The capacity a state kept at `position` is laid out for: the position, even so every owner's rows divide it.
pub fn capacityFor(position: u32, limit: u32) u32 {
    const even = std.math.add(u32, position, position & 1) catch return limit;
    return @min(even, limit);
}

/// The rows of an owner of `ratio` that hold a position's state: its whole groups and the one a carried row opens.
pub fn usedRows(position: u32, ratio: u32, capacity: u32) usize {
    return @min(std.math.divCeil(u32, position, ratio) catch unreachable, capacity / ratio);
}

/// Copy the rows `position` uses from a pool laid out for `src_capacity` into one laid out for `dst_capacity`.
pub fn copy(l: Layout, dst: []u8, dst_capacity: u32, src: []const u8, src_capacity: u32, position: u32) Error!void {
    return copyOwners(l, l.ratios.len, dst, dst_capacity, src, src_capacity, position);
}

/// `copy` for the first `owners` owners only (a layer-split restore: the owners below the cut), the rest left as they are.
pub fn copyOwners(l: Layout, owners: usize, dst: []u8, dst_capacity: u32, src: []const u8, src_capacity: u32, position: u32) Error!void {
    if (position > dst_capacity or position > src_capacity or owners > l.ratios.len) return error.SnapshotLayout;
    if (dst.len != l.storageBytes(dst_capacity) or src.len != l.storageBytes(src_capacity)) return error.SnapshotLayout;
    for (l.ratios[0..owners], 0..) |r, id| {
        const rows = @min(usedRows(position, r, src_capacity), usedRows(position, r, dst_capacity));
        var from = l.blockOffset(id, src_capacity);
        var to = l.blockOffset(id, dst_capacity);
        for (l.strides) |stride| {
            @memcpy(dst[to..][0 .. rows * stride], src[from..][0 .. rows * stride]);
            from += @as(usize, src_capacity / r) * stride;
            to += @as(usize, dst_capacity / r) * stride;
        }
    }
}
