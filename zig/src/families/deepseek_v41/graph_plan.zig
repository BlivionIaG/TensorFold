//! The (rows, context bucket) shapes a serving engine captures at boot, from the request path's own bucket functions.

const std = @import("std");
const graphs = @import("graphs.zig");
const rowtab = @import("rowtab.zig");

/// Distinct row counts a plan lists at most (`graphs.Settings.rows_max` is at most 64).
pub const rows_cap = 64;

pub const Shape = struct { rows: u32, ctx: u32 };

/// Shapes in capture order: context bucket by bucket (every stream grows through the low ones), rows ascending within.
pub const Plan = struct {
    rows: []const u32,
    ctx: u32, // context buckets 0 .. ctx - 1

    pub fn count(p: Plan) usize {
        return p.rows.len * p.ctx;
    }

    pub fn at(p: Plan, i: usize) Shape {
        return .{ .rows = p.rows[i % p.rows.len], .ctx = @intCast(i / p.rows.len) };
    }
};

/// The most rows one slot's window holds: a draft block and its verify row, at least the 16 `aot-needs` lists.
pub fn slotRows(block: u32, rows_max: u32) u32 {
    return @min(rows_max, @max(16, block + 1));
}

/// The most rows a row-mode window holds: every slot's full window, within the rows cap.
pub fn roundRows(cap: u32, slots: u32) u32 {
    return @min(cap, slots * rowtab.seg_max);
}

/// One slot's windows (`Forward.window`) are keyed by their own row count: 1 .. `most`.
pub fn windowRows(out: *[rows_cap]u32, most: u32) []const u32 {
    const n = @min(most, rows_cap);
    for (out[0..n], 0..) |*r, i| r.* = @intCast(i + 1);
    return out[0..n];
}

/// Row mode's windows pad to a bucket (`rowtab.bucketRows`): the distinct buckets of windows of 1 .. `most` rows.
pub fn rowBuckets(out: *[rows_cap]u32, most: u32, cap: u32) []const u32 {
    var n: usize = 0;
    var want: u32 = 1;
    while (want <= most) {
        const r = rowtab.bucketRows(want, cap) orelse break;
        out[n] = r;
        n += 1;
        want = r + 1;
    }
    return out[0..n];
}

/// The context buckets 0 .. n - 1 a window can land in: up to position `last`, a window ending by `limit` - 1.
pub fn ctxBuckets(last: u64, limit: u64, s: graphs.Settings) u32 {
    return graphs.keyFor(1, @min(last, limit -| 1), s).ctx + 1;
}
