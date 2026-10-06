//! Prompt reuse across turns: copies of a stream's caches at prompt cut points, kept by ids under a byte budget, so a
//! later prompt that extends one prefills only its new tokens (Python's PrefixCache, with the same eviction order).

const std = @import("std");
const state = @import("state.zig");
const Allocator = std.mem.Allocator;

/// A snapshot this close to the prompt's end saves too little to keep next to the entry before it.
pub const min_gap = 256;

const Entry = struct {
    ids: []u32,
    caches: state.Caches,
    bytes: usize,
    /// A later prompt resumed from it: it outlives the entries nobody has asked for.
    hit: bool = false,
};

pub const Cache = struct {
    gpa: Allocator,
    /// Oldest first.
    entries: std.ArrayList(Entry) = .empty,
    /// Most entries kept (0: none).
    keep: usize,
    /// Most bytes the entries hold together.
    budget: usize,

    pub fn init(gpa: Allocator, keep: usize, budget: usize) Cache {
        return .{ .gpa = gpa, .keep = keep, .budget = budget };
    }

    pub fn deinit(c: *Cache) void {
        for (c.entries.items) |*e| c.release(e);
        c.entries.deinit(c.gpa);
    }

    fn release(c: *Cache, e: *Entry) void {
        c.gpa.free(e.ids);
        e.caches.deinit(c.gpa);
    }

    /// Bytes the entries hold.
    pub fn held(c: *const Cache) usize {
        var n: usize = 0;
        for (c.entries.items) |e| n += e.bytes;
        return n;
    }

    /// The longest entry the prompt strictly extends (one prompt token is always left to prefill), now the newest.
    pub fn longest(c: *Cache, prompt: []const u32) ?*const Entry {
        var best: ?usize = null;
        for (c.entries.items, 0..) |e, i| {
            if (e.ids.len >= prompt.len or !std.mem.eql(u32, e.ids, prompt[0..e.ids.len])) continue;
            if (best == null or e.ids.len > c.entries.items[best.?].ids.len) best = i;
        }
        const i = best orelse return null;
        var e = c.entries.orderedRemove(i);
        e.hit = true;
        c.entries.appendAssumeCapacity(e);
        return &c.entries.items[c.entries.items.len - 1];
    }

    /// Whether an entry holds exactly `ids`.
    pub fn has(c: *const Cache, ids: []const u32) bool {
        for (c.entries.items) |e| if (std.mem.eql(u32, e.ids, ids)) return true;
        return false;
    }

    /// Keep `caches` (taken: freed here when not kept) as the newest entry for `ids`, then evict to the limits.
    pub fn add(c: *Cache, ids: []const u32, caches: state.Caches) !void {
        var own = caches;
        errdefer own.deinit(c.gpa);
        if (c.keep == 0 or own.held() > c.budget) return own.deinit(c.gpa);
        const copy = try c.gpa.dupe(u32, ids);
        errdefer c.gpa.free(copy);
        try c.entries.ensureUnusedCapacity(c.gpa, 1);
        var i: usize = 0;
        while (i < c.entries.items.len) {
            if (std.mem.eql(u32, c.entries.items[i].ids, ids)) {
                var old = c.entries.orderedRemove(i);
                c.release(&old);
            } else i += 1;
        }
        c.entries.appendAssumeCapacity(.{ .ids = copy, .caches = own, .bytes = own.held() });
        while (c.entries.items.len > c.keep or (c.entries.items.len > 1 and c.held() > c.budget)) c.drop();
    }

    /// Drop the oldest entry never resumed from (not the newest), else the oldest.
    fn drop(c: *Cache) void {
        var at: usize = 0;
        for (c.entries.items[0 .. c.entries.items.len - 1], 0..) |e, i| if (!e.hit) {
            at = i;
            break;
        };
        var gone = c.entries.orderedRemove(at);
        c.release(&gone);
    }
};

/// Cuts sit on multiples of the chunked recurrence's chunk: a resumed span then chunks a prompt as a fresh one does.
pub const step = 64;

/// Where a prefill past `cached` keeps a state: the request's shared system blocks and rendered history, then one
/// token before the prompt's end unless a cut already sits within `min_gap` of it, each down to a multiple of `step`.
/// Ascending; written into `out`.
pub fn cuts(out: []u32, prompt_len: usize, cached: usize, history: u32, shared: []const u32) []const u32 {
    var n: usize = 0;
    const found = [_]u32{history};
    for ([_][]const u32{ shared, &found }) |list| for (list) |raw| {
        const p = raw - raw % step;
        if (p <= cached or p >= prompt_len or n == out.len) continue;
        if (std.mem.indexOfScalar(u32, out[0..n], p) != null) continue;
        out[n] = p;
        n += 1;
    };
    std.mem.sort(u32, out[0..n], {}, std.sort.asc(u32));
    const end = (@max(1, prompt_len - 1) / step) * step;
    const near = n > 0 and prompt_len - out[n - 1] < min_gap;
    if (cached < end and end < prompt_len and !near and n < out.len and std.mem.indexOfScalar(u32, out[0..n], @intCast(end)) == null) {
        out[n] = @intCast(end);
        n += 1;
    }
    return out[0..n];
}

test "cuts keep shared blocks, the history and the entry end" {
    var buf: [8]u32 = undefined;
    try std.testing.expectEqualSlices(u32, &.{ 576, 960, 1984 }, cuts(&buf, 2000, 0, 1000, &.{ 600, 600 }));
    try std.testing.expectEqualSlices(u32, &.{960}, cuts(&buf, 1100, 600, 1000, &.{600}));
    try std.testing.expectEqualSlices(u32, &.{64}, cuts(&buf, 100, 0, 0, &.{}));
    try std.testing.expectEqualSlices(u32, &.{}, cuts(&buf, 10, 0, 0, &.{}));
    try std.testing.expectEqualSlices(u32, &.{}, cuts(&buf, 1, 0, 0, &.{}));
}
