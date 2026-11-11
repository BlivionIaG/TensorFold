//! The building blocks of the cache policies: an insertion-ordered key set over the 10,240 expert keys (what the Python original keeps in
//! `collections.OrderedDict`) as an intrusive doubly linked list, and the per-call "requested in this call" mark (the original's `kset`).
const std = @import("std");

pub const NL = 40;
pub const NE = 256;
pub const TOPK = 8;
pub const NK = NL * NE;
/// The longest key list of one call: a lockstep group's union of its sequences' picks.
pub const group_size = 4;
pub const max_keys = group_size * TOPK;

const none: u16 = 0xffff;
const head: u16 = NK; // the sentinel node

const Link = struct { prev: u16, next: u16 };

/// Keys in insertion order, oldest first; `append` / `toEnd` make a key the newest. A key is in at most one position.
pub const OList = struct {
    link: []Link,
    count: usize = 0,

    pub fn init(a: std.mem.Allocator) !OList {
        const link = try a.alloc(Link, NK + 1);
        @memset(link, .{ .prev = none, .next = none });
        link[head] = .{ .prev = head, .next = head };
        return .{ .link = link };
    }

    pub fn contains(self: *const OList, k: u16) bool {
        return self.link[k].next != none;
    }

    /// `od[k] = None` for a key that is not in: the newest.
    pub fn append(self: *OList, k: u16) void {
        const last = self.link[head].prev;
        self.link[k] = .{ .prev = last, .next = head };
        self.link[last].next = k;
        self.link[head].prev = k;
        self.count += 1;
    }

    /// `od[k] = None`: appended if absent, left where it is if present.
    pub fn put(self: *OList, k: u16) void {
        if (!self.contains(k)) self.append(k);
    }

    pub fn remove(self: *OList, k: u16) void {
        const l = self.link[k];
        self.link[l.prev].next = l.next;
        self.link[l.next].prev = l.prev;
        self.link[k].next = none;
        self.count -= 1;
    }

    /// `od.move_to_end(k)`.
    pub fn toEnd(self: *OList, k: u16) void {
        if (self.link[k].next == head) return;
        self.remove(k);
        self.append(k);
    }

    /// The oldest key.
    pub fn first(self: *const OList) ?u16 {
        const f = self.link[head].next;
        return if (f == head) null else f;
    }

    /// The key after `k` in age order (null after the newest).
    pub fn after(self: *const OList, k: u16) ?u16 {
        const n = self.link[k].next;
        return if (n == head) null else n;
    }

    /// `od.popitem(last=False)`.
    pub fn popFirst(self: *OList) u16 {
        const f = self.first().?;
        self.remove(f);
        return f;
    }
};

/// The keys requested by the call in progress (the original's `kset = set(keys)`), cleared in O(1) by a new epoch.
pub const Mark = struct {
    stamp: []u32,
    epoch: u32 = 0,

    pub fn init(a: std.mem.Allocator) !Mark {
        const stamp = try a.alloc(u32, NK);
        @memset(stamp, 0);
        return .{ .stamp = stamp };
    }

    pub fn begin(self: *Mark, keys: []const u16) void {
        self.epoch +%= 1;
        if (self.epoch == 0) {
            @memset(self.stamp, 0);
            self.epoch = 1;
        }
        for (keys) |k| self.stamp[k] = self.epoch;
    }

    pub fn has(self: *const Mark, k: u16) bool {
        return self.stamp[k] == self.epoch;
    }
};

/// `first_free(od, kset)`: the oldest entry of `od` that this call did not request, or null.
pub fn firstFree(od: *const OList, mark: *const Mark) ?u16 {
    var v = od.first();
    while (v) |k| : (v = od.after(k)) if (!mark.has(k)) return k;
    return null;
}

/// The original would die on `del od[None]`: a cache smaller than one call's keys has no victim.
pub fn noVictim() noreturn {
    @panic("expert_cache_sim: no evictable line (the cache is smaller than the keys of one call)");
}
