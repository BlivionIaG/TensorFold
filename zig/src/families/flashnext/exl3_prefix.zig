//! The prefix store for the EXL3 forward: whole-sequence snapshots keyed by their token prefix, with a byte budget.
//!
//! A snapshot of a hybrid model (GDN recurrent state, attention KV, PLE history) can only be taken at the exact length it
//! describes: the recurrent state cannot be rewound. So an entry is stored at the length where the caller took it and is
//! used only by a prompt that has those tokens as a prefix AND is long enough to still compute its last token, which is
//! `exl3_cache.resumeAt`: the kept prefix stops one token before the prompt end, so a prompt always computes at least one
//! token and has logits to sample from. The snapshot type is the caller's; it needs `deinit(*Snap, gpa)`.

const std = @import("std");
const cache = @import("exl3_cache.zig");

pub fn PrefixStore(comptime Snap: type) type {
    return struct {
        const Self = @This();

        pub const Entry = struct {
            tokens: []u32,
            snap: Snap,
            bytes: u64,
            used: u64,
        };

        pub const Hit = struct {
            /// Tokens already covered by the snapshot; the caller computes `query[len..]`.
            len: usize,
            snap: *const Snap,
        };

        gpa: std.mem.Allocator,
        budget: u64,
        total: u64 = 0,
        tick: u64 = 0,
        items: std.ArrayList(Entry) = .empty,

        pub fn init(gpa: std.mem.Allocator, budget: u64) Self {
            return .{ .gpa = gpa, .budget = budget };
        }

        pub fn deinit(self: *Self) void {
            for (self.items.items) |*e| {
                self.gpa.free(e.tokens);
                e.snap.deinit(self.gpa);
            }
            self.items.deinit(self.gpa);
        }

        /// Takes ownership of `snap` on success. A snapshot larger than the whole budget is refused and stays the caller's.
        /// The same token list replaces its older entry. Least recently used entries go first to make room.
        pub fn put(self: *Self, tokens: []const u32, snap: Snap, bytes: u64) error{ Empty, OverBudget, OutOfMemory }!void {
            if (tokens.len == 0) return error.Empty;
            if (bytes > self.budget) return error.OverBudget;
            for (self.items.items, 0..) |*e, i| {
                if (std.mem.eql(u32, e.tokens, tokens)) {
                    self.drop(i);
                    break;
                }
            }
            while (self.total + bytes > self.budget) self.drop(self.oldest());
            const copy = try self.gpa.dupe(u32, tokens);
            errdefer self.gpa.free(copy);
            self.tick += 1;
            try self.items.append(self.gpa, .{ .tokens = copy, .snap = snap, .bytes = bytes, .used = self.tick });
            self.total += bytes;
        }

        fn oldest(self: *const Self) usize {
            var best: usize = 0;
            for (self.items.items, 0..) |e, i| {
                if (e.used < self.items.items[best].used) best = i;
            }
            return best;
        }

        fn drop(self: *Self, i: usize) void {
            var e = self.items.orderedRemove(i);
            self.total -= e.bytes;
            self.gpa.free(e.tokens);
            e.snap.deinit(self.gpa);
        }

        /// The longest stored entry that is a prefix of `query` and stops at or before `resumeAt(query.len)`.
        pub fn lookup(self: *Self, query: []const u32) ?Hit {
            const limit: usize = cache.resumeAt(@intCast(query.len));
            var best: ?usize = null;
            for (self.items.items, 0..) |e, i| {
                if (e.tokens.len > limit) continue;
                if (!std.mem.eql(u32, e.tokens, query[0..e.tokens.len])) continue;
                if (best == null or e.tokens.len > self.items.items[best.?].tokens.len) best = i;
            }
            const i = best orelse return null;
            self.tick += 1;
            self.items.items[i].used = self.tick;
            return .{ .len = self.items.items[i].tokens.len, .snap = &self.items.items[i].snap };
        }
    };
}

const TestSnap = struct {
    freed: *usize,
    fn deinit(s: *TestSnap, gpa: std.mem.Allocator) void {
        _ = gpa;
        s.freed.* += 1;
    }
};

test "a prompt hits the longest stored prefix that leaves its last token to compute" {
    var freed: usize = 0;
    var store = PrefixStore(TestSnap).init(std.testing.allocator, 100);
    defer store.deinit();
    try store.put(&.{ 1, 2, 3 }, .{ .freed = &freed }, 10);
    try store.put(&.{ 1, 2, 3, 4, 5 }, .{ .freed = &freed }, 10);
    try std.testing.expectEqual(@as(usize, 5), store.lookup(&.{ 1, 2, 3, 4, 5, 6 }).?.len);
    // the 5-token entry would cover the whole 5-token prompt: resumeAt(5) = 4 refuses it, the 3-token entry serves
    try std.testing.expectEqual(@as(usize, 3), store.lookup(&.{ 1, 2, 3, 4, 5 }).?.len);
    // a prompt equal to the 3-token entry cannot use it either
    try std.testing.expect(store.lookup(&.{ 1, 2, 3 }) == null);
    try std.testing.expect(store.lookup(&.{ 1, 2, 9, 4 }) == null);
    try std.testing.expect(store.lookup(&.{}) == null);
}

test "the budget evicts the least recently used entry and refuses one larger than the budget" {
    var freed: usize = 0;
    var store = PrefixStore(TestSnap).init(std.testing.allocator, 25);
    defer store.deinit();
    try store.put(&.{1}, .{ .freed = &freed }, 10);
    try store.put(&.{2}, .{ .freed = &freed }, 10);
    _ = store.lookup(&.{ 1, 7 }); // touches {1}
    try store.put(&.{3}, .{ .freed = &freed }, 10); // evicts {2}
    try std.testing.expectEqual(@as(usize, 1), freed);
    try std.testing.expect(store.lookup(&.{ 2, 7 }) == null);
    try std.testing.expect(store.lookup(&.{ 1, 7 }) != null);
    try std.testing.expectEqual(@as(u64, 20), store.total);
    try std.testing.expectError(error.OverBudget, store.put(&.{4}, .{ .freed = &freed }, 26));
    try std.testing.expectError(error.Empty, store.put(&.{}, .{ .freed = &freed }, 1));
    // same tokens replace the older entry
    try store.put(&.{1}, .{ .freed = &freed }, 5);
    try std.testing.expectEqual(@as(usize, 2), freed);
    try std.testing.expectEqual(@as(u64, 15), store.total);
}
