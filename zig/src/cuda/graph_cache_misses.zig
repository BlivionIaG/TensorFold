//! Which request-path captures a graph cache logs: each key's first, then every `every`th past `keys_max` keys.

const std = @import("std");

/// Distinct keys a cache remembers having logged; past them the log is thinned to every `every`th capture.
pub const keys_max = 512;
pub const every = 64;

pub fn Log(comptime Key: type) type {
    return struct {
        const Self = @This();
        seen: std.AutoArrayHashMapUnmanaged(Key, void) = .empty,
        logged: u64 = 0,

        pub fn deinit(l: *Self, gpa: std.mem.Allocator) void {
            l.seen.deinit(gpa);
        }

        /// Whether the `nth` request-path capture (counting from 1), of `key`, is logged: a key's first only.
        pub fn first(l: *Self, gpa: std.mem.Allocator, key: Key, nth: u64) bool {
            if (l.seen.contains(key)) return false;
            const room = l.seen.count() < keys_max;
            if (room) {
                l.seen.put(gpa, key, {}) catch {}; // a map that cannot grow only repeats a line
            } else if (nth % every != 0) return false;
            l.logged += 1;
            return true;
        }
    };
}

test "misses: a key is logged once however often it is captured again" {
    const t = std.testing;
    var l: Log(u32) = .{};
    defer l.deinit(t.allocator);
    try t.expect(l.first(t.allocator, 7, 1));
    try t.expect(!l.first(t.allocator, 7, 2));
    try t.expect(l.first(t.allocator, 8, 3));
    try t.expectEqual(@as(u64, 2), l.logged);
}

test "misses: past the key table only every 64th capture is logged" {
    const t = std.testing;
    var l: Log(u32) = .{};
    defer l.deinit(t.allocator);
    for (0..keys_max) |k| try t.expect(l.first(t.allocator, @intCast(k), k + 1));
    try t.expectEqual(@as(u64, keys_max), l.logged);
    var shown: u64 = 0;
    for (0..4 * every) |i| {
        if (l.first(t.allocator, @intCast(10_000 + i), keys_max + i + 1)) shown += 1;
    }
    try t.expect(shown >= 3 and shown <= 5);
}
