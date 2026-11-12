//! A forward's scratch handed out front to back and reset per forward, so a captured graph replays the same addresses.

const std = @import("std");
const runtime = @import("runtime.zig");
const DeviceBuffer = @import("memory.zig").DeviceBuffer;

/// What an arena handed out and took back, in order, for the hazard check of a recorded round (`hazard.zig`); an arena with no log keeps none.
pub const Log = struct {
    pub const Item = union(enum) {
        /// `bytes` bytes at `at` (the 256-aligned address `take` returned).
        take: struct { at: u64, bytes: usize },
        /// the arena's `used` after a `release` or `reset`.
        release: usize,
    };

    gpa: std.mem.Allocator,
    items: std.ArrayList(Item) = .empty,

    pub fn deinit(l: *Log) void {
        l.items.deinit(l.gpa);
    }
};

pub const Arena = struct {
    /// Null for a counting arena.
    buf: ?DeviceBuffer,
    base: u64,
    len: usize,
    used: usize = 0,
    peak: usize = 0,
    log: ?*Log = null,

    pub fn init(r: *const runtime.Runtime, bytes: usize) runtime.Error!Arena {
        const buf = try DeviceBuffer.alloc(r, bytes);
        return .{ .buf = buf, .base = @intFromPtr(buf.ptr), .len = bytes };
    }

    /// An arena with no memory behind it, for a forward on the counting stream: its peak is what the forward takes.
    pub fn counting() Arena {
        return .{ .buf = null, .base = 1 << 40, .len = 1 << 60 };
    }

    pub fn deinit(self: *Arena) void {
        if (self.buf) |*b| b.free();
        self.* = undefined;
    }

    /// `bytes` 256-byte aligned; refused past the end (the caller sized the arena for its largest forward).
    pub fn take(self: *Arena, bytes: usize) error{OutOfDeviceMemory}!u64 {
        const at = std.mem.alignForward(usize, self.used, 256);
        if (at + bytes > self.len) {
            std.log.err("scratch arena of {d} bytes cannot take {d} more at {d}", .{ self.len, bytes, at });
            return error.OutOfDeviceMemory;
        }
        if (self.log) |l| l.items.append(l.gpa, .{ .take = .{ .at = self.base + at, .bytes = bytes } }) catch return error.OutOfDeviceMemory;
        self.used = at + bytes;
        self.peak = @max(self.peak, self.used);
        return self.base + at;
    }

    /// `n` values of `T`.
    pub fn of(self: *Arena, comptime T: type, n: usize) error{OutOfDeviceMemory}!u64 {
        return self.take(n * @sizeOf(T));
    }

    pub fn reset(self: *Arena) void {
        self.used = 0;
        if (self.log) |l| l.items.append(l.gpa, .{ .release = 0 }) catch {};
    }

    /// The current mark, to hand scratch back after a step that needs it only briefly.
    pub fn mark(self: *const Arena) usize {
        return self.used;
    }

    pub fn release(self: *Arena, at: usize) void {
        std.debug.assert(at <= self.used);
        self.used = at;
        if (self.log) |l| l.items.append(l.gpa, .{ .release = at }) catch {};
    }
};

test "scratch comes front to back, 256-byte aligned, and is refused past the end" {
    var a = Arena.counting();
    a.len = 1000;
    try std.testing.expectEqual(@as(u64, 1 << 40), try a.take(10));
    try std.testing.expectEqual(@as(u64, (1 << 40) + 256), try a.of(f32, 4));
    const m = a.mark();
    _ = try a.take(300);
    a.release(m);
    try std.testing.expectEqual(@as(usize, 272), a.used);
    try std.testing.expectEqual(@as(usize, 512 + 300), a.peak);
    a.reset();
    try std.testing.expectEqual(@as(usize, 0), a.used);
}
