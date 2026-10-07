//! A forward's scratch handed out front to back and reset per forward, so a captured graph replays the same addresses.

const std = @import("std");
const Driver = @import("driver.zig").Driver;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const Error = @import("driver.zig").Error;

pub const Arena = struct {
    buf: DeviceBuffer,
    used: usize = 0,
    peak: usize = 0,

    pub fn init(d: *const Driver, bytes: usize) Error!Arena {
        return .{ .buf = try DeviceBuffer.alloc(d, bytes) };
    }

    pub fn deinit(self: *Arena) void {
        self.buf.free();
        self.* = undefined;
    }

    /// `bytes` 256-byte aligned; refused past the end (the caller sized the arena for its largest forward).
    pub fn take(self: *Arena, bytes: usize) error{OutOfDeviceMemory}!u64 {
        const at = std.mem.alignForward(usize, self.used, 256);
        if (at + bytes > self.buf.len) {
            std.log.err("scratch arena of {d} bytes cannot take {d} more at {d}", .{ self.buf.len, bytes, at });
            return error.OutOfDeviceMemory;
        }
        self.used = at + bytes;
        self.peak = @max(self.peak, self.used);
        return self.buf.ptr + at;
    }

    /// `n` values of `T`.
    pub fn of(self: *Arena, comptime T: type, n: usize) error{OutOfDeviceMemory}!u64 {
        return self.take(n * @sizeOf(T));
    }

    pub fn reset(self: *Arena) void {
        self.used = 0;
    }

    /// The current mark, to hand scratch back after a step that needs it only briefly.
    pub fn mark(self: *const Arena) usize {
        return self.used;
    }

    pub fn release(self: *Arena, at: usize) void {
        std.debug.assert(at <= self.used);
        self.used = at;
    }
};
