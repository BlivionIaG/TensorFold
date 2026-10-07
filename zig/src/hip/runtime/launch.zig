//! Kernel launches: argument values packed where hipModuleLaunchKernel reads them, plus cooperative grids.

const std = @import("std");
const abi = @import("abi.zig");
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const Error = @import("driver.zig").Error;

pub const Dim3 = abi.Dim3;

/// Argument values in kernel order; each must have exactly the C type of its kernel parameter.
pub const Args = struct {
    pub const max_args = 48;
    pub const max_bytes = 2048;

    storage: [max_bytes]u8 align(16) = undefined,
    offsets: [max_args]u16 = undefined,
    sizes: [max_args]u16 = undefined,
    ptrs: [max_args]?*anyopaque = undefined,
    used: usize = 0,
    count: usize = 0,
    /// An argument did not fit: the launch is refused, in every build mode.
    overflow: bool = false,

    /// Copies `value` (a pointer-sized address, scalar, or extern struct passed by value) at its C alignment.
    pub fn add(self: *Args, value: anytype) void {
        const T = @TypeOf(value);
        comptime std.debug.assert(@sizeOf(T) > 0);
        const at = std.mem.alignForward(usize, self.used, @alignOf(T));
        if (self.count >= max_args or at + @sizeOf(T) > max_bytes) {
            self.overflow = true;
            return;
        }
        @memcpy(self.storage[at..][0..@sizeOf(T)], std.mem.asBytes(&value));
        self.offsets[self.count] = @intCast(at);
        self.sizes[self.count] = @sizeOf(T);
        self.count += 1;
        self.used = at + @sizeOf(T);
    }

    /// Argument `i` read back as an unsigned integer (4- and 8-byte values only).
    pub fn integer(self: *const Args, i: usize) ?u64 {
        if (i >= self.count) return null;
        const at = self.storage[self.offsets[i]..];
        return switch (self.sizes[i]) {
            4 => std.mem.readInt(u32, at[0..4], .little),
            8 => std.mem.readInt(u64, at[0..8], .little),
            else => null,
        };
    }

    /// The address array, rebuilt here so a moved Args never hands out stale addresses.
    pub fn pointers(self: *Args) ?[*]?*anyopaque {
        if (self.count == 0) return null;
        for (self.offsets[0..self.count], 0..) |off, i| self.ptrs[i] = &self.storage[off];
        return &self.ptrs;
    }
};

pub const Config = struct {
    grid: Dim3,
    block: Dim3,
    shared: u32 = 0,
    cooperative: bool = false,

    pub fn validate(self: Config) Error!void {
        const g = self.grid;
        const b = self.block;
        if (g.x == 0 or g.y == 0 or g.z == 0 or b.x == 0 or b.y == 0 or b.z == 0) return error.Invalid;
        if (@as(u64, b.x) * b.y * b.z > 1024) return error.Invalid;
        if (@as(u64, g.x) * b.x > std.math.maxInt(u32) or @as(u64, g.y) * b.y > std.math.maxInt(u32) or @as(u64, g.z) * b.z > std.math.maxInt(u32)) return error.Invalid;
    }
};

/// hipModuleLaunchKernel, or hipModuleLaunchCooperativeKernel for a grid whose blocks must all be resident.
pub fn launch(f: Function, cfg: Config, stream: Stream, args: *Args) Error!void {
    try cfg.validate();
    if (args.overflow) return error.Invalid;
    const d = f.d;
    const g = cfg.grid;
    const b = cfg.block;
    if (cfg.cooperative) {
        return d.check(d.api.hipModuleLaunchCooperativeKernel(f.handle, g.x, g.y, g.z, b.x, b.y, b.z, cfg.shared, stream.handle, args.pointers()), "hipModuleLaunchCooperativeKernel");
    }
    try d.check(d.api.hipModuleLaunchKernel(f.handle, g.x, g.y, g.z, b.x, b.y, b.z, cfg.shared, stream.handle, args.pointers(), null), "hipModuleLaunchKernel");
}

test "args keep C alignment and sizes" {
    var a: Args = .{};
    a.add(@as(u64, 0x1122334455667788));
    a.add(@as(i32, -3));
    a.add(@as(bool, true));
    a.add(@as(f64, 2.5));
    const S = extern struct { p: u64, n: c_int, s: f32 };
    a.add(S{ .p = 7, .n = 8, .s = 9 });
    try std.testing.expectEqualSlices(u16, &.{ 0, 8, 12, 16, 24 }, a.offsets[0..a.count]);
    const ptrs = a.pointers().?;
    try std.testing.expectEqual(@as(i32, -3), @as(*align(1) const i32, @ptrCast(ptrs[1].?)).*);
    try std.testing.expectEqual(@as(f64, 2.5), @as(*align(1) const f64, @ptrCast(ptrs[3].?)).*);
}

test "a block over 1,024 threads, an empty grid or a grid past 2^32 threads is refused" {
    try std.testing.expectError(error.Invalid, (Config{ .grid = .{}, .block = .{ .x = 2048 } }).validate());
    try std.testing.expectError(error.Invalid, (Config{ .grid = .{ .x = 0 }, .block = .{} }).validate());
    try std.testing.expectError(error.Invalid, (Config{ .grid = .{ .x = 1 << 31 }, .block = .{ .x = 4 } }).validate());
}

test "an argument past the pack's room marks it, whatever the build mode" {
    var a: Args = .{};
    for (0..Args.max_args) |_| a.add(@as(u32, 1));
    try std.testing.expect(!a.overflow);
    a.add(@as(u32, 1));
    try std.testing.expect(a.overflow and a.count == Args.max_args);
}
