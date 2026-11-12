//! Stream value operations, an optional capability of the HIP runtime: hipStreamWriteValue32 and hipStreamWaitValue32.
const std = @import("std");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");

pub const Error = runtime.Error || error{NoValueOps};

/// hipStreamWaitValue32's greater-or-equal compare.
const wait_gte: c_uint = 0;

const WaitValue32 = *const fn (abi.Stream, ?*anyopaque, u32, c_uint, u32) callconv(.c) abi.Result;
const WriteValue32 = *const fn (abi.Stream, ?*anyopaque, u32, c_uint) callconv(.c) abi.Result;

pub const ValueOps = struct {
    wait32: WaitValue32,
    write32: WriteValue32,

    /// Both entry points from the runtime, or null when it does not export them.
    pub fn from(lib: anytype) ?ValueOps {
        const wait32 = lib.lookup(WaitValue32, "hipStreamWaitValue32") orelse return null;
        const write32 = lib.lookup(WriteValue32, "hipStreamWriteValue32") orelse return null;
        return .{ .wait32 = wait32, .write32 = write32 };
    }

    /// The runtime's value operations; `error.NoValueOps` when it has none, which an offloaded model needs.
    pub fn resolve(r: *const runtime.Runtime) Error!ValueOps {
        const lib: *@TypeOf(r.lib) = @constCast(&r.lib);
        return from(lib) orelse error.NoValueOps;
    }

    /// Queues `*word = value` on `stream`: host memory the device can see, written in stream order.
    pub fn write(v: ValueOps, stream: abi.Stream, word: *u32, value: u32) runtime.Error!void {
        try runtime.check(v.write32(stream, word, value, 0));
    }

    /// Queues a stream wait until `*word >= value`, the host answering it.
    pub fn waitAtLeast(v: ValueOps, stream: abi.Stream, word: *u32, value: u32) runtime.Error!void {
        try runtime.check(v.wait32(stream, word, value, wait_gte, 0xffff_ffff));
    }
};

const Fake = struct {
    missing: ?[]const u8 = null,
    var last: [3]u64 = undefined;
    var calls: usize = 0;

    fn wait(_: abi.Stream, word: ?*anyopaque, value: u32, flags: c_uint, mask: u32) callconv(.c) abi.Result {
        last = .{ @intFromPtr(word), value, @as(u64, flags) << 32 | mask };
        calls += 1;
        return 0;
    }

    fn write(_: abi.Stream, word: ?*anyopaque, value: u32, flags: c_uint) callconv(.c) abi.Result {
        last = .{ @intFromPtr(word), value, flags };
        calls += 1;
        return 0;
    }

    pub fn lookup(f: *const Fake, comptime T: type, name: [:0]const u8) ?T {
        if (f.missing) |m| if (std.mem.eql(u8, m, name)) return null;
        if (T == WaitValue32) return &wait;
        return &write;
    }
};

test "a runtime without either value operation has no capability" {
    try std.testing.expect(ValueOps.from(&Fake{}) != null);
    try std.testing.expect(ValueOps.from(&Fake{ .missing = "hipStreamWaitValue32" }) == null);
    try std.testing.expect(ValueOps.from(&Fake{ .missing = "hipStreamWriteValue32" }) == null);
}

test "a write is queued as written and a wait asks for at least the value under a full mask" {
    const ops = ValueOps.from(&Fake{}).?;
    var word: u32 = 0;
    try ops.write(null, &word, 7);
    try std.testing.expectEqual([3]u64{ @intFromPtr(&word), 7, 0 }, Fake.last);
    try ops.waitAtLeast(null, &word, 9);
    try std.testing.expectEqual([3]u64{ @intFromPtr(&word), 9, 0xffff_ffff }, Fake.last);
    try std.testing.expectEqual(@as(usize, 2), Fake.calls);
}
