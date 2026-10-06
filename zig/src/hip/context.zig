//! One device, made current on the calling thread (HIP keeps a current device per thread, no context handle).

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const Context = struct {
    d: *const Driver,
    ordinal: c_int,
    device: abi.Device,

    pub fn init(d: *const Driver, ordinal: c_int) Error!Context {
        var dev: abi.Device = 0;
        try d.check(d.api.hipDeviceGet(&dev, ordinal), "hipDeviceGet");
        try d.check(d.api.hipSetDevice(ordinal), "hipSetDevice");
        return .{ .d = d, .ordinal = ordinal, .device = dev };
    }

    /// Waits for all work; resources made on the device must be released first.
    pub fn deinit(self: *Context) void {
        _ = self.d.api.hipDeviceSynchronize();
        self.* = undefined;
    }

    /// Makes the device current on another thread before it calls HIP.
    pub fn makeCurrent(self: *const Context) Error!void {
        try self.d.check(self.d.api.hipSetDevice(self.ordinal), "hipSetDevice");
    }

    pub fn synchronize(self: *const Context) Error!void {
        try self.d.check(self.d.api.hipDeviceSynchronize(), "hipDeviceSynchronize");
    }

    pub fn attribute(self: *const Context, a: abi.DeviceAttribute) Error!c_int {
        var v: c_int = 0;
        try self.d.check(self.d.api.hipDeviceGetAttribute(&v, a, self.ordinal), "hipDeviceGetAttribute");
        return v;
    }

    /// The gfx major and minor as 10 * major + minor (gfx1030: 103, gfx1100: 110, gfx1151: 115); the stepping is not in it.
    pub fn capability(self: *const Context) Error!u32 {
        const major = try self.attribute(.compute_capability_major);
        const minor = try self.attribute(.compute_capability_minor);
        return @intCast(10 * major + minor);
    }

    pub fn name(self: *const Context, buf: []u8) Error![]const u8 {
        if (buf.len < 2) return error.Invalid;
        try self.d.check(self.d.api.hipDeviceGetName(buf.ptr, @intCast(buf.len), self.device), "hipDeviceGetName");
        return std.mem.sliceTo(buf, 0);
    }

    pub const MemInfo = struct { free: usize, total: usize };

    /// The current device's free and total bytes.
    pub fn memInfo(self: *const Context) Error!MemInfo {
        var m: MemInfo = .{ .free = 0, .total = 0 };
        try self.d.check(self.d.api.hipMemGetInfo(&m.free, &m.total), "hipMemGetInfo");
        return m;
    }
};
