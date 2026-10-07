//! One device, made current on the calling thread (HIP keeps a current device per thread, no context handle).

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const Caps = @import("../caps.zig").Caps;

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

    /// The device's gfx name ("gfx1100", "gfx906:sramecc+:xnack-"), found in its properties record by its prefix.
    pub fn archName(self: *const Context, out: []u8) Error![]const u8 {
        var props: [16384]u8 align(8) = @splat(0);
        try self.d.check(self.d.api.hipGetDevicePropertiesR0600(&props, self.ordinal), "hipGetDeviceProperties");
        var at: usize = 256;
        while (at + 4 < props.len) : (at += 1) {
            if (!std.mem.startsWith(u8, props[at..], "gfx") or !std.ascii.isHex(props[at + 3])) continue;
            const text = std.mem.sliceTo(props[at..], 0);
            if (text.len > out.len) return error.Invalid;
            @memcpy(out[0..text.len], text);
            return out[0..text.len];
        }
        return error.NotFound;
    }

    /// What this GPU can do, from its gfx name; a GPU outside the caps table is refused with its name.
    pub fn caps(self: *const Context) Error!Caps {
        var buf: [64]u8 = undefined;
        const arch = try self.archName(&buf);
        return Caps.of(arch) orelse {
            std.log.err("{s} is not in the table of GPUs this build supports", .{arch});
            return error.Invalid;
        };
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
