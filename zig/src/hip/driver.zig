//! The HIP runtime opened at run time: libamdhip64's entry points in one table, and checked calls.

const std = @import("std");
const abi = @import("abi.zig");

pub const Error = error{ DriverUnavailable, MissingSymbol, HipFailed, OutOfDeviceMemory, NotReady, NotFound, Invalid };

/// The loader's search path first, then ROCm's own directory.
const paths = [_][]const u8{ "libamdhip64.so.7", "libamdhip64.so", "/opt/rocm/lib/libamdhip64.so" };

pub const Driver = struct {
    lib: std.DynLib,
    api: abi.Api,

    pub fn open() Error!Driver {
        for (paths) |p| return openPath(p) catch |e| switch (e) {
            error.DriverUnavailable => continue,
            else => return e,
        };
        return error.DriverUnavailable;
    }

    /// Resolves every field of `abi.Api` by its exact name; a missing symbol refuses the whole library.
    pub fn openPath(path: []const u8) Error!Driver {
        var lib = std.DynLib.open(path) catch return error.DriverUnavailable;
        errdefer lib.close();
        var api: abi.Api = undefined;
        const info = @typeInfo(abi.Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return error.MissingSymbol;
            };
        }
        const d: Driver = .{ .lib = lib, .api = api };
        try d.check(api.hipInit(0), "hipInit");
        return d;
    }

    pub fn close(self: *Driver) void {
        self.lib.close();
    }

    /// Logs a failed call with HIP's own name and text for the code; NOT_READY is a state, not a failure.
    pub fn check(self: *const Driver, res: abi.Result, what: []const u8) Error!void {
        if (res == abi.success) return;
        if (res == abi.error_not_ready) return error.NotReady;
        std.log.err("{s}: {s} ({d}) {s}", .{ what, self.errorName(res), res, self.errorText(res) });
        return switch (res) {
            abi.error_out_of_memory => error.OutOfDeviceMemory,
            abi.error_not_found => error.NotFound,
            else => error.HipFailed,
        };
    }

    pub fn errorName(self: *const Driver, res: abi.Result) []const u8 {
        return if (self.api.hipGetErrorName(res)) |p| std.mem.span(p) else "hipErrorUnknown";
    }

    pub fn errorText(self: *const Driver, res: abi.Result) []const u8 {
        return if (self.api.hipGetErrorString(res)) |p| std.mem.span(p) else "";
    }

    /// HIP's version as 10,000,000 * major + 100,000 * minor + patch (7.14: 71460850).
    pub fn version(self: *const Driver) Error!c_int {
        var v: c_int = 0;
        try self.check(self.api.hipRuntimeGetVersion(&v), "hipRuntimeGetVersion");
        return v;
    }

    pub fn deviceCount(self: *const Driver) Error!c_int {
        var n: c_int = 0;
        try self.check(self.api.hipGetDeviceCount(&n), "hipGetDeviceCount");
        return n;
    }
};
