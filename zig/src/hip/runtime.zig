//! Complete model-free HIP runtime admission; partial symbol tables never escape.
const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi.zig");
pub const Error = error{ DriverUnavailable, MissingSymbol, HipFailed, Invalid };

pub const Runtime = struct {
    lib: std.DynLib,
    api: abi.Api,

    /// The runtime library by its platform name (ROCm's libamdhip64.so, the macOS port's libamdhip64.dylib), found by the loader's
    /// search path; TF_HIP_LIB names the file instead (an install outside the search path, or another runtime).
    pub fn open() Error!Runtime {
        if (std.c.getenv("TF_HIP_LIB")) |path| return openPath(std.mem.span(path));
        return openPath(library_name);
    }

    pub const library_name = if (builtin.os.tag == .macos) "libamdhip64.dylib" else "libamdhip64.so";

    pub fn openPath(path: []const u8) Error!Runtime {
        var lib = std.DynLib.open(path) catch return error.DriverUnavailable;
        errdefer lib.close();
        const api = try @import("symbols.zig").resolve(abi.Api, &lib);
        try check(api.hipInit(0));
        return .{ .lib = lib, .api = api };
    }

    /// HIP's name for a failed call's result, for logs (`hipErrorOutOfMemory` ...).
    pub fn errorName(self: *const Runtime, result: abi.Result) []const u8 {
        const name = self.api.hipGetErrorName(result) orelse return "unknown HIP error";
        return std.mem.span(name);
    }

    /// The GPUs this process sees.
    pub fn deviceCount(self: *const Runtime) Error!c_int {
        var n: c_int = 0;
        try check(self.api.hipGetDeviceCount(&n));
        return n;
    }

    pub fn close(self: *Runtime) void {
        self.lib.close();
        self.* = undefined;
    }
};

pub fn check(result: abi.Result) Error!void {
    if (result != 0) return error.HipFailed;
}

test "complete runtime refuses absent library" {
    try std.testing.expectError(error.DriverUnavailable, Runtime.openPath("/nonexistent/hip-runtime.so"));
}
