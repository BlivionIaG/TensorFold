//! Complete model-free HIP runtime admission; partial symbol tables never escape.
const std = @import("std");
const abi = @import("abi.zig");
pub const Error = error{ DriverUnavailable, MissingSymbol, HipFailed, OutOfDeviceMemory, NotFound, NotReady, Invalid };

pub const Runtime = struct {
    lib: std.DynLib,
    api: abi.Api,

    pub fn open() Error!Runtime {
        return openPath("libamdhip64.so");
    }

    pub fn openPath(path: []const u8) Error!Runtime {
        var lib = std.DynLib.open(path) catch return error.DriverUnavailable;
        errdefer lib.close();
        const api = try @import("symbols.zig").resolve(abi.Api, &lib);
        const r: Runtime = .{ .lib = lib, .api = api };
        try r.check(api.hipInit(0));
        return r;
    }

    pub fn close(self: *Runtime) void {
        self.lib.close();
        self.* = undefined;
    }

    /// Logs a failed result with HIP's own name and text; not-ready is a state, not a failure.
    pub fn check(self: *const Runtime, result: abi.Result) Error!void {
        classify(result) catch |err| {
            if (err != error.NotReady) log.warn("{s} ({d}): {s}", .{ text(self.api.hipGetErrorName(result)), result, text(self.api.hipGetErrorString(result)) });
            return err;
        };
    }

    /// Error names only, for tests that mock a few calls; their expected failures stay quiet.
    pub fn forTests() Runtime {
        if (!@import("builtin").is_test) @compileError("test-only runtime");
        std.testing.log_level = .err;
        const Names = struct {
            fn name(_: abi.Result) callconv(.c) ?[*:0]const u8 {
                return "hipErrorMock";
            }
        };
        var r: Runtime = undefined;
        r.api.hipGetErrorName = Names.name;
        r.api.hipGetErrorString = Names.name;
        return r;
    }
};

const log = std.log.scoped(.hip);

fn text(s: ?[*:0]const u8) []const u8 {
    return if (s) |p| std.mem.span(p) else "?";
}

/// Out of memory, not found and not ready stay distinct; every other failure is HipFailed.
pub fn classify(result: abi.Result) Error!void {
    return switch (result) {
        abi.success => {},
        abi.error_out_of_memory => error.OutOfDeviceMemory,
        abi.error_not_found => error.NotFound,
        abi.error_not_ready => error.NotReady,
        else => error.HipFailed,
    };
}

test "complete runtime refuses absent library" {
    try std.testing.expectError(error.DriverUnavailable, Runtime.openPath("/nonexistent/hip-runtime.so"));
}

test "HIP results keep out-of-memory, not-found and not-ready apart" {
    try classify(0);
    try std.testing.expectError(error.OutOfDeviceMemory, classify(2));
    try std.testing.expectError(error.NotFound, classify(500));
    try std.testing.expectError(error.NotReady, classify(600));
    try std.testing.expectError(error.HipFailed, classify(1));
    try std.testing.expectError(error.HipFailed, classify(-1));
}

test "a failed result is named through HIP and not-ready is not" {
    const Mock = struct {
        var named: usize = 0;
        fn name(_: abi.Result) callconv(.c) ?[*:0]const u8 {
            named += 1;
            return null;
        }
    };
    std.testing.log_level = .err;
    var r: Runtime = undefined;
    r.api.hipGetErrorName = Mock.name;
    r.api.hipGetErrorString = Mock.name;
    Mock.named = 0;
    try r.check(0);
    try std.testing.expectError(error.NotReady, r.check(600));
    try std.testing.expectEqual(@as(usize, 0), Mock.named);
    try std.testing.expectError(error.OutOfDeviceMemory, r.check(2));
    try std.testing.expectEqual(@as(usize, 2), Mock.named);
}
