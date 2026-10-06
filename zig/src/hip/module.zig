//! Loaded GPU code: code objects or offload bundles, their kernels by symbol name and their device globals.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const Module = struct {
    d: *const Driver,
    handle: abi.Module,

    /// A code object or an offload bundle of several (8-byte aligned); HIP picks the one built for this GPU.
    pub fn load(d: *const Driver, image: []const u8) Error!Module {
        if (image.len == 0) {
            std.log.err("empty GPU image: this binary was built without kernels (-Dhipcc or -Dhsaco)", .{});
            return error.Invalid;
        }
        if (@intFromPtr(image.ptr) % 8 != 0) return error.Invalid;
        var m: abi.Module = null;
        try d.check(d.api.hipModuleLoadData(&m, image.ptr), "hipModuleLoadData");
        return .{ .d = d, .handle = m };
    }

    pub fn unload(self: *Module) void {
        _ = self.d.api.hipModuleUnload(self.handle);
        self.* = undefined;
    }

    /// A kernel by its exact (mangled or extern "C") symbol; the function lives as long as the module.
    pub fn function(self: Module, name: [:0]const u8) Error!Function {
        var f: abi.Function = null;
        self.d.check(self.d.api.hipModuleGetFunction(&f, self.handle, name.ptr), "hipModuleGetFunction") catch |e| {
            std.log.err("kernel symbol not in module: {s}", .{name});
            return e;
        };
        return .{ .d = self.d, .handle = f };
    }

    pub const Global = struct { ptr: abi.DevicePtr, len: usize };

    pub fn global(self: Module, name: [:0]const u8) Error!Global {
        var g: Global = .{ .ptr = 0, .len = 0 };
        try self.d.check(self.d.api.hipModuleGetGlobal(&g.ptr, &g.len, self.handle, name.ptr), "hipModuleGetGlobal");
        return g;
    }
};

pub const Function = struct {
    d: *const Driver,
    handle: abi.Function,

    pub fn attribute(self: Function, a: abi.FunctionAttribute) Error!c_int {
        var v: c_int = 0;
        try self.d.check(self.d.api.hipFuncGetAttribute(&v, a, self.handle), "hipFuncGetAttribute");
        return v;
    }

    /// Resident blocks a CU holds of this kernel at `threads` a block and `shared` dynamic bytes.
    pub fn occupancy(self: Function, threads: u32, shared: usize) Error!u32 {
        var n: c_int = 0;
        try self.d.check(self.d.api.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&n, self.handle, @intCast(threads), shared), "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
        return @intCast(@max(n, 0));
    }
};
