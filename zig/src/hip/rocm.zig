//! The ROCm kernels of this GPU's family, embedded in the binary: the Python ROCm engine's kernels and the torch-op
//! kernels launched from Zig on code objects (launches.zig), or through the library of C launchers opened from memory
//! (TF_HIP_LAUNCH=library), each call checked.

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi.zig");
const kernels = @import("kernels.zig");
const driver = @import("driver.zig");
const launches = @import("launches.zig");

pub const Error = error{ LibraryUnavailable, MissingSymbol, KernelFailed } || driver.Error;

/// How the kernels are launched: from Zig on the family's code objects, or by the embedded library's C launchers.
pub const Launch = enum { zig, library };

const S = abi.Stream;
const P = ?*anyopaque;
const C = ?*const anyopaque;
const F = ?[*]f32;
const CF = ?[*]const f32;
const I = ?[*]i32;
const CI = ?[*]const i32;

/// Each field is the library's exact export; every launcher returns 0 or 1 (its message in tf_last_error).
pub const Api = struct {
    tf_last_error: *const fn () callconv(.c) [*:0]const u8,
    tf_op_error: *const fn () callconv(.c) [*:0]const u8,
    tf_wmma_build: *const fn () callconv(.c) c_int,
    tf_rms: *const fn (C, CF, P, c_int, c_int, c_int, f32, S) callconv(.c) c_int,
    tf_conv_decode: *const fn (CF, CF, F, F, c_int, c_int, c_int, S) callconv(.c) c_int,
    tf_conv_rows: *const fn (CF, CF, F, F, F, c_int, c_int, c_int, S) callconv(.c) c_int,
    tf_rope_decode: *const fn (CF, F, c_int, c_int, c_int, c_int, f32, S, CI, c_int) callconv(.c) c_int,
    tf_moe_router: *const fn (C, c_int, CF, F, c_int, c_int, c_int, S) callconv(.c) c_int,
    tf_moe_select: *const fn (CF, I, F, I, I, c_int, c_int, c_int, c_int, S) callconv(.c) c_int,
    tf_moe_act: *const fn (CF, P, c_int, c_int, c_int, f32, S) callconv(.c) c_int,
    tf_moe_combine: *const fn (CF, CF, P, c_int, c_int, c_int, c_int, S) callconv(.c) c_int,
    tf_gdn_gate: *const fn (C, C, c_int, CF, CF, F, F, c_int, c_int, S) callconv(.c) c_int,
    tf_causal: *const fn (CF, C, C, F, c_int, c_int, c_int, c_int, c_int, c_int, f32, c_int, c_longlong, c_longlong, c_longlong, c_longlong, c_longlong, c_longlong, c_int, F, F, F, S, CI) callconv(.c) c_int,
    tf_gated_delta: *const fn (CF, CF, CF, CF, CF, F, F, c_int, c_int, c_int, c_int, c_int, c_int, S, F) callconv(.c) c_int,
    tf_affine_dot2_splits: *const fn (c_int, c_int, c_int, c_int, c_int) callconv(.c) c_int,
    tf_affine: *const fn (C, C, C, C, c_int, P, c_int, c_int, c_int, c_int, c_int, c_int, c_int, S, F, c_int, c_int) callconv(.c) c_int,
    tf_affine_routed: *const fn (C, C, C, C, c_int, P, CI, c_int, CI, c_int, c_int, c_int, c_int, c_int, c_int, c_int, S) callconv(.c) c_int,
    tf_embed_rows: *const fn (C, C, C, c_int, CI, c_int, c_int, c_int, c_int, P, c_int, S) callconv(.c) c_int,
    tf_cast: *const fn (C, c_int, P, c_int, c_longlong, S) callconv(.c) c_int,
    tf_silu_mul: *const fn (C, C, P, c_int, c_longlong, S) callconv(.c) c_int,
    tf_add: *const fn (C, C, P, c_int, c_longlong, S) callconv(.c) c_int,
    tf_attn_gate: *const fn (CF, C, P, c_int, c_int, c_int, c_int, c_int, S) callconv(.c) c_int,
    tf_gnorm_silu: *const fn (CF, C, P, c_int, c_longlong, S) callconv(.c) c_int,
    tf_copy_cols: *const fn (C, c_longlong, c_int, P, c_int, c_int, c_int, S) callconv(.c) c_int,
    tf_rope_prefill: *const fn (C, c_int, P, c_int, c_longlong, c_longlong, c_int, c_int, c_int, c_int, c_int, f32, S) callconv(.c) c_int,
    tf_kv_write: *const fn (C, P, c_int, c_int, c_int, c_int, c_int, c_int, S) callconv(.c) c_int,
    tf_conv_prefill: *const fn (C, c_int, CF, CF, F, F, c_int, c_int, c_int, S) callconv(.c) c_int,
    tf_gdn_gate_prefill: *const fn (C, C, c_int, CF, CF, F, F, c_int, c_int, S) callconv(.c) c_int,
    tf_dense_rows: *const fn (C, c_int, CF, P, c_int, c_int, c_int, S) callconv(.c) c_int,
    tf_argmax_rows: *const fn (C, c_int, c_int, c_int, I, S) callconv(.c) c_int,
    tf_topk_rows: *const fn (C, c_int, c_int, CI, c_int, I, P, S) callconv(.c) c_int,
    tf_moe_route: *const fn (CI, c_int, c_int, c_int, I, I, c_int, S) callconv(.c) c_int,
    tf_tp_error: *const fn () callconv(.c) [*:0]const u8,
    tf_add_wide: *const fn (C, CF, P, c_int, c_longlong, S) callconv(.c) c_int,
    tf_moe_localize: *const fn (CI, CI, I, c_int, c_int, S) callconv(.c) c_int,
    tf_moe_foreign_items: *const fn (I, c_int, c_int, S) callconv(.c) c_int,
    tf_moe_zero_foreign: *const fn (F, CI, c_int, c_longlong, c_int, S) callconv(.c) c_int,
};

pub const Family = enum { rdna2, rdna3 };

/// gfx major 10 (gfx103x) is RDNA2; 11 and 12 take the WMMA build.
pub fn familyOf(capability: u32) ?Family {
    return switch (capability / 10) {
        10 => if (capability == 103) .rdna2 else null,
        11, 12 => .rdna3,
        else => null,
    };
}

pub const Library = struct {
    lib: std.DynLib,
    api: Api,
    family: Family,
    /// The Zig launches on the current device; null runs every call through `api`.
    zig: ?launches.Launcher = null,

    /// TF_HIP_LAUNCH picks the path: `zig` (the default) or `library`.
    pub fn launchMode() Launch {
        const v = std.c.getenv("TF_HIP_LAUNCH") orelse return .zig;
        return std.meta.stringToEnum(Launch, std.mem.span(v)) orelse .zig;
    }

    /// The embedded library of `family`, written to an anonymous file and opened; every export resolved. The Zig
    /// launches load their code objects on the calling thread's current device.
    pub fn open(d: *const driver.Driver, family: Family) Error!Library {
        return openMode(d, family, launchMode());
    }

    pub fn openMode(d: *const driver.Driver, family: Family, mode: Launch) Error!Library {
        var lib = try openLibrary(family);
        errdefer lib.close();
        if (mode == .zig) {
            const images = switch (family) {
                .rdna2 => kernels.rdna2_modules,
                .rdna3 => kernels.rdna3_modules,
            };
            if (images[0].len == 0) {
                std.log.warn("no {t} code objects in this binary: launching through the library", .{family});
            } else {
                lib.zig = try launches.Launcher.load(d, family == .rdna3, images);
            }
        }
        return lib;
    }

    fn openLibrary(family: Family) Error!Library {
        const bytes = switch (family) {
            .rdna2 => kernels.rdna2,
            .rdna3 => kernels.rdna3,
        };
        if (bytes.len == 0) {
            std.log.err("this binary was built without the {t} kernel library (-Dhipcc or -Dhsaco)", .{family});
            return error.LibraryUnavailable;
        }
        if (builtin.os.tag != .linux) return error.LibraryUnavailable;
        const linux = std.os.linux;
        const rc = linux.memfd_create("tensorfold-hip", 0);
        if (linux.errno(rc) != .SUCCESS) return error.LibraryUnavailable;
        const fd: i32 = @intCast(rc);
        defer _ = linux.close(fd);
        var at: usize = 0;
        while (at < bytes.len) {
            const n = linux.write(fd, bytes[at..].ptr, bytes.len - at);
            if (linux.errno(n) != .SUCCESS or n == 0) return error.LibraryUnavailable;
            at += n;
        }
        var path_buf: [64]u8 = undefined;
        const path = std.fmt.bufPrint(&path_buf, "/proc/self/fd/{d}", .{fd}) catch unreachable;
        var lib = std.DynLib.open(path) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("the {t} kernel library has no {s}", .{ family, name });
                return error.MissingSymbol;
            };
        }
        return .{ .lib = lib, .api = api, .family = family };
    }

    pub fn close(self: *Library) void {
        if (self.zig) |*z| z.unload();
        self.lib.close();
    }

    fn ArgsOf(comptime name: []const u8) type {
        return std.meta.ArgsTuple(@typeInfo(@FieldType(Api, name)).pointer.child);
    }

    /// One launcher by its C name: the Zig launch when this build has it, else the library's, checked.
    pub fn call(self: *const Library, comptime name: []const u8, args: ArgsOf(name)) Error!void {
        if (self.zig) |*z| {
            if (@hasDecl(launches.Launcher, name)) return @call(.auto, @field(launches.Launcher, name), .{z} ++ args);
        }
        try self.check(@call(.auto, @field(self.api, name), args), name);
    }

    /// A launcher's status: 0 passes, else the library's message is logged under `what`.
    pub fn check(self: *const Library, rc: c_int, what: []const u8) Error!void {
        if (rc == 0) return;
        const kernel = std.mem.span(self.api.tf_last_error());
        const op = std.mem.span(self.api.tf_op_error());
        const tp = std.mem.span(self.api.tf_tp_error());
        std.log.err("{s}: {s}", .{ what, if (kernel.len > 0) kernel else if (op.len > 0) op else tp });
        return error.KernelFailed;
    }
};

test "the gfx major picks the library family" {
    try std.testing.expectEqual(Family.rdna2, familyOf(103).?);
    try std.testing.expectEqual(Family.rdna3, familyOf(110).?);
    try std.testing.expectEqual(Family.rdna3, familyOf(115).?);
    try std.testing.expect(familyOf(101) == null);
}
