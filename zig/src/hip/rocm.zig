//! The embedded ROCm kernels of this GPU's family, launched from Zig on code objects or through the C library, each call checked.

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("runtime/abi.zig");
const kernels = @import("kernels.zig");
const driver = @import("runtime/driver.zig");
const launches = @import("launches.zig");
const Policy = @import("policy.zig").Policy;
const Caps = @import("caps.zig").Caps;

pub const Error = error{ LibraryUnavailable, MissingSymbol, KernelFailed } || driver.Error;


pub const Launch = @import("policy.zig").Launch;

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
    tf_qk_rope: *const fn (C, c_int, c_longlong, c_int, CF, f32, c_int, c_int, c_int, c_int, f32, CI, F, P, c_int, S) callconv(.c) c_int,
    tf_rope_prefill: *const fn (C, c_int, P, c_int, c_longlong, c_longlong, c_int, c_int, c_int, c_int, c_int, f32, S) callconv(.c) c_int,
    tf_kv_write: *const fn (C, P, c_int, c_int, c_int, c_int, c_int, c_int, S) callconv(.c) c_int,
    tf_kv_write_at: *const fn (C, P, c_int, c_int, c_int, c_int, c_int, CI, S) callconv(.c) c_int,
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
    tf_token_prob: *const fn (C, c_int, c_int, c_int, CI, F, S) callconv(.c) c_int,
};

/// The code objects built for a GPU: the library and modules a caps row names.
pub const Family = @import("caps.zig").Family;

pub const Library = struct {
    lib: std.DynLib,
    api: Api,
    caps: Caps,
    /// What the run may use: the kernels and the ops read their switches here.
    policy: Policy,
    /// The Zig launches on the current device; null runs every call through `api`.
    zig: ?launches.Launcher = null,

    /// The embedded library of the GPU's family, opened from an anonymous file; with the policy's `launch` at `zig` the Zig launches load on the calling thread's device.
    pub fn open(d: *const driver.Driver, caps: Caps, policy: Policy) Error!Library {
        const family = caps.family;
        var lib = try openLibrary(caps, policy);
        errdefer lib.close();
        if (policy.launch == .zig) {
            const images = switch (family) {
                .rdna2 => kernels.rdna2_modules,
                .rdna3 => kernels.rdna3_modules,
                .gcn5 => return error.LibraryUnavailable,
            };
            if (images[0].len == 0) {
                std.log.warn("no {t} code objects in this binary: launching through the library", .{family});
            } else {
                lib.zig = try launches.Launcher.load(d, caps, policy, images);
            }
        }
        return lib;
    }

    fn openLibrary(caps: Caps, policy: Policy) Error!Library {
        const family = caps.family;
        const bytes = switch (family) {
            .rdna2 => kernels.rdna2,
            .rdna3 => kernels.rdna3,
            .gcn5 => &[0]u8{},
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
        return .{ .lib = lib, .api = api, .caps = caps, .policy = policy };
    }

    pub fn close(self: *Library) void {
        if (self.zig) |*z| z.unload();
        self.lib.close();
    }

    /// What each launch of a product chooses at every row count on this GPU under this policy, one line an (op, path).
    pub fn explain(self: *const Library, w: *std.Io.Writer, bits: u8, group: u16) std.Io.Writer.Error!void {
        const z = self.zig orelse return w.writeAll("kernel choice: the C library's own, by launch=library\n");
        try z.affine.reg.explain(z.affine.env(), .mlx, w, bits, group);
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

test "the caps name the library family" {
    try std.testing.expectEqual(Family.rdna2, Caps.of("gfx1030").?.family);
    try std.testing.expectEqual(Family.rdna3, Caps.of("gfx1100").?.family);
    try std.testing.expectEqual(Family.rdna3, Caps.of("gfx1151").?.family);
    try std.testing.expect(Caps.of("gfx1010") == null);
}
