//! gfx1030 ROCm host dispatch. Mac stays Metal. gfx1030 selects this backend.
//! hipcc 7.14.0 compiles the device code. Family runtimes, MoE, attention, and MTP are not ported here.
const std = @import("std");
const produce = @import("produce.zig");
const graph = @import("graph.zig");
const mixer = @import("mixer.zig");

pub const DType = enum { fp16, bf16, fp32, i8 };
pub const Schedule = enum { fdot2_f16, fdot2_bf16, sdot4, scalar_gemv, wmma };
pub const Op = enum { w4a16, packed_int };

pub const KvScore = enum { caller_window, full_span_dense_causal };

pub const Residual = struct {
    norms_e_proj_plus_h_proj: bool,
    adds_attention: bool,
};

pub const Backend = enum { metal, rocm };

pub const Error = error{
    ArchRefused,
    FatbinArchMismatch,
    FatbinMissing,
    FatbinRefused,
    Bf16Refused,
    ActivationNotFp16,
    ActivationNotPackedInt,
    ScheduleRefused,
    ScalarGemvRefused,
    WmmaRefused,
    PackedIntNotSdot4,
    DequantThenFdot2Refused,
    CodebookRefused,
    ShapeRefused,
    MtpResidualRefused,
    FullSpanDenseCausal,
    NotPorted,
};

pub const Launch = struct {
    symbol: []const u8,
    fatbin: []const u8,
    schedule: Schedule,
};

pub const Request = struct {
    arch: []const u8,
    fatbin_arch: []const u8,
    op: Op,
    activation: DType,
    schedule: Schedule,
    codebook: []const u8,
    /// Packed ints widened to fp16 and then dotted. That is not a produce path.
    dequant_then_fdot2: bool = false,
    /// Clang offload bundle bytes. Empty means the fatbin was not built.
    bundle: []const u8 = "",
};

/// Mac keeps Metal. Any other host selects the Zig ROCm backend only for gfx1030.
pub fn select(macos: bool, arch_name: []const u8) Error!Backend {
    if (macos) return .metal;
    produce.acceptArch(arch_name) catch return error.ArchRefused;
    return .rocm;
}

pub fn prepare(req: Request) Error!Launch {
    produce.acceptArch(req.arch) catch return error.ArchRefused;
    produce.acceptArch(req.fatbin_arch) catch return error.ArchRefused;
    if (!std.mem.eql(u8, req.arch, req.fatbin_arch)) return error.FatbinArchMismatch;
    if (req.schedule == .scalar_gemv) return error.ScalarGemvRefused;
    if (req.schedule == .wmma) return error.WmmaRefused;
    if (req.schedule == .fdot2_bf16 or req.activation == .bf16) return error.Bf16Refused;
    if (req.dequant_then_fdot2) return error.DequantThenFdot2Refused;
    if (!std.mem.eql(u8, req.codebook, produce.codebook)) return error.CodebookRefused;
    const launch: Launch = switch (req.op) {
        .w4a16 => blk: {
            if (req.activation != .fp16) return error.ActivationNotFp16;
            if (req.schedule != .fdot2_f16) return error.ScheduleRefused;
            break :blk .{
                .symbol = "tf_gfx1030_w4a16_fdot2",
                .fatbin = produce.fatbin_rel,
                .schedule = .fdot2_f16,
            };
        },
        .packed_int => blk: {
            if (req.activation != .i8) return error.ActivationNotPackedInt;
            if (req.schedule != .sdot4) return error.PackedIntNotSdot4;
            break :blk .{
                .symbol = "tf_gfx1030_sdot4",
                .fatbin = produce.fatbin_rel,
                .schedule = .sdot4,
            };
        },
    };
    if (req.bundle.len == 0) return error.FatbinMissing;
    produce.acceptBundle(req.bundle) catch |err| switch (err) {
        error.ForeignArch => return error.ArchRefused,
        else => return error.FatbinRefused,
    };
    return launch;
}

fn loadedBundle() Error![]const u8 {
    return produce.loadFatbin() catch |err| switch (err) {
        error.ArchRefused, error.ForeignArch => return error.ArchRefused,
        else => return error.FatbinRefused,
    };
}

pub fn bindW4A16(m: i32, n: i32, k: i32, activation: DType, arch_name: []const u8) Error!Launch {
    if (m < 1 or n < 1 or k < 2 or (k & 1) != 0) return error.ShapeRefused;
    return prepare(.{
        .arch = arch_name,
        .fatbin_arch = arch_name,
        .op = .w4a16,
        .activation = activation,
        .schedule = .fdot2_f16,
        .codebook = produce.codebook,
        .bundle = try loadedBundle(),
    });
}

pub fn bindPackedInt(arch_name: []const u8) Error!Launch {
    return prepare(.{
        .arch = arch_name,
        .fatbin_arch = arch_name,
        .op = .packed_int,
        .activation = .i8,
        .schedule = .sdot4,
        .codebook = produce.codebook,
        .bundle = try loadedBundle(),
    });
}

pub fn residual(contract: Residual) Error!void {
    if (contract.norms_e_proj_plus_h_proj and contract.adds_attention) return error.MtpResidualRefused;
}

pub fn scoreKv(mode: KvScore) Error!void {
    if (mode == .full_span_dense_causal) return error.FullSpanDenseCausal;
}

pub fn launchAttention(mode: KvScore) Error!void {
    try scoreKv(mode);
    return error.NotPorted;
}

fn expectAbsent(src: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, src, needle) != null) {
        std.debug.print("device source contains {s}\n", .{needle});
        return error.ForbiddenDeviceText;
    }
}

test "macOS stays Metal and gfx1030 selects ROCm only with one fatbin" {
    try std.testing.expectEqual(Backend.metal, try select(true, "gfx1030"));
    try std.testing.expectEqual(Backend.metal, try select(true, "gfx900"));
    try std.testing.expectEqual(Backend.rocm, try select(false, "gfx1030"));
    try std.testing.expectError(error.ArchRefused, select(false, "gfx900"));
    try std.testing.expectError(error.ArchRefused, select(false, "gfx906"));
    try std.testing.expectError(error.ArchRefused, select(false, "gfx1013"));
}

test "fp16 W4A16 launches fdot2; bf16, scalar gemv, and WMMA do not" {
    const launch = try bindW4A16(2, 32, 64, .fp16, "gfx1030");
    try std.testing.expectEqualStrings("tf_gfx1030_w4a16_fdot2", launch.symbol);
    try std.testing.expectEqualStrings(produce.fatbin_rel, launch.fatbin);
    try std.testing.expectError(error.FatbinMissing, prepare(.{
        .arch = "gfx1030",
        .fatbin_arch = "gfx1030",
        .op = .w4a16,
        .activation = .fp16,
        .schedule = .fdot2_f16,
        .codebook = "3inst",
    }));
    try std.testing.expectError(error.FatbinRefused, prepare(.{
        .arch = "gfx1030",
        .fatbin_arch = "gfx1030",
        .op = .w4a16,
        .activation = .fp16,
        .schedule = .fdot2_f16,
        .codebook = "3inst",
        .bundle = &[_]u8{ 0x7f, 'E', 'L', 'F' },
    }));
    try std.testing.expectError(error.Bf16Refused, bindW4A16(2, 32, 64, .bf16, "gfx1030"));
    try std.testing.expectError(error.ActivationNotFp16, bindW4A16(2, 32, 64, .fp32, "gfx1030"));
    try std.testing.expectError(error.ArchRefused, bindW4A16(2, 32, 64, .fp16, "gfx900"));
    try std.testing.expectError(error.ArchRefused, bindW4A16(2, 32, 64, .fp16, "gfx906"));
    try std.testing.expectError(error.ArchRefused, bindW4A16(2, 32, 64, .fp16, "gfx1013"));
    try std.testing.expectError(error.ShapeRefused, bindW4A16(1, 8, 31, .fp16, "gfx1030"));
    try std.testing.expectError(error.ScalarGemvRefused, prepare(.{
        .arch = "gfx1030",
        .fatbin_arch = "gfx1030",
        .op = .w4a16,
        .activation = .fp16,
        .schedule = .scalar_gemv,
        .codebook = "3inst",
    }));
    try std.testing.expectError(error.WmmaRefused, prepare(.{
        .arch = "gfx1030",
        .fatbin_arch = "gfx1030",
        .op = .w4a16,
        .activation = .fp16,
        .schedule = .wmma,
        .codebook = "3inst",
    }));
    try std.testing.expectError(error.Bf16Refused, prepare(.{
        .arch = "gfx1030",
        .fatbin_arch = "gfx1030",
        .op = .w4a16,
        .activation = .fp16,
        .schedule = .fdot2_bf16,
        .codebook = "3inst",
    }));
    try std.testing.expectError(error.DequantThenFdot2Refused, prepare(.{
        .arch = "gfx1030",
        .fatbin_arch = "gfx1030",
        .op = .w4a16,
        .activation = .fp16,
        .schedule = .fdot2_f16,
        .codebook = "3inst",
        .dequant_then_fdot2 = true,
    }));
}

test "packed int is sdot4, not a widened fdot2" {
    const launch = try bindPackedInt("gfx1030");
    try std.testing.expectEqualStrings(produce.fatbin_rel, launch.fatbin);
    try std.testing.expectEqualStrings("tf_gfx1030_sdot4", launch.symbol);
    try std.testing.expect(launch.schedule == .sdot4);
    try std.testing.expectError(error.DequantThenFdot2Refused, prepare(.{
        .arch = "gfx1030",
        .fatbin_arch = "gfx1030",
        .op = .packed_int,
        .activation = .i8,
        .schedule = .sdot4,
        .codebook = "3inst",
        .dequant_then_fdot2 = true,
    }));
    try std.testing.expectError(error.PackedIntNotSdot4, prepare(.{
        .arch = "gfx1030",
        .fatbin_arch = "gfx1030",
        .op = .packed_int,
        .activation = .i8,
        .schedule = .fdot2_f16,
        .codebook = "3inst",
    }));
    try std.testing.expectError(error.CodebookRefused, prepare(.{
        .arch = "gfx1030",
        .fatbin_arch = "gfx1030",
        .op = .w4a16,
        .activation = .fp16,
        .schedule = .fdot2_f16,
        .codebook = "mcg",
    }));
}

test "MTP residual and full-span dense causal KV are refused" {
    try std.testing.expectError(error.MtpResidualRefused, residual(.{
        .norms_e_proj_plus_h_proj = true,
        .adds_attention = true,
    }));
    try residual(.{ .norms_e_proj_plus_h_proj = false, .adds_attention = false });
    try std.testing.expectError(error.FullSpanDenseCausal, scoreKv(.full_span_dense_causal));
    try std.testing.expectError(error.NotPorted, launchAttention(.caller_window));
}

test "device sources emit fdot2 and sdot4 explicitly and name no other arch" {
    const w4 = @embedFile("hip/w4a16_fdot2.hip");
    const packed_src = @embedFile("hip/sdot4.hip");
    try std.testing.expect(std.mem.indexOf(u8, w4, "__builtin_amdgcn_fdot2") != null);
    try std.testing.expect(std.mem.indexOf(u8, w4, "89226354u") != null);
    try std.testing.expect(std.mem.indexOf(u8, w4, "act_kind != 1") != null);
    try expectAbsent(w4, "__builtin_amdgcn_fdot2_f32_bf16");
    try expectAbsent(w4, "v_dot2_f32_bf16");
    try expectAbsent(w4, "__builtin_amdgcn_sdot4");
    try expectAbsent(w4, "wmma");
    try expectAbsent(w4, "mma.sync");
    try expectAbsent(w4, "-ffp-contract=off");
    try expectAbsent(w4, "gfx900");
    try expectAbsent(w4, "gfx906");
    try expectAbsent(w4, "gfx1013");
    try std.testing.expect(std.mem.indexOf(u8, packed_src, "__builtin_amdgcn_sdot4") != null);
    try expectAbsent(packed_src, "__builtin_amdgcn_fdot2");
    try expectAbsent(packed_src, "hip_fp16");
    try expectAbsent(packed_src, "__half");
    try expectAbsent(packed_src, "wmma");
    try expectAbsent(packed_src, "gfx900");
    try expectAbsent(packed_src, "gfx906");
    try expectAbsent(packed_src, "gfx1013");
    _ = graph;
    _ = mixer;
}
