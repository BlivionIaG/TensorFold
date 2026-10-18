//! The HIP launches gate 5 checks: each one's threads and LDS, and what hipcc made of it on every RDNA chip we serve.
const std = @import("std");
const limits = @import("limits.zig");

const U = limits.HipUsage;

/// A HIP launch and what hipcc made of it for gfx1030, gfx1100, gfx1151 and gfx1201 (limits.rdna's order).
pub const Record = struct { launch: limits.HipLaunch, usage: [limits.rdna.len]U };

const rms_ns = "_ZN2tf4rocm10rms_kernelI";

/// The 4-bit stream tile with the relu^2 epilogue, `rows` rows and `cb` columns a lane: DotF16 on RDNA2, else DotBF16.
fn relu2Tile(comptime rows: u32, comptime cb: u32) limits.HipLaunch {
    const fmt = "_ZN2tf4rocm24affine_dot2_stream_relu2INS0_{s}ELi4ELi{d}ELi{d}EEEvNS0_6AffineENS0_11StreamSidesEii";
    const half = std.fmt.comptimePrint(fmt, .{ "6DotF16", rows, cb });
    const brain = std.fmt.comptimePrint(fmt, .{ "7DotBF16", rows, cb });
    return .{
        .name = std.fmt.comptimePrint("relu2 stream tile 4-bit {d}x{d}", .{ rows, cb }),
        .symbol = brain,
        .image = "affine_tiles",
        .threads = 128,
        .symbols = .{ half, brain, brain, brain },
    };
}

/// HIP kernels as launched; tf-hip-kernel-contract limits DIR re-derives the usage records from the code objects.
const own_records = [_]Record{
    .{
        .launch = .{ .name = "RMS norm tf_rms_rows", .symbol = "tf_rms_rows", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 58, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 43, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 42, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 42, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "RMS norm rms_kernel<__half>",
            .symbol = rms_ns ++ "6__halfEEvPKT_PKfPS3_if",
            .image = "act",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "RMS norm rms_kernel<hip_bfloat16>",
            .symbol = rms_ns ++ "12hip_bfloat16EEvPKT_PKfPS3_if",
            .image = "act",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "mamba tf_mamba_conv", .symbol = "tf_mamba_conv", .image = "mamba", .threads = 256 },
        .usage = .{
            .{ .vgprs = 27, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 26, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 26, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 26, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "mamba tf_mamba_dt", .symbol = "tf_mamba_dt", .image = "mamba", .threads = 256 },
        .usage = .{
            .{ .vgprs = 16, .sgprs = 25, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 25, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 25, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 27, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "mamba tf_mamba_scan", .symbol = "tf_mamba_scan", .image = "mamba", .threads = 128 },
        .usage = .{
            .{ .vgprs = 127, .sgprs = 41, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 126, .sgprs = 41, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 125, .sgprs = 42, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 42, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "mamba tf_mamba_gnorm", .symbol = "tf_mamba_gnorm", .image = "mamba", .threads = 128 },
        .usage = .{
            .{ .vgprs = 23, .sgprs = 23, .lds = 16, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 24, .lds = 16, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 23, .lds = 16, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 24, .lds = 16, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "nemotron tf_nemo_select", .symbol = "tf_nemo_select", .image = "ops", .threads = 32 },
        .usage = .{
            .{ .vgprs = 27, .sgprs = 26, .lds = 8320, .private = 0, .bound = 32, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 26, .lds = 8320, .private = 0, .bound = 32, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 26, .lds = 8320, .private = 0, .bound = 32, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 26, .lds = 8320, .private = 0, .bound = 32, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "nemotron tf_nemo_select_rank",
            .symbol = "tf_nemo_select_rank",
            .image = "ops",
            .threads = 512,
        },
        .usage = .{
            .{ .vgprs = 27, .sgprs = 30, .lds = 8320, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 30, .lds = 8320, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 30, .lds = 8320, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 29, .lds = 8320, .private = 0, .bound = 512, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "nemotron tf_nemo_relu2", .symbol = "tf_nemo_relu2", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 8, .sgprs = 9, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "nemotron tf_nemo_localize",
            .symbol = "tf_nemo_localize",
            .image = "ops",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 4, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 9, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "nemotron tf_nemo_combine", .symbol = "tf_nemo_combine", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 9, .sgprs = 20, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 20, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 20, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 20, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_score_f16",
            .symbol = "tf_plan_score_f16",
            .image = "plan",
            .threads = 128,
            .dynamic = 16 * 128 * 4,
        },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 33, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 29, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_score_bf16",
            .symbol = "tf_plan_score_bf16",
            .image = "plan",
            .threads = 128,
            .dynamic = 16 * 128 * 4,
        },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 33, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 29, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_softmax_stats",
            .symbol = "tf_softmax_stats",
            .image = "attention",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 9, .sgprs = 16, .lds = 1024, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 18, .lds = 1024, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 16, .lds = 1024, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 18, .lds = 1024, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_apply_f16",
            .symbol = "tf_plan_apply_f16",
            .image = "plan",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_apply_bf16",
            .symbol = "tf_plan_apply_bf16",
            .image = "plan",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_sum_partials",
            .symbol = "tf_sum_partials",
            .image = "attention",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 5, .sgprs = 14, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 5, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 5, .sgprs = 14, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 5, .sgprs = 16, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "add rms tf_add_rms_f16", .symbol = "tf_add_rms_f16", .image = "act", .threads = 1024 },
        .usage = .{
            .{ .vgprs = 43, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 45, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 45, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 44, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "add rms tf_add_rms_bf16", .symbol = "tf_add_rms_bf16", .image = "act", .threads = 1024 },
        .usage = .{
            .{ .vgprs = 44, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 45, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 45, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 44, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "mamba tf_mamba_conv_window",
            .symbol = "tf_mamba_conv_window",
            .image = "mamba",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 32, .sgprs = 42, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 38, .sgprs = 42, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 37, .sgprs = 42, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 37, .sgprs = 44, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "mamba tf_mamba_scan_snap",
            .symbol = "tf_mamba_scan_snap",
            .image = "mamba",
            .threads = 128,
        },
        .usage = .{
            .{ .vgprs = 137, .sgprs = 50, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 50, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 51, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 131, .sgprs = 52, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(1, 2),
        .usage = .{
            .{ .vgprs = 110, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(1, 4),
        .usage = .{
            .{ .vgprs = 189, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(1, 8),
        .usage = .{
            .{ .vgprs = 177, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 230, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 230, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 230, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(2, 2),
        .usage = .{
            .{ .vgprs = 130, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 166, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 166, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 166, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(2, 4),
        .usage = .{
            .{ .vgprs = 196, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 250, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 250, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 250, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(2, 8),
        .usage = .{
            .{ .vgprs = 248, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 221, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 221, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 221, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(4, 2),
        .usage = .{
            .{ .vgprs = 211, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 202, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 202, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 202, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(4, 4),
        .usage = .{
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 160, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 160, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 160, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_kv_write_act",
            .symbol = "tf_plan_kv_write_act",
            .image = "plan",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 20, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 20, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 20, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 20, .sgprs = 28, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_score_act_f16",
            .symbol = "tf_plan_score_act_f16",
            .image = "plan",
            .threads = 128,
            .dynamic = 16 * 128 * 4,
        },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 28, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_score_act_bf16",
            .symbol = "tf_plan_score_act_bf16",
            .image = "plan",
            .threads = 128,
            .dynamic = 16 * 128 * 4,
        },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 28, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_sum_partials_act",
            .symbol = "tf_sum_partials_act",
            .image = "plan",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 7, .sgprs = 16, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 16, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
};

pub const records = own_records ++ @import("hip_records_stream.zig").records;

/// Gate 5's planted HIP controls: a launch past its __launch_bounds__, and 72 KiB of LDS where a workgroup has 64.
pub const controls = [_]Record{
    .{
        .launch = .{
            .name = "control: past its bounds",
            .symbol = "tf_ctl_bounded",
            .image = "contract_controls",
            .threads = 512,
        },
        .usage = .{
            .{ .vgprs = 4, .sgprs = 10, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 10, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 7, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "control: 72 KiB of LDS",
            .symbol = "tf_ctl_big_lds",
            .image = "contract_controls",
            .threads = 256,
            .dynamic = 32 * 1024,
        },
        .usage = .{
            .{ .vgprs = 6, .sgprs = 9, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 8, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 8, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 8, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
};

/// Every HIP record's problems on every chip we serve, printed; returns how many (a missing record is one).
pub fn check(a: std.mem.Allocator, list: []const Record, quiet: bool) !usize {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var n: usize = 0;
    for (list) |r| for (limits.rdna, r.usage) |t, u| {
        if (u.vgprs == 0) {
            try out.print(a, "{s} on {s}: no compiled usage recorded\n", .{ r.launch.name, t.name });
            n += 1;
        } else n += try limits.hip(r.launch, u, t, &out, a);
    };
    if (!quiet and out.items.len > 0) std.debug.print("{s}", .{out.items});
    return n;
}

test "every registered HIP launch fits every RDNA chip we serve, and each planted control breaks one" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 0), try check(a, &records, false));
    for (controls) |r| try std.testing.expect(try check(a, &.{r}, true) > 0);
    for (controls) |r| {
        var good = r;
        for (&good.usage) |*u| u.* = .{ .vgprs = 16 };
        good.launch.threads = 256;
        good.launch.dynamic = 0;
        try std.testing.expectEqual(@as(usize, 0), try check(a, &.{good}, true)); // red only for what it plants
    }
}
