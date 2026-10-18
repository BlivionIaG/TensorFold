//! Gate 5's records of the pair of relu^2 stream tiles in one launch (affine_dot2_stream_relu2_pair).
const std = @import("std");
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

/// The 4-bit pair of relu^2 stream tiles in one launch, `rows` rows and `cb` columns a lane, as relu2Tile.
fn pairTile(comptime rows: u32, comptime cb: u32) limits.HipLaunch {
    const fmt = "_ZN2tf4rocm29affine_dot2_stream_relu2_pairINS0_{s}ELi4ELi{d}ELi{d}EEEvNS0_6AffineES3_iii";
    const half = std.fmt.comptimePrint(fmt, .{ "6DotF16", rows, cb });
    const brain = std.fmt.comptimePrint(fmt, .{ "7DotBF16", rows, cb });
    return .{
        .name = std.fmt.comptimePrint("relu2 pair tile 4-bit {d}x{d}", .{ rows, cb }),
        .symbol = brain,
        .image = "affine_tiles",
        .threads = 128,
        .symbols = .{ half, brain, brain, brain },
    };
}

pub const records = [_]Record{
    .{
        .launch = pairTile(1, 2),
        .usage = .{
            .{ .vgprs = 111, .sgprs = 58, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = pairTile(1, 4),
        .usage = .{
            .{ .vgprs = 190, .sgprs = 58, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = pairTile(1, 8),
        .usage = .{
            .{ .vgprs = 177, .sgprs = 58, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 232, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 232, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 232, .sgprs = 70, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = pairTile(2, 2),
        .usage = .{
            .{ .vgprs = 137, .sgprs = 58, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 166, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 166, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 166, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = pairTile(2, 4),
        .usage = .{
            .{ .vgprs = 196, .sgprs = 58, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 251, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 251, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 251, .sgprs = 70, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = pairTile(2, 8),
        .usage = .{
            .{ .vgprs = 250, .sgprs = 58, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 223, .sgprs = 74, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 223, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 223, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = pairTile(4, 2),
        .usage = .{
            .{ .vgprs = 212, .sgprs = 58, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 78, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 70, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = pairTile(4, 4),
        .usage = .{
            .{ .vgprs = 256, .sgprs = 58, .lds = 0, .private = 240, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 78, .lds = 0, .private = 272, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 78, .lds = 0, .private = 272, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 70, .lds = 0, .private = 272, .bound = 128, .wave = 32 },
        },
    },
};
