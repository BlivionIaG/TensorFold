//! Compiler-derived launch usage for every swiglu stream specialization on the four served RDNA targets.
const Record = @import("hip_records.zig").Record;
const tile = @import("hip_records_stream.zig").tile;

pub const records = [_]Record{
    .{
        .launch = tile(2, 1, 2, true),
        .usage = .{
            .{ .vgprs = 98, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 1, 4, true),
        .usage = .{
            .{ .vgprs = 170, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 1, 8, true),
        .usage = .{
            .{ .vgprs = 253, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 2, true),
        .usage = .{
            .{ .vgprs = 111, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 4, true),
        .usage = .{
            .{ .vgprs = 148, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 8, true),
        .usage = .{
            .{ .vgprs = 235, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 4, 2, true),
        .usage = .{
            .{ .vgprs = 209, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 199, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 199, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 199, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 4, 4, true),
        .usage = .{
            .{ .vgprs = 243, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 225, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 225, .sgprs = 43, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 225, .sgprs = 43, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 2, true),
        .usage = .{
            .{ .vgprs = 136, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 4, true),
        .usage = .{
            .{ .vgprs = 230, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 183, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 183, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 183, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 8, true),
        .usage = .{
            .{ .vgprs = 184, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 243, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 243, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 243, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 2, true),
        .usage = .{
            .{ .vgprs = 119, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 4, true),
        .usage = .{
            .{ .vgprs = 178, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 8, true),
        .usage = .{
            .{ .vgprs = 244, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 4, 2, true),
        .usage = .{
            .{ .vgprs = 139, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 4, 4, true),
        .usage = .{
            .{ .vgprs = 187, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 187, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 187, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 187, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 2, true),
        .usage = .{
            .{ .vgprs = 104, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 4, true),
        .usage = .{
            .{ .vgprs = 186, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 179, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 179, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 179, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 8, true),
        .usage = .{
            .{ .vgprs = 180, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 272, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 272, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 272, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 2, true),
        .usage = .{
            .{ .vgprs = 127, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 4, true),
        .usage = .{
            .{ .vgprs = 187, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 208, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 208, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 208, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 8, true),
        .usage = .{
            .{ .vgprs = 243, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 4, 2, true),
        .usage = .{
            .{ .vgprs = 209, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 200, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 200, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 200, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 4, 4, true),
        .usage = .{
            .{ .vgprs = 253, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 144, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 43, .lds = 0, .private = 144, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 43, .lds = 0, .private = 144, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 2, true),
        .usage = .{
            .{ .vgprs = 138, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 125, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 125, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 125, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 4, true),
        .usage = .{
            .{ .vgprs = 246, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 198, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 198, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 198, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 8, true),
        .usage = .{
            .{ .vgprs = 215, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 208, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 208, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 208, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 2, true),
        .usage = .{
            .{ .vgprs = 97, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 126, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 126, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 126, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 4, true),
        .usage = .{
            .{ .vgprs = 160, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 199, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 199, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 199, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 8, true),
        .usage = .{
            .{ .vgprs = 254, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 193, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 193, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 193, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 4, 2, true),
        .usage = .{
            .{ .vgprs = 135, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 140, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 140, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 140, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 4, 4, true),
        .usage = .{
            .{ .vgprs = 217, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 189, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 189, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 189, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 2, true),
        .usage = .{
            .{ .vgprs = 140, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 4, true),
        .usage = .{
            .{ .vgprs = 251, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 209, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 209, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 209, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 8, true),
        .usage = .{
            .{ .vgprs = 233, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 272, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 272, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 272, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 2, true),
        .usage = .{
            .{ .vgprs = 127, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 4, true),
        .usage = .{
            .{ .vgprs = 221, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 8, true),
        .usage = .{
            .{ .vgprs = 191, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 200, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 200, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 200, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 4, 2, true),
        .usage = .{
            .{ .vgprs = 181, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 145, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 145, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 145, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 4, 4, true),
        .usage = .{
            .{ .vgprs = 245, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 190, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 190, .sgprs = 44, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 190, .sgprs = 44, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 2, true),
        .usage = .{
            .{ .vgprs = 120, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 4, true),
        .usage = .{
            .{ .vgprs = 218, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 214, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 214, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 214, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 8, true),
        .usage = .{
            .{ .vgprs = 239, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 272, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 272, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 272, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 2, true),
        .usage = .{
            .{ .vgprs = 127, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 4, true),
        .usage = .{
            .{ .vgprs = 187, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 250, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 250, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 250, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 8, true),
        .usage = .{
            .{ .vgprs = 248, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 640, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 640, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 640, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 4, 2, true),
        .usage = .{
            .{ .vgprs = 209, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 202, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 202, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 202, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 4, 4, true),
        .usage = .{
            .{ .vgprs = 253, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 144, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 43, .lds = 0, .private = 144, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 43, .lds = 0, .private = 144, .bound = 128, .wave = 32 },
        },
    },
};
