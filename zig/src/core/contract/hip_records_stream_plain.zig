//! Compiler-derived launch usage for every plain stream specialization on the four served RDNA targets.
const Record = @import("hip_records.zig").Record;
const tile = @import("hip_records_stream.zig").tile;

pub const records = [_]Record{
    .{
        .launch = tile(2, 1, 2, false),
        .usage = .{
            .{ .vgprs = 100, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 1, 4, false),
        .usage = .{
            .{ .vgprs = 177, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 170, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 170, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 170, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 1, 8, false),
        .usage = .{
            .{ .vgprs = 146, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 254, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 254, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 254, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 2, false),
        .usage = .{
            .{ .vgprs = 111, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 4, false),
        .usage = .{
            .{ .vgprs = 143, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 220, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 220, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 220, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 8, false),
        .usage = .{
            .{ .vgprs = 240, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 210, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 210, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 210, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 4, 2, false),
        .usage = .{
            .{ .vgprs = 202, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 201, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 201, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 201, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 4, 4, false),
        .usage = .{
            .{ .vgprs = 245, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 228, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 228, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 228, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 2, false),
        .usage = .{
            .{ .vgprs = 138, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 123, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 123, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 123, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 4, false),
        .usage = .{
            .{ .vgprs = 230, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 8, false),
        .usage = .{
            .{ .vgprs = 181, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 227, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 227, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 227, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 2, false),
        .usage = .{
            .{ .vgprs = 106, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 4, false),
        .usage = .{
            .{ .vgprs = 160, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 208, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 208, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 208, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 8, false),
        .usage = .{
            .{ .vgprs = 229, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 4, 2, false),
        .usage = .{
            .{ .vgprs = 130, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 4, 4, false),
        .usage = .{
            .{ .vgprs = 180, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 188, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 188, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 188, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 2, false),
        .usage = .{
            .{ .vgprs = 110, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 126, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 126, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 126, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 4, false),
        .usage = .{
            .{ .vgprs = 189, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 8, false),
        .usage = .{
            .{ .vgprs = 177, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 230, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 230, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 230, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 2, false),
        .usage = .{
            .{ .vgprs = 127, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 4, false),
        .usage = .{
            .{ .vgprs = 176, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 221, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 221, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 221, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 8, false),
        .usage = .{
            .{ .vgprs = 248, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 221, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 221, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 221, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 4, 2, false),
        .usage = .{
            .{ .vgprs = 211, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 202, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 202, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 202, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 4, 4, false),
        .usage = .{
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 2, false),
        .usage = .{
            .{ .vgprs = 140, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 4, false),
        .usage = .{
            .{ .vgprs = 249, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 201, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 201, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 201, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 8, false),
        .usage = .{
            .{ .vgprs = 212, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 160, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 160, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 160, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 2, false),
        .usage = .{
            .{ .vgprs = 108, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 125, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 125, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 125, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 4, false),
        .usage = .{
            .{ .vgprs = 137, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 187, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 187, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 187, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 8, false),
        .usage = .{
            .{ .vgprs = 229, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 198, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 198, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 198, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 4, 2, false),
        .usage = .{
            .{ .vgprs = 129, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 139, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 139, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 139, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 4, 4, false),
        .usage = .{
            .{ .vgprs = 178, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 187, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 187, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 187, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 2, false),
        .usage = .{
            .{ .vgprs = 142, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 131, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 131, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 131, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 4, false),
        .usage = .{
            .{ .vgprs = 254, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 212, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 212, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 212, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 8, false),
        .usage = .{
            .{ .vgprs = 230, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 256, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 256, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 256, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 2, false),
        .usage = .{
            .{ .vgprs = 130, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 4, false),
        .usage = .{
            .{ .vgprs = 215, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 8, false),
        .usage = .{
            .{ .vgprs = 194, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 4, 2, false),
        .usage = .{
            .{ .vgprs = 163, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 146, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 146, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 146, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 4, 4, false),
        .usage = .{
            .{ .vgprs = 239, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 193, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 193, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 193, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 2, false),
        .usage = .{
            .{ .vgprs = 122, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 4, false),
        .usage = .{
            .{ .vgprs = 221, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 8, false),
        .usage = .{
            .{ .vgprs = 236, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 304, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 304, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 304, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 2, false),
        .usage = .{
            .{ .vgprs = 127, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 4, false),
        .usage = .{
            .{ .vgprs = 172, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 224, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 224, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 224, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 8, false),
        .usage = .{
            .{ .vgprs = 253, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 656, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 656, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 42, .lds = 0, .private = 656, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 4, 2, false),
        .usage = .{
            .{ .vgprs = 211, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 4, 4, false),
        .usage = .{
            .{ .vgprs = 256, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 255, .sgprs = 46, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 255, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
            .{ .vgprs = 255, .sgprs = 42, .lds = 0, .private = 128, .bound = 128, .wave = 32 },
        },
    },
};
