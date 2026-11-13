//! Gate 5's records of the split offload's boundary rows (split/rows.hip): window rows to and from the activation kind.
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

const U = limits.HipUsage;

/// A split/rows.hip launch and its (VGPRs, SGPRs, LDS bytes) on each chip, wave32 with no scratch.
fn rows(comptime symbol: []const u8, threads: u32, u: [limits.rdna.len][3]u32) Record {
    var usage: [limits.rdna.len]U = undefined;
    for (&usage, u) |*x, v| x.* = .{ .vgprs = v[0], .sgprs = v[1], .lds = v[2], .bound = threads };
    return .{ .launch = .{ .name = "split rows " ++ symbol, .symbol = symbol, .image = "split_rows", .threads = threads }, .usage = usage };
}

pub const records = [_]Record{
    rows("tf_split_import_row", 256, .{ .{ 8, 12, 0 }, .{ 8, 18, 0 }, .{ 8, 8, 0 }, .{ 8, 8, 0 } }),
    rows("tf_split_export_row", 256, .{ .{ 4, 12, 0 }, .{ 5, 18, 0 }, .{ 4, 9, 0 }, .{ 4, 9, 0 } }),
    rows("tf_split_consume_stamped", 1024, .{ .{ 11, 25, 4 }, .{ 11, 25, 4 }, .{ 11, 25, 4 }, .{ 11, 25, 4 } }),
};
