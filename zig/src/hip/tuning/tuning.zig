//! The registry's costs a GPU: the .zon tables beside this file, read at build time. No autotuning at run time, so ranks
//! and runs choose alike. A backend names which table a GPU takes.

const std = @import("std");

/// One entry's cost and the products it takes (the file's header says what each field means).
pub const Row = struct {
    id: []const u8,
    cost: f32,
    rows: u32 = 0,
    rows_matrix: u32 = 0,
    min_rows: u32 = 0,
    items: u32 = 0,
    plain: bool = true,
    routed: bool = true,
};

pub const Table = struct { gfx: []const u8, rows: []const Row };

fn build(comptime zon: anytype) Table {
    const rows = comptime blk: {
        var out: [zon.entries.len]Row = undefined;
        for (zon.entries, 0..) |e, i| {
            var row: Row = .{ .id = e.id, .cost = e.cost };
            for (@typeInfo(Row).@"struct".field_names) |name| {
                if (std.mem.eql(u8, name, "id") or std.mem.eql(u8, name, "cost")) continue;
                if (@hasField(@TypeOf(e), name)) @field(row, name) = @field(e, name);
            }
            out[i] = row;
        }
        break :blk out;
    };
    return .{ .gfx = zon.gfx, .rows = &rows };
}

pub const gfx1030 = build(@import("gfx1030.zon"));
pub const gfx1100 = build(@import("gfx1100.zon"));

pub fn find(t: *const Table, id: []const u8) ?Row {
    for (t.rows) |r| if (std.mem.eql(u8, r.id, id)) return r;
    return null;
}

test "the tables name each entry once" {
    for ([_]*const Table{ &gfx1030, &gfx1100 }) |t| {
        for (t.rows, 0..) |r, i| {
            for (t.rows[i + 1 ..]) |o| try std.testing.expect(!std.mem.eql(u8, r.id, o.id));
        }
    }
    try std.testing.expect(find(&gfx1100, "mlx.project.decode.matrix") != null);
    try std.testing.expect(find(&gfx1030, "mlx.project.decode.matrix") == null);
}
