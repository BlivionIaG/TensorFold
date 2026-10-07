//! What a GPU can do, from its gfx name; a GPU outside the table is refused rather than guessed.

const std = @import("std");

pub const Matrix = enum { none, wmma11, wmma12 };

pub const Caps = struct {
    gfx: u32,
    wave: u8,
    dot2_f16: bool,
    dot2_bf16: bool,
    sdot4: bool,
    sdot8: bool,
    matrix: Matrix,
    lds_bytes: u32 = 64 * 1024,

    /// The caps of `name` ("gfx1030", "gfx1151", "gfx1100:xnack-" ...), or null for a GPU outside the table.
    pub fn of(name: []const u8) ?Caps {
        const gfx = parse(name) orelse return null;
        for (table) |row| if (row.match(gfx)) return row.caps(gfx);
        return null;
    }
};

fn parse(name: []const u8) ?u32 {
    if (!std.mem.startsWith(u8, name, "gfx")) return null;
    const end = std.mem.indexOfScalar(u8, name, ':') orelse name.len;
    return std.fmt.parseInt(u32, name[3..end], 16) catch null;
}

const Row = struct {
    lo: u32,
    hi: u32,
    dot2_bf16: bool,
    matrix: Matrix,

    fn match(r: Row, gfx: u32) bool {
        return gfx >= r.lo and gfx <= r.hi;
    }

    fn caps(r: Row, gfx: u32) Caps {
        return .{ .gfx = gfx, .wave = 32, .dot2_f16 = true, .dot2_bf16 = r.dot2_bf16, .sdot4 = true, .sdot8 = true, .matrix = r.matrix };
    }
};

/// gfx ids in hex as the names spell them (gfx1030 = 0x1030): RDNA2, RDNA3, RDNA3.5 and RDNA4 parts.
const table = [_]Row{
    .{ .lo = 0x1030, .hi = 0x1036, .dot2_bf16 = false, .matrix = .none },
    .{ .lo = 0x1100, .hi = 0x1103, .dot2_bf16 = true, .matrix = .wmma11 },
    .{ .lo = 0x1150, .hi = 0x1153, .dot2_bf16 = true, .matrix = .wmma11 },
    .{ .lo = 0x1200, .hi = 0x1201, .dot2_bf16 = true, .matrix = .wmma12 },
};

test "the table gives each generation its instructions" {
    const rdna2 = Caps.of("gfx1030").?;
    try std.testing.expect(rdna2.dot2_f16 and !rdna2.dot2_bf16 and rdna2.matrix == .none and rdna2.wave == 32);
    try std.testing.expect(Caps.of("gfx1100").?.dot2_bf16 and Caps.of("gfx1100").?.matrix == .wmma11);
    try std.testing.expectEqual(Matrix.wmma11, Caps.of("gfx1151").?.matrix);
    try std.testing.expectEqual(Matrix.wmma12, Caps.of("gfx1201").?.matrix);
    try std.testing.expectEqual(@as(u32, 0x1100), Caps.of("gfx1100:xnack-").?.gfx);
}

test "every RDNA2, RDNA3 and RDNA3.5 part is listed, and its neighbours outside them are refused" {
    for ([_][]const u8{ "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036" }) |n| try std.testing.expect(Caps.of(n) != null);
    for ([_][]const u8{ "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1153" }) |n| try std.testing.expect(Caps.of(n) != null);
    for ([_][]const u8{ "gfx1010", "gfx1037", "gfx1104", "gfx1154", "gfx906", "gfx90a", "sm_90", "gfx" }) |n| try std.testing.expect(Caps.of(n) == null);
}
