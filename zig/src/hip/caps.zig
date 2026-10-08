//! What a GPU can do, from its gfx name; kernels and the registry test these capabilities, never the GPU's name.

const std = @import("std");

pub const Matrix = enum { none, wmma11, wmma12 };
pub const Act = enum { f16, bf16 };
/// The code objects built for a GPU: the build compiles one set a family.
pub const Family = enum { gcn5, rdna2, rdna3 };

pub const Caps = struct {
    gfx: u32,
    wave: u8,
    dot2_f16: bool,
    dot2_bf16: bool,
    sdot4: bool,
    sdot8: bool,
    matrix: Matrix,
    act: Act,
    family: Family,
    lds_bytes: u32 = 64 * 1024,

    /// The caps of `name` ("gfx1030", "gfx1100", "gfx90a:sramecc+:xnack-" ...), or null for a GPU outside the table.
    pub fn of(name: []const u8) ?Caps {
        const gfx = parse(name) orelse return null;
        for (table) |row| if (row.match(gfx)) return row.caps(gfx);
        return null;
    }

    /// The word every tp rank compares at join: a group refuses mixed GPUs.
    pub fn id(c: Caps) u32 {
        return c.gfx;
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
    wave: u8,
    dot2_f16: bool,
    dot2_bf16: bool,
    sdot: bool,
    matrix: Matrix,
    act: Act,
    family: Family,

    fn match(r: Row, gfx: u32) bool {
        return gfx >= r.lo and gfx <= r.hi;
    }

    fn caps(r: Row, gfx: u32) Caps {
        return .{ .gfx = gfx, .wave = r.wave, .dot2_f16 = r.dot2_f16, .dot2_bf16 = r.dot2_bf16, .sdot4 = r.sdot, .sdot8 = r.sdot, .matrix = r.matrix, .act = r.act, .family = r.family };
    }
};

/// gfx ids in hex as the names spell them (gfx1030 = 0x1030); a row covers one generation's parts.
const table = [_]Row{
    .{ .lo = 0x900, .hi = 0x900, .wave = 64, .dot2_f16 = false, .dot2_bf16 = false, .sdot = false, .matrix = .none, .act = .f16, .family = .gcn5 },
    .{ .lo = 0x906, .hi = 0x906, .wave = 64, .dot2_f16 = true, .dot2_bf16 = false, .sdot = true, .matrix = .none, .act = .f16, .family = .gcn5 },
    .{ .lo = 0x1030, .hi = 0x1036, .wave = 32, .dot2_f16 = true, .dot2_bf16 = false, .sdot = true, .matrix = .none, .act = .f16, .family = .rdna2 },
    .{ .lo = 0x1100, .hi = 0x1103, .wave = 32, .dot2_f16 = true, .dot2_bf16 = true, .sdot = true, .matrix = .wmma11, .act = .bf16, .family = .rdna3 },
    .{ .lo = 0x1150, .hi = 0x1153, .wave = 32, .dot2_f16 = true, .dot2_bf16 = true, .sdot = true, .matrix = .wmma11, .act = .bf16, .family = .rdna3 },
    .{ .lo = 0x1200, .hi = 0x1201, .wave = 32, .dot2_f16 = true, .dot2_bf16 = true, .sdot = true, .matrix = .wmma12, .act = .bf16, .family = .rdna3 },
};

test "the table gives each generation its instructions" {
    const v620 = Caps.of("gfx1030").?;
    try std.testing.expect(v620.dot2_f16 and !v620.dot2_bf16 and v620.matrix == .none and v620.act == .f16 and v620.wave == 32);
    const w7800 = Caps.of("gfx1100").?;
    try std.testing.expect(w7800.dot2_bf16 and w7800.matrix == .wmma11 and w7800.family == .rdna3);
    try std.testing.expectEqual(Matrix.wmma12, Caps.of("gfx1201").?.matrix);
    try std.testing.expectEqual(@as(u8, 64), Caps.of("gfx906:sramecc+:xnack-").?.wave);
    try std.testing.expect(!Caps.of("gfx900").?.dot2_f16);
    try std.testing.expect(Caps.of("gfx90a") == null);
    try std.testing.expect(Caps.of("sm_90") == null);
    try std.testing.expect(Caps.of("gfx1151").?.id() != Caps.of("gfx1100").?.id());
}
