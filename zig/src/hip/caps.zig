//! What an RDNA (or Vega) GPU can do, from its exact gfx name; kernels test these capabilities, never the name.
const std = @import("std");

pub const Matrix = enum { none, wmma11, wmma12 };
pub const Act = enum { f16, bf16 };
pub const Generation = enum { gcn5, rdna2, rdna3, rdna3_5, rdna4 };

pub const Caps = struct {
    wave: u8,
    dot2_f16: bool,
    dot2_bf16: bool,
    sdot4: bool,
    sdot8: bool,
    matrix: Matrix,
    act: Act,
    /// dot2 rounds a sum with a -0 lane apart from the same sum with +0, so masked lanes must enter as +0.
    dot2_signed_zero: bool,
    generation: Generation,
    /// The routed tiles take expert pointers (kernels/hip/tiles/plan.hpp TF_EXPERT_PTRS): the expert offload runs here.
    expert_ptrs: bool = false,

    /// The caps of `name` ("gfx1030", "gfx1151:xnack-" ...), or null for a GPU outside the table.
    pub fn of(name: []const u8) ?Caps {
        const end = std.mem.indexOfScalar(u8, name, ':') orelse name.len;
        for (table) |row| {
            for (row.names) |n| if (std.mem.eql(u8, n, name[0..end])) return row.caps;
        }
        return null;
    }

    /// The gfx names of `generation`, in table order.
    pub fn names(generation: Generation) []const []const u8 {
        for (table) |row| if (row.caps.generation == generation) return row.names;
        unreachable;
    }
};

const Row = struct { names: []const []const u8, caps: Caps };

/// One row a generation; every part of it shares the instructions the kernels use.
const table = [_]Row{
    // Vega 10: wave64, no dot2/dot4 (gfx906+), no matrix cores, no bf16; its dot2 is two fused FMAs (common/fdot2.hpp).
    .{ .names = &.{"gfx900"}, .caps = .{ .wave = 64, .dot2_f16 = false, .dot2_bf16 = false, .sdot4 = false, .sdot8 = false, .matrix = .none, .act = .f16, .dot2_signed_zero = false, .generation = .gcn5 } },
    .{ .names = &.{ "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036" }, .caps = .{ .wave = 32, .dot2_f16 = true, .dot2_bf16 = false, .sdot4 = true, .sdot8 = true, .matrix = .none, .act = .f16, .dot2_signed_zero = false, .generation = .rdna2 } },
    .{ .names = &.{ "gfx1100", "gfx1101", "gfx1102", "gfx1103" }, .caps = .{ .wave = 32, .dot2_f16 = true, .dot2_bf16 = true, .sdot4 = true, .sdot8 = true, .matrix = .wmma11, .act = .bf16, .dot2_signed_zero = false, .generation = .rdna3 } },
    .{ .names = &.{ "gfx1150", "gfx1151", "gfx1152", "gfx1153" }, .caps = .{ .wave = 32, .dot2_f16 = true, .dot2_bf16 = true, .sdot4 = true, .sdot8 = true, .matrix = .wmma11, .act = .bf16, .dot2_signed_zero = true, .generation = .rdna3_5 } },
    .{ .names = &.{ "gfx1200", "gfx1201" }, .caps = .{ .wave = 32, .dot2_f16 = true, .dot2_bf16 = true, .sdot4 = true, .sdot8 = true, .matrix = .wmma12, .act = .bf16, .dot2_signed_zero = false, .generation = .rdna4 } },
};

test "each generation gets its instructions" {
    const rdna2 = Caps.of("gfx1030").?;
    try std.testing.expect(rdna2.dot2_f16 and !rdna2.dot2_bf16 and rdna2.matrix == .none and rdna2.act == .f16);
    const rdna3 = Caps.of("gfx1100").?;
    try std.testing.expect(rdna3.dot2_bf16 and rdna3.matrix == .wmma11 and rdna3.act == .bf16);
    try std.testing.expectEqual(Generation.rdna3_5, Caps.of("gfx1151:xnack-").?.generation);
    try std.testing.expectEqual(Matrix.wmma12, Caps.of("gfx1201").?.matrix);
    try std.testing.expect(Caps.of("gfx1152").?.dot2_signed_zero and !Caps.of("gfx1201").?.dot2_signed_zero);
    try std.testing.expect(!rdna2.dot2_signed_zero and !rdna3.dot2_signed_zero);
    for (table) |row| try std.testing.expectEqual(@as(u8, if (row.caps.generation == .gcn5) 64 else 32), row.caps.wave);
}

test "Vega 10 is wave64 with no dot instructions" {
    const vega = Caps.of("gfx900").?;
    try std.testing.expectEqual(Generation.gcn5, vega.generation);
    try std.testing.expect(vega.wave == 64 and !vega.dot2_f16 and !vega.dot2_bf16 and !vega.sdot4 and !vega.sdot8 and vega.matrix == .none and vega.act == .f16);
    try std.testing.expect(!vega.dot2_signed_zero);
    try std.testing.expectEqual(vega.wave, Caps.of("gfx900:xnack-").?.wave);
}

test "names outside the table are refused" {
    for ([_][]const u8{ "gfx1010", "gfx1037", "gfx1104", "gfx1202", "gfx906", "gfx908", "gfx90a", "gfx", "", "sm_90", "gfx11000" }) |name|
        try std.testing.expect(Caps.of(name) == null);
}

test "every name is in exactly one generation" {
    var count: usize = 0;
    for (std.enums.values(Generation)) |g| {
        for (Caps.names(g)) |n| {
            try std.testing.expectEqual(g, Caps.of(n).?.generation);
            count += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 18), count);
}
