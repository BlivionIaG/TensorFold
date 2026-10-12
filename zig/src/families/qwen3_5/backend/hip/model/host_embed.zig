//! The embedding left on the host (an APU whose head is its own matrix): rows dequantized there as tf_embed_rows does.

const std = @import("std");
const quant = @import("core").quant;

pub const Kind = enum { bf16, f16, f32 };

/// An MLX affine embedding's words and group tables as the checkpoint maps them; its pages stay reclaimable.
pub const Table = struct {
    words: []const u8,
    scales: []const u8,
    biases: []const u8,
    /// The tables' dtype: bf16, f16 or f32.
    tables: Kind,
    bits: u8,
    group: u16,
    rows: usize,
    k: usize,

    /// The table of `p` when its bytes are the checkpoint's own (not a converted copy) and its tables one dtype.
    pub fn of(p: quant.Host) ?Table {
        const h = switch (p) {
            .mlx => |x| x,
            .dense => return null,
        };
        if (h.scales.dtype != h.biases.dtype) return null;
        const kind: Kind = switch (h.scales.dtype) {
            .bf16 => .bf16,
            .f16 => .f16,
            .f32 => .f32,
            else => return null,
        };
        const rows: usize = @intCast(h.words.shape[0]);
        const k = @as(usize, @intCast(h.words.shape[1])) * 32 / h.bits;
        if (rows == 0 or k % h.group != 0) return null;
        return .{ .words = h.words.bytes, .scales = h.scales.bytes, .biases = h.biases.bytes, .tables = kind, .bits = h.bits, .group = h.group, .rows = rows, .k = k };
    }

    fn word(t: *const Table, i: usize) u32 {
        return std.mem.readInt(u32, t.words[4 * i ..][0..4], .little);
    }

    fn entry(t: *const Table, bytes: []const u8, i: usize) f32 {
        return switch (t.tables) {
            .f32 => @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)),
            .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)))),
            .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16),
        };
    }

    /// Code `c` of row `id`, one crossing two words included (code_at).
    fn code(t: *const Table, id: usize, c: usize) u32 {
        const words_row = t.k * t.bits / 32;
        const bit = c * t.bits;
        const at = id * words_row + bit / 32;
        const shift: u6 = @intCast(bit % 32);
        var v: u64 = t.word(at) >> @intCast(shift);
        if (@as(usize, shift) + t.bits > 32 and bit / 32 + 1 < words_row) v |= @as(u64, t.word(at + 1)) << @intCast(32 - @as(usize, shift));
        return @intCast(v & ((@as(u64, 1) << @intCast(t.bits)) - 1));
    }

    /// Rows `ids` into `out` in fp16 or bf16: code * scale, then + bias, in fp32, rounded once, as the device does.
    pub fn gather(t: *const Table, ids: []const u32, out: []u8, bf16: bool) void {
        const groups = t.k / t.group;
        for (ids, 0..) |raw, r| {
            const id: usize = if (raw < t.rows) raw else 0; // the device would read past the table
            for (0..t.k) |c| {
                const s = t.entry(t.scales, id * groups + c / t.group);
                const b = t.entry(t.biases, id * groups + c / t.group);
                var y: f32 = @as(f32, @floatFromInt(t.code(id, c))) * s;
                y = y + b;
                const at = 2 * (r * t.k + c);
                std.mem.writeInt(u16, out[at..][0..2], if (bf16) toBf16(y) else @bitCast(@as(f16, @floatCast(y))), .little);
            }
        }
    }
};

fn toBf16(v: f32) u16 {
    if (v != v) return 0x7fc0;
    const u: u32 = @bitCast(v);
    return @intCast((u +% (((u >> 16) & 1) + 0x7fff)) >> 16);
}

test "host rows are the device kernel's: 4- and 8-bit codes, bf16 and f16 tables, fp16 and bf16 out" {
    // one row of 8 codes at 4 bits in one word, groups of 4: codes 1..8
    const words = [_]u8{ 0x21, 0x43, 0x65, 0x87 };
    var scales: [4]u8 = undefined;
    var biases: [4]u8 = undefined;
    std.mem.writeInt(u16, scales[0..2], @bitCast(@as(f16, 0.5)), .little);
    std.mem.writeInt(u16, scales[2..4], @bitCast(@as(f16, 0.25)), .little);
    std.mem.writeInt(u16, biases[0..2], @bitCast(@as(f16, -1.0)), .little);
    std.mem.writeInt(u16, biases[2..4], @bitCast(@as(f16, 0.125)), .little);
    const t: Table = .{ .words = &words, .scales = &scales, .biases = &biases, .tables = .f16, .bits = 4, .group = 4, .rows = 1, .k = 8 };
    var out: [16]u8 = undefined;
    t.gather(&.{0}, &out, false);
    const want = [_]f16{ -0.5, 0.0, 0.5, 1.0, 1.375, 1.625, 1.875, 2.125 };
    for (want, 0..) |w, i| try std.testing.expectEqual(w, @as(f16, @bitCast(std.mem.readInt(u16, out[2 * i ..][0..2], .little))));
    t.gather(&.{0}, &out, true);
    try std.testing.expectEqual(@as(u16, 0xbf00), std.mem.readInt(u16, out[0..2], .little));
    // 8 bits, bf16 tables: code 200 times 0.5 plus 1
    const w8 = [_]u8{ 200, 0, 0, 0 };
    const s8 = [_]u8{ 0x00, 0x3f };
    const b8 = [_]u8{ 0x80, 0x3f };
    const t8: Table = .{ .words = &w8, .scales = &s8, .biases = &b8, .tables = .bf16, .bits = 8, .group = 4, .rows = 1, .k = 4 };
    var o8: [8]u8 = undefined;
    t8.gather(&.{0}, &o8, false);
    try std.testing.expectEqual(@as(f16, 101.0), @as(f16, @bitCast(std.mem.readInt(u16, o8[0..2], .little))));
}
