//! ExLlamaV3 n-gram row codec. Not the EXL3 tile bitstream.
//! One fp16 scale word, then 160 tail-biting mul1 values of K bits, K in 2..8.
//! Bit 0 of a state is its least significant bit, matching v0.6.6 `_ple_rows`.

const std = @import("std");
// This file is its own `zig test` root, so the import cannot spell `../../core`.
const exl3_format = @import("exl3_format.zig");
const shape = @import("exl3_shape.zig");

const values = shape.ngram_dh;

/// Int16 words in one row at `bits`: the fp16 scale, then 160 values. `bits` outside 2..8 is `NgramWidth`.
pub fn wordsPerRow(bits: usize) error{NgramWidth}!usize {
    if (bits < 2 or bits > 8) return error.NgramWidth;
    return 1 + values * bits / 16;
}

/// Mul1 state of value `value_index` in one row. `bits` is 2..8 and `words` has `wordsPerRow` length.
/// Bit m comes from position `value_index - m / bits` (add 160 when negative), then bit `m % bits` of that group.
pub fn state(words: []const i16, bits: usize, value_index: usize) u16 {
    std.debug.assert(bits >= 2 and bits <= 8);
    std.debug.assert(value_index < values);
    std.debug.assert(words.len == 1 + values * bits / 16);
    var acc: u16 = 0;
    for (0..16) |m| {
        var pos: isize = @as(isize, @intCast(value_index)) - @as(isize, @intCast(m / bits));
        if (pos < 0) pos += @as(isize, @intCast(values));
        const sb: usize = @as(usize, @intCast(pos)) * bits + (m % bits);
        const word: u16 = @bitCast(words[1 + (sb >> 4)]);
        const within: u4 = @intCast(sb & 15);
        const bit: u16 = (word >> within) & 1;
        acc |= bit << @as(u4, @intCast(m));
    }
    return acc;
}

/// Decode one row into `out`. `bias` and `out` each have length 160.
/// `out[i]` is one f16 rounding of `f32(codebook) * f32(scale) + f32(bias[i])`. The codebook value is already f16.
pub fn decodeRow(words: []const i16, bits: usize, bias: []const f16, out: []f16) error{NgramWidth}!void {
    const width = try wordsPerRow(bits);
    if (words.len != width or bias.len != values or out.len != values) return error.NgramWidth;
    const scale: f16 = @bitCast(words[0]);
    for (0..values) |i| {
        const cb = exl3_format.codebookValue(.mul1, state(words, bits, i));
        const sum: f32 = @as(f32, @floatCast(cb)) * @as(f32, @floatCast(scale)) + @as(f32, @floatCast(bias[i]));
        out[i] = @floatCast(sum);
    }
}

fn fillIndex(words: []i16) void {
    for (words, 0..) |*w, i| w.* = @bitCast(@as(u16, @intCast(i)));
}

/// Second copy of the row walk, so `state` is checked against the spec rather than itself.
fn specState(words: []const i16, bits: usize, value_index: usize) u16 {
    var acc: u16 = 0;
    for (0..16) |m| {
        var pos: isize = @as(isize, @intCast(value_index)) - @as(isize, @intCast(m / bits));
        if (pos < 0) pos += @as(isize, @intCast(values));
        const sb: usize = @as(usize, @intCast(pos)) * bits + (m % bits);
        const word: u16 = @bitCast(words[1 + (sb >> 4)]);
        const within: u4 = @intCast(sb & 15);
        const bit: u16 = (word >> within) & 1;
        acc |= bit << @as(u4, @intCast(m));
    }
    return acc;
}

test "wordsPerRow is one scale word plus 160 K-bit values" {
    try std.testing.expectEqual(@as(usize, 160), values);
    try std.testing.expectEqual(@as(usize, 21), try wordsPerRow(2));
    try std.testing.expectEqual(@as(usize, 31), try wordsPerRow(3));
    try std.testing.expectEqual(@as(usize, 51), try wordsPerRow(5));
    try std.testing.expectEqual(@as(usize, 51), try wordsPerRow(shape.ngram_bits));
    try std.testing.expectEqual(@as(usize, 81), try wordsPerRow(8));
    try std.testing.expectError(error.NgramWidth, wordsPerRow(0));
    try std.testing.expectError(error.NgramWidth, wordsPerRow(1));
    try std.testing.expectError(error.NgramWidth, wordsPerRow(9));
}

test "patterned rows at 2, 3, 5 and 8 bits match an independent tail-biting walk" {
    var storage: [81]i16 = @splat(0);
    var bias: [160]f16 = @splat(0);
    var out: [160]f16 = @splat(0);
    for ([_]usize{ 2, 3, 5, 8 }) |bits| {
        const width = try wordsPerRow(bits);
        const words = storage[0..width];
        fillIndex(words);
        for (0..values) |i| try std.testing.expectEqual(specState(words, bits, i), state(words, bits, i));
        words[0] = @bitCast(@as(f16, 1.0));
        try decodeRow(words, bits, &bias, &out);
        for (out, 0..) |y, i| {
            const cb = exl3_format.codebookValue(.mul1, state(words, bits, i));
            try std.testing.expectEqual(@as(u16, @bitCast(cb)), @as(u16, @bitCast(y)));
        }
    }
    fillIndex(&storage);
    try std.testing.expectEqual(@as(u16, 0x5001), state(storage[0..21], 2, 0));
    try std.testing.expectEqual(@as(u16, 0x9001), state(storage[0..31], 3, 0));
    try std.testing.expectEqual(@as(u16, 0x8001), state(storage[0..51], 5, 0));
    try std.testing.expectEqual(@as(u16, 0x0020), state(storage[0..51], 5, 1));
    try std.testing.expectEqual(@as(u16, 0x0400), state(storage[0..51], 5, 2));
    try std.testing.expectEqual(@as(u16, 0x0001), state(storage[0..81], 8, 0));
}

test "value 0 at 5 bits reads a wrapped tail bit" {
    const bits: usize = 5;
    var words: [51]i16 = @splat(0);
    const m: usize = 5;
    const raw: isize = 0 - @as(isize, @intCast(m / bits));
    try std.testing.expect(raw < 0);
    const pos: usize = @intCast(raw + @as(isize, @intCast(values)));
    try std.testing.expectEqual(@as(usize, 159), pos);
    const sb = pos * bits + (m % bits);
    const word_index = 1 + (sb >> 4);
    const within: u4 = @intCast(sb & 15);
    try std.testing.expectEqual(@as(usize, 50), word_index);
    try std.testing.expectEqual(@as(u4, 11), within);
    words[word_index] = @bitCast(@as(u16, 1) << within);

    const direct: u16 = (@as(u16, @bitCast(words[word_index])) >> within) & 1;
    const got: u16 = (state(&words, bits, 0) >> @as(u4, @intCast(m))) & 1;
    try std.testing.expectEqual(@as(u16, 1), direct);
    try std.testing.expectEqual(direct, got);
    try std.testing.expectEqual(@as(u16, 1) << @as(u4, @intCast(m)), state(&words, bits, 0));

    // Clamping the negative position to 0 reads the head of the row, which this pattern leaves clear.
    const head: u16 = @bitCast(words[1]);
    const nowrap: u16 = (head >> @as(u4, @intCast(m % bits))) & 1;
    try std.testing.expectEqual(@as(u16, 0), nowrap);
    try std.testing.expect(got != nowrap);
}

test "scale and bias round in f32, then once to f16" {
    const bits: usize = 5;
    var words: [51]i16 = @splat(0);
    // codebook(0) is f16 0xC2E8. Times f16 0x3C01, plus f16 0x1400, is f16 0xC2E9.
    // Rounding the product to f16 before the add yields 0xC2EA.
    const scale: f16 = @bitCast(@as(u16, 0x3C01));
    words[0] = @bitCast(scale);
    var bias: [160]f16 = @splat(0);
    bias[0] = @bitCast(@as(u16, 0x1400));
    for (bias[1..], 1..) |*b, i| b.* = @bitCast(@as(u16, @intCast(0x2000 + i)));
    var out: [160]f16 = @splat(0);
    try decodeRow(&words, bits, &bias, &out);

    try std.testing.expectEqual(@as(u16, 0), state(&words, bits, 0));
    try std.testing.expectEqual(@as(u16, 0xC2E8), @as(u16, @bitCast(exl3_format.codebookValue(.mul1, 0))));
    try std.testing.expectEqual(@as(u16, 0xC2E9), @as(u16, @bitCast(out[0])));
    for (out, 0..) |y, i| {
        const cb = exl3_format.codebookValue(.mul1, state(&words, bits, i));
        const want: f16 = @floatCast(@as(f32, @floatCast(cb)) * @as(f32, @floatCast(scale)) + @as(f32, @floatCast(bias[i])));
        try std.testing.expectEqual(@as(u16, @bitCast(want)), @as(u16, @bitCast(y)));
    }
    const cb0 = exl3_format.codebookValue(.mul1, 0);
    const early: f16 = @floatCast(@as(f32, @floatCast(cb0)) * @as(f32, @floatCast(scale)));
    const early_sum: f16 = @floatCast(@as(f32, @floatCast(early)) + @as(f32, @floatCast(bias[0])));
    try std.testing.expectEqual(@as(u16, 0xC2EA), @as(u16, @bitCast(early_sum)));
    try std.testing.expect(early_sum != out[0]);
}

test "a row that is not one scale plus 160 values is refused" {
    var words: [51]i16 = @splat(0);
    var bias: [160]f16 = @splat(0);
    var out: [160]f16 = @splat(0);
    try std.testing.expectError(error.NgramWidth, decodeRow(words[0..50], 5, &bias, &out));
    try std.testing.expectError(error.NgramWidth, decodeRow(words[0..0], 5, &bias, &out));
    var long_words: [52]i16 = @splat(0);
    try std.testing.expectError(error.NgramWidth, decodeRow(&long_words, 5, &bias, &out));
    try std.testing.expectError(error.NgramWidth, decodeRow(&words, 2, &bias, &out));
    try std.testing.expectError(error.NgramWidth, decodeRow(&words, 5, bias[0..159], &out));
    try std.testing.expectError(error.NgramWidth, decodeRow(&words, 5, &bias, out[0..159]));
    var bias_long: [161]f16 = @splat(0);
    var out_long: [161]f16 = @splat(0);
    try std.testing.expectError(error.NgramWidth, decodeRow(&words, 5, &bias_long, &out));
    try std.testing.expectError(error.NgramWidth, decodeRow(&words, 5, &bias, &out_long));
    try std.testing.expectError(error.NgramWidth, decodeRow(&words, 0, &bias, &out));
    try std.testing.expectError(error.NgramWidth, decodeRow(&words, 1, &bias, &out));
    try std.testing.expectError(error.NgramWidth, decodeRow(&words, 9, &bias, &out));
}
