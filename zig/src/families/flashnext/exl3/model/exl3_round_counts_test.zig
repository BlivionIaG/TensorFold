const std = @import("std");
const round = @import("exl3_round.zig");
const offload = @import("../offload/exl3_offload.zig");

test "round frequencies saturate history, break saturated ties and leave refused work unobserved" {
    var slots: [2]u16 = undefined;
    var states: [2]offload.Line = undefined;
    var stamps: [2]u32 = undefined;
    var layers: [2]u8 = undefined;
    var v = offload.Vram.init(&slots, &states, &stamps, &layers);
    var counts: [offload.shape.layers][offload.shape.experts]u32 = @splat(@splat(0));
    counts[3][3] = std.math.maxInt(u32) - 1;
    counts[3][7] = std.math.maxInt(u32);
    counts[4][3] = 17;
    var r = try round.Round.init(std.testing.allocator, slots.len, 8);
    defer r.deinit(std.testing.allocator);
    try round.plan(&v, 3, &.{ 7, 7, 3, 3, 3, 512, -1 }, &counts, 1, 1, .{}, &r);
    try std.testing.expectEqual(std.math.maxInt(u32), counts[3][3]);
    try std.testing.expectEqual(std.math.maxInt(u32), counts[3][7]);
    try std.testing.expectEqual(@as(u32, 17), counts[4][3]);
    try std.testing.expectEqualSlices(u16, &.{ 3, 7 }, r.ids[0..r.n_ids]);
    try std.testing.expectEqual(@as(u32, 5), r.miss_slots);
    try std.testing.expectEqual(@as(u32, 1), r.shared_slots);
    states = @splat(.inflight);
    const previous_slots = slots;
    const previous_states = states;
    try std.testing.expectError(error.Busy, round.plan(&v, 3, &.{9}, &counts, 2, 2, .{}, &r));
    try std.testing.expectEqual(@as(u32, 0), counts[3][9]);
    try std.testing.expectError(error.Length, round.plan(&v, 3, &.{ 9, 513 }, &counts, 3, 3, .{}, &r));
    try std.testing.expectEqual(@as(u32, 0), counts[3][9]);
    try std.testing.expectEqualDeep(previous_slots, slots);
    try std.testing.expectEqualDeep(previous_states, states);
}
