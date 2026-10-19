const std = @import("std");
const family = @import("family");
const c = family.spec;
fn depth(s: c.Shape) usize { return s.k / 2 + s.k / 32 + 8; }
test "original two bit fixture is weak at actual PR31 runner seed" {
    const a = std.testing.allocator;
    var f = try family.affine.Affine.init(a, .{ .rows = 1, .n = 4096, .k = 2048 }, 2, 32);
    defer f.deinit();
    const seed = 0x6b63 + 0x9e37 * 15;
    try f.fill(.stress, seed);
    // Every two-bit column is identical; preserve actual RNG consumption, then inspect the first output only.
    f.shape.n = 1;
    var want: [1]f64 = undefined;
    var mag: [1]f64 = undefined;
    var got: [1]f64 = undefined;
    try f.reference(&want, &mag);
    const spec: c.Spec = .{ .name = "stream", .accum = .f32, .depth = depth, .out = .f32, .served = &.{} };
    var margin: f64 = std.math.inf(f64);
    for ([_]family.rows.Order{ .sequential, .pairwise, .strided }) |order| {
        try f.simulate(.f16, order, &got);
        const miss = if (std.math.isFinite(got[0])) @abs(got[0] - want[0]) / c.boundExact(spec, want[0])
            else std.math.inf(f64);
        margin = @min(margin, miss);
        std.debug.print("OLD index15 group32 {t}: want{d} got{d} miss{d}x\n", .{ order, want[0], got[0], miss });
    }
    try std.testing.expectEqual(@as(f64, 0), margin);
    std.debug.print("REPRODUCED original fixture fp16 minimum margin0x; required100x\n", .{});
}
